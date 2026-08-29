"""Stage 51 implementation -- ESPN CFB athlete CORE records (identity + bio).

Core v2 ``/athletes/{id}``: the canonical identity record. Athlete-keyed and
FLAT, because a core record is per-athlete state, not per-season -- the endpoint
takes no season param at all.

**Why this exists separately from stage 05.** The player_stats payload carries
statistics and nothing that identifies who they belong to; the only carrier of
the athlete id there is the FILENAME. Identity and bio have to come from
somewhere, and one request per athlete against this endpoint is the cheapest
complete source. Same rationale as the MBB sibling
(``espn_mbb_09_player_core_scrape.py``), which is worth reading -- most of what
is known about this endpoint's traps was learned there.

Everything useful is inline in one response, verified live 2026-08-29 on athlete
4426339 (Spencer Rattler): ``fullName``, ``height`` (numeric inches) +
``displayHeight``, ``weight``, ``jersey``, ``position.displayName``,
``dateOfBirth``, ``birthPlace``, ``active``, ``links``.

**Do NOT hydrate ``team`` or ``college``.** Both come back as ``{"$ref": url}``
and nothing more (confirmed on the live probe). Following them triples the
request count across the entire athlete universe for ids that are already
embedded in the ref URL and can be parsed out downstream for free.

**``team`` is the athlete's CURRENT team**, not their team in any past season --
its ref is literally ``/seasons/{current}/teams/{id}``. Season-team belongs to
stage 05's ``statistics[].teamId`` or to the game rosters. Bio is likewise a
CURRENT snapshot that ESPN overwrites in place: era-correct height, weight or
jersey is not obtainable from this endpoint, or from any other ESPN endpoint.
Do not present a banked record as the athlete's state in a historical season.

Athlete ids are stage 04's game rosters, shared with stage 05 rather than
re-derived -- see :func:`cfb_raw_scrape.scrape_cfb_player_stats.athletes_by_season`.
ESPN's Core v2 ``/seasons/{y}/athletes`` index is NOT a shortcut: the MBB
sibling measured it at 100 ``{"$ref"}`` links per page, 78 pages for a single
season, hydrating nothing. Its only advantage is listing roster-only players who
never appeared in a game.

**Cadence: monthly, not daily.** A core record is per-athlete state that changes
on a roster cycle, not a game cycle. It shares the out-of-band cadence with
stage 50 recruits and is deliberately absent from ``daily_cfb_scraper.sh``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    most_recent_cfb_season,
    run_pool,
    write_json_atomic,
)
from cfb_raw_scrape.scrape_cfb_player_stats import (
    MIN_SEASON,
    athletes_by_season,
    stale_athletes,
)
from sportsdataverse.cfb import espn_cfb_player_core

DATASET = "player_core"

__all__ = [
    "DATASET",
    "MIN_SEASON",
    "is_complete",
    "main",
    "out_path",
    "write_one",
]


def out_path(athlete_id: int | str) -> str:
    return f"cfb/{DATASET}/json/{athlete_id}.json"


def _is_active(payload: dict) -> bool:
    """Whether ESPN still considers this athlete active.

    The record of an ACTIVE athlete is a moving target -- team, jersey and
    weight all change -- while a former player's is effectively frozen. This
    catches the case the season-staleness rule alone misses: a transfer who has
    not yet appeared in a game this season is absent from the game rosters, so
    no live season flags them, but their team ref has already changed.
    """
    return bool(payload.get("active"))


def is_valid(payload: dict) -> bool:
    """A core record must at least identify somebody.

    ESPN returns an envelope either way, so "we got JSON back" is not "we got a
    record" -- the id and a name are the minimum that makes the file useful.
    """
    return bool(payload.get("id")) and bool(
        payload.get("fullName") or payload.get("displayName")
    )


def is_complete(athlete_id: int | str, stale: set[str] | None = None) -> bool:
    """Banked, identifiable, not in a live season, and no longer active."""
    if stale and str(athlete_id) in stale:
        return False
    p = Path(out_path(athlete_id))
    if not p.is_file():
        return False
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return is_valid(payload) and not _is_active(payload)


def write_one(athlete_id: int | str, logger) -> str:
    """Fetch and bank one athlete's core record. Refuses an unidentifiable one."""
    payload = espn_cfb_player_core(athlete_id=athlete_id, return_parsed=False)
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    if not is_valid(payload):
        raise RuntimeError(
            f"{DATASET} athlete {athlete_id}: payload carries no id/name -- "
            "refusing to bank an unidentifiable record"
        )
    write_json_atomic(payload, out_path(athlete_id))
    return str(payload.get("fullName") or payload.get("displayName"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args(argv)
    start = max(args.start_year, MIN_SEASON)
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")

    logger = get_logger(f"cfb_{DATASET}", end)
    try:
        per = athletes_by_season(start, end)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    stale = stale_athletes(per)
    ids = sorted(set().union(*per.values()) if per else set(), key=int)
    todo = [a for a in ids if rescrape or not is_complete(a, stale)]
    logger.info(
        "%s %s-%s: %d athletes, %d to fetch (%d banked, inactive and final)",
        DATASET,
        start,
        end,
        len(ids),
        len(todo),
        len(ids) - len(todo),
    )
    failures: list[str] = []

    def _one(aid, _log=logger, _f=failures):
        try:
            write_one(aid, _log)
        except Exception as exc:  # noqa: BLE001 -- one athlete must not stop the sweep
            _log.error("%s", exc)
            _f.append(str(aid))

    run_pool(_one, todo, kind="thread", desc=f"{DATASET} {start}-{end}")
    if failures:
        logger.error(
            "%s: %d athlete(s) failed: %s", DATASET, len(failures), failures[:10]
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
