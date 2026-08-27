"""Scrape the per-season ESPN team + conference reference into the raw store.

Season-keyed, not game-keyed, so this does not use the schedule master or the
game-oriented helpers in ``_cfb_raw_utils``.

WHAT IT CAPTURES (per season, one bundle file)
----------------------------------------------
* ``divisions``       the group-80 (FBS) and group-81 (FCS) team-ref lists, kept
                      as ordered id lists -- membership in one of those two lists
                      is the ONLY authoritative source of a team's division.
* ``group_children``  the raw ``/groups/{80,81}/children`` payloads (conference refs).
* ``conferences``     one raw conference payload per referenced group id.
* ``teams``           one raw SEASON-SCOPED team payload per team id.

Everything is persisted UNPARSED. The season bundle is a container of verbatim
ESPN payloads, not a reshape -- the compiler in ``cfbfastR-cfb-data`` does the
tidying, so a parser fix is re-applied offline instead of costing a re-scrape.

WHY ONE FILE PER SEASON rather than one per team: the consumer reads this tree
over HTTP from raw.githubusercontent.com. Per-team files would make the compiler
issue ~7,300 requests for a full backfill instead of ~26.

Layout::

    cfb/teams/json/{season}.json    season bundle; ``incomplete == []`` means done
    cfb/reference/positions.json    league position reference (season-independent)

Two ESPN gotchas are load-bearing here:

* A team's own ``groups`` ``$ref`` points at **season type 3**
  (``/seasons/{s}/types/3/groups/{g}``) while the group-80 children live under
  type 2. The group *id* is the same in both, so conference ids are collected as
  a union over both sources and each is fetched through its own ref's season type.
* The generated ``espn_cfb_positions`` wrapper hardcodes ``params={}``, so the
  74-item position list cannot be paged through the wrapper (it serves 25/page).
  The index page is fetched through sdv-py's ``dl_utils.download`` chokepoint
  with ``limit=200``; every item is then resolved through the SDK wrapper.

Pacing knobs (ESPN Core v2 403s under aggressive parallelism):
``CFB_TEAMS_WORKERS`` (4), ``CFB_TEAMS_RETRIES`` (4), ``CFB_TEAMS_BACKOFF`` (2.0s).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import sportsdataverse as sdv

from _cfb_raw_utils import get_logger, most_recent_cfb_season, run_pool, write_json_atomic

DATASET = "teams"
FIRST_SEASON = 2001
#: group 80 = FBS (Division I-A), group 81 = FCS (Division I-AA).
DIVISION_GROUPS = {"fbs": 80, "fcs": 81}
SEASON_TYPE = 2
POSITIONS_URL = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/positions"
_ID_RE = re.compile(r"/(?:teams|groups|positions)/(\d+)")
_TYPE_RE = re.compile(r"/types/(\d+)/groups/")


def _env_num(name: str, default: float, cast):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = cast(raw)
    except (TypeError, ValueError):
        return default
    return val if val >= 0 else default


def _retry(fn, *args, logger=None, what: str = "", **kwargs):
    """Call fn with bounded retry. Returns None when every attempt fails.

    ``dl_utils.download`` already retries 403/429/5xx internally; this guards the
    layer above it (transport resets, a JSON body that never arrives) so one bad
    team cannot abort a season.
    """
    attempts = max(1, int(_env_num("CFB_TEAMS_RETRIES", 4, int)))
    backoff = _env_num("CFB_TEAMS_BACKOFF", 2.0, float)
    for attempt in range(1, attempts + 1):
        try:
            out = fn(*args, **kwargs)
            if out:
                return out
            raise ValueError("empty payload")
        except Exception as exc:  # noqa: BLE001 - transport errors vary by backend
            if attempt == attempts:
                if logger is not None:
                    logger.error("%s failed after %d attempts: %s", what, attempts, exc)
                return None
            time.sleep(backoff * attempt)
    return None


def _ref_id(ref: str) -> str | None:
    m = _ID_RE.search(ref or "")
    return m.group(1) if m else None


def _ref_ids(payload: dict) -> list[str]:
    out = []
    for item in (payload or {}).get("items") or []:
        rid = _ref_id((item or {}).get("$ref", ""))
        if rid is not None:
            out.append(rid)
    return out


def _workers() -> int:
    return max(1, int(_env_num("CFB_TEAMS_WORKERS", 4, int)))


def scrape_season(season: int, logger) -> dict:
    """Build one season bundle. Never raises; records unfetchable ids instead."""
    incomplete: list[str] = []

    divisions: dict[str, list[str]] = {}
    group_children: dict[str, dict] = {}
    for label, gid in DIVISION_GROUPS.items():
        payload = _retry(
            sdv.cfb.espn_cfb_season_group_teams,
            season=season,
            season_type=SEASON_TYPE,
            group_id=gid,
            limit=900,
            return_parsed=False,
            logger=logger,
            what=f"{season} group {gid} teams",
        )
        divisions[str(gid)] = _ref_ids(payload)
        if payload is None:
            incomplete.append(f"group_{gid}_teams")
        logger.info("%s %s (group %s): %d teams", season, label, gid, len(divisions[str(gid)]))

        children = _retry(
            sdv.cfb.espn_cfb_season_group_children,
            season=season,
            season_type=SEASON_TYPE,
            group_id=gid,
            limit=900,
            return_parsed=False,
            logger=logger,
            what=f"{season} group {gid} children",
        )
        if children is None:
            incomplete.append(f"group_{gid}_children")
        else:
            group_children[str(gid)] = children

    team_ids = sorted({t for ids in divisions.values() for t in ids}, key=int)

    def _team(tid: str):
        return tid, _retry(
            sdv.cfb.espn_cfb_season_team,
            season=season,
            team_id=tid,
            return_parsed=False,
            logger=logger,
            what=f"{season} team {tid}",
        )

    teams: dict[str, dict] = {}
    for tid, payload in run_pool(_team, team_ids, kind="thread", workers=_workers(), desc=f"{DATASET} {season} teams"):
        if payload is None:
            incomplete.append(f"team_{tid}")
        else:
            teams[tid] = payload

    # Conference ids: the union of the 80/81 children and every id a team's own
    # `groups` ref names. Neither source alone is complete -- an independent's
    # group never shows up as a child, and the children list can carry a
    # conference no captured team belongs to.
    conf_refs: dict[str, tuple[int, str]] = {}

    def _note_conf(ref: str) -> None:
        cid = _ref_id(ref)
        if not cid:
            return
        m = _TYPE_RE.search(ref)
        conf_refs.setdefault(cid, (int(m.group(1)) if m else SEASON_TYPE, cid))

    for children in group_children.values():
        for item in children.get("items") or []:
            _note_conf((item or {}).get("$ref", "") or "")
    for payload in teams.values():
        _note_conf(((payload or {}).get("groups") or {}).get("$ref", "") or "")

    def _conf(item):
        stype, cid = item
        return cid, _retry(
            sdv.cfb.espn_cfb_season_group,
            season=season,
            season_type=stype,
            group_id=cid,
            return_parsed=False,
            logger=logger,
            what=f"{season} conference {cid}",
        )

    conferences: dict[str, dict] = {}
    for cid, payload in run_pool(
        _conf,
        sorted(conf_refs.values(), key=lambda t: int(t[1])),
        kind="thread",
        workers=_workers(),
        desc=f"{DATASET} {season} conferences",
    ):
        if payload is None:
            incomplete.append(f"conference_{cid}")
        else:
            conferences[cid] = payload

    logger.info(
        "%s: %d teams, %d conferences, %d incomplete",
        season,
        len(teams),
        len(conferences),
        len(incomplete),
    )
    return {
        "season": season,
        "season_type": SEASON_TYPE,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "divisions": divisions,
        "group_children": group_children,
        "conferences": conferences,
        "teams": teams,
        "incomplete": sorted(incomplete),
    }


def season_path(season: int) -> Path:
    return Path(f"cfb/{DATASET}/json/{season}.json")


def is_complete(season: int) -> bool:
    """A bundle counts as captured only when nothing in it failed to fetch."""
    p = season_path(season)
    if not p.is_file():
        return False
    try:
        with p.open(encoding="utf-8") as f:
            return not json.load(f).get("incomplete")
    except (OSError, ValueError):
        return False


def scrape_positions(logger, *, force: bool = False) -> Path:
    """Capture the season-independent league position reference (74 items)."""
    out = Path("cfb/reference/positions.json")
    if out.is_file() and not force:
        logger.info("positions: already captured at %s", out)
        return out
    from sportsdataverse.dl_utils import download

    # The generated wrapper hardcodes params={} and ESPN pages this 25 at a time,
    # so the INDEX is fetched through sdv-py's HTTP chokepoint with an explicit
    # limit; each item is then resolved through the SDK wrapper.
    resp = download(url=POSITIONS_URL, params={"limit": 200})
    index = resp.json() if resp is not None else {}
    ids = _ref_ids(index)
    logger.info("positions: %s refs (count=%s)", len(ids), index.get("count"))

    def _pos(pid: str):
        return pid, _retry(
            sdv.cfb.espn_cfb_position,
            position_id=pid,
            return_parsed=False,
            logger=logger,
            what=f"position {pid}",
        )

    items = {}
    for pid, payload in run_pool(_pos, ids, kind="thread", workers=_workers(), desc="positions"):
        if payload is not None:
            items[pid] = payload
    write_json_atomic(
        {
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count": index.get("count"),
            "positions": items,
        },
        out,
    )
    logger.info("positions: wrote %d/%d to %s", len(items), len(ids), out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=FIRST_SEASON)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument(
        "-r",
        "--rescrape",
        type=str,
        default="false",
        help="re-fetch seasons already captured (default: skip them)",
    )
    ap.add_argument("--positions-only", action="store_true")
    args = ap.parse_args()
    end = args.end_year or most_recent_cfb_season()
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")

    logger = get_logger(f"cfb_{DATASET}", "positions")
    scrape_positions(logger, force=rescrape)
    if args.positions_only:
        return

    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        if not rescrape and is_complete(season):
            logger.info("%s: already captured, skipping", season)
            continue
        bundle = scrape_season(season, logger)
        write_json_atomic(bundle, season_path(season))
        logger.info("%s: wrote %s", season, season_path(season))


if __name__ == "__main__":
    main()
