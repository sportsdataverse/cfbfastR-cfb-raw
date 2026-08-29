"""Stage 05 implementation -- ESPN CFB per-athlete CAREER stats.

**Athlete-keyed, not season-keyed, and that is forced by the endpoint.** Probed
live 2026-08-29 on athlete 4426339: ``espn_cfb_player_stats_v3(season=2023)``,
``(season=2024)`` and ``(season=None)`` return the SAME payload, carrying every
season the athlete played (2019-2023). ESPN accepts the ``season`` query param
and ignores it -- the per-season breakdown lives inside
``categories[].statistics[]``.

So one fetch per athlete returns all of their seasons. Writing
``{season}/{athlete_id}.json`` would store the same payload N times and imply a
season scoping that does not exist. Output is ``{athlete_id}.json``.

The MBB sibling (``espn_mbb_06_player_stats_scrape.py``) reached the same shape
for the same reason, and its docstring carries a warning this one repeats: use
``espn_cfb_player_stats_v3``, NOT ``espn_cfb_player_stats``. Despite the name,
the unsuffixed function is the Core v2 ``/athletes/{id}/statistics`` endpoint --
a different API with a different payload (``$ref``/``season``/``athlete``/
``splits``). Do not "simplify" the import.

**Athlete ids come from this repo, not from a release.** MBB reads the
``espn_mens_college_basketball_player_boxscores`` release; the college-football
equivalent does not exist (probed 2026-08-29: HTTP 404). It is not needed --
stage 04's ``cfb/game_rosters/json/{game_id}.json`` already holds every athlete
who dressed for every game, as ``data[].athlete_id``. Measured over the whole
tree: **151,143 unique athletes**, 6.9k-9.1k per season through 2013 and
20.8k-27.5k from 2014 on (ESPN widened roster coverage that year -- the jump is
real, not a gap).

Payload shape, read from a real response::

    {athlete: {...}, categories: [{name, displayName,
        statistics: [{season: {year}, teamSlug, stats: [...]}]}]}
"""

from __future__ import annotations

import argparse
import glob
import json
from datetime import date
from pathlib import Path

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    most_recent_cfb_season,
    run_pool,
    write_json_atomic,
)
from sportsdataverse.cfb import espn_cfb_player_stats_v3

DATASET = "player_stats"

#: Stage 04's output -- the athlete-id source (see module docstring).
GAME_ROSTERS_GLOB = "cfb/game_rosters/json/*.json"

#: ESPN CFB game-roster coverage starts in 2004, the same floor as the rest of
#: this repo's ESPN trees.
MIN_SEASON = 2004

#: A payload with no categories is a throttled 200, not an athlete who never
#: recorded a stat -- ESPN returns the athlete envelope either way.
MIN_CATEGORIES = 1


def out_path(athlete_id: int | str) -> str:
    return f"cfb/{DATASET}/json/{athlete_id}.json"


def is_season_final(season: int, today: date | None = None) -> bool:
    """True once `season` can no longer change -- February of the following year.

    Same rule as stages 03 and 07: a season labelled N runs into January N+1.
    """
    today = today or date.today()
    return (today.year, today.month) >= (season + 1, 2)


def athletes_by_season(
    start: int, end: int, pattern: str = GAME_ROSTERS_GLOB
) -> dict[int, set[str]]:
    """Map season -> athlete ids, read from stage 04's game rosters.

    Returns:
        ``{season: {athlete_id, ...}}`` restricted to seasons in ``[start, end]``.

    Raises:
        FileNotFoundError: when stage 04 has not run. An empty mapping would
            report "0 athletes" as a successful sweep.
    """
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(
            f"no game rosters at {pattern} -- run stage 04 before {DATASET}"
        )
    per: dict[int, set[str]] = {}
    for f in files:
        try:
            d = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        season = d.get("season")
        if season is None or not (start <= int(season) <= end):
            continue
        # An unplayed game ships data: [] -- 1,655 of them in the tree, mostly
        # future fixtures. Not an error, just nothing to enumerate.
        for row in d.get("data") or []:
            aid = row.get("athlete_id")
            if aid:
                per.setdefault(int(season), set()).add(str(aid))
    return per


def stale_athletes(
    per_season: dict[int, set[str]], today: date | None = None
) -> set[str]:
    """Athlete ids whose banked payload cannot be trusted to be final.

    **A career payload GROWS.** An athlete who is still playing gains a new
    entry in ``categories[].statistics[]`` every season, so "the file exists"
    is not "the file is current" -- skipping on presence alone freezes that
    athlete at whatever season we first fetched them. That is exactly how the
    2027 recruiting class froze at 4,779 rows, and what stage 07's is_complete
    guards against for a live season's standings.

    An athlete is final only when EVERY season we saw them in is final.
    """
    stale: set[str] = set()
    for season, ids in per_season.items():
        if not is_season_final(season, today):
            stale |= ids
    return stale


def count_categories(payload: dict) -> int:
    return len(payload.get("categories") or [])


def is_complete(athlete_id: int | str, stale: set[str] | None = None) -> bool:
    """Banked, non-empty, and not still accruing seasons."""
    if stale and str(athlete_id) in stale:
        return False
    p = Path(out_path(athlete_id))
    if not p.is_file():
        return False
    try:
        return (
            count_categories(json.loads(p.read_text(encoding="utf-8")))
            >= MIN_CATEGORIES
        )
    except (OSError, json.JSONDecodeError):
        return False


def write_one(athlete_id: int | str, logger) -> int:
    """Fetch and bank one athlete's career stats. Refuses an empty payload."""
    payload = espn_cfb_player_stats_v3(athlete_id=athlete_id, return_parsed=False)
    if isinstance(payload, (bytes, str)):
        payload = json.loads(payload)
    n = count_categories(payload)
    if n < MIN_CATEGORIES:
        raise RuntimeError(
            f"{DATASET} athlete {athlete_id}: 0 categories -- refusing to bank "
            "what is almost certainly a throttled 200"
        )
    write_json_atomic(payload, out_path(athlete_id))
    return n


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
        "%s %s-%s: %d athletes, %d to fetch (%d banked and final, %d still accruing)",
        DATASET,
        start,
        end,
        len(ids),
        len(todo),
        len(ids) - len(todo),
        len(stale & set(ids)),
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
