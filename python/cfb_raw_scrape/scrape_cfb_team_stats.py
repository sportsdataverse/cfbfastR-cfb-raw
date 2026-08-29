"""Stage 06 implementation -- ESPN CFB per-team SEASON statistics (Core v2).

**The season lives in the PATH, and that is the whole reason this stage uses
Core v2 rather than the endpoint its basketball siblings use.**

The MBB/WNBA team-stats stages call ``common/v3 .../teams/{id}/statistics``
with ``?season=YYYY``. That URL does not serve college football -- probed live
2026-08-29 for Alabama (333), it returns 404 for season 2024, 2019, and with no
season at all. There is also no generated ``espn_cfb_team_season_stats``
wrapper in sportsdataverse-py for the Core v2 route, so this stage goes through
``dl_utils.download`` directly.

Core v2 does serve it, with both the season and the season type as PATH
segments::

    .../seasons/{season}/types/{seasontype}/teams/{team_id}/statistics

That is worth more than convenience. A season passed as a QUERY param can be
accepted and silently ignored -- which is exactly what ``/athletes/{id}/stats``
does (see stage 05) and what the roster endpoint's ``limit`` did before stage 03
was fixed. A path segment cannot be ignored: a wrong season is a 404, not a
plausible-looking payload for the wrong year. Verified rather than assumed,
Alabama ``totalYards``:

===========  =========  =========
season       type 2     type 3
===========  =========  =========
2024         5073       5333
2023         5215       5503
2015         5493       6406
2004         3656       3920
2003         404        404
===========  =========  =========

Two things fall out of that table. The values differ per season, so the path IS
honoured. And **type 3 is CUMULATIVE, not postseason-only** -- it is larger than
type 2 in every season, i.e. regular season plus bowls, the same "overall"
semantics stage 07 found on the standings endpoint (``seasonType=3``,
``name="overall"``). Both types are captured; do not treat type 3 as a
postseason-only split.

The 2003 404 sets the floor at 2004, matching the rest of this repo's ESPN
trees.

Payload shape, read from a real response (11 categories, 285 stats)::

    {$ref, season: {...}, team: {...}, seasonType: {...},
     splits: {categories: [{name, displayName,
         stats: [{name, displayName, value, displayValue, ...}]}]}}
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    most_recent_cfb_season,
    run_pool,
    write_json_atomic,
)
from sportsdataverse.dl_utils import download

DATASET = "team_stats"

STATS_URL = (
    "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
    "/seasons/{season}/types/{seasontype}/teams/{team_id}/statistics"
)

#: 2 = regular season, 3 = CUMULATIVE incl. postseason (see the module
#: docstring -- type 3 totals exceed type 2 in every season measured).
SEASON_TYPES = (2, 3)

#: Probed: 2003 returns 404 for both season types, 2004 returns data.
MIN_SEASON = 2004

#: Which stage-01 division lists to enumerate -- FBS and FCS, the same pair
#: stage 03 and stage 07 cover. Group 35 (D2/D3) is named by ESPN but never
#: populated.
STATS_GROUPS = ("80", "81")

#: A payload with zero stats is a throttled 200 or an empty shell, not a team
#: that recorded nothing.
MIN_CATEGORIES = 1
MIN_STATS = 1


def out_path(season: int, seasontype: int, team_id: int | str) -> str:
    return f"cfb/{DATASET}/json/{season}/{seasontype}/{team_id}.json"


def teams_path(season: int) -> str:
    return f"cfb/teams/json/{season}.json"


def team_ids(season: int, groups: tuple[str, ...] = STATS_GROUPS) -> list[str]:
    """Team ids for `season`, from stage 01's output.

    Raises:
        FileNotFoundError: when stage 01 has not run for this season -- a real
            precondition, not something to paper over with an empty list.
    """
    p = Path(teams_path(season))
    if not p.is_file():
        raise FileNotFoundError(
            f"{p} missing -- run stage 01 (teams) for {season} before {DATASET}"
        )
    divisions = json.loads(p.read_text(encoding="utf-8")).get("divisions") or {}
    ids: list[str] = []
    for g in groups:
        ids.extend(divisions.get(g) or [])
    return sorted(dict.fromkeys(ids), key=int)


def _fetch(season: int, seasontype: int, team_id: int | str) -> dict:
    resp = download(
        url=STATS_URL.format(season=season, seasontype=seasontype, team_id=team_id)
    )
    return resp.json() if resp is not None else {}


def summarize(payload: dict) -> dict:
    """Count categories and the stats inside them.

    Categories alone are not enough: ESPN can ship the category scaffolding
    with every ``stats`` list empty, which counts as a non-zero category count
    and zero actual data.
    """
    cats = (payload.get("splits") or {}).get("categories") or []
    return {
        "categories": len(cats),
        "stats": sum(len(c.get("stats") or []) for c in cats),
    }


def is_season_final(season: int, today: date | None = None) -> bool:
    """True once `season` can no longer change -- February of the following year."""
    today = today or date.today()
    return (today.year, today.month) >= (season + 1, 2)


def is_complete(
    season: int, seasontype: int, team_id: int | str, today: date | None = None
) -> bool:
    """Banked, non-empty, AND the season is over.

    Season stats accumulate every week, so a mid-season file is a snapshot.
    Treating it as done freezes the team at whatever week we first fetched it.
    """
    if not is_season_final(season, today):
        return False
    p = Path(out_path(season, seasontype, team_id))
    if not p.is_file():
        return False
    try:
        counts = summarize(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return False
    return counts["categories"] >= MIN_CATEGORIES and counts["stats"] >= MIN_STATS


def write_one(season: int, seasontype: int, team_id: int | str, logger) -> dict:
    """Fetch and bank one (season, type, team). Raises rather than banking empty."""
    payload = _fetch(season, seasontype, team_id)
    counts = summarize(payload)
    if counts["categories"] < MIN_CATEGORIES or counts["stats"] < MIN_STATS:
        raise RuntimeError(
            f"{DATASET} {season}/{seasontype} team {team_id}: refusing to bank an "
            f"empty payload ({counts['categories']} categories, "
            f"{counts['stats']} stats)"
        )
    payload["season_requested"] = season
    payload["season_type_requested"] = seasontype
    write_json_atomic(payload, out_path(season, seasontype, team_id))
    return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    ap.add_argument(
        "-t",
        "--season_types",
        default=",".join(str(t) for t in SEASON_TYPES),
        help="ESPN season types (2 = regular, 3 = cumulative incl. postseason)",
    )
    args = ap.parse_args(argv)
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    types = [int(t.strip()) for t in args.season_types.split(",") if t.strip()]

    rc = 0
    for season in range(max(args.start_year, MIN_SEASON), end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        try:
            ids = team_ids(season)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            rc = 1
            continue
        jobs = [
            (t, tid)
            for t in types
            for tid in ids
            if rescrape or not is_complete(season, t, tid)
        ]
        logger.info(
            "%s %s: %d teams x %d types, %d to fetch (%d banked and final)",
            DATASET,
            season,
            len(ids),
            len(types),
            len(jobs),
            len(ids) * len(types) - len(jobs),
        )
        failures: list[str] = []

        def _one(job, _s=season, _log=logger, _f=failures):
            seasontype, tid = job
            try:
                write_one(_s, seasontype, tid, _log)
            except Exception as exc:  # noqa: BLE001 -- one team must not stop the sweep
                _log.error("%s", exc)
                _f.append(f"{seasontype}/{tid}")

        run_pool(_one, jobs, kind="thread", desc=f"{DATASET} {season}")
        if failures:
            logger.error(
                "%s %s: %d failed: %s", DATASET, season, len(failures), failures[:10]
            )
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
