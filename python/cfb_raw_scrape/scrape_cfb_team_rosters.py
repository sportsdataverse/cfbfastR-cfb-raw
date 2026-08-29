"""Stage 03 implementation -- ESPN CFB per-team season rosters.

Depends on stage 01: team ids come from ``cfb/teams/json/{season}.json``'s
``divisions`` map, which lists ids per ESPN group (80 = FBS 148 teams,
81 = FCS 130). That is the dependency the pipeline order encodes -- 01 teams
before 03 team_rosters.

**THIS ENDPOINT IS CURRENT-SEASON ONLY, AND THAT IS THE WHOLE DESIGN.** Probed
live 2026-08-29: ``espn_cfb_team_roster(team_id=333)`` returns
``season = {"year": 2026, ...}`` with no way to ask for another -- passing
``season=`` raises ``TypeError`` from ``download()``, the same as the standings
endpoint's ``week`` and ``seasontype``.

Two consequences the code enforces rather than documents:

1. **A requested season that is not ESPN's current season cannot be served.**
   Writing a 2026 roster into ``cfb/team_rosters/json/2024/`` would look like a
   successful backfill and be a fabrication. :func:`write_one` compares the
   payload's own season against the requested one and refuses on mismatch.
2. **The current season is never "done".** Rosters churn all year -- transfers,
   injuries, suspensions. A banked file is a snapshot, so the current season
   always re-fetches; only a finished season is skippable.

Historical rosters are NOT available here. ``docs/ESPN_ROSTERS.md`` §17 records
why: ``roster_2004-2022`` are byte-equivalent mirrors of the CFBD-sourced
``cfb_rosters_{season}.parquet`` that ``sportsdataverse.cfb.load_cfb_rosters``
already fetches, and only ``rosters_2023+`` are genuinely ESPN-derived. Filling
history is a CFBD splice, not a deeper crawl of this endpoint.

Payload shape, read from a real Alabama (333) response::

    {season: {year, type, name}, team: {...}, coach: [...],
     athletes: [{position: "offense",  items: [ ...52 players... ]},
                {position: "defense",  items: [ ...42... ]},
                {position: "specialTeam", items: [ ...6... ]},
                {position: "injuredReserveOrOut" | "suspended" | "practiceSquad",
                 items: []}]}    # 100 players across 6 groups
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import sportsdataverse as sdv

from cfb_raw_scrape._cfb_raw_utils import (
    get_logger,
    most_recent_cfb_season,
    run_pool,
    write_json_atomic,
)

DATASET = "team_rosters"

#: Which stage-01 division lists to enumerate. D2/D3 (group 35) carries 405 team
#: ids, but those teams have no ESPN roster endpoint -- the same asymmetry as
#: standings, where group 35 is named and always empty.
ROSTER_GROUPS = ("80", "81")

#: A roster with no players is a throttled 200, not a team with no team.
MIN_PLAYERS = 1


def out_path(season: int, team_id: int | str) -> str:
    return f"cfb/{DATASET}/json/{season}/{team_id}.json"


def teams_path(season: int) -> str:
    return f"cfb/teams/json/{season}.json"


def team_ids(season: int, groups: tuple[str, ...] = ROSTER_GROUPS) -> list[str]:
    """Team ids for `season`, from stage 01's output.

    Raises:
        FileNotFoundError: when stage 01 has not run for this season. That is a
            real precondition, not something to paper over with an empty list --
            an empty list would report "0 rosters" as success.
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


def _fetch(team_id: int | str) -> dict:
    return sdv.cfb.espn_cfb_team_roster(team_id=team_id, return_parsed=False)


def count_players(payload: dict) -> int:
    """Players across every position group (offense / defense / specialTeam / ...)."""
    return sum(len(g.get("items") or []) for g in (payload.get("athletes") or []))


def is_season_final(season: int, today: date | None = None) -> bool:
    """True once `season` can no longer change -- February of the following year.

    Same rule as stage 07 standings: a season labelled N runs into January N+1.
    """
    today = today or date.today()
    return (today.year, today.month) >= (season + 1, 2)


def is_complete(season: int, team_id: int | str, today: date | None = None) -> bool:
    """Banked, non-empty, and the season is over. Any of the three failing is False."""
    if not is_season_final(season, today):
        return False
    p = Path(out_path(season, team_id))
    if not p.is_file():
        return False
    try:
        return count_players(json.loads(p.read_text(encoding="utf-8"))) >= MIN_PLAYERS
    except (OSError, json.JSONDecodeError):
        return False


def write_one(season: int, team_id: int | str, logger) -> int:
    """Fetch and bank one team's roster. Refuses a mismatched or empty payload."""
    payload = _fetch(team_id)
    got = (payload.get("season") or {}).get("year")
    if got is not None and int(got) != int(season):
        raise RuntimeError(
            f"{DATASET} {season} team {team_id}: endpoint served season {got}. "
            "This endpoint has no season parameter and only serves the current "
            "season -- banking it under a different one would fabricate history "
            "(see the module docstring; use the CFBD splice for pre-2023)."
        )
    players = count_players(payload)
    if players < MIN_PLAYERS:
        raise RuntimeError(
            f"{DATASET} {season} team {team_id}: 0 players -- refusing to bank "
            "what is almost certainly a throttled 200"
        )
    payload["season_requested"] = season
    write_json_atomic(payload, out_path(season, team_id))
    return players


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    ap.add_argument(
        "-g",
        "--groups",
        default=",".join(ROSTER_GROUPS),
        help="ESPN group ids to enumerate from stage 01 (default: 80 FBS, 81 FCS)",
    )
    args = ap.parse_args(argv)
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    groups = tuple(g.strip() for g in args.groups.split(",") if g.strip())

    rc = 0
    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        try:
            ids = team_ids(season, groups)
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            rc = 1
            continue
        todo = [t for t in ids if rescrape or not is_complete(season, t)]
        logger.info(
            "%s %s: %d teams, %d to fetch (%d already banked and final)",
            DATASET,
            season,
            len(ids),
            len(todo),
            len(ids) - len(todo),
        )
        failures: list[str] = []

        def _one(team_id, _s=season, _log=logger, _f=failures):
            try:
                write_one(_s, team_id, _log)
            except Exception as exc:  # noqa: BLE001 -- one team must not stop the sweep
                _log.error("%s", exc)
                _f.append(str(team_id))

        run_pool(_one, todo, kind="thread", desc=f"{DATASET} {season}")
        if failures:
            logger.error(
                "%s %s: %d team(s) failed: %s",
                DATASET,
                season,
                len(failures),
                failures[:10],
            )
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
