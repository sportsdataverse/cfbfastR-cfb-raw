"""Stage 52 entry point -- ESPN's own CFB recruiting classes.

Cold-start execution order:

    01 teams              reference; no upstream
    02 schedules          the game-id gate for everything below
    03 team_rosters       needs 01 teams for the id list; CURRENT SEASON ONLY
    04 pbp / json         fetches rosters + participants inline, writes json/final
    05 player_stats       athlete-keyed career stats (ESPN ignores season)
    06 team_stats         Core v2, season + type in the PATH
    07 standings
    08 qbr
    09 power_index
    50 recruits           preflight-gated monthly cadence, not the daily loop
    51 player_core        identity + bio; monthly, athlete-keyed
    52 espn_recruits      ESPN's OWN recruiting classes; monthly (50 is 247Sports)

Implementation: ``cfb_raw_scrape.scrape_cfb_espn_recruits``.

A DIFFERENT PROVIDER from stage 50, which scrapes 247Sports. Different universe,
different grades, different rankings -- neither is the other's oracle.

Cadence is MONTHLY alongside 50 and 51.
"""

from cfb_raw_scrape.scrape_cfb_espn_recruits import main

if __name__ == "__main__":
    raise SystemExit(main())
