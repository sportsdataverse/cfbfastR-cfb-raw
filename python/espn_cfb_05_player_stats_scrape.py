"""Stage 05 entry point -- ESPN CFB per-athlete career stats.

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
    51 player_core        (reserved -- monthly, not built yet)

Implementation: ``cfb_raw_scrape.scrape_cfb_player_stats``.
"""

from cfb_raw_scrape.scrape_cfb_player_stats import main

if __name__ == "__main__":
    raise SystemExit(main())
