"""Stage 04 -- ESPN CFB per-game rosters.

Thin shim over ``scrape_cfb_game_rosters``: the directory listing IS the pipeline.

**The numbers are this repo's COLD-START EXECUTION ORDER** -- the order the
stages must run in from an empty tree, renumbered 2026-08-29. Reading the
directory top to bottom gives you a working pipeline:

    01 teams              reference; no upstream
    02 schedules          the game-id gate for everything below
    03 team_rosters       (reserved -- not built yet)
    04 game_rosters       backfill-only; 06 already fetches this per game
    05 play_participants  backfill-only; 06 already fetches this per game
    06 pbp / json         fetches rosters + participants inline, writes json/final
    07 player_stats       (reserved -- not built yet)
    08 team_stats         (reserved -- not built yet)
    09 standings          (reserved -- not built yet)
    10 qbr
    11 power_index
    50 recruits           preflight-gated monthly cadence, not the daily loop
    51 player_core        (reserved -- monthly, not built yet)


**04 and 05 are not in the daily loop.** ``scrape_cfb_json`` already calls
``_rosters`` and ``_participants`` per game and embeds both in the json/final
payload, so running the standalone stages daily would fetch the same data twice
-- doubling a ~250 Core v2 ``$ref`` fan-out per game against an endpoint that
403s under load. They stay as stages for backfilling their own
``cfb/game_rosters`` and ``cfb/play_participants`` trees (20,696 files each).

**This DIVERGES from the nba / mbb / wnba / wbb family numbering on purpose.**
Those repos number by cross-repo dataset identity, where 04 means game_rosters
everywhere. CFB numbers by its own dependency chain instead, because the CFB
pipeline has joins the other leagues do not -- teams feeding the extended
schedule interface, and game_rosters plus play_participants feeding json/final.
Do not "fix" a CFB number to match a sibling repo; check this list first.

A reserved number stays EMPTY until its stage is built. The daily driver's
sequence in ``scripts/daily_cfb_scraper.sh`` remains the executable truth.

Example:
    One season::

        source scripts/_venv.sh
        "$PY" python/espn_cfb_04_game_rosters_scrape.py -s 2026 -e 2026
"""

from __future__ import annotations

from scrape_cfb_game_rosters import main

if __name__ == "__main__":
    raise SystemExit(main())
