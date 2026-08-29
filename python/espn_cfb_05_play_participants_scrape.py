"""Stage 05 -- ESPN CFB per-play participants (CFB extra).

Thin shim over ``scrape_cfb_participants``: the directory listing IS the pipeline.

**The numbers are this repo's COLD-START EXECUTION ORDER** -- the order the
stages must run in from an empty tree, renumbered 2026-08-29. Reading the
directory top to bottom gives you a working pipeline:

    01 teams              reference; no upstream
    02 schedules          the game-id gate for everything below
    03 team_rosters       (reserved -- not built yet)
    04 game_rosters
    05 play_participants
    06 pbp / json         joins game_rosters + play_participants into json/final
    07 player_stats       (reserved -- not built yet)
    08 team_stats         (reserved -- not built yet)
    09 standings          (reserved -- not built yet)
    10 qbr
    11 power_index
    50 recruits           preflight-gated monthly cadence, not the daily loop
    51 player_core        (reserved -- monthly, not built yet)

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
        "$PY" python/espn_cfb_05_play_participants_scrape.py -s 2026 -e 2026
"""

from __future__ import annotations

from scrape_cfb_participants import main

if __name__ == "__main__":
    raise SystemExit(main())
