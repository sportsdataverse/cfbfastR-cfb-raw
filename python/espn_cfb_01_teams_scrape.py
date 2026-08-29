"""Stage 01 -- ESPN CFB per-season team + conference reference (CFB extra).

Thin shim over ``scrape_cfb_teams``: the directory listing IS the pipeline.

**The numbers are this repo's COLD-START EXECUTION ORDER** -- the order the
stages must run in from an empty tree, renumbered 2026-08-29. Reading the
directory top to bottom gives you a working pipeline:

    01 teams              reference; no upstream
    02 schedules          the game-id gate for everything below
    03 team_rosters       needs 01 teams for the id list; CURRENT SEASON ONLY
    04 pbp / json         fetches rosters + participants inline, writes json/final
    05 player_stats       (reserved -- not built yet)
    06 team_stats         (reserved -- not built yet)
    07 standings
    08 qbr
    09 power_index
    50 recruits           preflight-gated monthly cadence, not the daily loop
    51 player_core        (reserved -- monthly, not built yet)


**Rosters and participants are not stages.** ``scrape_cfb_pbp`` (04) calls
``_rosters`` and ``_participants`` per game and embeds both in the json/final
payload, so separate stages meant fetching the same data twice -- a second ~250
Core v2 ``$ref`` fan-out per game against an endpoint that 403s under load. They
were deleted 2026-08-29 and the sequence CLOSED over them: a deleted stage's
number is reclaimed, only RESERVED numbers hold their place.

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
        "$PY" python/espn_cfb_01_teams_scrape.py -s 2026 -e 2026
"""

from __future__ import annotations

from cfb_raw_scrape.scrape_cfb_teams import main

if __name__ == "__main__":
    raise SystemExit(main())
