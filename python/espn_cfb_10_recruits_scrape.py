"""Stage 10 -- ESPN CFB recruiting classes (CFB extra).

Thin shim over ``scrape_cfb_recruits``: the directory listing IS the pipeline, and the stage
number is the cross-repo identity for this dataset across the ESPN family
(nba / mbb / wnba / wbb / cfb).

**The numbers carry holes on purpose.** 01-09 are the shared ESPN family slots;
CFB does not scrape 03 standings, 05 draft, 06 player_stats, 07 team_stats,
08 team_rosters or 09 player_core, so those numbers stay EMPTY rather than being
compacted -- a number means the same dataset in every repo, which is worth more
than a dense sequence. CFB-only datasets start at 10.

The number is intended BUILD order, not run order: the daily driver's sequence
in ``scripts/daily_cfb_scraper.sh`` is the executable truth.

Example:
    One season::

        source scripts/_venv.sh
        "$PY" python/espn_cfb_10_recruits_scrape.py -s 2026 -e 2026
"""

from __future__ import annotations

from scrape_cfb_recruits import main

if __name__ == "__main__":
    raise SystemExit(main())
