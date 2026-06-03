#!/bin/bash
# Full historical backfill (default 2004 -> most-recent season).
set -uo pipefail
START_YEAR=${1:-2004}
END_YEAR=${2:-$(uv run python -c "import sys; sys.path.insert(0,'python'); from _cfb_raw_utils import most_recent_cfb_season as m; print(m())")}
bash scripts/daily_cfb_scraper.sh -s "$START_YEAR" -e "$END_YEAR" -r false
