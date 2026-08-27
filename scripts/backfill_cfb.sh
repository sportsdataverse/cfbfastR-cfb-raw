#!/bin/bash
# Full historical backfill (default 2004 -> most-recent season).
set -uo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
START_YEAR=${1:-2004}
END_YEAR=${2:-$("$PY" -c "import sys; sys.path.insert(0,'python'); from cfb_raw_scrape._cfb_raw_utils import most_recent_cfb_season as m; print(m())")}
bash scripts/daily_cfb_scraper.sh -s "$START_YEAR" -e "$END_YEAR" -r false
