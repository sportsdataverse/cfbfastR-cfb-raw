#!/usr/bin/env bash
# Scrape 247 recruit classes into the raw store.
#
# A signed class is IMMUTABLE, so this is idempotent by design: any year with a
# complete manifest is skipped instantly and only the current cycle re-scrapes.
# A full cold backfill is ~4 min/class year; a daily run is one year.
#
# THE FLOOR IS 2002, AND IT IS MEASURED. 247 returns rows back to 1996, but
# composite ratings only become usable in 2002:
#     1996/1998  0 rated recruits at all
#     2000       48% rated on page 1, 0% by page 4
#     2001       52% rated on page 1, 0% by page 4
#     2002       88% / 99% / 100% across pages 1 / 4 / 8
# Below 2002 a class is mostly unrated, which does not error -- it silently
# understates every team's talent.
#
# Usage:
#   scripts/10_scrape_recruits.sh                 # current cycle only
#   scripts/10_scrape_recruits.sh 2002 2026       # cold backfill
#   scripts/10_scrape_recruits.sh 2026 2026 --rescrape
#
# Pace/retry are env-tunable, shared with sdv-py so both sides tune together:
#   SDV_PY_247_RETRIES (3)  SDV_PY_247_DELAY (0.5s)  SDV_PY_247_BACKOFF (2.0s)
set -euo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

cd "$(dirname "$0")/.."

FLOOR=2002
CURRENT="$(date +%Y)"
# 10# forces base 10: bash reads a zero-padded "08" as octal and errors out.
[[ "$((10#$(date +%m)))" -ge 8 ]] && CURRENT=$((CURRENT + 1))  # signing class runs a year ahead

START="${1:-$CURRENT}"
END="${2:-$START}"
shift 2 2>/dev/null || true

if [[ "$START" -lt "$FLOOR" ]]; then
  echo "!! start $START is below the measured floor $FLOOR -- clamping (see header)" >&2
  START="$FLOOR"
fi

# Per-class-year loop so each year's run record commits WITH that year, rather
# than one aggregate log at the end of a backfill that may never reach its exit.
# The canonical name comes from the Python side (get_logger -> cfb_recruits_
# logfile_<year>.log); the whole-run tee is a temp file, because a timestamped
# aggregate is session state and not a pipeline artifact.
# shellcheck source=scripts/_commit.sh
source "$(dirname "${BASH_SOURCE[0]}")/_commit.sh"

mkdir -p logs
RUN_LOG=$(mktemp "/tmp/cfb_recruits_run.XXXXXX.log")
PUSH="${PUSH:-true}"
RC=0

echo "scraping classes ${START}-${END}"
echo "watch with: tail -f $RUN_LOG"

for y in $(seq "$START" "$END"); do
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    "$PY" python/espn_cfb_10_recruits_scrape.py -s "$y" -e "$y" "$@" 2>&1 | tee -a "$RUN_LOG"
  year_rc=${PIPESTATUS[0]}
  [ "$year_rc" -eq 0 ] || { echo "!! class $y exited $year_rc"; RC=1; }
  # Commit the season log even when the year FAILED -- the log is how the
  # failure gets diagnosed, so that is exactly when it must survive.
  if [ "$PUSH" = "true" ]; then
    sdv_commit_log cfb_recruits "$y" || RC=1
  fi
done

echo "EXIT=$RC" | tee -a "$RUN_LOG"
exit "$RC"
