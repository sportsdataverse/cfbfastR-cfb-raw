#!/bin/bash
# Re-run the games the write guard skipped as degraded.
#
# write_json_guarded refuses a write that would replace a good banked summary
# with a 5xx/empty stub, and download_game then skips the rest of that game.
# Those games keep their previous data (nothing is lost) but are NOT refreshed,
# so a "full rescrape" has a per-season remainder of ~20 games.
#
# ESPN's 5xx here are transient: in the 2004 pilot all 11 degraded games fetched
# cleanly on a second attempt. Run this after the main rescrape finishes, when
# there is no concurrent load from the season loop.
#
# USAGE
#   bash scripts/retry_degraded_games.sh              # harvest + retry
#   DRY_RUN=1 bash scripts/retry_degraded_games.sh    # just show the list
#
# ENV
#   CFB_PY        interpreter override (default: this repo's .venv; see scripts/_venv.sh)
#   MAX_PASSES    retry rounds (default 2) -- a game still degraded after these
#                 is left with its banked copy and reported
set -uo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

# Interpreter resolved by scripts/_venv.sh (sourced above); CFB_PY still overrides.
MAX_PASSES=${MAX_PASSES:-2}
# Degraded ids are harvested from every season's log, so the retry pass needs a
# season range to intersect them against (--ids-file keeps only the ids that are
# in that season's schedule). Defaults span the full corpus; narrow it with
# SEASON_START/SEASON_END when retrying a single season.
SEASON_START=${SEASON_START:-2004}
SEASON_END=${SEASON_END:-$("$PY" -c "import sys; sys.path.insert(0,'python'); from cfb_raw_scrape._cfb_raw_utils import most_recent_cfb_season as m; print(m())")}
DRY_RUN=${DRY_RUN:-0}

export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

mkdir -p logs
LIST="logs/degraded_retry_list.txt"
LOG="logs/retry_degraded.log"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# Harvest fresh each run: the main scrape appends as it goes, so a stale list
# would silently under-retry.
grep -h -oE "degraded summary for [0-9]+" logs/cfb_json_logfile_*.log 2>/dev/null \
  | awk '{print $NF}' | sort -u > "$LIST.all"
  $PY python/filter_stale.py "$LIST.all" "$LIST"
  rm -f "$LIST.all"
say "harvested $(wc -l < "$LIST" | tr -d ' ') degraded games into $LIST"

if [ "$DRY_RUN" = "1" ]; then
  cat "$LIST"
  exit 0
fi

for PASS in $(seq 1 "$MAX_PASSES"); do
  N=$(wc -l < "$LIST" | tr -d ' ')
  [ "$N" -eq 0 ] && { say "nothing left to retry"; break; }
  say "=== pass $PASS/$MAX_PASSES over $N games ==="

  $PY python/espn_cfb_04_pbp_scrape.py -s "$SEASON_START" -e "$SEASON_END" --ids-file "$LIST" 2>&1 | tee -a "$LOG"

  # Re-harvest so the next pass only carries what is still failing.
  grep -h -oE "degraded summary for [0-9]+" logs/cfb_json_logfile_*.log 2>/dev/null \
    | awk '{print $NF}' | sort -u > "$LIST.all"
    $PY python/filter_stale.py "$LIST.all" "$LIST.new"
    rm -f "$LIST.all"
  mv "$LIST.new" "$LIST"
done

say "=== retry complete; $(wc -l < "$LIST" | tr -d ' ') games still degraded (banked copies intact) ==="
echo "EXIT=0" | tee -a "$LOG"
