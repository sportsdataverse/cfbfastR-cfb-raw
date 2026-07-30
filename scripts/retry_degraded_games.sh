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
#   CFB_PY        interpreter (default "uv run python"; see rescrape_cfb_full.sh)
#   MAX_PASSES    retry rounds (default 2) -- a game still degraded after these
#                 is left with its banked copy and reported
set -uo pipefail

PY="${CFB_PY:-uv run python}"
MAX_PASSES=${MAX_PASSES:-2}
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

  $PY python/retry_degraded.py "$LIST" 2>&1 | tee -a "$LOG"

  # Re-harvest so the next pass only carries what is still failing.
  grep -h -oE "degraded summary for [0-9]+" logs/cfb_json_logfile_*.log 2>/dev/null \
    | awk '{print $NF}' | sort -u > "$LIST.all"
    $PY python/filter_stale.py "$LIST.all" "$LIST.new"
    rm -f "$LIST.all"
  mv "$LIST.new" "$LIST"
done

say "=== retry complete; $(wc -l < "$LIST" | tr -d ' ') games still degraded (banked copies intact) ==="
echo "EXIT=0" | tee -a "$LOG"
