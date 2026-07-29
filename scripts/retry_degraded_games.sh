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
  | awk '{print $NF}' | sort -u > "$LIST"
say "harvested $(wc -l < "$LIST" | tr -d ' ') degraded games into $LIST"

if [ "$DRY_RUN" = "1" ]; then
  cat "$LIST"
  exit 0
fi

for PASS in $(seq 1 "$MAX_PASSES"); do
  N=$(wc -l < "$LIST" | tr -d ' ')
  [ "$N" -eq 0 ] && { say "nothing left to retry"; break; }
  say "=== pass $PASS/$MAX_PASSES over $N games ==="

  $PY - "$LIST" <<'PYEOF' 2>&1 | tee -a "$LOG"
import os, sys
sys.path.insert(0, "python")
import polars as pl
from scrape_cfb_json import download_game

ids = [int(x) for x in open(sys.argv[1]).read().split()]
master = pl.read_parquet("cfb/cfb_schedule_master.parquet")
season_of = dict(
    zip(master["game_id"].cast(pl.Int64).to_list(), master["season"].cast(pl.Int64).to_list())
)

ok = degraded = err = 0
for gid in ids:
    season = season_of.get(gid)
    if season is None:
        print(f"  {gid}: not in schedule master, skipping")
        continue
    before = os.path.getsize(f"cfb/json/raw/{gid}.json") if os.path.exists(f"cfb/json/raw/{gid}.json") else 0
    res = download_game(gid, season, True)
    after = os.path.getsize(f"cfb/json/raw/{gid}.json") if os.path.exists(f"cfb/json/raw/{gid}.json") else 0
    if after < 0.5 * max(before, 1):
        print(f"  {gid} ({season}): DATA LOSS {before} -> {after}")
    if res == "ok":
        ok += 1
    elif res == "degraded":
        degraded += 1
    else:
        err += 1
print(f"RETRY RESULT ok={ok} still_degraded={degraded} error={err}")
PYEOF

  # Re-harvest so the next pass only carries what is still failing.
  grep -h -oE "degraded summary for [0-9]+" logs/cfb_json_logfile_*.log 2>/dev/null \
    | awk '{print $NF}' | sort -u > "$LIST.new"
  mv "$LIST.new" "$LIST"
done

say "=== retry complete; $(wc -l < "$LIST" | tr -d ' ') games still degraded (banked copies intact) ==="
echo "EXIT=0" | tee -a "$LOG"
