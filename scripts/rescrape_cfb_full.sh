#!/bin/bash
# Full 2004->2025 CFB rescrape (regular season + bowls), resumable.
#
# Every season in range is rescraped from scratch (-r true), covering the whole
# per-game pipeline in scrape_cfb_json.py: raw summary, enriched final,
# play_participants, game_rosters, power_index, betting, team_box_extra.
#
# WHY THIS IS SLOW: espn_cfb_game_rosters resolves each athlete $ref with its
# own request (~220/game post-2014, ~65/game pre-2014). The full range is on the
# order of 3M HTTP calls. Budget days, not hours, and expect to restart it --
# hence the season-level checkpoint below.
#
# USAGE
#   bash scripts/rescrape_cfb_full.sh                # 2004..2025, resumes
#   bash scripts/rescrape_cfb_full.sh 2004 2004      # pilot one season
#   FORCE=1 bash scripts/rescrape_cfb_full.sh        # ignore the checkpoint
#
# TUNING (env only -- never edit this file to re-pace a run)
#   CFB_SCRAPE_WORKERS   game-pool workers            (default 4)
#   CFB_EXTRAS_RETRIES   retries on aux/extras fetch  (default 2)
#   CFB_RESCRAPE_GIT     1 = commit+push each season  (default 0, off)
#
# WATCH IT LIVE (run in a second terminal):
#   tail -f logs/rescrape_cfb_full.log
set -uo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

START_YEAR=${1:-2004}
END_YEAR=${2:-2025}
FORCE=${FORCE:-0}

export CFB_SCRAPE_WORKERS=${CFB_SCRAPE_WORKERS:-4}
export CFB_EXTRAS_RETRIES=${CFB_EXTRAS_RETRIES:-2}
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

CFB_RESCRAPE_GIT=${CFB_RESCRAPE_GIT:-0}

# CFB_PY EXISTS BECAUSE `uv run` RE-SYNCS THE ENV before every invocation.
# uv.lock pins sportsdataverse to the PyPI RELEASE (0.0.72) even though
# pyproject declares git@main, so `uv run` silently reinstalls the released
# build over any locally-installed one -- observed mid-run on 2026-07-28,
# which restarted a season on code missing every fix from that day.
# Point CFB_PY at the venv directly to pin one interpreter state.
# Interpreter resolved by scripts/_venv.sh (sourced above); CFB_PY still overrides.

mkdir -p logs
LOG="logs/rescrape_cfb_full.log"
CKPT="logs/rescrape_checkpoint.txt"
touch "$CKPT"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

say "=== rescrape start ${START_YEAR}..${END_YEAR} workers=${CFB_SCRAPE_WORKERS} retries=${CFB_EXTRAS_RETRIES} force=${FORCE} git=${CFB_RESCRAPE_GIT} py=${PY} ==="

# Abort before touching the network if the interpreter is not running the build
# we expect. A silent uv re-sync once restarted a season on code missing every
# fix of that day, producing wrong player ids with no error.
say "--- preflight: interpreter + sportsdataverse build ---"
$PY python/preflight_build.py >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  say "PREFLIGHT FAILED -- refusing to start. See $LOG"
  tail -12 "$LOG"
  exit 1
fi
tail -4 "$LOG"

if [ "${CFB_PROXY:-0}" = "1" ]; then
  # Report the pool + remaining bandwidth once at start. Never prints the proxy
  # URL, which embeds login:password.
  $PY -c "import sys; sys.path.insert(0,'python'); import proxy_pool; proxy_pool.apply_to_env(); print('  ' + proxy_pool.status())" >> "$LOG" 2>&1
  tail -1 "$LOG"
fi

for YEAR in $(seq "$START_YEAR" "$END_YEAR"); do
  if [ "$FORCE" != "1" ] && grep -qx "$YEAR" "$CKPT"; then
    say "season $YEAR already complete (checkpoint) -- skipping"
    continue
  fi

  say "--- season $YEAR: refreshing schedule ---"
  $PY python/espn_cfb_01_schedules_scrape.py -s "$YEAR" -e "$YEAR" -r true 2>&1 | tee -a "$LOG"
  SCHED_RC=${PIPESTATUS[0]}
  if [ "$SCHED_RC" -ne 0 ]; then
    say "season $YEAR SCHEDULE FAILED rc=$SCHED_RC -- not checkpointing, continuing to next season"
    continue
  fi

  say "--- season $YEAR: rescraping all games (raw+final+participants+rosters+extras) ---"
  $PY python/espn_cfb_02_pbp_scrape.py -s "$YEAR" -e "$YEAR" -r true 2>&1 | tee -a "$LOG"
  RC=${PIPESTATUS[0]}

  if [ "$RC" -ne 0 ]; then
    say "season $YEAR SCRAPE FAILED rc=$RC -- not checkpointing; re-run to retry this season"
    continue
  fi

  # Post-season sanity: how much of the season actually landed non-empty?
  $PY python/verify_season_fill.py -s "$YEAR" 2>&1 | tee -a "$LOG"

  echo "$YEAR" >> "$CKPT"
  say "season $YEAR COMPLETE (checkpointed)"

  if [ "$CFB_RESCRAPE_GIT" = "1" ]; then
    git add cfb/ >/dev/null 2>&1 || true
    git commit -q -m "data(cfb): full rescrape season $YEAR" || say "season $YEAR: nothing to commit"
    git push -q || say "season $YEAR: push failed (continuing)"
  fi
done

say "=== rescrape finished ${START_YEAR}..${END_YEAR} ==="
echo "EXIT=0" | tee -a "$LOG"
