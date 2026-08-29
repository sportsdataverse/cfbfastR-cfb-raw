#!/bin/bash
# Backfill stage 06 team_stats, 2004 -> present.
#
# ~264 teams x 2 season types x 22 seasons = ~11.6k Core v2 requests. Core v2
# returns 403 under aggressive concurrency, so pace with CFB_TEAM_STATS_WORKERS
# (default 6) rather than editing anything.
#
#   CFB_TEAM_STATS_WORKERS=4 ./scripts/backfill_team_stats.sh 2004 2025
#
# Resumable: every (season, type, team) already banked and final is skipped, so
# a Ctrl-C and re-run costs only the unfinished remainder. Commits per season so
# an interrupted run never strands uncommitted captures.
set -uo pipefail
cd "$(dirname "$0")/.."
source scripts/_venv.sh
. scripts/_commit.sh

START="${1:-2004}"
END="${2:-$("$PY" -c 'from cfb_raw_scrape._cfb_raw_utils import most_recent_cfb_season as m; print(m())')}"
LOG="logs/backfill_team_stats.log"
mkdir -p logs

echo "=== team_stats backfill ${START}-${END} | workers=${CFB_TEAM_STATS_WORKERS:-6} | $(date -u +%FT%TZ) ===" >> "$LOG"
RC=0
for season in $(seq "$START" "$END"); do
  echo "--- season ${season} start $(date -u +%FT%TZ) ---" >> "$LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 \
    "$PY" python/espn_cfb_06_team_stats_scrape.py -s "$season" -e "$season" >> "$LOG" 2>&1 \
    || { echo "!! season ${season} returned non-zero" >> "$LOG"; RC=1; }
  sdv_commit_push "CFB team_stats backfill (${season})" cfb || RC=1
  sdv_commit_log cfb_team_stats "$season" || RC=1
done
echo "=== done $(date -u +%FT%TZ) EXIT=${RC} ===" >> "$LOG"
exit "$RC"
