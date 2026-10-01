#!/bin/bash
# Reprocess a run of seasons one at a time, each under the git_pull sweep's lock, then (with
# --data) rebuild and publish cfbfastR-cfb-data for the same seasons.
#
# OPERATOR RUNBOOK (after a sportsdataverse lock bump that moved PROCESSING_VERSION)
#   bash scripts/reprocess_chain.sh                          # this season, then newest-first to 2004
#   bash scripts/reprocess_chain.sh --data                   # ... then cfb-data, season by season
#   SEASONS="2011 2010 2009" bash scripts/reprocess_chain.sh # resume an explicit list
#   detached: setsid nohup bash scripts/reprocess_chain.sh --data >/dev/null 2>&1 </dev/null &
# It prints its log path and a watch command, and ends with "chain done" and EXIT=<rc>. Each
# season is scripts/reprocess_cfb.sh, which skips finals already at the current
# processing_version, so re-running resumes.
#
# BEFORE YOU START (each of these cost a restart on 2026-10-01)
#   1. Land every sportsdataverse-py PR the reprocess should carry. The stamp is
#      <version>+<sdv-py commit>.<SCHEMA_REV>: a lock bump after the run re-stales the corpus.
#   2. Lock bump + SCHEMA_REV (python/cfb_raw_scrape/_cfb_raw_utils.py) by PR, here and in
#      cfbfastR-cfb-data; `uv sync` both; check the stamp:
#        .venv/bin/python -c 'import sys; sys.path.insert(0, "python"); from cfb_raw_scrape._cfb_raw_utils import PROCESSING_VERSION; print(PROCESSING_VERSION)'
#   3. See who else holds the locks: `fuser -v /tmp/git_pull_sdv.lock /tmp/cfbfastR-cfb-data-build.lock`.
#      Other sessions' rebuild jobs take both; this chain waits for them rather than failing.
#   4. Stopping it: kill by PID (`ps -eo pid,args | awk '$3=="scripts/reprocess_chain.sh"'`),
#      never `pkill -f` with a pattern your own shell's command line also contains.
#
# WHAT IT HANDLES
#   - The lock. The git_pull sweep (:40 every 4 h) and other jobs take /tmp/git_pull_sdv.lock;
#     each season waits up to LOCK_WAIT (3 h) and a timeout names the holder.
#   - Lock fds: each season runs with 9>&- (the subshell keeps the lock). git daemonizes
#     `gc --auto` and credential-cache--daemon (https push, ~15 min); an inherited fd held the
#     lock after the season ended (a repack of the 37 GB pack store held it 45 min).
#   - Auto gc, off for every git process here: a repack mid-chain only competes for IO.
#     scripts/_commit.sh does the same.
#   - The daily window: no season starts between PAUSE_FROM and PAUSE_TO ET (the 04:05 scrape).
#   - A failed season does not stop the chain; the end lists a SEASONS= rerun, and --data is
#     skipped when any season failed (it would compile stale finals).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

DATA=false
case "${1:-}" in
  --data) DATA=true ;;
  "") ;;
  *) echo "usage: [SEASONS=\"Y ...\"] bash scripts/reprocess_chain.sh [--data]" >&2; exit 2 ;;
esac
SEASONS=${SEASONS:-$(seq "${CFB_LAST:-$(date +%Y)}" -1 "${CFB_FIRST:-2004}")}
LOCK=${GIT_PULL_LOCK:-/tmp/git_pull_sdv.lock}
LOCK_WAIT=${LOCK_WAIT:-10800}
PAUSE_FROM=${PAUSE_FROM:-340}
PAUSE_TO=${PAUSE_TO:-515}
CFB_DATA_ROOT=${CFB_DATA_ROOT:-$(pwd)/../cfbfastR-cfb-data}
# git 2.25 (the droplet) ignores GIT_CONFIG_COUNT; GIT_CONFIG_PARAMETERS is what `git -c` sets
export GIT_CONFIG_PARAMETERS="${GIT_CONFIG_PARAMETERS:+$GIT_CONFIG_PARAMETERS }'gc.auto=0'"
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

mkdir -p logs
LOG="logs/cfb_reprocess_chain_$(date -u +%Y%m%d_%H%M%S).log"   # git-ignored (not *_logfile_YYYY)
exec > >(tee -a "$LOG") 2>&1
log() { echo "$(date -u +%FT%TZ) $*"; }
log "seasons: $(echo $SEASONS) | data: $DATA | lock wait: ${LOCK_WAIT}s"
echo "log:   $(pwd)/$LOG"
echo "watch: grep -a 'season\|data \|chain done\|held by' \"$(pwd)/$LOG\""

holders() { for p in $(fuser "$1" 2>/dev/null); do ps -o pid=,etime=,args= -p "$p" | cut -c1-160; done; }
pause() {
  local hm
  while :; do
    hm=$((10#$(TZ=America/New_York date +%H%M)))
    if [ "$hm" -ge "$PAUSE_FROM" ] && [ "$hm" -lt "$PAUSE_TO" ]; then log "pause window (${hm} ET)"; sleep 300; else return 0; fi
  done
}

failed=""
for y in $SEASONS; do
  pause
  log "season $y start"
  ( flock -w "$LOCK_WAIT" 9 || { log "season $y: no lock after ${LOCK_WAIT}s, held by:"; holders "$LOCK"; exit 1; }
    bash scripts/reprocess_cfb.sh -s "$y" -e "$y" 9>&- ) 9>"$LOCK"
  rc=$?
  log "season $y exit $rc"
  [ "$rc" -eq 0 ] || failed="$failed $y"
done

rc=0
if [ -n "$failed" ]; then
  rc=1
  log "failed seasons:$failed -- rerun: SEASONS=\"${failed# }\" bash scripts/reprocess_chain.sh$($DATA && echo " --data")"
fi
if $DATA; then
  if [ -n "$failed" ]; then
    log "data skipped: the reprocess failed for$failed"
  else
    for y in $SEASONS; do
      pause
      log "data $y start"
      # cron_daily_cfb.sh takes the cfb-data build lock and this lock itself, refuses a dirty or
      # off-main checkout, pulls, syncs and publishes; its output is long, so it gets a file
      ( cd "$CFB_DATA_ROOT" && bash scripts/cron_daily_cfb.sh -s "$y" -e "$y" ) \
        > "logs/cfb_data_chain_${y}_$(date -u +%Y%m%d_%H%M%S).log" 2>&1
      drc=$?
      log "data $y exit $drc"
      [ "$drc" -eq 0 ] || { rc=1; log "data $y failed: see logs/cfb_data_chain_${y}_*.log"; }
    done
  fi
fi
log "chain done"
echo "EXIT=$rc"
exit "$rc"
