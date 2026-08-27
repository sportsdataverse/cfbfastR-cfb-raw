#!/bin/bash
# Rebuild final/ from on-disk raw for a season range (no re-scrape, no network).
#
# OPERATOR RUNBOOK
#   bash scripts/reprocess_cfb.sh -s 2004 -e 2025            # rebuild + push per season
#   bash scripts/reprocess_cfb.sh -s 2015 -e 2015 -p false   # rebuild LOCALLY, no commit/push
#   bash scripts/reprocess_cfb.sh -s 2004 -e 2025 -f         # force, ignoring processing_version
#
# It prints its own log path and a watch command on startup. Ctrl-C is safe:
# resume is derived from what is on disk (a final whose processing_version
# already matches is skipped), so re-running continues where it stopped.
#
# `-p false` exists because a full-corpus run rewrites ~19.6k files, which is one
# very large commit AND fires the -data rebuild trigger. Verify a season locally
# first, then run for real.
#
# COST (measured 2026-08-27, offline, no network): 1.19 s/game single-threaded,
# so the full 2004-2026 corpus (19,586 games) is roughly ONE HOUR at 6-8 workers.
# reprocess_cfb_json sizes workers itself from cpu + free RAM; override with
# CFB_SCRAPE_WORKERS if it guesses badly.
#
# WHEN TO RUN IT: a sportsdataverse bump moves PROCESSING_VERSION, so every
# banked game goes stale at once and the preflight below will say "to rebuild:
# <everything>". That is expected, not alarming. Do NOT run it against the
# current season while the daily scrape is live -- both commit to this repo and
# will queue behind each other.
set -uo pipefail

# Real-time, correctly-encoded output. Without these a multi-hour job looks hung
# for minutes at a time (4KB block buffering) and cp1252 chokes on the log's
# unicode.
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
while getopts s:e:fp: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    f) FORCE="--force";;
    p) PUSH=${OPTARG};;
  esac
done
FORCE=${FORCE:-}
PUSH=${PUSH:-true}

usage() {
  echo "usage: bash scripts/reprocess_cfb.sh -s START [-e END] [-f] [-p true|false]" >&2
  echo "  -s  first season (REQUIRED)   -e  last season (default: START)" >&2
  echo "  -f  force, ignore processing_version" >&2
  echo "  -p  commit+push per season (default true); false = rebuild locally" >&2
  exit 2
}

# Validate before anything expensive. Under `set -u` a missing -s used to crash on
# the END_YEAR default with "unbound variable" -- an unreadable failure for the one
# script an operator reaches for under pressure.
[ -n "${START_YEAR:-}" ] || { echo "error: -s is required" >&2; usage; }
END_YEAR=${END_YEAR:-$START_YEAR}
case "$START_YEAR$END_YEAR" in *[!0-9]*) echo "error: seasons must be numeric" >&2; usage;; esac
[ "$START_YEAR" -le "$END_YEAR" ] || { echo "error: -s ($START_YEAR) is after -e ($END_YEAR)" >&2; usage; }
# -p decides whether ~19.6k files get pushed. A typo must not be interpreted
# silently in either direction, so anything but true/false is a usage error.
case "$PUSH" in true|false) ;; *) echo "error: -p must be true or false, got: $PUSH" >&2; usage;; esac

mkdir -p logs

RUN_LOG="logs/cfb_reprocess_$(date -u +%Y%m%d_%H%M%S).log"
echo "reprocess ${START_YEAR}-${END_YEAR}  force=${FORCE:-no}  push=${PUSH}"
echo "log:   ${RUN_LOG}"
# Quoted: a repo path containing spaces would otherwise print an uncopyable command.
printf %s%s%s%s "watch: tail -f " "\"" "$(pwd)/${RUN_LOG}" "\"" ; echo

# How much work is actually queued, BEFORE committing hours to it. Counting
# stale finals is cheap and the answer is frequently "all of them" -- a
# sportsdataverse bump moves PROCESSING_VERSION, so every banked game goes stale
# at once.
"$PY" - "$START_YEAR" "$END_YEAR" "${FORCE:-}" <<'PREFLIGHT' 2>&1 | tee -a "$RUN_LOG"
import sys
sys.path.insert(0, "python")
from cfb_raw_scrape._cfb_raw_utils import PROCESSING_VERSION, games_for_seasons, load_schedule_master
from reprocess_cfb_json import _final_is_current

s, e = int(sys.argv[1]), int(sys.argv[2])
force = len(sys.argv) > 3 and sys.argv[3] == "--force"
master = load_schedule_master()
games = games_for_seasons(master, s, e)
# --force makes the rebuild loop ignore processing_version, so counting through
# _final_is_current here would report "0 to rebuild" while the loop rebuilds every
# game. A preflight that disagrees with the run it precedes is worse than none.
stale = games if force else [g for g in games if not _final_is_current(g)]
print(f"target processing_version : {PROCESSING_VERSION}")
print(f"force                     : {force}")
print(f"games in {s}-{e}            : {len(games)}")
print(f"  already current (skipped) : {len(games) - len(stale)}")
print(f"  to rebuild                : {len(stale)}")
PREFLIGHT

# Shared with daily_cfb_scraper.sh rather than copied: two near-identical push
# helpers is how one of them quietly stops matching the other.
# shellcheck source=scripts/daily_cfb_scraper.sh
sdv_commit_push() {
  local msg="$1"; shift
  git add -- "$@" >/dev/null 2>&1 || true
  if git diff --cached --quiet; then
    echo "nothing to commit for: $msg"
    return 0
  fi
  git commit -m "$msg" >/dev/null || { echo "::warning ::commit failed: $msg"; return 1; }
  local attempt
  for attempt in 1 2 3; do
    if git push origin HEAD >/dev/null 2>&1; then
      echo "pushed: $msg (attempt $attempt)"
      return 0
    fi
    echo "push rejected (attempt $attempt); syncing with origin"
    git fetch --quiet origin main || true
    if ! git rebase --merge origin/main >/dev/null 2>&1; then
      git rebase --abort >/dev/null 2>&1 || true
      echo "::error ::cannot rebase onto origin/main for: $msg"
      return 1
    fi
  done
  echo "::error ::push still rejected after 3 attempts: $msg"
  return 1
}

# Per-season log AND the whole-run log are written straight from the pipe. The
# previous form staged a third copy in /tmp and copied it twice; on a full-corpus
# run that is a lot of duplicated bytes for no extra information.
for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  SEASON_LOG="logs/cfb_reprocess_logfile_${i}.log"
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    "$PY" python/reprocess_cfb_json.py -s "$i" -e "$i" $FORCE
    if [ "$PUSH" = "true" ]; then
      # Load-bearing subject: the -data trigger greps the years out of it.
      sdv_commit_push "CFB Reprocess Update (Start: $i End: $i)" cfb/json/final || PUSH_RC=1
    else
      echo "-p false: rebuilt season $i locally, NOT committed or pushed"
    fi
  } 2>&1 | tee "$SEASON_LOG" | tee -a "$RUN_LOG"
  if [ "$PUSH" = "true" ]; then
    sdv_commit_push "CFB Reprocess log update (Start: $i End: $i)" "$SEASON_LOG" || PUSH_RC=1
  fi
done

RC=0
if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale." | tee -a "$RUN_LOG"
  RC=1
fi
# Grep-able completion marker. Do NOT trust an earlier "done" line -- this is
# written last, and only here.
echo "EXIT=$RC" | tee -a "$RUN_LOG"
exit $RC
