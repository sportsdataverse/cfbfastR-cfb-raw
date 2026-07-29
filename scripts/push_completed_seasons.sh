#!/bin/bash
# Commit + push each season as the rescrape finishes it.
#
# Decoupled from scripts/rescrape_cfb_full.sh on purpose: bash reads a script
# incrementally, so editing the scraper mid-run risks corrupting execution.
# This watcher tails the scraper's checkpoint file instead and pushes each
# season the moment it is recorded there -- and a season is only checkpointed
# after its scrape returned rc=0 AND the fill verifier ran, so "checkpointed"
# already means "verified".
#
# Safe to start, stop, and restart at any point: logs/pushed_seasons.txt is the
# idempotence ledger, so a restart re-pushes nothing.
#
# USAGE
#   bash scripts/push_completed_seasons.sh            # watch until scraper is done
#   ONESHOT=1 bash scripts/push_completed_seasons.sh  # push what's ready, then exit
#
# ENV
#   PUSH_POLL_SECONDS   checkpoint poll interval (default 60)
#   PUSH_STOP_AFTER     last season expected; watcher exits once pushed (default 2025)
set -uo pipefail

POLL=${PUSH_POLL_SECONDS:-60}
STOP_AFTER=${PUSH_STOP_AFTER:-2025}
ONESHOT=${ONESHOT:-0}

CKPT="logs/rescrape_checkpoint.txt"
LEDGER="logs/pushed_seasons.txt"
LOG="logs/push_seasons.log"
mkdir -p logs
touch "$CKPT" "$LEDGER"

say() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

# A stale .git/index.lock (left by an interrupted git op, or a concurrent
# session) makes every git command fail. Wait for it briefly, and only clear it
# if it is genuinely stale -- never yank a lock a live process is holding.
wait_for_index_lock() {
  local waited=0
  while [ -f .git/index.lock ] && [ "$waited" -lt 120 ]; do
    sleep 5
    waited=$((waited + 5))
  done
  if [ -f .git/index.lock ]; then
    local age
    age=$(( $(date +%s) - $(stat -c %Y .git/index.lock 2>/dev/null || date +%s) ))
    if [ "$age" -gt 300 ]; then
      say "clearing stale index.lock (age ${age}s)"
      rm -f .git/index.lock
    else
      say "index.lock held by a live process -- skipping this tick"
      return 1
    fi
  fi
  return 0
}

push_season() {
  local YEAR=$1
  say "season $YEAR: preparing commit"

  wait_for_index_lock || return 1

  # NOTE: commit FIRST, rebase second. Pulling with --autostash before the
  # commit would stash the whole rescrape output (~2.9k files for 2004) while
  # the scraper is still actively writing into cfb/, and the un-stash can then
  # collide with files that changed underneath it. Committing first turns those
  # working-tree changes into a commit, so the later rebase only has to move
  # commits around and there is nothing large to stash.

  # Stage explicit paths only -- never a blind `git add -A`. A FAILED add must
  # not be mistaken for "nothing to commit": that is how the first version of
  # this script marked 2004 pushed while 2769 files sat uncommitted.
  if ! git add cfb/ logs/ scripts/ python/ 2>>"$LOG"; then
    say "season $YEAR: git add FAILED -- will retry next tick"
    return 1
  fi

  if git diff --cached --quiet; then
    # Genuinely nothing staged. Only benign if the season's files are already
    # committed; otherwise something is wrong and we must NOT mark it pushed.
    if [ "$(git status --porcelain -- cfb/ | wc -l | tr -d ' ')" -eq 0 ]; then
      say "season $YEAR: already committed, marking pushed"
      echo "$YEAR" >> "$LEDGER"
      return 0
    fi
    say "season $YEAR: nothing staged but cfb/ is dirty -- NOT marking pushed, retrying next tick"
    return 1
  fi

  local NFILES VERIFY
  NFILES=$(git diff --cached --name-only | wc -l | tr -d ' ')
  VERIFY=$(grep -E "^VERIFY $YEAR " logs/rescrape_cfb_full.log 2>/dev/null | tail -5)
  local REFUSED
  REFUSED=$(grep -c "REFUSED degraded write" logs/rescrape_cfb_full.log 2>/dev/null || echo 0)

  # Pre-commit hooks in this repo rewrite staged files (doctoc, whitespace) and
  # abort the commit when they do, so a single attempt is not enough -- re-add
  # and retry once.
  if ! git commit -q -F - <<EOF
data(cfb): full rescrape season $YEAR

Regular season + bowls, rescraped from scratch: raw summary, enriched final,
play_participants, game_rosters, power_index, betting, team_box_extra.

Files staged: $NFILES
Degraded writes refused so far (banked copy kept): $REFUSED

$VERIFY
EOF
  then
    say "season $YEAR: commit aborted (hooks likely rewrote files) -- re-adding and retrying"
    git add cfb/ logs/ scripts/ python/ 2>>"$LOG" || true
    if ! git commit -q -m "data(cfb): full rescrape season $YEAR ($NFILES files)"; then
      say "season $YEAR: COMMIT FAILED AGAIN -- leaving staged, will retry next tick"
      return 1
    fi
  fi

  # Never trust the commit landed just because the hook output looked green.
  local HEAD_SHA
  HEAD_SHA=$(git rev-parse HEAD)
  if ! git log -1 --format=%s | grep -q "season $YEAR"; then
    say "season $YEAR: HEAD is not the season commit -- NOT marking pushed"
    return 1
  fi

  # PUSH FIRST, rebase only if the remote actually moved.
  #
  # The previous version rebased unconditionally, and `--autostash` then had to
  # stash the working tree. On Windows that FAILS while the scraper holds
  # logs/cfb_json_logfile_<season>.log open ("unable to unlink old ..."), which
  # left a half-finished rebase plus a pile of orphaned autostashes and blocked
  # every subsequent push. We are normally the only pusher here, so a plain push
  # succeeds and no rebase -- and no stash -- is needed at all.
  if ! git push -q 2>>"$LOG"; then
    say "season $YEAR: PUSH REJECTED (remote moved, or network) -- commit $HEAD_SHA is local"
    say "season $YEAR: NOT auto-rebasing: the scraper holds logs open and a rebase here"
    say "season $YEAR: corrupts the tree. Retrying next tick; if this persists, rebase by hand"
    say "season $YEAR: once the scrape is idle."
    return 1
  fi

  # Confirm the remote actually has it before writing the ledger. The ledger is
  # the idempotence record; a false entry means the season is never retried.
  git fetch -q origin main 2>>"$LOG" || true
  if ! git merge-base --is-ancestor "$HEAD_SHA" origin/main 2>/dev/null; then
    say "season $YEAR: push reported ok but $HEAD_SHA is not on origin/main -- NOT marking pushed"
    return 1
  fi

  say "season $YEAR: PUSHED + VERIFIED ON REMOTE ($NFILES files, $HEAD_SHA)"
  echo "$YEAR" >> "$LEDGER"
}

say "=== push watcher start (poll=${POLL}s stop_after=${STOP_AFTER} oneshot=${ONESHOT}) ==="

while true; do

  while read -r YEAR; do
    [ -z "$YEAR" ] && continue
    grep -qx "$YEAR" "$LEDGER" && continue
    push_season "$YEAR" || true
  done < "$CKPT"

  if grep -qx "$STOP_AFTER" "$LEDGER"; then
    say "=== final season $STOP_AFTER pushed -- watcher exiting ==="
    break
  fi
  [ "$ONESHOT" = "1" ] && { say "oneshot done"; break; }
  sleep "$POLL"
done
