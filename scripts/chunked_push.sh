#!/bin/bash
# Commit + push a large final/ rebuild in season-sized chunks.
#
# WHY THIS EXISTS
# ---------------
# A single ~20k-file commit produces a pack GitHub refuses twice over:
#
#   fatal: protocol error: bad line length 65515          (HTTP/2)
#   error: RPC failed; curl 6 Recv failure                (HTTP/1.1, postBuffer 500MB)
#
# `scripts/reprocess_cfb.sh` already avoids this by committing and pushing PER
# SEASON. This script restores that shape for an after-the-fact bulk rebuild
# (e.g. a model retrain that touches every game), where the work is already in
# the worktree and there is no per-season loop to hang the commits off.
#
# NOTES THAT COST TIME TO LEARN
# -----------------------------
# * The file list comes from `git diff --name-only HEAD`, which takes ~12s over
#   20k modified JSON. `git status` over the same tree takes >10 MINUTES.
# * Never run two of these (or any two git jobs) against the same repo at once
#   -- they collide on .git/index.lock and one silently rolls back.
# * `--no-verify` is deliberate: these are machine-generated payloads already
#   validated by the reprocess pipeline, and running the full hook suite over
#   20k files per chunk makes the run untenable.
#
# USAGE
#   scripts/chunked_push.sh                 # default 1000-file chunks
#   SIZE=500 scripts/chunked_push.sh        # smaller chunks if a push still fails
#   SUBJECT="feat(reprocess): ..." scripts/chunked_push.sh
#
# Resumable: each chunk is its own commit+push, so a mid-run failure leaves the
# completed chunks on the remote and the remainder still in the worktree. Just
# rerun it.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || exit 1

LIST=$(mktemp)
CHUNK=$(mktemp)
trap 'rm -f "$LIST" "$CHUNK"' EXIT
SIZE=${SIZE:-1000}
PATHSPEC=${PATHSPEC:-cfb/json/final}
SUBJECT=${SUBJECT:-"feat(reprocess): rebuild final/"}
BODY=${BODY:-"Bulk final/ rebuild split into season-sized commits: a single ~20k-file commit produces a pack GitHub resets on both HTTP/2 and HTTP/1.1."}

git diff --name-only HEAD -- "$PATHSPEC" > "$LIST" || exit 1
total=$(wc -l < "$LIST")
echo "files to push: $total in chunks of $SIZE (pathspec: $PATHSPEC)"
[ "$total" -eq 0 ] && { echo "nothing to do"; exit 0; }

i=0; n=0
while [ "$i" -lt "$total" ]; do
  n=$((n + 1))
  tail -n +$((i + 1)) "$LIST" | head -n "$SIZE" > "$CHUNK"
  cnt=$(wc -l < "$CHUNK")
  [ "$cnt" -eq 0 ] && break
  git add --pathspec-from-file="$CHUNK" 2>/dev/null
  if git diff --cached --quiet; then
    echo "chunk $n: already current ($((i + cnt))/$total)"
    i=$((i + SIZE)); continue
  fi
  git commit -q --no-verify -m "$SUBJECT (chunk $n)" -m "$BODY" || {
    echo "chunk $n: commit failed"; exit 1
  }
  if git push -q origin HEAD:main 2>/dev/null; then
    echo "chunk $n: pushed $cnt ($((i + cnt))/$total)"
  else
    echo "chunk $n: PUSH FAILED at $((i + cnt))/$total -- rerun to resume (try SIZE=500)"
    exit 1
  fi
  i=$((i + SIZE))
done
echo "ALL CHUNKS PUSHED ($total files)"
