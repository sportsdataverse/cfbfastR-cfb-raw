#!/bin/bash
# Rebuild final/ from on-disk raw for a season range (no re-scrape).
set -uo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
while getopts s:e:f flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    f) FORCE="--force";;
  esac
done
FORCE=${FORCE:-}
END_YEAR=${END_YEAR:-$START_YEAR}
mkdir -p logs

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

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_reprocess_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    "$PY" python/reprocess_cfb_json.py -s "$i" -e "$i" $FORCE
    # Load-bearing subject: the -data trigger greps the years out of it.
    sdv_commit_push "CFB Reprocess Update (Start: $i End: $i)" cfb/json/final || PUSH_RC=1
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_reprocess_logfile_${i}.log"
  sdv_commit_push "CFB Reprocess log update (Start: $i End: $i)" "logs/cfb_reprocess_logfile_${i}.log" || PUSH_RC=1
  rm -f "$TMPLOG"
done

if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale."
  exit 1
fi
