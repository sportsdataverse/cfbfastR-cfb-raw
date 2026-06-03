#!/bin/bash
# Rebuild final/ from on-disk raw for a season range (no re-scrape).
set -uo pipefail
while getopts s:e:f flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    f) FORCE="--force";;
  esac
done
FORCE=${FORCE:-}
mkdir -p logs

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_reprocess_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    uv run python python/reprocess_cfb_json.py -s "$i" -e "$i" $FORCE
    git pull >/dev/null
    git add cfb/json/final/* >/dev/null 2>&1 || true
    git commit -m "CFB Reprocess Update (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_reprocess_logfile_${i}.log"
  git pull --rebase >/dev/null || true
  git add "logs/cfb_reprocess_logfile_${i}.log"
  git commit -m "CFB Reprocess log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
done
