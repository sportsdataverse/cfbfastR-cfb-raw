#!/bin/bash
# Scrape raw CFB datasets per season (schedules -> json[+all aux/extras]).
set -uo pipefail

while getopts s:e:r: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    r) RESCRAPE=${OPTARG};;
  esac
done
RESCRAPE=${RESCRAPE:-false}
mkdir -p logs

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_raw_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    uv run python python/scrape_cfb_schedules.py -s "$i" -e "$i" -r "$RESCRAPE"
    uv run python python/scrape_cfb_json.py      -s "$i" -e "$i" -r "$RESCRAPE"
    git pull >/dev/null
    git add cfb/* >/dev/null 2>&1 || true
    git commit -m "CFB Raw Update (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  git pull --rebase >/dev/null || true
  git add "logs/cfb_raw_logfile_${i}.log"
  git commit -m "CFB Raw log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
done
