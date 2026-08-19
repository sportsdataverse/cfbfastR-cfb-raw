#!/bin/bash
# Scrape raw CFB datasets per season (schedules -> json[+all aux/extras]).
set -uo pipefail

while getopts s:e:r:h: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    r) RESCRAPE=${OPTARG};;
    h) HOLLOW=${OPTARG};;
  esac
done
RESCRAPE=${RESCRAPE:-false}
HOLLOW=${HOLLOW:-false}
END_YEAR=${END_YEAR:-$START_YEAR}
mkdir -p logs

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_raw_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    uv run python python/scrape_cfb_schedules.py -s "$i" -e "$i" -r "$RESCRAPE"
    uv run python python/scrape_cfb_json.py      -s "$i" -e "$i" -r "$RESCRAPE" --hollow "$HOLLOW"
    git pull >/dev/null
    git add cfb/* >/dev/null 2>&1 || true
    git commit -m "CFB Raw Update (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  # NOT `pull --rebase`: git's default am backend base64-encodes every blob it
  # replays, and this repo's .git is ~20 GB of parquet/JSON -- it stalls. The
  # merge backend replays by tree instead. (rebase.backend=merge is not an
  # option: it landed in git 2.26 and the scrape host runs 2.25.1.)
  git fetch --quiet origin main && git rebase --merge origin/main >/dev/null || true
  git add "logs/cfb_raw_logfile_${i}.log"
  git commit -m "CFB Raw log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
done

# Recruiting: CLASS-year keyed, not game-season keyed, so it runs ONCE here
# rather than inside the loop above -- iterating it per game season would
# re-scrape the same signing class N times.
#
# Cheap by construction: a signed class is immutable, so every prior year is
# skipped on sight and only the current cycle actually fetches (~4 min).
#
# Failure-isolated on purpose. Recruiting is a side dataset; 247 being down or
# rate-limiting must never fail the game-data run that is this script's job.
{
  bash scripts/10_scrape_recruits.sh || echo "!! recruits scrape failed (non-fatal)"
  git pull >/dev/null 2>&1 || true
  git add cfb/recruits >/dev/null 2>&1 || true
  git commit -m "CFB Recruits Update" >/dev/null 2>&1 || echo "No recruit changes to commit"
  git push >/dev/null 2>&1 || true
} 2>&1 | tee -a "logs/cfb_recruits_daily.log"
