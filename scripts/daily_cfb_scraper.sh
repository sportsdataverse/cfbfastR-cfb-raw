#!/bin/bash
# Scrape raw CFB datasets per season (schedules -> json[+all aux/extras]).
set -uo pipefail

# Resolve this repo's interpreter (never `uv run` in a long job -- it re-syncs).
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"

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


# Commit + push, surviving a remote that moved while the build was running.
#
# The previous form pulled BEFORE staging, which can only ever abort: the build
# has just rewritten the tracked parquet/csv files, so `git pull` refuses with
# "Your local changes would be overwritten by merge". It then committed anyway,
# pushed into a non-fast-forward rejection, and swallowed all of it in
# `>/dev/null` with no rc check -- a GREEN job that published nothing. See
# wehoop-wnba-data runs 32192069433 + 32192069566 (2026-08-18).
#
# Order matters: stage and commit FIRST so the tree is clean, and only then
# reconcile with origin. `rebase --merge` rather than `pull --rebase` because
# git's default am backend base64-encodes every parquet blob it replays.
# shellcheck source=scripts/_commit.sh
source "$(dirname "${BASH_SOURCE[0]}")/_commit.sh"

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_raw_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    "$PY" python/espn_cfb_01_schedules_scrape.py -s "$i" -e "$i" -r "$RESCRAPE"
    "$PY" python/espn_cfb_02_pbp_scrape.py      -s "$i" -e "$i" -r "$RESCRAPE" --hollow "$HOLLOW"
    sdv_commit_push "CFB Raw Update (Start: $i End: $i)" cfb || PUSH_RC=1
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  # NOT `pull --rebase`: git's default am backend base64-encodes every blob it
  # replays, and this repo's .git is ~20 GB of parquet/JSON -- it stalls. The
  # merge backend replays by tree instead. (rebase.backend=merge is not an
  # option: it landed in git 2.26 and the scrape host runs 2.25.1.)
  sdv_commit_push "CFB Raw log update (Start: $i End: $i)" "logs/cfb_raw_logfile_${i}.log" || PUSH_RC=1
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
  # Deliberately NOT `|| PUSH_RC=1`: a 247 outage must not fail the game-data
  # run. But it does get the retry + sync, so a moved origin no longer silently
  # discards the signing class.
  sdv_commit_push "CFB Recruits Update" cfb/recruits || echo "!! recruits push failed (non-fatal)"
} 2>&1 | tee -a "$(mktemp "/tmp/cfb_recruits_daily.XXXXXX.log")"

# A rejected push is a FAILED run, not a green one. Release assets upload on a
# separate path and can succeed while the repo mirror is left stale.
if [ "${PUSH_RC:-0}" != "0" ]; then
  echo "::error ::At least one commit failed to reach origin; the repo mirror is stale."
  exit 1
fi
