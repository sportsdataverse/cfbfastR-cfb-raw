#!/bin/bash
# Monthly CFB stages: 50 recruits (247Sports), 51 player_core, 52 espn_recruits.
#
# WHY THESE ARE NOT IN THE DAILY LOOP. All three are per-entity state that moves
# on a roster/recruiting cycle, not a game cycle. A recruiting class changes on
# commitments and signing day; a core record changes on transfers and roster
# churn. Running them daily is wasted work against hosts that rate-limit, and
# running them never is how they go stale -- which is exactly what happened to
# the 2027 class, frozen at 4,779 rows from 2026-08-06 until this month.
#
# EACH STAGE DECIDES ITS OWN WORK. There is no preflight here computing what is
# stale, deliberately: that logic belongs in the stage, where the daily driver
# and a manual run get it too. Every stage already implements the same contract
# -- is_complete() returns False for anything still open, so a stage invoked
# over a wide range fetches only what genuinely needs refreshing and skips the
# rest instantly. This script's job is cadence, not policy.
#
# DRY RUN: `CFB_PY=echo ./scripts/monthly_cfb_scraper.sh` -- NOT `PY=echo`.
# _venv.sh sets PY unconditionally, so a PY= prefix is clobbered and the stages
# run for real. CFB_PY is the override _venv.sh actually honours. (Learned the
# hard way: a "dry run" of this script scraped and committed two live classes.)
# Note even a stubbed run still executes the git helpers -- do not dry-run this
# while another committing job is working in the same checkout.
#
#   ./scripts/monthly_cfb_scraper.sh              # current + previous class year
#   ./scripts/monthly_cfb_scraper.sh -s 2006 -e 2028   # full backfill
#
# Pace: CFB_TEAM_STATS_WORKERS-style knobs live on the stages; 247 retry/backoff
# is SDV_PY_247_RETRIES / SDV_PY_247_BACKOFF.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# shellcheck source=scripts/_venv.sh
source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
# shellcheck source=scripts/_commit.sh
source "$(dirname "${BASH_SOURCE[0]}")/_commit.sh"

while getopts s:e:r: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    r) RESCRAPE=${OPTARG};;
  esac
done
RESCRAPE=${RESCRAPE:-false}
# A class year rolls in August, so the CURRENT class is next calendar year once
# the season starts. Default covers the open class and the one just signed --
# the only two that can still move.
# PYTHONPATH is REQUIRED on every inline python call here: the package lives in
# python/ and is not installed, so without it `from cfb_raw_scrape...` raises
# ModuleNotFoundError, the $() yields EMPTY, and the stage is invoked with -s ''.
# Caught by running this script for real -- see the dry-run note above.
export PYTHONPATH="${PYTHONPATH:-}${PYTHONPATH:+:}python"
THIS_CLASS=$("$PY" -c 'from datetime import date; d=date.today(); print(d.year + (1 if d.month >= 8 else 0))')
START_YEAR=${START_YEAR:-$((THIS_CLASS - 1))}
END_YEAR=${END_YEAR:-$THIS_CLASS}
mkdir -p logs

LOG="logs/monthly_cfb_scraper.log"
echo "=== monthly ${START_YEAR}-${END_YEAR} rescrape=${RESCRAPE} $(date -u +%FT%TZ) ===" >> "$LOG"
RC=0

run_stage() {  # <script> <label> <first> <last>
  local script="$1" label="$2" first="$3" last="$4"
  echo "--- ${label} ${first}-${last} $(date -u +%FT%TZ) ---" >> "$LOG"
  PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 TQDM_DISABLE=1 \
    "$PY" "python/${script}" -s "$first" -e "$last" -r "$RESCRAPE" >> "$LOG" 2>&1 \
    || { echo "!! ${label} returned non-zero" >> "$LOG"; RC=1; }
}

# 50 + 52 are class-year keyed. 51 is SEASON keyed (it reads game rosters), so
# it takes the played season, not the class year -- passing a class year would
# ask for rosters from a season that has not happened.
run_stage espn_cfb_50_recruits_scrape.py      "50 recruits (247)" "$START_YEAR" "$END_YEAR" bare
run_stage espn_cfb_52_espn_recruits_scrape.py "52 espn_recruits"  "$START_YEAR" "$END_YEAR" value

PLAYED_SEASON=$("$PY" -c 'from cfb_raw_scrape._cfb_raw_utils import most_recent_cfb_season as m; print(m())')
run_stage espn_cfb_51_player_core_scrape.py   "51 player_core"    "$PLAYED_SEASON" "$PLAYED_SEASON" value

sdv_commit_push "CFB monthly update (recruits ${START_YEAR}-${END_YEAR}, player_core ${PLAYED_SEASON})" cfb || RC=1
for stage in cfb_recruits cfb_espn_recruits; do
  for y in $(seq "$START_YEAR" "$END_YEAR"); do
    sdv_commit_log "$stage" "$y" || RC=1
  done
done
sdv_commit_log cfb_player_core "$PLAYED_SEASON" || RC=1

echo "=== done $(date -u +%FT%TZ) EXIT=${RC} ===" >> "$LOG"
exit "$RC"
