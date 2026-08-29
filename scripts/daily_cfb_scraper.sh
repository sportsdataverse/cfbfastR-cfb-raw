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
    # Cold-start execution order (see any stage docstring for the full list).
    # 01 teams first: reference data with no upstream, and the extended
    # schedule/team-info interface will read it.
    "$PY" python/espn_cfb_01_teams_scrape.py -s "$i" -e "$i" -r "$RESCRAPE" \
      || echo "!! teams scrape failed for $i (non-fatal)"
    # 02 schedules is the game-id gate -- everything below needs the master.
    "$PY" python/espn_cfb_02_schedules_scrape.py -s "$i" -e "$i" -r "$RESCRAPE"
    # 04 game_rosters and 05 play_participants are NOT in the daily loop, and
    # that is deliberate: scrape_cfb_pbp already fetches both per game
    # (_rosters / _participants) and embeds them in the json/final payload, so
    # running the standalone stages here would fetch the SAME data twice --
    # doubling a ~250 Core v2 $ref fan-out per game against an endpoint that
    # 403s under load. They remain as standalone stages for backfilling their
    # own cfb/game_rosters and cfb/play_participants trees.
    # 06 pbp -- the expensive stage, and the one the others feed.
    # 03 team_rosters -- reads 02 schedules? no: it reads 01 TEAMS for its id
    # list, so it must follow 01. Current-season only by construction; it
    # refuses to bank a season the endpoint cannot serve.
    "$PY" python/espn_cfb_03_team_rosters_scrape.py -s "$i" -e "$i" -r "$RESCRAPE" || echo "!! team_rosters scrape failed for $i (non-fatal)"
    "$PY" python/espn_cfb_04_pbp_scrape.py -s "$i" -e "$i" -r "$RESCRAPE" --hollow "$HOLLOW"
    # 10/11 are season-level and depend only on the schedule.
    # 07 standings -- season-keyed, one call for the whole conference tree.
    "$PY" python/espn_cfb_07_standings_scrape.py -s "$i" -e "$i" -r "$RESCRAPE" || echo "!! standings scrape failed for $i (non-fatal)"
    "$PY" python/espn_cfb_08_qbr_scrape.py -s "$i" -e "$i" \
      || echo "!! qbr scrape failed for $i (non-fatal)"
    "$PY" python/espn_cfb_09_power_index_scrape.py -s "$i" -e "$i" -r "$RESCRAPE" \
      || echo "!! power_index scrape failed for $i (non-fatal)"
    sdv_commit_push "CFB Raw Update (Start: $i End: $i)" cfb || PUSH_RC=1
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  # Every stage writes its own canonical per-season log via get_logger; commit
  # each in-loop so a season's run record lands with that season's data.
  # Verified against each stage's get_logger(): pbp logs as cfb_json, and qbr
  # has NO get_logger at all (0 tracked logs) so it is deliberately absent.
  # game_rosters/play_participants are not here because they are not in the
  # loop -- see the comment above.
  for stage in cfb_teams cfb_schedules cfb_team_rosters cfb_json cfb_standings cfb_power_index; do
    sdv_commit_log "$stage" "$i" || PUSH_RC=1
  done
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
  bash scripts/50_scrape_recruits.sh || echo "!! recruits scrape failed (non-fatal)"
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
