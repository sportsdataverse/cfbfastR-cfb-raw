# shellcheck shell=bash
# Shared commit+push helper. Source it; do not execute it.
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_commit.sh"
#
# Extracted 2026-08-29 from daily_cfb_scraper.sh, which had the only copy while
# reprocess_cfb.sh carried a byte-identical duplicate. Two copies of a
# push-retry routine is one copy too many: a fix to the rebase backend or the
# attempt count lands in one and not the other.
#
# THE PER-SEASON LOG CONVENTION (why this file exists at all)
# ----------------------------------------------------------
# Every stage that loops over seasons writes ONE logfile per season, named by
# the canonical pattern, and commits it INSIDE the loop:
#
#     logs/<stage>_logfile_<season>.log
#
# so a season's run record lands with that season's data rather than at the end
# of a multi-hour job that may never reach its own exit. `sdv_commit_log` below
# is the one-liner for it. Python stages get the same filename for free from
# `cfb_raw_scrape._cfb_raw_utils.get_logger(name, year)`.
#
# Tracked families: cfb_raw, cfb_json, cfb_schedules, cfb_reprocess,
# cfb_recruits, cfb_teams.
#
# ONE-OFF LOGS ARE NOT COMMITTED. An incident tool, a rescrape, a probe or a
# whole-run aggregate log writes to a temp file or a name .gitignore excludes --
# never to `logs/<stage>_logfile_<season>.log`. The per-season file is a
# reproducible artifact of a pipeline stage; a one-off is session state.

# Stage the given paths, commit, and push with a rebase-retry.
#
# Order matters: stage and commit FIRST so the tree is clean, and only then
# reconcile with origin. `rebase --merge` rather than `pull --rebase` because
# git's default am backend base64-encodes every parquet blob it replays.
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

# Commit one season's canonical logfile from inside a season loop.
#
#   sdv_commit_log cfb_recruits "$y"     -> logs/cfb_recruits_logfile_<y>.log
#
# Silently does nothing when the file is absent, so a stage that failed before
# opening its log does not also fail the commit.
sdv_commit_log() {
  local stage="$1" season="$2"
  local path="logs/${stage}_logfile_${season}.log"
  [ -f "$path" ] || { echo "no log to commit: $path"; return 0; }
  sdv_commit_push "CFB ${stage} log update (Start: ${season} End: ${season})" "$path"
}
