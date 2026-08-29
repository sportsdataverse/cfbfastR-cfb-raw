#!/usr/bin/env bash
# One interpreter resolver for this repo's drivers. Source it, do not execute:
#
#     source "$(dirname "${BASH_SOURCE[0]}")/_venv.sh"
#     "$PY" python/espn_cfb_02_schedules_scrape.py -s 2026
#
# CFB_PY EXISTS BECAUSE `uv run` RE-SYNCS THE ENV before every invocation, so it
# can swap the interpreter under a running job: observed mid-run on 2026-07-28,
# when it reinstalled the PyPI release over the git@main build and restarted a
# season on code missing every fix from that day. Keep `uv run` for tests and
# lint; never for a long-running entry point.
#
# CFB_PY was already the repo's convention (rescrape_cfb_full.sh,
# retry_degraded_games.sh); this file just makes it the single definition
# instead of a default repeated per script -- and drops `uv run python` as the
# fallback, which is what reintroduced the bug whenever CFB_PY was unset.
#
# Order: CFB_PY override -> this repo's .venv -> loud failure. Never a bare
# `python`: pilots found drivers silently binding a SIBLING repo's venv.

_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -n "${CFB_PY:-}" ]; then
  PY="$CFB_PY"
elif [ -x "$_repo_root/.venv/Scripts/python.exe" ]; then   # Windows layout
  PY="$_repo_root/.venv/Scripts/python.exe"
elif [ -x "$_repo_root/.venv/bin/python" ]; then           # POSIX layout
  PY="$_repo_root/.venv/bin/python"
else
  echo "::error ::no interpreter. Run 'uv sync --frozen' in $_repo_root, or set CFB_PY." >&2
  return 1 2>/dev/null || exit 1
fi

export PY
# stderr, not stdout: drivers pipe stdout into logs and some callers parse it,
# so a banner here would contaminate their output. CFB_VENV_QUIET=1 silences it.
[ -n "${CFB_VENV_QUIET:-}" ] || echo "interpreter: $PY" >&2
