"""Fail fast if the interpreter is not running the sportsdataverse build we expect.

WHY THIS EXISTS: `uv run` re-syncs the environment before every invocation, and
uv.lock pins sportsdataverse to the PyPI RELEASE even though pyproject declares
git@main. On 2026-07-28 that silently reinstalled the released build over a
locally-installed one mid-backfill and restarted a season on code missing every
fix of that day -- with no error, just wrong output.

A multi-hour backfill must not be able to start on the wrong code. Run this
before the season loop; a non-zero exit aborts the run.
"""

from __future__ import annotations

import sys

FAILURES: list[str] = []


def _require(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


def main() -> None:
    import sportsdataverse
    from sportsdataverse.cfb import cfb_pbp
    from sportsdataverse.cfb.cfb_pbp import CFBPlayProcess

    print(f"  python : {sys.executable}")
    print(f"  sdv    : {sportsdataverse.__file__}")

    # Symbols this backfill's correctness depends on. Each maps to a fix whose
    # absence silently produces wrong player ids rather than an error.
    for sym in ("_PLAYER_NAME_TEAM_SUFFIX", "_PLAYER_NAME_TAIL", "_norm_player_name"):
        _require(hasattr(cfb_pbp, sym), f"cfb_pbp missing {sym}")
    _require(
        hasattr(CFBPlayProcess, "_CFBPlayProcess__boxscore_records"),
        "CFBPlayProcess missing __boxscore_records (box-score id source)",
    )

    # Behavioural gate, not just a symbol check: a tail word must not resolve
    # through the surname fallback to a different athlete.
    if not FAILURES:
        import polars as pl

        proc = CFBPlayProcess(gameId=1)
        proc.join_participants = False
        proc.game_roster = [
            {"athlete_id": 701, "full_name": "Russell Wilson", "team_id": 70},
            {"athlete_id": 702, "full_name": "Alex Screen", "team_id": 70},
        ]
        out = proc._CFBPlayProcess__attach_player_ids(
            pl.DataFrame(
                [{"passer_player_name": "Russell Wilson screen", "pos_team": 70}]
            ),
        )
        got = (out["passer_player_name"][0], out["passer_player_id"][0])
        _require(
            got == ("Russell Wilson", 701),
            f"tail/fallback behaviour wrong: got {got}, expected ('Russell Wilson', 701)",
        )

    if FAILURES:
        print("  build  : FAILED")
        for f in FAILURES:
            print(f"    - {f}")
        print("  hint   : the venv was probably re-synced by `uv run`; reinstall the")
        print(
            "           intended build and invoke via CFB_PY=./.venv/Scripts/python.exe"
        )
        sys.exit(1)

    print("  build  : OK")


if __name__ == "__main__":
    main()
