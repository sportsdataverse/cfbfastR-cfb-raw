"""Post-scrape fill check for one season: did the data actually land?

Prints one OK/WARN line per dataset so a long rescrape is auditable from the
log alone. Exits non-zero only on a hard problem (missing files), not on the
structurally-empty pre-2014 participants -- ESPN ships no per-play
``participants[]`` before 2014, so an empty participants set is EXPECTED there
and reported as INFO rather than WARN.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _cfb_raw_utils import games_for_seasons, load_schedule_master

# Season from which ESPN populates per-play participants[]. Verified live
# against Core v2 /competitions/{id}/plays: 2004-2013 return a full play stream
# with zero participants; 2014+ carry them on ~90% of plays.
PARTICIPANTS_MIN_SEASON = 2014

# A stamped-but-empty record list is a few dozen bytes; a real one is >50KB.
EMPTY_BYTES = 2_000


def _tally(paths: list[str]) -> tuple[int, int, int]:
    """Return (ok, empty, missing) for a list of expected file paths."""
    ok = empty = missing = 0
    for p in paths:
        if not os.path.exists(p):
            missing += 1
        elif os.path.getsize(p) < EMPTY_BYTES:
            empty += 1
        else:
            ok += 1
    return ok, empty, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--season", type=int, required=True)
    ap.add_argument(
        "--warn-pct",
        type=float,
        default=5.0,
        help="warn when more than this %% of games are empty/missing",
    )
    args = ap.parse_args()
    season = args.season

    master = load_schedule_master()
    games = games_for_seasons(master, season, season)
    n = len(games)
    if n == 0:
        print(f"VERIFY {season}: FAIL no games in schedule master")
        sys.exit(1)

    datasets = {
        "raw": [f"cfb/json/raw/{g}.json" for g in games],
        "final": [f"cfb/json/final/{g}.json" for g in games],
        "game_rosters": [f"cfb/game_rosters/json/{g}.json" for g in games],
        "play_participants": [f"cfb/play_participants/json/{g}.json" for g in games],
    }

    bad_overall = False
    for name, paths in datasets.items():
        ok, empty, missing = _tally(paths)
        pct_bad = 100.0 * (empty + missing) / n

        expected_empty = (
            name == "play_participants" and season < PARTICIPANTS_MIN_SEASON
        )
        if expected_empty:
            level = "INFO"
            note = " (pre-2014: ESPN ships no participants[] -- expected)"
        elif pct_bad > args.warn_pct:
            level = "WARN"
            note = ""
            bad_overall = True
        else:
            level = "OK"
            note = ""

        print(
            f"VERIFY {season} {level:<4} {name:<18} "
            f"games={n:>4} ok={ok:>4} empty={empty:>4} missing={missing:>3} "
            f"bad={pct_bad:5.1f}%{note}"
        )

    # Cheap content probe: confirm a real final actually carries plays + names.
    #
    # Skip zero-play games. ESPN marks some games STATUS_FINAL while publishing
    # no play-by-play at all (2014 game 400548403: drives.previous=0,
    # playByPlayAvailable=None -- yet 193 participants rows, so the scrape was
    # fine). Probing one of those reports a meaningless id_given_name=0.0% and
    # reads like a total failure.
    probe = None
    for g in games:
        path = f"cfb/json/final/{g}.json"
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                candidate = json.load(fh)
        except Exception:  # noqa: BLE001 - probe is best-effort
            continue
        if candidate.get("plays"):
            probe = g
            break
    if probe is not None:
        try:
            with open(f"cfb/json/final/{probe}.json", encoding="utf-8") as fh:
                obj = json.load(fh)
            plays = obj.get("plays") or []
            named = sum(
                1
                for p in plays
                if p.get("rusher_player_name") or p.get("passer_player_name")
            )
            with_id = sum(
                1
                for p in plays
                if p.get("rusher_player_id") or p.get("passer_player_id")
            )
            pct = 100.0 * with_id / named if named else 0.0
            print(
                f"VERIFY {season} PROBE game={probe} plays={len(plays)} "
                f"named={named} with_id={with_id} id_given_name={pct:.1f}%"
            )
        except Exception as exc:  # noqa: BLE001 - probe is best-effort
            print(f"VERIFY {season} PROBE game={probe} failed: {exc}")

    sys.exit(1 if bad_overall else 0)


if __name__ == "__main__":
    main()
