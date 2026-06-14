"""One-time investigation: CFBD air_yards fill rate on CFB pass plays.

Run: uv run --env-file .env python python/cpoe/inspect_cfbd_air_yards.py
"""
from __future__ import annotations

import os
import sys
import requests

CFBD_BASE = "https://api.collegefootballdata.com"

SAMPLE_GAMES = [
    (2020, 1, "Alabama"),
    (2021, 1, "Ohio State"),
    (2022, 1, "Georgia"),
    (2023, 1, "Michigan"),
    (2024, 1, "Texas"),
]

AIR_YARDS_CANDIDATES = ["air_yards", "yards_to_sticks", "pass_length", "passLength"]


def get_plays(year: int, week: int, team: str, token: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    r = requests.get(
        f"{CFBD_BASE}/plays",
        params={"seasonType": "regular", "year": year, "week": week, "offense": team},
        headers=headers,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    token = os.environ.get("CFB_DATA_API_KEY") or os.environ.get("CFBD_DATA_API_KEY")
    if not token:
        print("ERROR: CFB_DATA_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    total_pass = 0
    total_with_air: dict[str, int] = {c: 0 for c in AIR_YARDS_CANDIDATES}
    first_keys_printed = False

    for year, week, team in SAMPLE_GAMES:
        try:
            plays = get_plays(year, week, team, token)
            pass_plays = [
                p for p in plays
                if "pass" in str(p.get("playType") or p.get("play_type") or "").lower()
                   or p.get("passAttempt") or p.get("pass_attempt")
            ]
            print(f"\n{year} wk{week} {team}: {len(pass_plays)} pass plays (of {len(plays)} total)")
            if pass_plays and not first_keys_printed:
                print("  Keys:", sorted(pass_plays[0].keys()))
                first_keys_printed = True
            for cand in AIR_YARDS_CANDIDATES:
                has = sum(1 for p in pass_plays if p.get(cand) is not None)
                pct = has / len(pass_plays) * 100 if pass_plays else 0
                print(f"    {cand:25s}: {has:4d}/{len(pass_plays)} ({pct:.0f}%)")
                total_with_air[cand] += has
            total_pass += len(pass_plays)
        except Exception as exc:
            print(f"  ERROR {year} {team}: {exc}")

    print(f"\n{'='*60}")
    print(f"TOTAL PASS PLAYS SAMPLED: {total_pass}")
    for cand, n in total_with_air.items():
        pct = n / total_pass * 100 if total_pass else 0
        verdict = "FEASIBLE (>=60%)" if pct >= 60 else "INFEASIBLE (<60%)"
        print(f"  {cand:25s}: {n:5d}/{total_pass} ({pct:.0f}%) — {verdict}")


if __name__ == "__main__":
    main()
