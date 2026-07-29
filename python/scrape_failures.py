"""Report CFB scrape failures for review after each season run.

Two failure modes, both filesystem-derived (immune to log noise / retries):

1. ``no_final``     - a schedule-master game with no enriched ``final`` JSON
                      (the scraper's ``filter_undone`` would re-attempt it).
2. ``hollow_extras`` - the game scraped (final exists) but its Core v2 extras are
                      empty: both ``game_rosters`` and ``play_participants`` have an
                      empty ``data`` payload. This is the signature of ESPN Core v2
                      rate-limiting (throttle) — invisible to a "missing games"
                      check because the game itself succeeded via Site v2.

Writes ``logs/scrape_failures.csv`` (season, game_id, issue) sorted by season then
game_id, and prints a per-season summary: scheduled / missing_final / hollow_extras
plus the power_index empty-rate (informational; FCS games + pre-2015 legitimately
lack FPI, so it's noisier than the rosters/participants signal).

Usage::

    uv run python python/scrape_failures.py            # all seasons in the master
    uv run python python/scrape_failures.py -s 2010    # one season
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

import pyarrow.parquet as pq

MASTER = "cfb/cfb_schedule_master.parquet"
FINAL_DIR = Path("cfb/json/final")
OUT = Path("logs/scrape_failures.csv")


def _payload_empty(name: str, gid: str, key: str) -> bool:
    """True if cfb/<name>/json/<gid>.json is missing or its <key> payload is empty."""
    try:
        d = json.loads((Path("cfb") / name / "json" / f"{gid}.json").read_text())
    except (OSError, ValueError):
        return True
    return not d.get(key)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--season", type=int, default=None, help="limit to one season")
    args = ap.parse_args()

    sm = pq.read_table(MASTER).to_pydict()
    gcol = "game_id" if "game_id" in sm else next(iter(sm))
    by_season: dict[int, list[str]] = collections.defaultdict(list)
    for g, s in zip(sm[gcol], sm["season"]):
        try:
            by_season[int(s)].append(str(int(g)))
        except (TypeError, ValueError):
            continue

    rows: list[tuple[int, str, str]] = []
    summary: list[tuple[int, int, int, int, int]] = []
    seasons = [args.season] if args.season else sorted(by_season)
    for season in seasons:
        gids = by_season.get(season, [])
        missing = hollow = pi_empty = 0
        for g in sorted(gids):
            if not (FINAL_DIR / f"{g}.json").exists():
                missing += 1
                rows.append((season, g, "no_final"))
                continue
            # game scraped; check whether its Core v2 extras came back hollow
            if _payload_empty("game_rosters", g, "data") and _payload_empty("play_participants", g, "data"):
                hollow += 1
                rows.append((season, g, "hollow_extras"))
            if _payload_empty("power_index", g, "items"):
                pi_empty += 1
        summary.append((season, len(gids), missing, hollow, pi_empty))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "game_id", "issue"])
        w.writerows(rows)

    print(f"{'season':>6} {'sched':>6} {'no_final':>9} {'hollow':>7} {'pi_empty':>9}")
    for season, total, missing, hollow, pi in summary:
        print(f"{season:>6} {total:>6} {missing:>9} {hollow:>7} {pi:>9}")
    print(f"\nwrote {OUT} - {len(rows)} reviewable failure(s) across {len(summary)} season(s)")
    print("(power_index emptiness is informational: FCS games + pre-2015 lack FPI by design)")


if __name__ == "__main__":
    main()
