"""Refresh the power_index dataset for a season range (standalone, outside daily loop)."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse

from cfb_raw_scrape._cfb_raw_utils import (
    filter_undone,
    games_for_seasons,
    get_logger,
    load_schedule_master,
    most_recent_cfb_season,
    run_pool,
    stamp,
    write_json_guarded,
)

# stage 04 banks the same file per game; share its fetch so the two can't drift
from cfb_raw_scrape.scrape_cfb_pbp import _power_index

DATASET = "power_index"


def write_one(game_id: int, season: int, week) -> None:
    # guarded: a degraded/empty answer must not clobber a banked file (stage 04 rule)
    write_json_guarded(
        stamp(_power_index(game_id), game_id=game_id, season=season, week=week),
        f"cfb/{DATASET}/json/{game_id}.json",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="true")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    master = load_schedule_master()
    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        games = filter_undone(
            games_for_seasons(master, season, season),
            dir=f"cfb/{DATASET}/json",
            rescrape=rescrape,
        )
        logger.info("%s %s: %d games", DATASET, season, len(games))
        weeks = dict(zip(master["game_id"].astype(int), master["week"]))
        run_pool(
            lambda g, _s=season: write_one(g, _s, weeks.get(g)),
            games,
            kind="thread",
            desc=f"{DATASET} {season}",
        )


if __name__ == "__main__":
    main()
