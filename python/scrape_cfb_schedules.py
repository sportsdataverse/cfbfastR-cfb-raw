"""Scrape CFB schedules per season -> cfb/schedules + master."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import pandas as pd
import sportsdataverse as sdv

from _cfb_raw_utils import get_logger, most_recent_cfb_season

SCHED_DIR = Path("cfb/schedules")
MASTER = "cfb/cfb_schedule_master.parquet"


def fetch_season(season: int) -> pd.DataFrame:
    # espn_cfb_schedule(dates=<year>) returns a full-season DataFrame with
    # game_id and season columns already present (verified via STEP 0).
    df = sdv.cfb.espn_cfb_schedule(dates=season, return_as_pandas=True)
    # Ensure season column exists (the API already includes it, but guard anyway)
    if "season" not in df.columns:
        df["season"] = season
    return df


def merge_master(new: pd.DataFrame, master_path: str = MASTER) -> None:
    p = Path(master_path)
    if p.exists():
        old = pd.read_parquet(p)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)


def write_season(df: pd.DataFrame, season: int) -> None:
    (SCHED_DIR / "parquet").mkdir(parents=True, exist_ok=True)
    (SCHED_DIR / "csv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCHED_DIR / "parquet" / f"cfb_schedule_{season}.parquet", index=False)
    df.to_csv(SCHED_DIR / "csv" / f"cfb_schedule_{season}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    for season in range(args.start_year, end + 1):
        logger = get_logger("cfb_schedules", season)
        try:
            df = fetch_season(season)
            write_season(df, season)
            merge_master(df)
            logger.info("schedules %s: %d games", season, len(df))
        except Exception:
            logger.exception("schedules failed for %s", season)


if __name__ == "__main__":
    main()
