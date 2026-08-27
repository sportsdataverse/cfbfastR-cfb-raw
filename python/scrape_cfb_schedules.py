"""Scrape CFB schedules per season -> cfb/schedules + master."""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import pandas as pd
import sportsdataverse as sdv

from cfb_raw_scrape._cfb_raw_utils import get_logger, most_recent_cfb_season

SCHED_DIR = Path("cfb/schedules")
MASTER = "cfb/cfb_schedule_master.parquet"


def fetch_season(season: int) -> pd.DataFrame:
    """Build a complete season schedule by iterating every (season_type, week)
    the ESPN calendar defines and accumulating, deduped on ``game_id``.

    A single ``dates=<year>`` scoreboard query is unreliable: it truncates at the
    API limit (~500, so a ~850-game season loses half) and mis-stamps
    season_type/week on the overflow rows. Fetching each calendar week with an
    explicit ``week`` + ``season_type`` returns correctly-attributed games and
    the whole regular season + postseason.
    """
    try:
        cal = sdv.cfb.espn_cfb_calendar(season=season, return_as_pandas=True)
        week_specs = [(int(r["season_type"]), int(r["week"])) for _, r in cal.iterrows()]
    except Exception:
        # Calendar unavailable (rare, very old seasons): regular weeks 1-16 + bowls.
        week_specs = [(2, w) for w in range(1, 17)] + [(3, 1)]

    frames = []
    for season_type, week in week_specs:
        df = sdv.cfb.espn_cfb_schedule(
            dates=season, week=week, season_type=season_type, return_as_pandas=True
        )
        if df is not None and len(df) > 0:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["game_id"], keep="last")
    if "season" not in out.columns:
        out["season"] = season
    return out


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
