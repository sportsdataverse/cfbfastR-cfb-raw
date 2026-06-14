"""Build the 5-feature matrix and 76-class label for the fourth-down yards-gained model.

Input: a polars DataFrame of final.json plays (or a concat of multiple games).
Output: polars DataFrame with FD_FEATURES columns + fd_label column.

Feature derivation:
  posteam_total  = (homeTeamSpread + overUnder)/2  if start.is_home else (overUnder-homeTeamSpread)/2
  posteam_spread = start.pos_team_spread  (already posteam perspective, set by CFBPlayProcess)
  down, distance, yards_to_goal = start.* columns directly

Label:
  fd_label = int(clip(yardsGained, -10, 65) + 10)  -- class 0..75
"""
from __future__ import annotations

import polars as pl

from .constants import (
    FD_CLIP_HIGH,
    FD_CLIP_LOW,
    FD_FEATURES,
    FD_FIRST_DOWN_PENALTY_COLS,
    FD_IS_HOME_COL,
    FD_LABEL_OFFSET,
    FD_OVERUNDER_COL,
    FD_SPREAD_COL,
    FD_YARDS_GAINED_COL,
)


def _first_down_penalty_col(df: pl.DataFrame) -> str | None:
    """Return whichever first-down-penalty column name is present in the frame."""
    for name in FD_FIRST_DOWN_PENALTY_COLS:
        if name in df.columns:
            return name
    return None


def derive_fd_features(plays: pl.DataFrame) -> pl.DataFrame:
    """Derive the 5 fourth-down model features and the fd_label from a plays DataFrame.

    No row filtering is performed here — all rows receive derived columns.
    Use this function to build the augmented frame; callers may then filter
    as needed (e.g., for train/test splits).

    Args:
        plays: polars DataFrame of final.json play records. Must contain:
            start.down, start.distance, start.yardsToEndzone,
            start.pos_team_spread, homeTeamSpread, overUnder,
            start.is_home, yardsGained.

    Returns:
        polars DataFrame with all original columns plus:
            down, distance, yards_to_goal, posteam_total, posteam_spread, fd_label.
    """
    if plays.is_empty():
        # Return empty frame with the expected schema
        return pl.DataFrame(
            schema={
                **{col: pl.Float64 for col in FD_FEATURES},
                "fd_label": pl.Int32,
            }
        )

    # --- derive posteam_total ---
    home_total = (pl.col(FD_SPREAD_COL) + pl.col(FD_OVERUNDER_COL)) / 2.0
    away_total = (pl.col(FD_OVERUNDER_COL) - pl.col(FD_SPREAD_COL)) / 2.0

    df = plays.with_columns(
        # posteam_total: home = (spread+total)/2, away = (total-spread)/2
        posteam_total=pl.when(pl.col(FD_IS_HOME_COL).cast(pl.Int32) == 1)
        .then(home_total)
        .otherwise(away_total),
        # posteam_spread: already the posteam-perspective spread from CFBPlayProcess
        posteam_spread=pl.col("start.pos_team_spread"),
        # rename source columns to canonical feature names
        down=pl.col("start.down").cast(pl.Float64),
        distance=pl.col("start.distance").cast(pl.Float64),
        yards_to_goal=pl.col("start.yardsToEndzone").cast(pl.Float64),
        # label: clip yardsGained to [-10, 65] then offset by +10 -> class 0..75
        fd_label=(
            pl.col(FD_YARDS_GAINED_COL)
            .cast(pl.Float64)
            .clip(FD_CLIP_LOW, FD_CLIP_HIGH)
            + FD_LABEL_OFFSET
        ).cast(pl.Int32),
    )

    return df
