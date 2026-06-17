"""Build the 5-feature matrix and 76-class label for the fourth-down yards-gained model.

Input: a polars DataFrame of final.json plays (or a concat of multiple games).
Output: (X: pd.DataFrame[5 cols], y: np.ndarray[int]) — no sample weights (decision #11).

Feature derivation:
  posteam_total  = (homeTeamSpread + overUnder)/2  if start.is_home else (overUnder-homeTeamSpread)/2
  posteam_spread = start.pos_team_spread  (already correct posteam perspective, set by CFBPlayProcess)
  down, distance, yards_to_goal = start.* columns directly

Label:
  label = int(clip(yardsGained, -10, 65) + 10)  — class 0..75
"""
from __future__ import annotations

import numpy as np
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


def _first_down_penalty_col(df: pl.DataFrame) -> str:
    """Return whichever first-down-penalty column name is present in the frame."""
    for name in FD_FIRST_DOWN_PENALTY_COLS:
        if name in df.columns:
            return name
    return FD_FIRST_DOWN_PENALTY_COLS[0]


def fd_features(plays: pl.DataFrame) -> tuple:
    """Filter plays and build the (X, y) pair for the fourth-down yards-gained model.

    Args:
        plays: polars DataFrame of final.json play records (all downs, all play types).

    Returns:
        X: pandas DataFrame with exactly the 5 columns in FD_FEATURES order.
        y: integer ndarray of class labels (0..75).
    """
    import pandas as pd

    if plays.is_empty() or len(plays.columns) == 0:
        return pd.DataFrame(columns=FD_FEATURES), np.array([], dtype=np.int32)

    fdp_col = _first_down_penalty_col(plays)

    # --- step 1: down filter (keep 3rd and 4th only) ---
    if "start.down" not in plays.columns:
        return pd.DataFrame(columns=FD_FEATURES), np.array([], dtype=np.int32)
    df = plays.filter(pl.col("start.down").is_in([3, 4]))

    # --- step 2: play-type filter (rush | pass | first-down-by-penalty) ---
    rush_expr = pl.col("rush").cast(pl.Boolean) if "rush" in df.columns else pl.lit(False)
    pass_expr = pl.col("pass").cast(pl.Boolean) if "pass" in df.columns else pl.lit(False)
    if fdp_col in df.columns:
        fdp_expr = pl.col(fdp_col).fill_null(False).cast(pl.Boolean)
    else:
        fdp_expr = pl.lit(False)
    df = df.filter(rush_expr | pass_expr | fdp_expr)

    # --- step 3: distance / yards_to_goal guards ---
    df = df.filter(
        (pl.col("start.distance") > 0)
        & (pl.col("start.yardsToEndzone") > 0)
        & (pl.col("start.distance") <= pl.col("start.yardsToEndzone"))
    )

    # --- step 4: spread / overUnder must be present ---
    df = df.filter(
        pl.col(FD_SPREAD_COL).is_not_null() & pl.col(FD_OVERUNDER_COL).is_not_null()
    )

    # --- step 5: yardsGained must be present ---
    df = df.filter(pl.col(FD_YARDS_GAINED_COL).is_not_null())

    if df.is_empty():
        return pd.DataFrame(columns=FD_FEATURES), np.array([], dtype=np.int32)

    # --- derive posteam_total ---
    home_total = (pl.col(FD_SPREAD_COL) + pl.col(FD_OVERUNDER_COL)) / 2.0
    away_total = (pl.col(FD_OVERUNDER_COL) - pl.col(FD_SPREAD_COL)) / 2.0
    df = df.with_columns(
        posteam_total=pl.when(pl.col(FD_IS_HOME_COL).cast(pl.Boolean) == True)  # noqa: E712
        .then(home_total)
        .otherwise(away_total),
        posteam_spread=pl.col("start.pos_team_spread"),
    )

    # --- build label ---
    df = df.with_columns(
        _label=(
            pl.col(FD_YARDS_GAINED_COL).cast(pl.Float64).clip(FD_CLIP_LOW, FD_CLIP_HIGH)
            + FD_LABEL_OFFSET
        ).cast(pl.Int32)
    )

    # --- select the 5 feature columns (in exact model order) ---
    col_map = {
        "down": "start.down",
        "distance": "start.distance",
        "yards_to_goal": "start.yardsToEndzone",
        "posteam_total": "posteam_total",
        "posteam_spread": "posteam_spread",
    }
    X = df.select([pl.col(col_map[f]).alias(f) for f in FD_FEATURES]).to_pandas()
    y = df["_label"].to_numpy()
    return X, y
