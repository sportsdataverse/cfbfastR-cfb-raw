"""Feature extraction for the CFB CP model (Approach A, 8 game-state features).

Input: a pandas DataFrame with columns produced by CFBPlayProcess.run_processing_pipeline()
       (or an equivalent ESPN PBP frame with `start.*` dot-notation columns).

Output: a pandas DataFrame containing FEATURE_COLS + TARGET_COL, one row per
        pass play (non-pass plays are filtered out).
"""
from __future__ import annotations

import pandas as pd

from .constants import FEATURE_COLS, PASS_PLAY_TYPES, TARGET_COL

# Mapping from ESPN dot-notation / cfbfastR column names to flat feature names.
_COL_MAP: dict[str, str] = {
    "start.down": "down",
    "start.distance": "distance",
    "start.yardsToEndzone": "yards_to_goal",
    "pos_score_diff_start": "score_diff",
    "start.TimeSecsRem": "seconds_remaining",
    "start.is_home": "is_home",
    "period": "period",
    "passing_down": "passing_down",
}


def _play_type_col(df: pd.DataFrame) -> str:
    """Return whichever play-type column is present."""
    for c in ("playType", "play_type", "type"):
        if c in df.columns:
            return c
    return ""


def extract_pass_features(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to pass plays and return the 8-feature matrix + target column.

    Args:
        df: Raw or processed PBP DataFrame.  Must contain columns matching
            the ESPN dot-notation names in ``_COL_MAP`` plus a play-type
            column and (optionally) a ``completion`` target column.

    Returns:
        pandas DataFrame with columns ``FEATURE_COLS + [TARGET_COL]``,
        reset index, dtypes coerced to float/int.  Empty if no pass plays
        or if input is empty.
    """
    if df.empty:
        return pd.DataFrame()

    # --- filter to pass plays ---
    pt_col = _play_type_col(df)
    if not pt_col:
        return pd.DataFrame()

    mask = df[pt_col].isin(PASS_PLAY_TYPES)
    plays = df[mask].copy()
    if plays.empty:
        return pd.DataFrame()

    # --- rename to flat feature names ---
    plays = plays.rename(columns=_COL_MAP)

    # --- build target column (1 = completion) ---
    if "completion" not in plays.columns:
        plays["completion"] = (
            plays.get(pt_col, pd.Series(dtype=str))
            .str.contains("Reception|Passing Touchdown", na=False)
            .astype(int)
        )
    else:
        plays["completion"] = plays["completion"].astype(int)

    # --- coerce boolean columns to int ---
    for col in ("is_home", "passing_down"):
        if col in plays.columns:
            plays[col] = plays[col].astype(int)

    keep = [c for c in FEATURE_COLS + [TARGET_COL] if c in plays.columns]
    return plays[keep].reset_index(drop=True)
