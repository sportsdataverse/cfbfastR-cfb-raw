"""Per-play derived columns: EqPPP, play_successful, play_explosive.

Faithful port of win-prob.ipynb cells 20 and 22.

Note on play_successful (OQ-2): 3rd-down plays default to False regardless of
yards gained. Conversions appear as 1st-down plays in the subsequent sequence.
This matches the notebook's np.select conditions exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constants import EXPLOSIVE_THRESHOLD, SR_DOWN1, SR_DOWN2, SR_DOWN4
from .ep_curve import eqppp as _eqppp


def add_play_features(
    df: pd.DataFrame,
    ep_data: list[float],
    st_types: list[str],
    bad_types: list[str],
) -> pd.DataFrame:
    """Add play_explosive, play_successful, and (optionally) EqPPP columns."""
    df = df.copy()

    is_bad = df["play_type"].isin(bad_types)
    is_st = df["play_type"].isin(st_types)

    # --- play_explosive ---
    df["play_explosive"] = np.select(
        [
            is_bad,
            is_st,
            df["yards_gained"] >= EXPLOSIVE_THRESHOLD,
        ],
        [False, False, True],
        default=False,
    )

    # --- play_successful (3rd down intentionally absent → default False) ---
    df["play_successful"] = np.select(
        [
            is_bad,
            is_st,
            (df["down"] == 1) & (df["yards_gained"] >= SR_DOWN1 * df["distance"]),
            (df["down"] == 2) & (df["yards_gained"] >= SR_DOWN2 * df["distance"]),
            (df["down"] >= 4) & (df["yards_gained"] >= SR_DOWN4 * df["distance"]),
        ],
        [False, False, True, True, True],
        default=False,
    )

    # --- EqPPP (zero for ST plays; skipped when ep_data is empty) ---
    if ep_data:
        df["EqPPP"] = df.apply(
            lambda x: 0.0
            if x["play_type"] in st_types
            else _eqppp(ep_data, int(x["yard_line"]), int(x["yards_gained"])),
            axis=1,
        )

    return df
