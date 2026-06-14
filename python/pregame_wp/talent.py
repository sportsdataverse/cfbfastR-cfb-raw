"""Roster talent + returning production helpers.

Port of win-prob.ipynb cells 47–55.
"""
from __future__ import annotations

import pandas as pd

from .constants import TALENT_FCS_PERCENTILE


def calculate_roster_talent(
    recruiting_df: pd.DataFrame,
    year: int,
    window: int = 4,
) -> pd.DataFrame:
    """Rolling 4-year mean recruiting composite per team, with FCS floor.

    Args:
        recruiting_df: DataFrame with columns ['team', 'year', 'rating'].
        year: Target year (inclusive upper bound).
        window: Number of prior years to average (default 4).

    Returns:
        DataFrame with columns ['team', 'talent'].
    """
    sub = recruiting_df[recruiting_df["year"] <= year].copy()
    # Keep the most recent `window` years per team
    sub = (
        sub.sort_values("year")
        .groupby("team")
        .apply(lambda g: g.tail(window), include_groups=False)
        .reset_index(level=0)
    )
    talent = (
        sub.groupby("team")["rating"]
        .mean()
        .reset_index()
        .rename(columns={"rating": "talent"})
    )
    # FCS floor: clamp to 2nd percentile of the FBS distribution
    floor = talent["talent"].quantile(TALENT_FCS_PERCENTILE)
    talent["talent"] = talent["talent"].clip(lower=floor)
    return talent


def calculate_returning_production(
    returning_df: pd.DataFrame,
) -> pd.DataFrame:
    """Snap-share-weighted returning production per team.

    Args:
        returning_df: DataFrame with columns ['team', 'returning', 'snap_share'].

    Returns:
        DataFrame with columns ['team', 'returning_production'].
    """
    result = (
        returning_df.assign(
            weighted=returning_df["returning"] * returning_df["snap_share"]
        )
        .groupby("team")
        .apply(
            lambda g: g["weighted"].sum() / g["snap_share"].sum(),
            include_groups=False,
        )
        .reset_index()
        .rename(columns={0: "returning_production"})
    )
    return result
