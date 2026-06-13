"""Select/rename final.json plays into the exact shipped model input matrices.

Returns pandas DataFrames (xgboost.DMatrix-friendly) with columns in the EXACT shipped
order, plus the label and weight arrays. WP label is win_indicator =
(start.pos_team.name == winner), i.e. the posteam NAME compared to the game winner;
no sample weights for WP (per the cfbscrapR-wpa recipe). EP uses ScoreDiff_W weights.
"""
from __future__ import annotations

import polars as pl

from . import constants as C


def _select(df: pl.DataFrame, source: dict[str, str]):
    out = df.select([pl.col(src).alias(name) for name, src in source.items()])
    return out.to_pandas()


def ep_matrix(df: pl.DataFrame):
    X = _select(df, C.EP_SOURCE)[C.EP_FEATURES]
    y = df["label"].to_numpy()
    w = df["ScoreDiff_W"].to_numpy()
    return X, y, w


def wp_matrix(df: pl.DataFrame, variant: str = "spread"):
    if variant == "spread":
        feats = C.WP_SPREAD_FEATURES
    elif variant == "naive":
        feats = C.WP_NAIVE_FEATURES
    else:
        raise ValueError(f"Unknown WP variant: {variant!r} (expected 'spread' or 'naive')")
    source = {k: v for k, v in C.WP_SOURCE.items() if k in feats}
    X = _select(df, source)[feats]
    y = (df["start.pos_team.name"] == df["winner"]).cast(pl.Int32).to_numpy()
    return X, y, None


def qbr_matrix(df: pl.DataFrame):
    """Per-(passer, game) weighted means of the 6 qbr_vars (mirrors CFBPlayProcess __process_qbr).

    `spread` is the posteam-perspective game spread. On final.json there is no flat `spread`
    column, but `start.pos_team_spread` IS the posteam spread -> alias it when `spread` is absent.
    Returns (X features, None, keys); the ESPN-QBR target is merged later in train_qbr.
    """
    if "spread" not in df.columns and "start.pos_team_spread" in df.columns:
        df = df.with_columns(spread=pl.col("start.pos_team_spread"))
    g = (
        df.filter(pl.col("passer_player_name").is_not_null())
        .group_by(["game_id", "season", "passer_player_name"])
        .agg(
            qbr_epa=(pl.col("qbr_epa") * pl.col("weight")).sum() / pl.col("weight").sum(),
            sack_epa=(pl.col("sack_epa") * pl.col("sack_weight")).sum() / pl.col("sack_weight").sum(),
            pass_epa=(pl.col("pass_epa") * pl.col("pass_weight")).sum() / pl.col("pass_weight").sum(),
            rush_epa=(pl.col("rush_epa") * pl.col("rush_weight")).sum() / pl.col("rush_weight").sum(),
            pen_epa=(pl.col("pen_epa") * pl.col("pen_weight")).sum() / pl.col("pen_weight").sum(),
            spread=pl.col("spread").first(),
        )
        .with_columns(pl.col(["sack_epa", "pass_epa", "rush_epa", "pen_epa"]).fill_null(0.0))
    )
    X = g.select(["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"]).to_pandas()
    keys = g.select(["game_id", "season", "passer_player_name"]).to_pandas()
    return X, None, keys
