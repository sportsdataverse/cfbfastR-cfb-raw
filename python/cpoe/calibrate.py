"""Leave-one-season-out (LOSO) calibration for the CFB CP model.

For each held-out season:
  - Train on all other seasons.
  - Predict completion probability on held-out plays.
  - Collect (predicted_cp, completion, distance_bucket) into a long-form DataFrame.

This module is the primary evaluation harness; the resulting cv_df feeds
validate.py (calibration tables) and figures.py (calibration plots).
"""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from . import constants as C
from .features import (
    assign_distance_bucket,
    build_feature_matrix,
    extract_cpoe_features,
    filter_pass_plays,
)
from .train import train_cp_model


def loso_calibrate(
    pbp_df: pl.DataFrame,
    seasons: list[int] | None = None,
    approach: str = "A",
    nrounds: int = C.CPOE_NROUNDS,
) -> pl.DataFrame:
    """Leave-one-season-out calibration.

    For each season in `seasons` (or all seasons present in pbp_df), train on
    the remaining seasons and predict on the held-out season.  Returns a
    long-form DataFrame with predicted_cp, cpoe, and distance_bucket columns
    suitable for validate.calibration_table().

    Args:
        pbp_df: Full plays DataFrame (unfiltered; filtering happens here).
        seasons: Optional list of seasons to include in the LOSO loop.
            None = all seasons found in pbp_df.
        approach: "A" or "B".
        nrounds: Boosting rounds per fold.

    Returns:
        polars DataFrame with columns:
            season, predicted_cp, cpoe, completion, distance_bucket,
            + all CPOE_FEATURES.
    """
    pass_df = filter_pass_plays(pbp_df)

    if seasons is not None:
        pass_df = pass_df.filter(pl.col("season").is_in(seasons))

    all_seasons = sorted(pass_df["season"].unique().to_list())
    folds: list[pl.DataFrame] = []

    features = C.CPOE_FEATURES if approach == "A" else C.CPOE_FEATURES_B

    for held_out in all_seasons:
        train_df = pass_df.filter(pl.col("season") != held_out)
        test_df = pass_df.filter(pl.col("season") == held_out)

        if train_df.is_empty() or test_df.is_empty():
            continue

        X_tr, y_tr, _ = build_feature_matrix(train_df, approach=approach)
        dtrain = xgb.DMatrix(X_tr, label=y_tr)
        model = xgb.train(C.CPOE_PARAMS, dtrain, num_boost_round=nrounds)

        feat_te = extract_cpoe_features(test_df, approach=approach)
        X_te = feat_te.select(features).to_pandas()
        preds = model.predict(xgb.DMatrix(X_te))

        fold = test_df.with_columns(
            pl.Series("predicted_cp", preds.tolist()).cast(pl.Float64),
            (
                pl.col("completion").cast(pl.Float64)
                - pl.Series("_tmp_pred", preds.tolist()).cast(pl.Float64)
            ).alias("cpoe"),
        )
        # Assign distance bucket for calibration stratification
        fold = assign_distance_bucket(fold)
        folds.append(fold)

    if not folds:
        return pl.DataFrame()

    return pl.concat(folds, how="diagonal_relaxed")
