"""CP model trainer — 8-feature binary:logistic XGBoost (Approach A).

Optionally extends to 9-feature (Approach B with CFBD air_yards) if Phase 0 Task 0.2
confirmed fill rate >= 60%.

Entry points:
  train_cp_model(pbp_df, output_path=None) → xgb.Booster
  compute_cpoe(df, model) → pl.DataFrame  (adds expected_completion + cpoe)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import xgboost as xgb

from . import constants as C
from .features import build_feature_matrix, extract_cpoe_features, filter_pass_plays


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_cp_model(
    pbp_df: pl.DataFrame,
    output_path: str | Path | None = None,
    approach: str = "A",
    nrounds: int = C.CPOE_NROUNDS,
) -> xgb.Booster:
    """Train a CP (completion probability) model from a plays DataFrame.

    Filters to genuine pass attempts (sacks and penalty-no-play excluded),
    builds the Approach A (or B) feature matrix, and trains a
    binary:logistic XGBoost model.

    Args:
        pbp_df: Full plays polars DataFrame from build_cp_training_frame /
            build_training_frame. Filtering is performed inside this function.
        output_path: Optional path; if provided, the model is saved as UBJ
            (XGBoost binary JSON).
        approach: "A" (8 features) or "B" (9 features, CFBD air_yards required).
        nrounds: Number of boosting rounds.

    Returns:
        Trained xgb.Booster.
    """
    pass_df = filter_pass_plays(pbp_df)
    X, y, _ = build_feature_matrix(pass_df, approach=approach)
    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(C.CPOE_PARAMS, dtrain, num_boost_round=nrounds)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(output_path))
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def compute_cpoe(df: pl.DataFrame, model: xgb.Booster, approach: str = "A") -> pl.DataFrame:
    """Add expected_completion and cpoe columns to a pass-plays DataFrame.

    cpoe = completion - expected_completion (positive means better than expected).

    Only genuine pass plays (pass_attempt==True, sack_vec==False,
    penalty_no_play==False) receive predictions; other rows are dropped.

    Args:
        df: plays DataFrame (filtered or unfiltered).
        model: Trained CP XGBoost Booster.
        approach: "A" or "B".

    Returns:
        polars DataFrame of pass plays with added columns:
            expected_completion (Float64), cpoe (Float64).
    """
    pass_df = filter_pass_plays(df)
    features = C.CPOE_FEATURES if approach == "A" else C.CPOE_FEATURES_B
    feat_df = extract_cpoe_features(pass_df, approach=approach)
    X = feat_df.select(features).to_pandas()
    preds = model.predict(xgb.DMatrix(X))

    return pass_df.with_columns(
        pl.Series("expected_completion", preds.tolist()).cast(pl.Float64),
        (
            pl.col("completion").cast(pl.Float64)
            - pl.Series("expected_completion_tmp", preds.tolist()).cast(pl.Float64)
        ).alias("cpoe"),
    )
