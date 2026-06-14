"""Fourth-down yards-gained model trainer.

Trains the 5-feature, 76-class multi:softprob XGBoost model that projects yards gained
on any play. No sample weights (the original model trains without them,
unlike the EP/WP models in Track 1). Feature input is derived via derive_fd_features().
"""
from __future__ import annotations

import polars as pl
import xgboost as xgb

from .constants import FD_FEATURES, FD_NROUNDS, FD_PARAMS
from .features import derive_fd_features, _first_down_penalty_col


def _filter_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Apply the standard fourth-down training filters to a plays DataFrame.

    Filters applied (in order):
      1. 3rd and 4th down only
      2. Rush, pass, or first-down-by-penalty plays
      3. distance > 0, yards_to_goal > 0, distance <= yards_to_goal
      4. homeTeamSpread and overUnder must be non-null
      5. yardsGained must be non-null
    """
    if df.is_empty():
        return df

    # --- step 1: down filter ---
    df = df.filter(pl.col("start.down").is_in([3, 4]))

    # --- step 2: play-type filter ---
    fdp_col = _first_down_penalty_col(df)
    rush_expr = pl.col("rush").cast(pl.Boolean) if "rush" in df.columns else pl.lit(False)
    pass_expr = pl.col("pass").cast(pl.Boolean) if "pass" in df.columns else pl.lit(False)
    if fdp_col is not None:
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
        pl.col("homeTeamSpread").is_not_null()
        & pl.col("overUnder").is_not_null()
    )

    # --- step 5: yardsGained must be present ---
    df = df.filter(pl.col("yardsGained").is_not_null())

    return df


def train_fd_model(
    pbp_df: pl.DataFrame,
    output_path: str | None = None,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Filter plays, derive features, and train the fourth-down yards-gained model.

    Args:
        pbp_df: polars DataFrame of final.json play records (all downs, all play types).
        output_path: If provided, save the trained model to this path (UBJ format).
        nrounds: Number of boosting rounds (default 157).

    Returns:
        Trained xgboost.Booster (multi:softprob, 5 features, 76 classes).

    Raises:
        ValueError: If no training rows survive the filter.
    """
    filtered = _filter_plays(pbp_df)
    if filtered.is_empty():
        raise ValueError(
            "No training rows survived the fourth-down feature filter. "
            "Check that plays include 3rd/4th-down rush/pass rows with "
            "overUnder, homeTeamSpread, and yardsGained present."
        )

    enriched = derive_fd_features(filtered)

    X = enriched.select(FD_FEATURES).to_pandas()
    y = enriched["fd_label"].to_numpy()

    dtrain = xgb.DMatrix(X[FD_FEATURES], label=y)
    model = xgb.train(FD_PARAMS, dtrain, num_boost_round=nrounds)

    if output_path is not None:
        model.save_model(output_path)

    return model


# Convenience alias matching the plan's train_from_plays name
def train_from_plays(
    plays: pl.DataFrame,
    nrounds: int = FD_NROUNDS,
) -> xgb.Booster:
    """Filter plays, build features, and train in one step.

    Args:
        plays: polars DataFrame of final.json play records.
        nrounds: Number of boosting rounds.

    Returns:
        Trained Booster.

    Raises:
        ValueError: If no training rows survive the filter.
    """
    return train_fd_model(plays, output_path=None, nrounds=nrounds)
