"""XGBRegressor training for the Pregame WP model (Track 4).

Architecture:
  Single-feature (5FRDiff) → 10-tree XGBRegressor → predicted MOV → normal-CDF WP.
  This matches the akeaswaran win-prob.ipynb notebook's training cell.

OQ-7 (mu=0): After fitting, std is derived from full training-set predictions.
mu is fixed at 0.0 (symmetric: each game has one positive and one negative entry).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from xgboost import XGBRegressor

from .constants import PREGAME_WP_PARAMS, XGB_SEED
from .data_prep import filter_outliers


def train_pregame_model(df: pl.DataFrame) -> XGBRegressor:
    """Fit a 10-tree XGBRegressor on (5FRDiff → PtsDiff) after outlier removal.

    Args:
        df: DataFrame with columns '5FRDiff' (feature) and 'PtsDiff' (target).
            Outlier rows are removed before fitting.

    Returns:
        Fitted XGBRegressor.

    Example:
        Quick start::

            import polars as pl
            import numpy as np
            from pregame_wp.model import train_pregame_model

            rng = np.random.default_rng(42)
            df = pl.DataFrame({
                "5FRDiff": rng.normal(0, 2, 500).tolist(),
                "PtsDiff": rng.normal(0, 14, 500).tolist(),
            })
            model = train_pregame_model(df)
            preds = model.predict(df[["5FRDiff"]].to_numpy())
    """
    df_clean = filter_outliers(df)

    X: np.ndarray = df_clean[["5FRDiff"]].to_numpy()
    y: np.ndarray = df_clean["PtsDiff"].to_numpy()

    model = XGBRegressor(**PREGAME_WP_PARAMS, random_state=XGB_SEED)
    model.fit(X, y)
    return model


def train_pregame_model_with_stats(
    df: pl.DataFrame,
) -> tuple[XGBRegressor, float, float]:
    """Fit model and return (model, mu, std) where mu=0.0 and std is from full training preds.

    OQ-7 resolution: mu is fixed at 0.0 (symmetric by construction).
    std is the standard deviation of predictions on the cleaned training set.

    Args:
        df: DataFrame with '5FRDiff' and 'PtsDiff' columns.

    Returns:
        Tuple of (fitted XGBRegressor, mu=0.0, std from training predictions).
    """
    df_clean = filter_outliers(df)

    X: np.ndarray = df_clean[["5FRDiff"]].to_numpy()
    y: np.ndarray = df_clean["PtsDiff"].to_numpy()

    model = XGBRegressor(**PREGAME_WP_PARAMS, random_state=XGB_SEED)
    model.fit(X, y)

    # OQ-7: mu = 0.0 (symmetric), std from full training-set predictions
    train_preds: np.ndarray = model.predict(X)
    mu = 0.0
    std = float(np.std(train_preds))

    return model, mu, std


def save_model(model: XGBRegressor, path: str, *, n_rows: int | None = None) -> None:
    """Save the trained model to a UBJ file (XGBoost native format) + model card.

    Args:
        model: Fitted XGBRegressor.
        path: File path (should end in .ubj).
        n_rows: Optional training row count recorded in the model card.
    """
    model.save_model(path)
    try:
        from model_training.model_card import write_xgb_model_card

        from .constants import PREGAME_WP_PARAMS
        write_xgb_model_card(
            path, model_type="pregame_wp", label="PtsDiff", model=model,
            features=["5FRDiff"], hyperparams=PREGAME_WP_PARAMS, n_rows=n_rows,
            extra={"wp_mapping": "norm.cdf((pred - 0.0) / std)"},
        )
    except Exception:  # noqa: BLE001 — card is best-effort; never block the save
        pass


def load_model(path: str) -> XGBRegressor:
    """Load a trained model from a UBJ file.

    Args:
        path: File path to the .ubj model file.

    Returns:
        Loaded XGBRegressor.
    """
    model = XGBRegressor()
    model.load_model(path)
    return model
