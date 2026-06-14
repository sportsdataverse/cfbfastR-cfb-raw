"""Training pipeline: outlier filter + XGBRegressor fit.

OQ-7 resolution: mu=0.0 (point-differential is symmetric), std = std of full
training-set predictions.  The notebook used test-split statistics which is
non-reproducible without a fixed seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats

from .constants import OUTLIER_Z_5FR, OUTLIER_Z_PTS, XGB_N_ESTIMATORS, XGB_SEED, WP_MU


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where 5FRDiff or PtsDiff exceeds the z-score thresholds."""
    mask_5fr = np.abs(stats.zscore(df["5FRDiff"])) < OUTLIER_Z_5FR
    mask_pts = np.abs(stats.zscore(df["PtsDiff"])) < OUTLIER_Z_PTS
    return df[mask_5fr & mask_pts].copy()


def train_pgwp_model(
    df: pd.DataFrame,
) -> tuple[xgb.XGBRegressor, float, float]:
    """Train a 10-tree XGBRegressor on 5FRDiff → PtsDiff.

    Returns:
        model: fitted XGBRegressor
        mu: 0.0 (OQ-7: symmetric by construction)
        std: std of full training-set predictions
    """
    X = df[["5FRDiff"]].values
    y = df["PtsDiff"].values

    model = xgb.XGBRegressor(
        n_estimators=XGB_N_ESTIMATORS,
        seed=XGB_SEED,
        verbosity=0,
    )
    model.fit(X, y)

    preds = model.predict(X)
    mu = WP_MU  # 0.0 — per OQ-7 resolution
    std = float(np.std(preds))

    return model, mu, std


def save_pgwp_model(
    model: xgb.XGBRegressor,
    std: float,
    path: str,
    season_range: tuple[int, int] | None = None,
) -> None:
    """Save model as UBJ + sidecar metadata JSON."""
    import json
    from pathlib import Path
    from datetime import date

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(p))

    card = {
        "mu": WP_MU,
        "std": std,
        "n_estimators": XGB_N_ESTIMATORS,
        "feature": "5FRDiff",
        "target": "PtsDiff",
        "trained_date": date.today().isoformat(),
        "season_range": list(season_range) if season_range else None,
        "note": "pgwp_model — NOT bundled into sdv-py. Track 4 analytic artifact.",
    }
    p.with_suffix(".json").write_text(json.dumps(card, indent=2))


def load_pgwp_model(path: str) -> tuple[xgb.XGBRegressor, float, float]:
    """Load model + sidecar metadata (mu, std)."""
    import json
    from pathlib import Path

    p = Path(path)
    model = xgb.XGBRegressor()
    model.load_model(str(p))
    card = json.loads(p.with_suffix(".json").read_text())
    return model, float(card["mu"]), float(card["std"])
