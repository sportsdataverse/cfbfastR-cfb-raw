"""GAM training for xREPA: LinearGAM(s(0) + s(1)) on prior-season epa_per_play and success.

Port of rb_eval_model.R:
    dakota_model = mgcv::gam(target ~ s(epa_per_play) + s(success), data=train_data, weights=weight)

The pygam LinearGAM is an approximate equivalent (B-spline basis vs mgcv thin-plate regression
splines). LOSO CV by season replicates the R cv_results loop.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

try:
    from pygam import LinearGAM, s
except ImportError as e:
    raise ImportError(
        "pygam is required for rb_eval training. Install with: uv sync --group gam"
    ) from e

try:
    import joblib
except ImportError as e:
    raise ImportError(
        "joblib is required for model persistence. Install with: uv sync --group dev"
    ) from e


_GAM_FEATURES = ["epa_per_play", "success"]


def train_xrepa(model_data: pl.DataFrame) -> "LinearGAM":
    """Fit LinearGAM(s(0) + s(1)) on (epa_per_play, success) -> target with sample weights."""
    df = model_data.drop_nulls(_GAM_FEATURES + ["target", "weight"])
    X = df.select(_GAM_FEATURES).to_numpy()
    y = df["target"].to_numpy()
    w = df["weight"].to_numpy()
    gam = LinearGAM(s(0) + s(1))
    gam.fit(X, y, weights=w)
    return gam


def loso_cv(model_data: pl.DataFrame) -> pl.DataFrame:
    """Leave-one-season-out cross-validation for xREPA.

    For each held-out season, train on all other seasons and predict on the held-out set.
    Mirrors R cv_results map_dfr loop (lines 98-115 in rb_eval_model.R).
    """
    seasons = sorted(model_data["season"].drop_nulls().unique().to_list())
    parts: list[pl.DataFrame] = []
    for season in seasons:
        train = model_data.filter(pl.col("season") != season).drop_nulls(
            _GAM_FEATURES + ["target", "weight"]
        )
        test = model_data.filter(pl.col("season") == season).drop_nulls(
            _GAM_FEATURES + ["target", "weight"]
        )
        if train.is_empty() or test.is_empty():
            continue
        gam = train_xrepa(train)
        X_test = test.select(_GAM_FEATURES).to_numpy()
        preds = gam.predict(X_test)
        parts.append(test.with_columns(pl.Series("exp_rb_epa", preds, dtype=pl.Float64)))
    if not parts:
        return model_data.with_columns(pl.lit(None).cast(pl.Float64).alias("exp_rb_epa"))
    return pl.concat(parts, how="diagonal_relaxed")


def save_model(
    gam: "LinearGAM",
    path: str | Path,
    season_range: tuple[int, int],
    n_rushers: int,
) -> Path:
    """Persist the fitted GAM via joblib and write a model_card.json sidecar.

    Returns:
        Path to the model_card.json sidecar.
    """
    import pygam
    from datetime import date

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(gam, path)
    card = {
        "pygam_version": pygam.__version__,
        "season_range": list(season_range),
        "n_rushers": n_rushers,
        "trained_date": date.today().isoformat(),
        "model_formula": "LinearGAM(s(0) + s(1))",
        "features": list(_GAM_FEATURES),
        "target": "unadjusted_epa",
        "note": "xREPA analytical artifact — NOT bundled into sdv-py.",
    }
    card_path = path.with_suffix(".json")
    card_path.write_text(json.dumps(card, indent=2))
    return card_path


def load_model(path: str | Path) -> "LinearGAM":
    """Load a persisted GAM from disk."""
    return joblib.load(Path(path))
