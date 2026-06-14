"""GAM model operations: lag features, weight, fit, LOSO CV, and persistence.

Port of rb_eval_model.R:
  - lrbs lag block (lines 54-86): add_lag_features, add_weight, build_model_data
  - GAM fitting (lines 104-115): fit_rb_eval_model
  - LOSO CV loop (lines 98-115): loso_cv
  - save/load: joblib + model_card.json sidecar

The LinearGAM(s(0) + s(1)) in pygam is an approximate port of:
    mgcv::gam(target ~ s(epa_per_play) + s(success), data=train_data, weights=weight)
(B-spline basis in pygam vs thin-plate regression splines in mgcv; functionally equivalent
for this data shape and size.)

Dependency note:
  pygam and joblib are in the [dependency-groups] gam / dev groups. They are imported
  lazily (inside functions) so that add_lag_features, add_weight, and build_model_data
  remain importable without the gam group installed — only the training and persistence
  functions require pygam / joblib at call time.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

from .constants import RB_EVAL_FEATURES, RB_EVAL_TARGET, RB_EVAL_PRED_COL


# ---------------------------------------------------------------------------
# Lag + weight helpers (no pygam dependency — always importable)
# ---------------------------------------------------------------------------

def add_lag_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add prior-season lag columns (lepa, lsuccess, lplays) by shifting within each rusher.

    Mirrors R `mutate(lepa = lag(epa_per_play, n=1), ...)` within rusher groups.
    The polars shift(1).over("rusher_player_name") on a frame sorted by
    (rusher_player_name, season) replicates the R lag() semantics exactly.

    Args:
        df: per-rusher-season aggregated frame with columns:
            rusher_player_name, season, n_plays, epa_per_play, success.

    Returns:
        Frame with lepa, lsuccess, lplays columns added. First-season rows per
        rusher will have null in the lag columns (drop_nulls in build_model_data).
    """
    df = df.sort(["rusher_player_name", "season"])
    return df.with_columns(
        lepa=pl.col("epa_per_play").shift(1).over("rusher_player_name"),
        lsuccess=pl.col("success").shift(1).over("rusher_player_name"),
        lplays=pl.col("n_plays").shift(1).over("rusher_player_name"),
    )


def add_weight(df: pl.DataFrame) -> pl.DataFrame:
    """Add Pythagorean weight column: weight = sqrt(n_plays^2 + lplays^2).

    Mirrors R: mutate(weight = sqrt(n_plays^2 + lplays^2)).

    Args:
        df: frame from add_lag_features with n_plays and lplays columns.

    Returns:
        Frame with weight column added.
    """
    return df.with_columns(
        weight=(
            pl.col("n_plays").cast(pl.Float64) ** 2
            + pl.col("lplays").cast(pl.Float64) ** 2
        ).sqrt(),
    )


def build_model_data(df: pl.DataFrame) -> pl.DataFrame:
    """Rename per-rusher-season columns to the GAM input contract and drop null-lag rows.

    GAM input contract:
      - epa_per_play (feature 0) <- lepa    (prior-season clamped EPA per play)
      - success      (feature 1) <- lsuccess (prior-season FO success rate)
      - unadjusted_epa (target)  <- current-season unclamped EPA per play

    This mirrors R lines 79-86:
      model_data <- lrbs %>% select(...) %>%
        rename(epa_per_play = lepa, success = lsuccess, ...)

    Args:
        df: output of add_weight (has lepa, lsuccess, lplays, weight, unadjusted_epa).

    Returns:
        Frame with GAM column names; rows with null lag values dropped.
    """
    select_cols = ["rusher_player_name", "season", "unadjusted_epa",
                   "lepa", "lsuccess", "weight"]
    if "lplays" in df.columns:
        select_cols.append("lplays")

    rename_map: dict[str, str] = {
        "lepa": "epa_per_play",
        "lsuccess": "success",
    }
    return (
        df.select([c for c in select_cols if c in df.columns])
        .rename(rename_map)
        .drop_nulls(["epa_per_play", "success", "weight"])
    )


# ---------------------------------------------------------------------------
# GAM training (lazy pygam import — requires uv sync --group gam)
# ---------------------------------------------------------------------------

def fit_rb_eval_model(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None = None,
):
    """Fit LinearGAM(s(0) + s(1)) on (epa_per_play, success) -> unadjusted_epa.

    Requires pygam (install with: uv sync --group gam).

    Args:
        X: feature matrix, shape (n, 2) — columns: [epa_per_play, success].
        y: target array, shape (n,) — unadjusted_epa values.
        weights: optional sample weights array, shape (n,). When None, equal weights.

    Returns:
        Fitted pygam LinearGAM.
    """
    try:
        from pygam import LinearGAM, s as gam_s
    except ImportError as exc:
        raise ImportError(
            "pygam is required for rb_eval model training. "
            "Install with: uv sync --group gam"
        ) from exc

    gam = LinearGAM(gam_s(0) + gam_s(1))
    if weights is not None:
        gam.fit(X, y, weights=weights)
    else:
        gam.fit(X, y)
    return gam


def loso_cv(model_data: pl.DataFrame) -> pl.DataFrame:
    """Leave-one-season-out cross-validation for xREPA.

    For each held-out season, trains on all other seasons and predicts on the held-out
    set. Mirrors the R cv_results map_dfr loop (lines 98-115 in rb_eval_model.R).

    Requires pygam (install with: uv sync --group gam).

    Args:
        model_data: GAM input frame from build_model_data, with columns
                    epa_per_play, success, unadjusted_epa, weight, season.

    Returns:
        Concatenation of per-season test frames with an added exp_rb_epa column
        (LOSO predictions). Seasons for which the training set is empty after
        null-dropping are skipped.
    """
    seasons = sorted(model_data["season"].drop_nulls().unique().to_list())
    parts: list[pl.DataFrame] = []

    for season in seasons:
        train = model_data.filter(pl.col("season") != season).drop_nulls(
            RB_EVAL_FEATURES + [RB_EVAL_TARGET, "weight"]
        )
        test = model_data.filter(pl.col("season") == season).drop_nulls(
            RB_EVAL_FEATURES + [RB_EVAL_TARGET, "weight"]
        )
        if train.is_empty() or test.is_empty():
            continue

        X_tr = train.select(RB_EVAL_FEATURES).to_numpy()
        y_tr = train[RB_EVAL_TARGET].to_numpy()
        w_tr = train["weight"].to_numpy()
        X_te = test.select(RB_EVAL_FEATURES).to_numpy()

        gam = fit_rb_eval_model(X_tr, y_tr, weights=w_tr)
        preds = gam.predict(X_te)

        parts.append(
            test.with_columns(
                pl.Series(RB_EVAL_PRED_COL, preds, dtype=pl.Float64)
            )
        )

    if not parts:
        return model_data.with_columns(
            pl.lit(None).cast(pl.Float64).alias(RB_EVAL_PRED_COL)
        )
    return pl.concat(parts, how="diagonal_relaxed")


# ---------------------------------------------------------------------------
# Model persistence (lazy joblib import — requires uv sync --group dev)
# ---------------------------------------------------------------------------

def save_model(
    model,
    path: str | Path,
    season_range: tuple[int, int],
    n_rushers: int,
    metadata: dict | None = None,
) -> Path:
    """Persist the fitted GAM to disk via joblib and write a model_card.json sidecar.

    Requires joblib (install with: uv sync --group dev).

    Args:
        model: fitted LinearGAM from fit_rb_eval_model.
        path: destination .pkl path.
        season_range: (first_season, last_season) of training data.
        n_rushers: number of rusher-season rows used for training.
        metadata: optional extra dict merged into the model card.

    Returns:
        Path to the model_card.json sidecar (path with .json suffix).
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required for model persistence. "
            "Install with: uv sync --group dev"
        ) from exc

    try:
        import pygam as _pygam
        pygam_version = _pygam.__version__
    except ImportError:
        pygam_version = "unknown"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

    card: dict = {
        "pygam_version": pygam_version,
        "model_formula": "LinearGAM(s(0) + s(1))",
        "features": RB_EVAL_FEATURES,
        "target": RB_EVAL_TARGET,
        "season_range": list(season_range),
        "n_rushers": n_rushers,
        "trained_date": date.today().isoformat(),
        "note": "xREPA analytical artifact — NOT bundled into sdv-py.",
    }
    if metadata:
        card.update(metadata)

    card_path = path.with_suffix(".json")
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card_path


def load_model(path: str | Path):
    """Load a persisted GAM from disk.

    Requires joblib (install with: uv sync --group dev).

    Args:
        path: path to the .pkl file written by save_model.

    Returns:
        Fitted LinearGAM.
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError(
            "joblib is required for model loading. "
            "Install with: uv sync --group dev"
        ) from exc
    return joblib.load(Path(path))
