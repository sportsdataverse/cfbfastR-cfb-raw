"""Shared model_card.json sidecar for the CFB model suite (Tracks 1-5).

Unifies the metadata sidecar that previously only Track 3 (rb_eval) wrote: a JSON
file beside each saved model capturing the training contract (model type, features,
label, hyperparameters, row count, data source, train date) plus an optional
metrics snapshot. Importable from every track via ``model_training.model_card``.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def write_xgb_model_card(
    model_path: Path | str,
    *,
    model_type: str,
    label: str,
    features: Optional[Sequence[str]] = None,
    model: Any = None,
    hyperparams: Optional[Dict[str, Any]] = None,
    n_rows: Optional[int] = None,
    seasons: Optional[Sequence[int]] = None,
    source: str = "cfb_final_json",
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``<model_path>.json`` describing how an XGBoost model was trained.

    Args:
        model_path: Path to the saved ``.ubj`` (card is its ``.json`` sibling).
        model_type: e.g. ``ep`` / ``wp_spread`` / ``qbr`` / ``fourth_down`` /
            ``pregame_wp`` / ``cpoe``.
        label: Target column name.
        features: Feature names; if omitted, read from ``model`` (Booster
            ``feature_names`` or sklearn ``feature_names_in_``).
        model: Optional fitted model used to introspect features when ``features``
            is None.
        hyperparams: Optional hyperparameter dict (from the track's constants).
        n_rows: Optional training row count.
        seasons: Optional training seasons (min/max recorded as the range).
        source: Data source label (default the committed final.json library).
        metrics: Optional metrics snapshot.
        extra: Optional extra keys merged into the card.

    Returns:
        Path to the written ``.json`` card.
    """
    try:
        import xgboost
        xgb_version = xgboost.__version__
    except Exception:  # noqa: BLE001
        xgb_version = "unknown"

    feats = list(features) if features is not None else _introspect_features(model)
    obj = (hyperparams or {}).get("objective")
    seasons_sorted = sorted(int(s) for s in seasons) if seasons else None

    card: Dict[str, Any] = {
        "model_type": model_type,
        "xgboost_version": xgb_version,
        "objective": obj,
        "features": feats,
        "n_features": len(feats),
        "label": label,
        "training_seasons": [seasons_sorted[0], seasons_sorted[-1]] if seasons_sorted else None,
        "n_training_rows": int(n_rows) if n_rows is not None else None,
        "hyperparameters": dict(hyperparams) if hyperparams else None,
        "source": source,
        "trained_date": date.today().isoformat(),
    }
    if metrics:
        card["metrics"] = metrics
    if extra:
        card.update(extra)

    card_path = Path(model_path).with_suffix(".json")
    card_path.parent.mkdir(parents=True, exist_ok=True)
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    return card_path


def _introspect_features(model: Any) -> list[str]:
    """Best-effort feature-name extraction from a Booster or sklearn estimator."""
    if model is None:
        return []
    names = getattr(model, "feature_names", None)
    if names:
        return list(names)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return list(names)
    return []
