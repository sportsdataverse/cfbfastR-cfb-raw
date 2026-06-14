"""Consolidated test suite for the fourth_down sub-package.

Tests use synthetic polars DataFrames (no live data required).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_play(**kwargs) -> dict:
    """Return a minimal valid play dict, overridable via kwargs."""
    base = {
        "start.down": 4,
        "start.distance": 5,
        "start.yardsToEndzone": 20,
        "start.pos_team_spread": 3.0,
        "homeTeamSpread": 3.0,
        "overUnder": 50.0,
        "start.is_home": 1,
        "yardsGained": 0.0,
        "rush": True,
        "pass": False,
        "firstD_by_penalty": False,
    }
    base.update(kwargs)
    return base


def _make_df(*plays) -> pl.DataFrame:
    return pl.DataFrame(list(plays), infer_schema_length=None)


# ---------------------------------------------------------------------------
# test_fd_label_clipping
# ---------------------------------------------------------------------------

def test_fd_label_clipping_low():
    """yardsGained=-15 (below clip low of -10) should yield label 0."""
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(_make_play(yardsGained=-15.0))
    out = derive_fd_features(df)
    assert out["fd_label"][0] == 0


def test_fd_label_clipping_high():
    """yardsGained=70 (above clip high of 65) should yield label 75."""
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(_make_play(yardsGained=70.0))
    out = derive_fd_features(df)
    assert out["fd_label"][0] == 75


def test_fd_label_clipping_zero():
    """yardsGained=0 should yield label 10 (clip(0,-10,65)+10 = 10)."""
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(_make_play(yardsGained=0.0))
    out = derive_fd_features(df)
    assert out["fd_label"][0] == 10


# ---------------------------------------------------------------------------
# test_posteam_total_home / away
# ---------------------------------------------------------------------------

def test_posteam_total_home():
    """is_home=1, homeTeamSpread=3, overUnder=50 -> posteam_total=(3+50)/2=26.5."""
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(_make_play(
        start__is_home=None,  # unused key — use the real column below
        **{"start.is_home": 1, "homeTeamSpread": 3.0, "overUnder": 50.0},
    ))
    out = derive_fd_features(df)
    assert abs(out["posteam_total"][0] - 26.5) < 1e-9


def test_posteam_total_away():
    """is_home=0, homeTeamSpread=3, overUnder=50 -> posteam_total=(50-3)/2=23.5."""
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(_make_play(**{"start.is_home": 0, "homeTeamSpread": 3.0, "overUnder": 50.0}))
    out = derive_fd_features(df)
    assert abs(out["posteam_total"][0] - 23.5) < 1e-9


# ---------------------------------------------------------------------------
# test_derive_fd_features_columns
# ---------------------------------------------------------------------------

def test_derive_fd_features_columns():
    """Output of derive_fd_features must contain all FD_FEATURES columns + fd_label."""
    from model_training.fourth_down.constants import FD_FEATURES
    from model_training.fourth_down.features import derive_fd_features

    df = _make_df(
        _make_play(yardsGained=5.0),
        _make_play(yardsGained=-3.0, **{"start.is_home": 0}),
    )
    out = derive_fd_features(df)
    for col in FD_FEATURES:
        assert col in out.columns, f"Missing column: {col}"
    assert "fd_label" in out.columns


# ---------------------------------------------------------------------------
# test_fd_model_roundtrip
# ---------------------------------------------------------------------------

def _synth_training_df(n: int = 100, seed: int = 42) -> pl.DataFrame:
    """Synthetic polars DataFrame that will survive derive_fd_features filtering."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        down = int(rng.integers(3, 5))
        distance = float(rng.integers(1, 15))
        yards_to_goal = float(rng.integers(int(distance), 99))
        yg = float(rng.integers(-10, 30))
        rows.append({
            "start.down": down,
            "start.distance": distance,
            "start.yardsToEndzone": yards_to_goal,
            "start.pos_team_spread": float(rng.uniform(-14, 14)),
            "homeTeamSpread": float(rng.uniform(-14, 14)),
            "overUnder": float(rng.uniform(40, 70)),
            "start.is_home": int(rng.integers(0, 2)),
            "yardsGained": yg,
            "rush": bool(rng.integers(0, 2)),
            "pass": bool(rng.integers(0, 2)),
            "firstD_by_penalty": False,
        })
    return pl.DataFrame(rows, infer_schema_length=None)


def test_fd_model_roundtrip():
    """Train on tiny synthetic df, save to temp file, reload, predict shape correct."""
    import xgboost as xgb
    from model_training.fourth_down.train import train_fd_model

    df = _synth_training_df(n=100)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = str(Path(tmpdir) / "fd_model_test.ubj")
        model = train_fd_model(df, output_path=out_path, nrounds=3)

        assert isinstance(model, xgb.Booster)
        assert model.num_features() == 5

        # reload from disk
        loaded = xgb.Booster()
        loaded.load_model(out_path)
        assert loaded.num_features() == 5

        # predict shape
        from model_training.fourth_down.features import derive_fd_features
        from model_training.fourth_down.constants import FD_FEATURES
        import pandas as pd

        out = derive_fd_features(df)
        X = out.select(FD_FEATURES).to_pandas()
        dmat = xgb.DMatrix(X)
        preds = loaded.predict(dmat)
        # multi:softprob with num_class=76 -> shape (n_rows, 76) or (n_rows * 76,)
        total_probs = preds.size
        n_rows = len(X)
        assert total_probs == n_rows * 76


# ---------------------------------------------------------------------------
# test_fourth_down_decision_stub
# ---------------------------------------------------------------------------

def test_fourth_down_decision_stub_raises_not_implemented():
    """Importing get_go_wp_py and calling it should raise NotImplementedError."""
    from model_training.fourth_down_decision import get_go_wp_py

    with pytest.raises(NotImplementedError):
        get_go_wp_py(None, None, None, None)
