"""Tests for the shared CFB-suite model_card helper."""
from __future__ import annotations

import json

import numpy as np
import pytest

from model_training.model_card import write_xgb_model_card


def test_card_explicit_features(tmp_path):
    card = write_xgb_model_card(
        tmp_path / "m.ubj", model_type="fourth_down", label="fd_label",
        features=["down", "distance", "yards_to_goal"],
        hyperparams={"objective": "multi:softprob", "eta": 0.025},
        n_rows=5000, extra={"num_class": 76},
    )
    assert card.name == "m.json"
    d = json.loads(card.read_text())
    assert d["model_type"] == "fourth_down"
    assert d["n_features"] == 3 and d["features"][0] == "down"
    assert d["objective"] == "multi:softprob"
    assert d["n_training_rows"] == 5000
    assert d["num_class"] == 76
    assert d["source"] == "cfb_final_json"
    assert "trained_date" in d


def test_card_introspects_booster_feature_names(tmp_path):
    xgb = pytest.importorskip("xgboost")
    X = np.random.default_rng(0).random((40, 3))
    y = (X[:, 0] > 0.5).astype(int)
    dm = xgb.DMatrix(X, label=y, feature_names=["a", "b", "c"])
    booster = xgb.train({"objective": "binary:logistic"}, dm, num_boost_round=3)
    booster.save_model(str(tmp_path / "b.ubj"))

    card = write_xgb_model_card(tmp_path / "b.ubj", model_type="cpoe",
                                label="completion", model=booster)
    d = json.loads(card.read_text())
    assert d["features"] == ["a", "b", "c"]  # pulled from booster.feature_names


def test_card_introspects_sklearn_feature_names(tmp_path):
    xgb = pytest.importorskip("xgboost")
    import polars as pl  # noqa: F401

    from xgboost import XGBRegressor
    import pandas as pd

    Xdf = pd.DataFrame({"5FRDiff": np.random.default_rng(1).random(50)})
    model = XGBRegressor(n_estimators=5).fit(Xdf, np.random.default_rng(2).random(50))

    card = write_xgb_model_card(tmp_path / "p.ubj", model_type="pregame_wp",
                                label="PtsDiff", model=model)
    d = json.loads(card.read_text())
    assert d["features"] == ["5FRDiff"]  # from feature_names_in_
