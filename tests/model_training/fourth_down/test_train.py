import json
import pathlib

import numpy as np
import pandas as pd
import polars as pl
import xgboost as xgb

from model_training.fourth_down import constants as C
from model_training.fourth_down.train import train_fourth_down

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training" / "fd_fixture_plays.json"


def _synth_fd_frame(n: int = 500) -> tuple:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "down": rng.integers(3, 5, n),
            "distance": rng.integers(1, 20, n).astype(float),
            "yards_to_goal": rng.integers(5, 99, n).astype(float),
            "posteam_total": rng.uniform(20, 70, n),
            "posteam_spread": rng.uniform(-40, 40, n),
        }
    )
    y = rng.integers(0, 76, n)
    return X, y


def test_train_returns_booster():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=3)
    assert isinstance(m, xgb.Booster)


def test_train_structure_5feat_76class_softprob():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=3)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 5
    assert cfg["objective"]["name"] == "multi:softprob"
    assert cfg["learner_model_param"]["num_class"] == "76"


def test_train_nrounds_produces_expected_tree_count():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=5)
    assert m.num_boosted_rounds() == 5


def test_train_feature_names_match_fd_features():
    X, y = _synth_fd_frame()
    m = train_fourth_down(X, y, nrounds=2)
    assert m.feature_names == C.FD_FEATURES


def test_train_from_plays_with_fixture():
    from model_training.fourth_down.train import train_from_plays

    plays = pl.DataFrame(json.loads(FIX.read_text()), infer_schema_length=None)
    m = train_from_plays(plays, nrounds=2)
    assert isinstance(m, xgb.Booster)
    assert m.num_features() == 5
