import pathlib

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from model_training.fourth_down import constants as C
from model_training.fourth_down.validate import assert_structure, calibration_fd

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training"
REF_MODEL = FIX / "fd_model.ubj"


@pytest.mark.skipif(not REF_MODEL.exists(), reason="fd_model.ubj fixture not on disk")
def test_assert_structure_passes_on_reference():
    ref = xgb.Booster()
    ref.load_model(str(REF_MODEL))
    assert_structure(ref)  # should not raise


def test_assert_structure_fails_on_wrong_feat_count():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((50, 4)), columns=["a", "b", "c", "d"])
    y = rng.integers(0, 76, 50)
    m = xgb.train(C.FD_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=2)
    with pytest.raises(AssertionError, match="num_features"):
        assert_structure(m)


def test_calibration_fd_shape():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "down": rng.integers(3, 5, 200),
            "distance": rng.integers(1, 20, 200).astype(float),
            "yards_to_goal": rng.integers(5, 99, 200).astype(float),
            "posteam_total": rng.uniform(20, 70, 200),
            "posteam_spread": rng.uniform(-40, 40, 200),
        }
    )
    y_yards = rng.integers(-10, 66, 200)
    from model_training.fourth_down.train import train_fourth_down

    y_label = (y_yards + 10).astype(int)
    m = train_fourth_down(X, y_label, nrounds=3)
    table = calibration_fd(m, X, y_yards)
    assert "pred_fd_prob" in table.columns and "empirical_fd_rate" in table.columns
    assert table["pred_fd_prob"].between(0, 1).all()
    assert table["empirical_fd_rate"].between(0, 1).all()
