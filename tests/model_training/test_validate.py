import pathlib

import numpy as np
import pandas as pd
import xgboost as xgb

from model_training.validate import prediction_parity

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "model_training"


def test_parity_against_self_is_zero():
    ref = xgb.Booster()
    ref.load_model(str(FIX / "xgb_ep_model.ubj"))
    X = pd.DataFrame(np.random.default_rng(0).random((50, 8)))
    report = prediction_parity(ref, ref, X)
    assert report["max_abs_diff"] == 0.0
    assert report["within_tol"] is True
