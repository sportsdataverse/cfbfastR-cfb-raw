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


def test_wp_naive_reference_predicts_on_sanity_fixture():
    import json

    from model_training import constants as C

    ref = xgb.Booster()
    ref.load_model(str(FIX / "xgb_wp_naive_model.ubj"))
    items = json.loads((FIX / "wpa-model-test-items.json").read_text())
    full = pd.DataFrame(items)
    # the keepers naive reference is 9-feat; the fixture is a superset. Restrict to the
    # reference's own leading feature count so the DMatrix width matches the model.
    nfeat = ref.num_features()
    # choose numeric feature columns deterministically: the WP_NAIVE contract order,
    # falling back to the first nfeat numeric columns of the fixture.
    candidate = [c for c in C.WP_NAIVE_FEATURES if c in full.columns]
    cols = (
        candidate[:nfeat]
        if len(candidate) >= nfeat
        else list(full.select_dtypes("number").columns)[:nfeat]
    )
    preds = ref.predict(xgb.DMatrix(full[cols]))
    assert ((preds >= 0) & (preds <= 1)).all()
