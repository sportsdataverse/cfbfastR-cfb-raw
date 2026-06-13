import json
import numpy as np
import polars as pl
from model_training import constants as C
from model_training.train_ep import train_ep


def _synth_ep_frame(n=400):
    rng = np.random.default_rng(0)
    rows = {src: rng.random(n) for src in C.EP_SOURCE.values()}
    rows["label"] = rng.integers(0, 7, n)
    rows["ScoreDiff_W"] = rng.random(n)
    return pl.DataFrame(rows)


def test_train_ep_produces_8feat_7class_softprob():
    model = train_ep(_synth_ep_frame(), nrounds=5)
    cfg = json.loads(model.save_config())["learner"]
    assert model.num_features() == 8
    assert cfg["objective"]["name"] == "multi:softprob"
    assert cfg["learner_model_param"]["num_class"] == "7"
