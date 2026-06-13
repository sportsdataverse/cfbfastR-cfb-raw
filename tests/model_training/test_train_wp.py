import json
import numpy as np
import polars as pl
from model_training import constants as C
from model_training.train_wp import train_wp


def _synth_wp_frame(n=400):
    rng = np.random.default_rng(1)
    rows = {src: rng.random(n) for src in C.WP_SOURCE.values()}
    rows["pos_team"] = ["A"] * n
    rows["start.pos_team.name"] = ["A"] * n
    rows["winner"] = rng.choice(["A", "B"], n)
    return pl.DataFrame(rows)


def test_wp_spread_is_13feat_logistic():
    m = train_wp(_synth_wp_frame(), variant="spread", nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 13 and cfg["objective"]["name"] == "binary:logistic"


def test_wp_naive_is_12feat():
    m = train_wp(_synth_wp_frame(), variant="naive", nrounds=5)
    assert m.num_features() == 12
