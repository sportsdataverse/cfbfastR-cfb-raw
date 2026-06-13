import json
import numpy as np
import pandas as pd
from model_training.train_qbr import train_qbr_from_matrix


def test_qbr_model_is_6feat_regression():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.random((300, 6)),
                     columns=["qbr_epa", "sack_epa", "pass_epa", "rush_epa", "pen_epa", "spread"])
    y = rng.random(300) * 100
    m = train_qbr_from_matrix(X, y, nrounds=5)
    cfg = json.loads(m.save_config())["learner"]
    assert m.num_features() == 6 and cfg["objective"]["name"] == "reg:squarederror"
