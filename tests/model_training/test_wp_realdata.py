import pathlib

import numpy as np
import pytest

from model_training.features import wp_matrix
from model_training.ingest import add_winner, build_training_frame

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_wp_label_and_matrix_on_real_final_json():
    df = add_winner(build_training_frame(FINAL_DIR, seasons=None))
    X, y, w = wp_matrix(df, variant="spread")
    assert X.shape[1] == 13
    assert w is None
    classes = set(np.unique(y).tolist())
    assert classes.issubset({0, 1})
    # a completed game has both a winning and losing posteam across its plays
    assert classes == {0, 1}
