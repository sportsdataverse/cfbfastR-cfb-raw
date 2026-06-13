import numpy as np
import polars as pl
from model_training import constants as C
from model_training.features import ep_matrix, wp_matrix


def _frame():
    base = {src: 1.0 for src in set(C.EP_SOURCE.values()) | set(C.WP_SOURCE.values())}
    base.update({"label": 0, "Total_W_Scaled": 0.5, "ScoreDiff_W": 0.5,
                 "season": 2024, "pos_team": "A", "start.pos_team.name": "A",
                 "winner": "A", "next_score_half": "Touchdown"})
    return pl.DataFrame([base, {**base, "label": 6, "winner": "B", "next_score_half": "No_Score"}])


def test_ep_matrix_shape_and_order():
    X, y, w = ep_matrix(_frame())
    assert X.shape[1] == 8 and list(X.columns) == C.EP_FEATURES
    assert y.tolist() == [0, 6]


def test_wp_spread_matrix_13_feats_and_binary_label():
    X, y, w = wp_matrix(_frame(), variant="spread")
    assert X.shape[1] == 13 and list(X.columns) == C.WP_SPREAD_FEATURES
    assert set(np.unique(y)).issubset({0, 1})


def test_wp_naive_drops_spread_time():
    X, _, _ = wp_matrix(_frame(), variant="naive")
    assert "spread_time" not in X.columns and X.shape[1] == 12
