"""Phase 1 Task 1.2 — constants module tests."""
from __future__ import annotations


def test_constants_imports():
    from cpoe import constants  # noqa: F401


def test_feature_cols_is_list():
    from cpoe.constants import FEATURE_COLS
    assert isinstance(FEATURE_COLS, list)
    assert len(FEATURE_COLS) == 8


def test_feature_cols_contains_required():
    from cpoe.constants import FEATURE_COLS
    required = {
        "down",
        "distance",
        "yards_to_goal",
        "score_diff",
        "seconds_remaining",
        "is_home",
        "period",
        "passing_down",
    }
    assert required == set(FEATURE_COLS)


def test_target_col_is_string():
    from cpoe.constants import TARGET_COL
    assert isinstance(TARGET_COL, str)
    assert TARGET_COL


def test_throw_depth_buckets_structure():
    from cpoe.constants import THROW_DEPTH_BUCKETS
    assert isinstance(THROW_DEPTH_BUCKETS, dict)
    assert "short" in THROW_DEPTH_BUCKETS
    assert "intermediate" in THROW_DEPTH_BUCKETS
    assert "long" in THROW_DEPTH_BUCKETS


def test_throw_depth_short_le_3():
    from cpoe.constants import THROW_DEPTH_BUCKETS
    lo, hi = THROW_DEPTH_BUCKETS["short"]
    assert lo == 0
    assert hi == 3


def test_throw_depth_intermediate_4_to_8():
    from cpoe.constants import THROW_DEPTH_BUCKETS
    lo, hi = THROW_DEPTH_BUCKETS["intermediate"]
    assert lo == 4
    assert hi == 8


def test_throw_depth_long_ge_9():
    from cpoe.constants import THROW_DEPTH_BUCKETS
    lo, hi = THROW_DEPTH_BUCKETS["long"]
    assert lo == 9
    assert hi is None


def test_xgb_params_is_dict():
    from cpoe.constants import XGB_PARAMS
    assert isinstance(XGB_PARAMS, dict)


def test_xgb_params_objective_binary_logistic():
    from cpoe.constants import XGB_PARAMS
    assert XGB_PARAMS.get("objective") == "binary:logistic"


def test_xgb_nrounds_positive_int():
    from cpoe.constants import XGB_NROUNDS
    assert isinstance(XGB_NROUNDS, int)
    assert XGB_NROUNDS > 0


def test_pass_play_types_is_set():
    from cpoe.constants import PASS_PLAY_TYPES
    assert isinstance(PASS_PLAY_TYPES, frozenset)
    assert len(PASS_PLAY_TYPES) > 0
