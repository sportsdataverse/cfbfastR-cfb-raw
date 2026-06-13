"""Phase 2 Task 2.1 — features module tests."""
from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture()
def raw_pass_plays() -> list[dict]:
    """Minimal ESPN-column pass-play dicts (camelCase or snake_case)."""
    return [
        {
            "playType": "Pass Reception",
            "start.down": 1,
            "start.distance": 10,
            "start.yardsToEndzone": 65,
            "pos_score_diff_start": 7,
            "start.TimeSecsRem": 1800,
            "start.is_home": True,
            "period": 2,
            "passing_down": 0,
            "completion": 1,
        },
        {
            "playType": "Pass Incompletion",
            "start.down": 3,
            "start.distance": 8,
            "start.yardsToEndzone": 30,
            "pos_score_diff_start": -3,
            "start.TimeSecsRem": 400,
            "start.is_home": False,
            "period": 4,
            "passing_down": 1,
            "completion": 0,
        },
        {
            "playType": "Rush",  # non-pass — should be filtered out
            "start.down": 1,
            "start.distance": 10,
            "start.yardsToEndzone": 50,
            "pos_score_diff_start": 0,
            "start.TimeSecsRem": 3600,
            "start.is_home": True,
            "period": 1,
            "passing_down": 0,
            "completion": 0,
        },
    ]


@pytest.fixture()
def pass_df(raw_pass_plays) -> pd.DataFrame:
    from cpoe.features import extract_pass_features
    return extract_pass_features(pd.DataFrame(raw_pass_plays))


def test_extract_pass_features_imports():
    from cpoe.features import extract_pass_features  # noqa: F401


def test_extract_returns_dataframe(pass_df):
    assert isinstance(pass_df, pd.DataFrame)


def test_extract_filters_non_pass(pass_df):
    """Rush play must be excluded."""
    assert len(pass_df) == 2


def test_extract_has_all_feature_cols(pass_df):
    from cpoe.constants import FEATURE_COLS
    for col in FEATURE_COLS:
        assert col in pass_df.columns, f"Missing column: {col}"


def test_extract_has_target_col(pass_df):
    from cpoe.constants import TARGET_COL
    assert TARGET_COL in pass_df.columns


def test_down_col_values(pass_df):
    assert list(pass_df["down"]) == [1, 3]


def test_distance_col_values(pass_df):
    assert list(pass_df["distance"]) == [10, 8]


def test_yards_to_goal_col_values(pass_df):
    assert list(pass_df["yards_to_goal"]) == [65, 30]


def test_score_diff_col_values(pass_df):
    assert list(pass_df["score_diff"]) == [7, -3]


def test_seconds_remaining_col_values(pass_df):
    assert list(pass_df["seconds_remaining"]) == [1800, 400]


def test_is_home_col_numeric(pass_df):
    vals = list(pass_df["is_home"])
    assert all(v in (0, 1) for v in vals)
    assert vals == [1, 0]


def test_period_col_values(pass_df):
    assert list(pass_df["period"]) == [2, 4]


def test_passing_down_col_numeric(pass_df):
    vals = list(pass_df["passing_down"])
    assert all(v in (0, 1) for v in vals)
    assert vals == [0, 1]


def test_completion_col_numeric(pass_df):
    vals = list(pass_df["completion"])
    assert all(v in (0, 1) for v in vals)
    assert vals == [1, 0]


def test_empty_input_returns_empty_df():
    import pandas as pd
    from cpoe.features import extract_pass_features
    result = extract_pass_features(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0


def test_no_pass_plays_returns_empty_df(raw_pass_plays):
    import pandas as pd
    from cpoe.features import extract_pass_features
    rush_only = [p for p in raw_pass_plays if p["playType"] == "Rush"]
    result = extract_pass_features(pd.DataFrame(rush_only))
    assert len(result) == 0
