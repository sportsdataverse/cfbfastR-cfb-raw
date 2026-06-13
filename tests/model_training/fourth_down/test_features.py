import json
import pathlib

import numpy as np
import polars as pl
import pytest

from model_training.fourth_down.features import fd_features

FIX = pathlib.Path(__file__).parent.parent.parent / "fixtures" / "model_training" / "fd_fixture_plays.json"


def _load_plays() -> pl.DataFrame:
    plays = json.loads(FIX.read_text())
    return pl.DataFrame(plays, infer_schema_length=None)


def test_filter_keeps_only_3rd_and_4th():
    X, y = fd_features(_load_plays())
    # play 3 (down=1) and plays 4/5 (null overUnder / null yardsGained) must be dropped
    assert len(X) == 2


def test_feature_columns_and_order():
    from model_training.fourth_down.constants import FD_FEATURES

    X, y = fd_features(_load_plays())
    assert list(X.columns) == FD_FEATURES


def test_posteam_total_home_offense():
    # play 1: is_home=1, homeTeamSpread=-7.0, overUnder=55.0
    # home_total = (-7 + 55) / 2 = 24.0
    X, y = fd_features(_load_plays())
    assert abs(X.iloc[0]["posteam_total"] - 24.0) < 1e-9


def test_posteam_total_away_offense():
    # play 2: is_home=0, homeTeamSpread=-7.0, overUnder=55.0
    # away_total = (55 - (-7)) / 2 = 31.0
    X, y = fd_features(_load_plays())
    assert abs(X.iloc[1]["posteam_total"] - 31.0) < 1e-9


def test_posteam_spread_passthrough():
    X, y = fd_features(_load_plays())
    assert X.iloc[0]["posteam_spread"] == -7.0
    assert X.iloc[1]["posteam_spread"] == 7.0


def test_label_clip_and_offset():
    # play 1: yardsGained=7 -> clip(-10,65) -> 7 -> +10 -> 17
    # play 2: yardsGained=-3 -> clip(-10,65) -> -3 -> +10 -> 7
    X, y = fd_features(_load_plays())
    assert y[0] == 17
    assert y[1] == 7


def test_label_dtype_is_integer():
    _, y = fd_features(_load_plays())
    assert y.dtype in (np.int32, np.int64, int)


def test_no_weights_returned():
    result = fd_features(_load_plays())
    assert len(result) == 2


def test_clip_low_yields_class_0():
    df = pl.DataFrame(
        [
            {
                "start.down": 4,
                "start.distance": 3,
                "start.yardsToEndzone": 10,
                "start.pos_team_spread": 2.0,
                "homeTeamSpread": 2.0,
                "overUnder": 50.0,
                "start.is_home": 1,
                "yardsGained": -20.0,
                "rush": True,
                "pass": False,
                "firstD_by_penalty": False,
            }
        ]
    )
    _, y = fd_features(df)
    assert y[0] == 0


def test_clip_high_yields_class_75():
    df = pl.DataFrame(
        [
            {
                "start.down": 3,
                "start.distance": 10,
                "start.yardsToEndzone": 80,
                "start.pos_team_spread": -14.0,
                "homeTeamSpread": -14.0,
                "overUnder": 60.0,
                "start.is_home": 1,
                "yardsGained": 80.0,
                "rush": False,
                "pass": True,
                "firstD_by_penalty": False,
            }
        ]
    )
    _, y = fd_features(df)
    assert y[0] == 75


def test_distance_greater_than_yards_to_goal_excluded():
    df = pl.DataFrame(
        [
            {
                "start.down": 4,
                "start.distance": 20,
                "start.yardsToEndzone": 5,
                "start.pos_team_spread": 0.0,
                "homeTeamSpread": 0.0,
                "overUnder": 50.0,
                "start.is_home": 1,
                "yardsGained": 3.0,
                "rush": True,
                "pass": False,
                "firstD_by_penalty": False,
            }
        ]
    )
    X, y = fd_features(df)
    assert len(X) == 0


def test_empty_input_returns_empty_frame():
    X, y = fd_features(pl.DataFrame())
    assert len(X) == 0 and len(y) == 0


def test_first_down_penalty_included_without_rush_pass():
    df = pl.DataFrame(
        [
            {
                "start.down": 4,
                "start.distance": 2,
                "start.yardsToEndzone": 10,
                "start.pos_team_spread": 3.0,
                "homeTeamSpread": -3.0,
                "overUnder": 50.0,
                "start.is_home": 0,
                "yardsGained": 2.0,
                "rush": False,
                "pass": False,
                "firstD_by_penalty": True,
            }
        ]
    )
    X, y = fd_features(df)
    assert len(X) == 1
