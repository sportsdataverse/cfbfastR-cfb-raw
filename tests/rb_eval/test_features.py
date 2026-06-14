import json
import pathlib

import polars as pl
import pytest

from rb_eval.features import add_fo_success

FIXTURE = pathlib.Path(__file__).parent.parent / "fixtures" / "rb_eval" / "synth_plays.json"


def _plays(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


# --- fo_success ---


def test_fo_success_down1_at_half_threshold():
    df = _plays(
        [
            {"start.down": 1, "start.distance": 10, "yds_rushed": 5},  # 5 >= 5.0 → True
            {"start.down": 1, "start.distance": 10, "yds_rushed": 4},  # 4 < 5.0 → False
            {"start.down": 1, "start.distance": 10, "yds_rushed": 0},  # 0 < 5.0 → False
        ]
    )
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False, False]


def test_fo_success_down2_at_seventy_percent():
    df = _plays(
        [
            {"start.down": 2, "start.distance": 10, "yds_rushed": 7},  # 7 >= 7.0 → True
            {"start.down": 2, "start.distance": 10, "yds_rushed": 6},  # 6 < 7.0 → False
        ]
    )
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False]


def test_fo_success_down3_at_full_distance():
    df = _plays(
        [
            {"start.down": 3, "start.distance": 3, "yds_rushed": 3},  # 3 >= 3 → True
            {"start.down": 3, "start.distance": 3, "yds_rushed": 2},  # 2 < 3 → False
            {"start.down": 4, "start.distance": 1, "yds_rushed": 1},  # 1 >= 1 → True
        ]
    )
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True, False, True]


def test_fo_success_down4_included():
    df = _plays([{"start.down": 4, "start.distance": 2, "yds_rushed": 2}])
    out = add_fo_success(df)
    assert out["fo_success"].to_list() == [True]


# --- filter_rush_plays ---


def test_filter_rush_plays_excludes_non_rush_and_team():
    df = pl.DataFrame(json.loads(FIXTURE.read_text()))
    from rb_eval.features import filter_rush_plays

    out = filter_rush_plays(df)
    assert "TEAM" not in (out["rusher_player_name"].to_list())
    assert out["rush"].to_list() == [True] * len(out)
    assert out["rusher_player_name"].null_count() == 0
    assert out["pos_team"].null_count() == 0
    assert out["epa"].null_count() == 0


def test_filter_adds_fo_success_and_is_rush_opp():
    df = pl.DataFrame(json.loads(FIXTURE.read_text()))
    from rb_eval.features import filter_rush_plays

    out = filter_rush_plays(df)
    assert "fo_success" in out.columns
    assert "is_rush_opp" in out.columns
    opps = out["is_rush_opp"].to_list()
    assert all(isinstance(v, bool) for v in opps)
