import numpy as np
import pandas as pd
from pregame_wp.talent import calculate_roster_talent, calculate_returning_production


def _make_recruiting(n=80, seed=0):
    rng = np.random.default_rng(seed)
    teams = ["A"] * 20 + ["B"] * 20 + ["C"] * 20 + ["D"] * 20
    years = [2020] * 80
    ratings = rng.uniform(70, 100, n).tolist()
    return pd.DataFrame({"team": teams, "year": years, "rating": ratings})


def test_roster_talent_returns_one_row_per_team():
    df = _make_recruiting()
    result = calculate_roster_talent(df, year=2020)
    assert set(result["team"]) == {"A", "B", "C", "D"}


def test_roster_talent_is_rolling_mean():
    # talent = mean(ratings) with FCS floor applied; talent >= raw mean always holds
    df = pd.DataFrame({
        "team": ["A"] * 20 + ["B"] * 20,
        "year": [2020] * 40,
        "rating": [90.0] * 20 + [85.0] * 20,
    })
    result = calculate_roster_talent(df, year=2020)
    a = result[result["team"] == "A"]["talent"].iloc[0]
    b = result[result["team"] == "B"]["talent"].iloc[0]
    # top team should be at its raw mean (floor never pulls it down)
    assert abs(a - 90.0) < 1e-9
    # all talents >= their raw means (floor only clips UP)
    assert b >= 85.0
    # relative ordering preserved
    assert a > b


def test_roster_talent_fcs_floor():
    # Team with very low composite gets clipped UP to the floor (raw mean < floor)
    df = pd.DataFrame({
        "team": ["A"] * 20 + ["B"] * 20 + ["Low"] * 20,
        "year": [2020] * 60,
        "rating": [95.0] * 20 + [90.0] * 20 + [50.0] * 20,
    })
    result = calculate_roster_talent(df, year=2020)
    low_raw = 50.0
    low_talent = result[result["team"] == "Low"]["talent"].iloc[0]
    # The floor must have pushed low_talent above its raw mean
    assert low_talent > low_raw


def _make_returning(seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "team": ["A", "A", "B", "B"],
        "returning": [0.6, 0.4, 0.8, 0.2],
        "snap_share": [0.55, 0.45, 0.7, 0.3],
    })


def test_returning_production_returns_one_row_per_team():
    df = _make_returning()
    result = calculate_returning_production(df)
    assert set(result["team"]) == {"A", "B"}


def test_returning_production_weighted():
    df = _make_returning()
    result = calculate_returning_production(df)
    # Team A: (0.6*0.55 + 0.4*0.45) / (0.55+0.45) = (0.33+0.18)/1.0 = 0.51
    a = result[result["team"] == "A"]["returning_production"].iloc[0]
    assert abs(a - 0.51) < 1e-9
