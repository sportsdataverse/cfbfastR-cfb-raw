import pandas as pd
from pregame_wp.play_features import add_play_features

ST_TYPES = ["Kickoff", "Punt", "Field Goal Good"]
BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]


def _play(play_type, down, distance, yards, yard_line=20):
    return {
        "play_type": play_type,
        "down": down,
        "distance": distance,
        "yards_gained": yards,
        "yard_line": yard_line,
    }


def test_down1_50pct_is_successful():
    df = pd.DataFrame([_play("Rush", 1, 10, 5)])  # 5 >= 0.5*10
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_successful"].iloc[0] is True or out["play_successful"].iloc[0] == True  # noqa: E712


def test_down1_below_50pct_is_not_successful():
    df = pd.DataFrame([_play("Rush", 1, 10, 4)])  # 4 < 5
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_successful"].iloc[0] == False  # noqa: E712


def test_down2_70pct_is_successful():
    df = pd.DataFrame([_play("Rush", 2, 10, 7)])  # 7 >= 0.7*10
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_successful"].iloc[0] == True  # noqa: E712


def test_down3_not_successful_by_default():
    # 3rd down: default is False regardless of yards (not in np.select conditions)
    df = pd.DataFrame([_play("Rush", 3, 5, 100)])
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_successful"].iloc[0] == False  # noqa: E712


def test_explosive_15_yards():
    df = pd.DataFrame([_play("Rush", 1, 10, 15)])
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_explosive"].iloc[0] == True  # noqa: E712


def test_bad_type_not_explosive():
    df = pd.DataFrame([_play("Interception", 1, 10, 15)])
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_explosive"].iloc[0] == False  # noqa: E712


def test_st_type_not_successful():
    df = pd.DataFrame([_play("Kickoff", 0, 0, 60)])
    out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
    assert out["play_successful"].iloc[0] == False  # noqa: E712


def test_eqppp_computed_for_off_play():
    from pregame_wp.ep_curve import load_ep_curve
    ep = load_ep_curve()
    df = pd.DataFrame([_play("Rush", 1, 10, 10, yard_line=20)])  # yl=20, gain=10 -> ep[30]-ep[20]
    out = add_play_features(df, ep, ST_TYPES, BAD_TYPES)
    assert abs(out["EqPPP"].iloc[0] - (ep[30] - ep[20])) < 1e-9


def test_eqppp_zero_for_st_play():
    from pregame_wp.ep_curve import load_ep_curve
    ep = load_ep_curve()
    df = pd.DataFrame([_play("Kickoff", 0, 0, 60, yard_line=35)])
    out = add_play_features(df, ep, ST_TYPES, BAD_TYPES)
    assert out["EqPPP"].iloc[0] == 0.0
