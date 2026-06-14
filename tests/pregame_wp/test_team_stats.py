import pandas as pd
import numpy as np
from pregame_wp.team_stats import (
    generate_team_play_stats,
    generate_team_drive_stats,
    generate_team_turnover_stats,
    generate_team_st_stats,
)

OFF_TYPES = ["Rush", "Pass Reception", "Pass Incompletion", "Rushing Touchdown"]
ST_TYPES = ["Kickoff", "Punt"]
BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]


# ---------------------------------------------------------------------------
# play stats fixture
# ---------------------------------------------------------------------------

def _make_plays():
    return pd.DataFrame([
        # Rush D1 gain 6 of 10 (successful), EqPPP=0.5, not explosive
        {"play_type": "Rush", "offense": "A", "defense": "B", "down": 1, "distance": 10,
         "yards_gained": 6, "EqPPP": 0.5, "play_successful": True, "play_explosive": False},
        # Rush D2 gain 3 of 7 (not successful: 3 < 0.7*7=4.9), EqPPP=0.2
        {"play_type": "Rush", "offense": "A", "defense": "B", "down": 2, "distance": 7,
         "yards_gained": 3, "EqPPP": 0.2, "play_successful": False, "play_explosive": False},
        # Pass D1 gain 15 (explosive, successful), EqPPP=1.2
        {"play_type": "Pass Reception", "offense": "A", "defense": "B", "down": 1, "distance": 5,
         "yards_gained": 15, "EqPPP": 1.2, "play_successful": True, "play_explosive": True},
        # Kickoff (ST — excluded from OffSR/OffER), EqPPP=0.0
        {"play_type": "Kickoff", "offense": "A", "defense": "B", "down": 0, "distance": 0,
         "yards_gained": 60, "EqPPP": 0.0, "play_successful": False, "play_explosive": False},
        # Rush D3 gain 0 (not successful by default), EqPPP=-0.1
        {"play_type": "Rush", "offense": "A", "defense": "B", "down": 3, "distance": 5,
         "yards_gained": 0, "EqPPP": -0.1, "play_successful": False, "play_explosive": False},
    ])


def test_off_sr_excludes_st():
    df = _make_plays()
    result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
    # 4 off plays (exc. Kickoff): 2 successful / 4 = 0.50
    assert abs(result["OffSR"].iloc[0] - 0.50) < 1e-9


def test_off_er_explosive_rate():
    df = _make_plays()
    result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
    # 1 explosive out of 4 off plays = 0.25
    assert abs(result["OffER"].iloc[0] - 0.25) < 1e-9


def test_avg_eqppp():
    df = _make_plays()
    result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
    # mean of [0.5, 0.2, 1.2, -0.1] = 0.45
    assert abs(result["AvgEqPPP"].iloc[0] - 0.45) < 1e-9


def test_iso_ppp_only_successful():
    df = _make_plays()
    result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
    # successful plays EqPPP: [0.5, 1.2], mean = 0.85
    assert abs(result["IsoPPP"].iloc[0] - 0.85) < 1e-9


# ---------------------------------------------------------------------------
# drive stats fixture
# ---------------------------------------------------------------------------

def _make_drives():
    return pd.DataFrame([
        # scoring opp: start=40, yards=25 -> 40+25=65 >= 60; scored=1, pts=7
        {"offense": "A", "drive_start_yardline": 40, "drive_yards": 25,
         "drive_scoring": 1, "drive_pts": 7},
        # scoring opp: start=50, yards=15 -> 65 >= 60; scored=0 (FG miss), pts=0
        {"offense": "A", "drive_start_yardline": 50, "drive_yards": 15,
         "drive_scoring": 0, "drive_pts": 0},
        # not scoring opp: start=20, yards=5 -> 25 < 60
        {"offense": "A", "drive_start_yardline": 20, "drive_yards": 5,
         "drive_scoring": 0, "drive_pts": 0},
    ])


def test_opp_rate():
    df = _make_drives()
    result = generate_team_drive_stats(df, "A")
    # 2 of 3 drives reach scoring opp threshold
    assert abs(result["OppRate"].iloc[0] - 2 / 3) < 1e-9


def test_opp_eff():
    df = _make_drives()
    result = generate_team_drive_stats(df, "A")
    # 1 of 2 scoring opps actually scored
    assert abs(result["OppEff"].iloc[0] - 0.5) < 1e-9


def test_opp_ppd():
    df = _make_drives()
    result = generate_team_drive_stats(df, "A")
    # mean pts across scoring opp drives: (7+0)/2 = 3.5
    assert abs(result["OppPPD"].iloc[0] - 3.5) < 1e-9


# ---------------------------------------------------------------------------
# turnover stats fixture
# ---------------------------------------------------------------------------

def _make_turnover_plays():
    return pd.DataFrame([
        # Pass broken up — PD
        {"play_type": "Pass Incompletion", "offense": "A", "defense": "B",
         "play_text": "incomplete pass, broken up by CB Joe"},
        # Interception
        {"play_type": "Interception", "offense": "A", "defense": "B",
         "play_text": "intercepted by LB Smith"},
        # Fumble recovered by defense
        {"play_type": "Fumble Recovery (Opponent)", "offense": "A", "defense": "B",
         "play_text": "fumble recovered by DE Jones"},
        # Clean rush — no event
        {"play_type": "Rush", "offense": "A", "defense": "B",
         "play_text": "rush for 5 yards"},
    ])


def test_exp_to_formula():
    df = _make_turnover_plays()
    result = generate_team_turnover_stats(df, "A", "B")
    # ExpTO = 0.22*(PDs+INTs) + 0.49*fums = 0.22*(1+1) + 0.49*1 = 0.44 + 0.49 = 0.93
    assert abs(result["ExpTO"].iloc[0] - (0.22 * 2 + 0.49 * 1)) < 1e-9


def test_actual_to_is_ints_plus_fumbles():
    df = _make_turnover_plays()
    result = generate_team_turnover_stats(df, "A", "B")
    # 1 INT + 1 fumble = 2
    assert result["ActualTO"].iloc[0] == 2


# ---------------------------------------------------------------------------
# ST stats — faithful OQ-5 port: PuntReturnEqPPP == PuntEqPPP (not punt_ret_eqppp)
# ---------------------------------------------------------------------------

def _make_st_plays():
    from pregame_wp.ep_curve import load_ep_curve
    ep = load_ep_curve()
    return pd.DataFrame([
        {"play_type": "Kickoff", "offense": "A", "defense": "B",
         "play_text": "kickoff for 65 yards, touchback",
         "kick_yards": 65, "return_yards": 0, "yard_line": 35},
        {"play_type": "Punt", "offense": "A", "defense": "B",
         "play_text": "punt for 45 yards",
         "kick_yards": 45, "return_yards": 0, "yard_line": 30},
    ]), ep


def test_punt_return_eqppp_equals_punt_eqppp():
    plays, ep = _make_st_plays()
    result = generate_team_st_stats(plays, "A", ep, {})
    # OQ-5 faithful port: PuntReturnEqPPP must equal PuntEqPPP
    assert result["PuntReturnEqPPP"].iloc[0] == result["PuntEqPPP"].iloc[0]
