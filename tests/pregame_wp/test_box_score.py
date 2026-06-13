import pandas as pd
from pregame_wp.box_score import calculate_box_score_from_frames
from pregame_wp.ep_curve import load_ep_curve, load_punt_sr

OFF_TYPES = ["Rush", "Pass Reception", "Pass Incompletion", "Rushing Touchdown"]
ST_TYPES = ["Kickoff", "Punt"]
BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]


def _make_synthetic_game():
    ep = load_ep_curve()
    punt_sr = load_punt_sr()

    plays = pd.DataFrame([
        # Offense A — 4 successful plays out of 6
        {"offense": "A", "defense": "B", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 6, "yard_line": 25, "play_text": "rush for 6"},
        {"offense": "A", "defense": "B", "play_type": "Pass Reception", "down": 1,
         "distance": 5, "yards_gained": 16, "yard_line": 31, "play_text": "pass for 16"},
        {"offense": "A", "defense": "B", "play_type": "Rush", "down": 2,
         "distance": 4, "yards_gained": 3, "yard_line": 47, "play_text": "rush for 3"},
        {"offense": "A", "defense": "B", "play_type": "Pass Incompletion", "down": 3,
         "distance": 1, "yards_gained": 0, "yard_line": 50, "play_text": "incomplete"},
        {"offense": "A", "defense": "B", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 5, "yard_line": 60, "play_text": "rush for 5"},
        {"offense": "A", "defense": "B", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 5, "yard_line": 65, "play_text": "rush for 5"},
        # Offense B — 2 successful plays out of 6 (weaker team)
        {"offense": "B", "defense": "A", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 2, "yard_line": 20, "play_text": "rush for 2"},
        {"offense": "B", "defense": "A", "play_type": "Rush", "down": 2,
         "distance": 8, "yards_gained": 1, "yard_line": 22, "play_text": "rush for 1"},
        {"offense": "B", "defense": "A", "play_type": "Pass Reception", "down": 1,
         "distance": 10, "yards_gained": 5, "yard_line": 25, "play_text": "pass for 5"},
        {"offense": "B", "defense": "A", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 2, "yard_line": 30, "play_text": "rush for 2"},
        {"offense": "B", "defense": "A", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 5, "yard_line": 35, "play_text": "rush for 5"},
        {"offense": "B", "defense": "A", "play_type": "Rush", "down": 1,
         "distance": 10, "yards_gained": 2, "yard_line": 40, "play_text": "rush for 2"},
    ])

    drives = pd.DataFrame([
        # A: one drive reaches scoring opp (start=35, yards=30 → 65), scores 7
        {"offense": "A", "defense": "B", "drive_start_yardline": 35,
         "drive_yards": 30, "drive_scoring": 1, "drive_pts": 7},
        # B: no scoring opp
        {"offense": "B", "defense": "A", "drive_start_yardline": 20,
         "drive_yards": 10, "drive_scoring": 0, "drive_pts": 0},
    ])

    return plays, drives, ep, punt_sr


def test_box_has_one_row_per_team():
    plays, drives, ep, punt_sr = _make_synthetic_game()
    result = calculate_box_score_from_frames(
        plays, drives, ep, punt_sr,
        eq_ppp_global_min=-2.0, eq_ppp_global_max=2.0,
    )
    assert set(result["Team"]) == {"A", "B"}


def test_stronger_team_has_higher_off_sr():
    plays, drives, ep, punt_sr = _make_synthetic_game()
    result = calculate_box_score_from_frames(
        plays, drives, ep, punt_sr,
        eq_ppp_global_min=-2.0, eq_ppp_global_max=2.0,
    )
    a_sr = result[result["Team"] == "A"]["OffSR"].iloc[0]
    b_sr = result[result["Team"] == "B"]["OffSR"].iloc[0]
    assert a_sr > b_sr


def test_5fr_present():
    plays, drives, ep, punt_sr = _make_synthetic_game()
    result = calculate_box_score_from_frames(
        plays, drives, ep, punt_sr,
        eq_ppp_global_min=-2.0, eq_ppp_global_max=2.0,
    )
    assert "5FR" in result.columns
    assert result["5FR"].notna().all()


def test_5fr_diff_present_and_antisymmetric():
    plays, drives, ep, punt_sr = _make_synthetic_game()
    result = calculate_box_score_from_frames(
        plays, drives, ep, punt_sr,
        eq_ppp_global_min=-2.0, eq_ppp_global_max=2.0,
    )
    a_diff = result[result["Team"] == "A"]["5FRDiff"].iloc[0]
    b_diff = result[result["Team"] == "B"]["5FRDiff"].iloc[0]
    assert abs(a_diff + b_diff) < 1e-9  # A's diff + B's diff = 0
