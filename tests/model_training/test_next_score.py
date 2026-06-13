import polars as pl
from model_training.next_score import label_next_score_half


def _plays(rows):
    return pl.DataFrame(rows)


def test_offense_touchdown_then_label_td():
    df = _plays([
        {"game_id": 1, "drive.id": 1, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Rush"},
        {"game_id": 1, "drive.id": 1, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Pass Incompletion"},
        {"game_id": 1, "drive.id": 2, "period": 1, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": True, "offense_score_play": True, "defense_score_play": False,
         "type.text": "Passing Touchdown"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["Touchdown", "Touchdown", "Touchdown"]
    assert out["label"].to_list() == [0, 0, 0]
    assert out["score_drive"].to_list() == [2, 2, 2]


def test_no_score_before_half_is_no_score():
    df = _plays([
        {"game_id": 1, "drive.id": 1, "period": 2, "pos_team": "A", "def_pos_team": "B",
         "scoring_play": False, "offense_score_play": False, "defense_score_play": False,
         "type.text": "Rush"},
    ])
    out = label_next_score_half(df)
    assert out["next_score_half"].to_list() == ["No_Score"]
    assert out["label"].to_list() == [6]
