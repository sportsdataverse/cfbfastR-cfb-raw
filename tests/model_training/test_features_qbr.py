import polars as pl
from model_training.features import qbr_matrix
from model_training.ingest import add_winner


def test_add_winner_from_final_scores():
    df = pl.DataFrame({
        "game_id": [1, 1], "homeTeamName": ["A", "A"], "awayTeamName": ["B", "B"],
        "homeScore": [28, 28], "awayScore": [10, 10], "is_home": [1, 0],
        "pos_team": ["A", "B"],
    })
    out = add_winner(df)
    assert out["winner"].to_list() == ["A", "A"]


def test_qbr_matrix_aggregates_per_qb_game():
    df = pl.DataFrame({
        "game_id": [1, 1], "passer_player_name": ["X", "X"], "season": [2024, 2024],
        "qbr_epa": [0.5, -0.2], "weight": [1.0, 1.0],
        "sack_epa": [None, -0.2], "pass_epa": [0.5, None], "rush_epa": [None, None],
        "pen_epa": [None, None], "sack_weight": [None, 1.0], "pass_weight": [1.0, None],
        "rush_weight": [None, None], "pen_weight": [None, None], "spread": [-3.0, -3.0],
    })
    X, y, w = qbr_matrix(df)
    assert X.shape[0] == 1 and list(X.columns) == ["qbr_epa", "sack_epa", "pass_epa",
                                                   "rush_epa", "pen_epa", "spread"]
