import polars as pl
from model_training.ingest import clean_plays


def test_drops_ot_zero_specialteams_and_fixes_kickoff_down():
    df = pl.DataFrame({
        "game_id": [1, 1, 2, 3, 4],
        "period":  [4, 5, 1, 0, 4],
        "start.down": [1, 1, 5, 1, 1],
        "type.text": ["Rush", "Rush", "Kickoff", "Rush", "Timeout"],
    })
    out = clean_plays(df)
    assert set(out["game_id"].to_list()).isdisjoint({2, 3})
    assert "Kickoff" not in out["type.text"].to_list()
    assert "Timeout" not in out["type.text"].to_list()
