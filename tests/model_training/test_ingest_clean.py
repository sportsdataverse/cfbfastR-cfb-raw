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


def test_partial_game_gate_uses_espn_clock_minutes_dotcol():
    # game 10 reaches a 0-minute 4th-qtr clock (complete) -> kept;
    # game 20's 4th qtr never reaches 0 (ESPN partial) -> dropped.
    # final.json carries the field as "clock.minutes" (dot-notation), not "clock_minutes".
    df = pl.DataFrame({
        "game_id":      [10, 10, 20, 20],
        "period":       [4, 4, 4, 4],
        "start.down":   [1, 2, 1, 2],
        "type.text":    ["Rush", "Pass", "Rush", "Pass"],
        "clock.minutes": [7, 0, 9, 5],
    })
    out = clean_plays(df)
    kept = set(out["game_id"].to_list())
    assert 10 in kept and 20 not in kept
