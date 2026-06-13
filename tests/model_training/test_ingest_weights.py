import polars as pl
from model_training.ingest import add_weights


def test_weights_formula_matches_nflscrapr():
    df = pl.DataFrame({
        "game_id": [1, 1, 1],
        "drive.id": [1, 2, 3],
        "score_drive": [3, 3, 3],
        "pos_score_diff_start": [0, -7, 21],
    })
    out = add_weights(df)
    assert out["Drive_Score_Dist_W"].to_list() == [0.0, 0.5, 1.0]
    sw = out["ScoreDiff_W"].to_list()
    assert abs(sw[0] - 1.0) < 1e-9 and abs(sw[2] - 0.0) < 1e-9
    assert out["Total_W_Scaled"].min() == 0.0 and out["Total_W_Scaled"].max() == 1.0
