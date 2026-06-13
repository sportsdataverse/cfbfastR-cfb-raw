import pathlib
import polars as pl
import pytest
from model_training.ingest import build_training_frame

FINAL_DIR = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL_DIR.glob("*.json")), reason="no backfill final.json on disk")
def test_build_frame_has_labels_and_weights():
    df = build_training_frame(FINAL_DIR, seasons=None)
    assert df.height > 0
    assert df["label"].is_in([0, 1, 2, 3, 4, 5, 6]).all()
    for col in ["Total_W_Scaled", "ScoreDiff_W", "next_score_half"]:
        assert col in df.columns
    from model_training.constants import EP_SOURCE
    for src in EP_SOURCE.values():
        assert src in df.columns
