import pathlib
import pytest
from model_training.cli import main

FINAL = pathlib.Path(__file__).resolve().parents[2] / "cfb" / "json" / "final"


@pytest.mark.skipif(not any(FINAL.glob("*.json")), reason="no backfill final.json")
def test_ingest_then_train_ep(tmp_path):
    pbp = tmp_path / "pbp_full.parquet"
    assert main(["ingest", "--final-dir", str(FINAL), "--out", str(pbp)]) == 0
    assert pbp.exists()
    assert main(["--stage", "2", "train-ep", "--pbp", str(pbp), "--out", str(tmp_path / "ep.ubj")]) == 0
    assert (tmp_path / "ep.ubj").exists()
