import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
from scrape_cfb_qbr import parse_qbr_payload

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "model_training" / "qbr_endpoint_sample.json"


def test_parse_extracts_game_id_athlete_and_qbr():
    payload = json.loads(FIX.read_text())
    rows = parse_qbr_payload(payload, year=2024, week=1)
    assert rows, "expected QBR rows"
    r = rows[0]
    assert {"game_id", "athlete_id", "year", "week", "QBR", "TQBR"} <= set(r.keys())
    assert str(r["game_id"]).isdigit()
