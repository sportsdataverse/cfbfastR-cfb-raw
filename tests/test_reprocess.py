import importlib.util
import json
import sys
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "reprocess_cfb_json.py"
spec = importlib.util.spec_from_file_location("reprocess_cfb_json", P)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)
sys.modules.setdefault("reprocess_cfb_json", rp)


class _FakeProc:
    def __init__(self, gameId=0, path_to_json=None, odds_override=None, **kw):
        self.gameId = gameId
        self.odds_override = odds_override
        self.odds_source = "injected"
        self.gameSpread = odds_override["gameSpread"] if odds_override else -7.5
        self.overUnder = odds_override["overUnder"] if odds_override else 52.5
        self.homeFavorite = True
        self.gameSpreadAvailable = True

    def cfb_pbp_disk(self):
        return {}

    def run_processing_pipeline(self):
        return {"plays": [{"id": 1}], "drives": {}, "advBoxScore": {}, "header": {}, "week": 1}


def _seed(base: Path):
    (base / "cfb/json/raw").mkdir(parents=True)
    (base / "cfb/json/raw/401.json").write_text(json.dumps(
        {"header": {"competitions": [{"competitors": [
            {"team": {"id": "333"}, "homeAway": "home"},
            {"team": {"id": "99"}, "homeAway": "away"}]}]}, "injuries": []}))
    for ds, payload in {
        "betting": {"game_spread": -10.5, "over_under": 60.0, "home_favorite": True,
                    "game_spread_available": True, "game_id": 401, "season": 2024},
        "rosters": {"game_id": 401, "season": 2024, "data": [{"athlete_id": 5}]},
        "play_participants": {"game_id": 401, "season": 2024, "data": [{"play_id": 1}]},
        "power_index": {"game_id": 401, "season": 2024, "fpi": 1},
        "team_box_extra": {"game_id": 401, "season": 2024},
    }.items():
        d = base / f"cfb/{ds}/json/2024"
        d.mkdir(parents=True)
        (d / "401.json").write_text(json.dumps(payload))


def test_reprocess_offline_injects_odds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    monkeypatch.setattr(rp, "CFBPlayProcess", _FakeProc)
    rp.reprocess_game(401, season=2024, force=True)
    final = json.loads((tmp_path / "cfb/json/final/401.json").read_text())
    assert final["betting"]["game_spread"] == -10.5     # injected from disk betting
    assert final["game_rosters"] == [{"athlete_id": 5}]
    assert final["processing_version"]


def test_version_gate_skips_current(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    monkeypatch.setattr(rp, "CFBPlayProcess", _FakeProc)
    (tmp_path / "cfb/json/final").mkdir(parents=True)
    (tmp_path / "cfb/json/final/401.json").write_text(json.dumps(
        {"processing_version": rp.PROCESSING_VERSION}))
    assert rp.reprocess_game(401, season=2024, force=False) == "skipped"
