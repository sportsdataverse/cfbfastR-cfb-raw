import importlib.util
import json
import sys
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "scrape_cfb_json.py"
spec = importlib.util.spec_from_file_location("scrape_cfb_json", P)
sj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sj)
# Register so pickle can resolve _worker.__module__ == "scrape_cfb_json" correctly.
sys.modules.setdefault("scrape_cfb_json", sj)


class _FakeProc:
    def __init__(self, gameId=0, raw=False, **kw):
        self.gameId = gameId
        self.raw = raw
        self.gameSpread = -7.5
        self.overUnder = 52.5
        self.homeFavorite = True
        self.gameSpreadAvailable = True
        self.odds_source = "summary_pickcenter"

    def espn_cfb_pbp(self):
        if self.raw:
            return {"header": {"competitions": [{"competitors": [
                        {"team": {"id": "333"}, "homeAway": "home"},
                        {"team": {"id": "99"}, "homeAway": "away"}]}]},
                    "injuries": [{"x": 1}], "gameNotes": [{"n": 1}], "pickcenter": []}
        return {}

    def run_processing_pipeline(self):
        return {"plays": [{"id": 1}], "advBoxScore": {}, "drives": {},
                "boxScore": {}, "header": {}, "season": 2024, "week": 1}


def test_download_game_writes_all_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sj, "CFBPlayProcess", _FakeProc)
    monkeypatch.setattr(sj, "_participants", lambda gid: [{"play_id": 1, "athlete_id": 5}])
    monkeypatch.setattr(sj, "_rosters", lambda gid: [{"athlete_id": 5}])
    monkeypatch.setattr(sj, "_officials", lambda gid: [{"name": "Ref"}])
    monkeypatch.setattr(sj, "_power_index", lambda gid: {"fpi": 1})
    monkeypatch.setattr(sj, "_odds_full", lambda gid: [{"provider": "x"}])
    monkeypatch.setattr(sj, "_propbets", lambda gid: [])

    sj.download_game(401, season=2024, rescrape=True)

    base = Path("cfb")
    assert (base / "json" / "raw" / "401.json").exists()
    final = json.loads((base / "json" / "final" / "401.json").read_text())
    assert final["id"] == 401 and final["season"] == 2024
    assert final["processing_version"]
    assert final["injuries"] == [{"x": 1}]
    assert final["betting"]["game_spread"] == -7.5
    assert final["officials"] == [{"name": "Ref"}]
    for ds in ("rosters", "play_participants", "betting", "officials",
               "power_index", "team_box_extra"):
        assert (base / ds / "json" / "2024" / "401.json").exists(), ds


def test_worker_is_module_level_and_calls_download_game(monkeypatch):
    # A module-level worker is required for ProcessPoolExecutor (lambdas aren't picklable).
    calls = []
    monkeypatch.setattr(sj, "download_game",
                        lambda gid, season, rescrape: calls.append((gid, season, rescrape)) or "ok")
    result = sj._worker((401, 2024, True))
    assert result == "ok"
    assert calls == [(401, 2024, True)]
    # must be a real module-level function (picklable), not a lambda/closure
    import pickle
    assert sj._worker.__name__ == "_worker"
    pickle.dumps(sj._worker)  # raises if not picklable
