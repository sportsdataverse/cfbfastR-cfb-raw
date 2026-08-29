import json

from cfb_raw_scrape import scrape_cfb_power_index


def test_power_index_refresher_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = scrape_cfb_power_index
    monkeypatch.setattr(m, "_fetch", lambda gid: {"fpi": 1})
    m.write_one(401, 2024)
    out = json.loads((tmp_path / "cfb/power_index/json/401.json").read_text())
    assert out["game_id"] == 401 and out["fpi"] == 1
