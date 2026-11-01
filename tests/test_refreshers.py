import json

from cfb_raw_scrape import scrape_cfb_power_index


def test_power_index_refresher_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = scrape_cfb_power_index
    # patch the fetch stage 04 shares; a stub of a private _fetch hid its removed
    # sdv-py function (espn_cfb_event_powerindex) for weeks
    monkeypatch.setattr(m, "_power_index", lambda gid: {"fpi": 1})
    m.write_one(401, 2024, 7)
    out = json.loads((tmp_path / "cfb/power_index/json/401.json").read_text())
    assert out["game_id"] == 401 and out["fpi"] == 1 and out["week"] == 7


def test_power_index_fetch_resolves():
    import sportsdataverse as sdv

    # the shared fetch calls a function that exists in the pinned sdv-py
    assert callable(sdv.cfb.espn_cfb_game_powerindex)
