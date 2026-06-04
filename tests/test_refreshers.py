import importlib.util
import json
import sys
from pathlib import Path


def _load(modname):
    P = Path(__file__).parents[1] / "python" / f"{modname}.py"
    spec = importlib.util.spec_from_file_location(modname, P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules.setdefault(modname, mod)
    return mod


def test_power_index_refresher_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _load("scrape_cfb_power_index")
    monkeypatch.setattr(m, "_fetch", lambda gid: {"fpi": 1})
    m.write_one(401, 2024)
    out = json.loads((tmp_path / "cfb/power_index/json/401.json").read_text())
    assert out["game_id"] == 401 and out["fpi"] == 1
