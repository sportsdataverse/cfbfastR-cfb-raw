import json
from pathlib import Path
import importlib.util

UTILS = Path(__file__).parents[1] / "python" / "_cfb_raw_utils.py"
spec = importlib.util.spec_from_file_location("_cfb_raw_utils", UTILS)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_write_json_atomic_creates_dirs_and_no_tmp(tmp_path):
    target = tmp_path / "a" / "b" / "401.json"
    u.write_json_atomic({"id": 401, "x": [1, 2]}, target)
    assert target.exists()
    assert json.loads(target.read_text())["id"] == 401
    assert not list(target.parent.glob("*.tmp")), "temp file left behind"


def test_processing_version_format():
    v = u.PROCESSING_VERSION
    assert "+" in v and v.split("+")[1].isdigit()


def test_stamp_adds_identity():
    out = u.stamp({"k": 1}, game_id=401, season=2024, week=1)
    assert out["game_id"] == 401 and out["season"] == 2024 and out["week"] == 1
    assert out["k"] == 1


def test_stamp_list_wraps_with_meta():
    out = u.stamp([{"a": 1}], game_id=401, season=2024, week=1)
    assert out["game_id"] == 401
    assert out["data"] == [{"a": 1}]
