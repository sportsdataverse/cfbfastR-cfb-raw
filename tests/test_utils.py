import json
from pathlib import Path
import importlib.util
import pandas as pd

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


def test_write_json_atomic_emits_valid_json_for_nan_inf(tmp_path):
    # Python's json writes bare NaN/Infinity (invalid JSON; R/JS/Go reject it).
    # write_json_atomic must coerce nan/inf -> null so cross-language consumers can parse.
    target = tmp_path / "g.json"
    obj = {"a": float("nan"), "b": float("inf"), "c": float("-inf"),
           "plays": [{"name": "x", "epa": float("nan"), "yds": 3.5}], "ok": 1}
    u.write_json_atomic(obj, target)
    raw = target.read_text()
    assert "NaN" not in raw and "Infinity" not in raw, "invalid JSON literal emitted"
    # strict parse (no NaN tolerance) must succeed
    loaded = json.loads(raw, parse_constant=_reject_constant)
    assert loaded["a"] is None and loaded["b"] is None and loaded["c"] is None
    assert loaded["plays"][0]["epa"] is None
    assert loaded["plays"][0]["yds"] == 3.5
    assert loaded["ok"] == 1


def _reject_constant(c):  # json.loads(parse_constant=) fires only for NaN/Infinity tokens
    raise AssertionError(f"non-standard JSON constant present: {c}")


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


def test_filter_undone_drops_existing(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402, 403], dir=str(final_dir), rescrape=False)
    assert out == [402, 403]


def test_filter_undone_rescrape_keeps_all(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402], dir=str(final_dir), rescrape=True)
    assert out == [401, 402]


def test_games_for_seasons_filters_range(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2, 3], "season": [2003, 2004, 2005]}).to_parquet(master)
    games = u.games_for_seasons(u.load_schedule_master(str(master)), 2004, 2005)
    assert sorted(games) == [2, 3]


def test_season_type_from_raw_variants():
    assert u.season_type_from_raw({"header": {"season": {"type": 2}}}) == 2
    assert u.season_type_from_raw({"header": {"season": {"type": {"id": "3"}}}}) == 3
    assert u.season_type_from_raw({"header": {"competitions": [{"type": {"id": 2}}]}}) == 2
    assert u.season_type_from_raw({"header": {}}) is None
