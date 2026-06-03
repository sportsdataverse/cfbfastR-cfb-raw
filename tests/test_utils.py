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
