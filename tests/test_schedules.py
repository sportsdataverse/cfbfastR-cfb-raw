import pandas as pd

from cfb_raw_scrape import scrape_cfb_schedules as s


def test_merge_master_dedupes_on_game_id(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2], "season": [2004, 2004]}).to_parquet(master)
    new = pd.DataFrame({"game_id": [2, 3], "season": [2004, 2004]})
    s.merge_master(new, str(master))
    out = pd.read_parquet(master).sort_values("game_id")
    assert out["game_id"].tolist() == [1, 2, 3]  # game_id 2 not duplicated


def test_merge_master_takes_int_ids_onto_string_master(tmp_path):
    # sdv-py's schedule ids went Int64 while the master holds strings: the mixed
    # column made every merge_master write fail (2026-10-31 .. 11-01)
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": pd.array([1], "int32"), "id": ["1"], "venue_id": ["9"]}).to_parquet(master)
    new = pd.DataFrame(
        {"game_id": pd.array([2], "Int64"), "id": pd.array([2], "Int64"), "venue_id": pd.array([None], "Int64")}
    )
    s.merge_master(s.conform_to_master(new, str(master)), str(master))
    out = pd.read_parquet(master).sort_values("game_id")
    assert out["id"].tolist() == ["1", "2"]
    assert out["venue_id"].isna().tolist() == [False, True]  # null stays null, not "<NA>"
