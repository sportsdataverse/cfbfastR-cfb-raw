import pandas as pd

from cfb_raw_scrape import scrape_cfb_schedules as s


def test_merge_master_dedupes_on_game_id(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2], "season": [2004, 2004]}).to_parquet(master)
    new = pd.DataFrame({"game_id": [2, 3], "season": [2004, 2004]})
    s.merge_master(new, str(master))
    out = pd.read_parquet(master).sort_values("game_id")
    assert out["game_id"].tolist() == [1, 2, 3]  # game_id 2 not duplicated
