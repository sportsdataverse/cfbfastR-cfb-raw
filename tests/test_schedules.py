import importlib.util
from pathlib import Path
import pandas as pd

P = Path(__file__).parents[1] / "python" / "scrape_cfb_schedules.py"
spec = importlib.util.spec_from_file_location("scrape_cfb_schedules", P)
s = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s)


def test_merge_master_dedupes_on_game_id(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2], "season": [2004, 2004]}).to_parquet(master)
    new = pd.DataFrame({"game_id": [2, 3], "season": [2004, 2004]})
    s.merge_master(new, str(master))
    out = pd.read_parquet(master).sort_values("game_id")
    assert out["game_id"].tolist() == [1, 2, 3]  # game_id 2 not duplicated
