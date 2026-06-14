import math

import polars as pl
import pytest
from rb_eval.aggregate import summarize_rusher_seasons


def _rush_plays(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def _base_play(rusher="A", season=2010, yds=5, down=1, dist=10, epa=0.5):
    return {
        "rusher_player_name": rusher,
        "season": season,
        "yds_rushed": yds,
        "start.down": down,
        "start.distance": dist,
        "epa": epa,
        "is_rush_opp": yds >= 4,
        "fo_success": (
            yds >= 0.5 * dist if down == 1
            else yds >= 0.7 * dist if down == 2
            else yds >= dist
        ),
        "highlight_yards": max(
            0.0,
            (0.5 * (min(yds, 8) - 4) if yds >= 4 else 0.0)
            + (yds - min(yds, 8) if yds > 8 else 0.0),
        ),
        "pos_team": 100,
    }


def test_epa_clamped_at_minus_4_5():
    play_big_loss = {
        **_base_play("A", 2010, yds=1, epa=-6.0),
        "is_rush_opp": False,
        "fo_success": False,
        "highlight_yards": 0.0,
    }
    plays = [_base_play("A", 2010) for _ in range(100)] + [play_big_loss]
    df = _rush_plays(plays)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "A").row(0, named=True)
    assert row["unadjusted_epa"] < row["epa"]  # unclamped < clamped (unclamped more negative)


def test_n_plays_filter_excludes_below_100():
    plays_50 = [_base_play("LowVol", 2010) for _ in range(50)]
    plays_101 = [_base_play("HighVol", 2010) for _ in range(101)]
    df = _rush_plays(plays_50 + plays_101)
    out = summarize_rusher_seasons(df)
    assert "LowVol" not in out["rusher_player_name"].to_list()
    assert "HighVol" in out["rusher_player_name"].to_list()


def test_n_opps_zero_guard_for_highlight_yards():
    plays = [
        {
            **_base_play("ToughYard", 2010, yds=2, epa=-0.1),
            "is_rush_opp": False,
            "highlight_yards": 0.0,
        }
        for _ in range(101)
    ]
    df = _rush_plays(plays)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "ToughYard").row(0, named=True)
    assert row["highlight_yards"] == 0.0
    assert row["n_opps"] == 0


def test_success_rate_formula():
    plays_success = [
        {
            **_base_play("S", 2010, yds=5, epa=0.3),
            "is_rush_opp": True,
            "fo_success": True,
            "highlight_yards": 0.5,
        }
        for _ in range(50)
    ]
    plays_fail = [
        {
            **_base_play("S", 2010, yds=2, epa=-0.1),
            "is_rush_opp": False,
            "fo_success": False,
            "highlight_yards": 0.0,
        }
        for _ in range(51)
    ]
    df = _rush_plays(plays_success + plays_fail)
    out = summarize_rusher_seasons(df)
    row = out.filter(pl.col("rusher_player_name") == "S").row(0, named=True)
    assert math.isclose(row["success"], 50 / 101, rel_tol=1e-9)


# --- lag + weight ---

from rb_eval.aggregate import add_season_lag


def _rusher_seasons(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_lag_shifts_prior_season_values():
    df = _rusher_seasons(
        [
            {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
             "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
            {"rusher_player_name": "A", "season": 2011, "n_plays": 120, "epa": 0.2,
             "success": 0.5, "highlight_yards": 0.6, "unadjusted_epa": 0.22, "n_opps": 70},
        ]
    )
    out = add_season_lag(df)
    row_2011 = out.filter(pl.col("season") == 2011).row(0, named=True)
    assert math.isclose(row_2011["lepa"], 0.1, rel_tol=1e-9)
    assert math.isclose(row_2011["lsuccess"], 0.4, rel_tol=1e-9)
    assert math.isclose(row_2011["lhlite_yds"], 0.5, rel_tol=1e-9)
    assert row_2011["lplays"] == 110


def test_first_season_has_null_lag():
    df = _rusher_seasons(
        [
            {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
             "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
        ]
    )
    out = add_season_lag(df)
    row = out.row(0, named=True)
    assert row["lepa"] is None
    assert row["lsuccess"] is None


def test_weight_formula():
    df = _rusher_seasons(
        [
            {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
             "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
            {"rusher_player_name": "A", "season": 2011, "n_plays": 120, "epa": 0.2,
             "success": 0.5, "highlight_yards": 0.6, "unadjusted_epa": 0.22, "n_opps": 70},
        ]
    )
    out = add_season_lag(df)
    row_2011 = out.filter(pl.col("season") == 2011).row(0, named=True)
    expected = math.sqrt(120**2 + 110**2)
    assert math.isclose(row_2011["weight"], expected, rel_tol=1e-9)


def test_non_consecutive_seasons_produce_adjacent_lag():
    df = _rusher_seasons(
        [
            {"rusher_player_name": "A", "season": 2012, "n_plays": 130, "epa": 0.3,
             "success": 0.55, "highlight_yards": 0.7, "unadjusted_epa": 0.32, "n_opps": 80},
            {"rusher_player_name": "A", "season": 2010, "n_plays": 110, "epa": 0.1,
             "success": 0.4, "highlight_yards": 0.5, "unadjusted_epa": 0.12, "n_opps": 60},
        ]
    )
    out = add_season_lag(df)
    row_2012 = out.filter(pl.col("season") == 2012).row(0, named=True)
    # sort is by (rusher, season): 2010 is adjacent → lag IS assigned
    assert row_2012["lepa"] is not None
    assert row_2012["lplays"] == 110


# --- orchestration ---

from rb_eval.aggregate import build_rusher_seasons, build_model_data


def _big_rush_frame(n_per_rusher: int = 110, seasons: list[int] | None = None) -> pl.DataFrame:
    seasons = seasons or [2010, 2011, 2012]
    rows = []
    for rusher in ["Alpha", "Beta"]:
        for season in seasons:
            for _ in range(n_per_rusher):
                rows.append(
                    {
                        "rusher_player_name": rusher,
                        "season": season,
                        "yds_rushed": 5,
                        "start.down": 1,
                        "start.distance": 10,
                        "epa": 0.3,
                        "is_rush_opp": True,
                        "fo_success": True,
                        "highlight_yards": 0.5,
                        "pos_team": 100,
                    }
                )
    return pl.DataFrame(rows)


def test_build_rusher_seasons_has_expected_columns():
    df = _big_rush_frame()
    out = build_rusher_seasons(df)
    for col in [
        "rusher_player_name", "season", "n_plays", "n_opps",
        "unadjusted_epa", "epa", "success", "highlight_yards",
        "lepa", "lsuccess", "weight",
    ]:
        assert col in out.columns, f"missing column: {col}"


def test_build_model_data_renames_to_gam_contract():
    df = _big_rush_frame()
    seasons_df = build_rusher_seasons(df)
    md = build_model_data(seasons_df)
    for col in ["target", "epa_per_play", "success", "highlight_yards", "weight", "season"]:
        assert col in md.columns, f"missing model_data column: {col}"
    assert md["epa_per_play"].null_count() == 0
    assert md["success"].null_count() == 0
