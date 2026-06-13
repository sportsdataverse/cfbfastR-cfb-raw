"""Phase 3 Task 3.1 — LOSO CV module tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cpoe.constants import FEATURE_COLS, TARGET_COL


@pytest.fixture()
def multi_season_df() -> pd.DataFrame:
    """3-season synthetic dataset (20 rows each)."""
    rng = np.random.default_rng(7)
    n = 20
    parts = []
    for season in (2021, 2022, 2023):
        data = {
            "season": season,
            "down": rng.integers(1, 5, n),
            "distance": rng.integers(1, 20, n),
            "yards_to_goal": rng.integers(1, 100, n),
            "score_diff": rng.integers(-28, 29, n),
            "seconds_remaining": rng.integers(0, 3600, n),
            "is_home": rng.integers(0, 2, n),
            "period": rng.integers(1, 5, n),
            "passing_down": rng.integers(0, 2, n),
            "completion": rng.integers(0, 2, n),
        }
        parts.append(pd.DataFrame(data))
    return pd.concat(parts, ignore_index=True)


def test_loso_imports():
    from cpoe.loso import run_loso_cv  # noqa: F401


def test_loso_returns_dict(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    assert isinstance(result, dict)


def test_loso_has_folds_key(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    assert "folds" in result


def test_loso_has_summary_key(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    assert "summary" in result


def test_loso_folds_count(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    # 3 seasons → 3 folds
    assert len(result["folds"]) == 3


def test_loso_fold_has_metrics(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    fold = result["folds"][0]
    assert "season" in fold
    assert "log_loss" in fold
    assert "brier_score" in fold
    assert "n_plays" in fold


def test_loso_log_loss_range(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    for fold in result["folds"]:
        assert 0.0 <= fold["log_loss"] <= 10.0


def test_loso_brier_score_range(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    for fold in result["folds"]:
        assert 0.0 <= fold["brier_score"] <= 1.0


def test_loso_summary_mean_log_loss(multi_season_df):
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df)
    assert "mean_log_loss" in result["summary"]
    assert "mean_brier_score" in result["summary"]


def test_loso_single_season_raises(multi_season_df):
    from cpoe.loso import run_loso_cv
    one_season = multi_season_df[multi_season_df["season"] == 2021]
    with pytest.raises(ValueError, match="at least 2"):
        run_loso_cv(one_season)


def test_loso_preds_column_in_folds(multi_season_df):
    """Each fold record must include per-play predictions array."""
    from cpoe.loso import run_loso_cv
    result = run_loso_cv(multi_season_df, return_preds=True)
    for fold in result["folds"]:
        assert "cp_pred" in fold
        assert len(fold["cp_pred"]) == fold["n_plays"]
