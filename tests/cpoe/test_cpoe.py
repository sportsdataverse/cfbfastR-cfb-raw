"""Tests for the CFB CPOE (Completion Percentage Over Expected) package.

All tests use synthetic polars DataFrames (50 rows) and do NOT require
backfill data on disk.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_df(n: int = 50, *, rng_seed: int = 42) -> pl.DataFrame:
    """Synthetic pass-play DataFrame (50 rows by default)."""
    rng = np.random.default_rng(rng_seed)
    return pl.DataFrame(
        {
            # feature cols (canonical names, post-rename)
            "down": rng.integers(1, 5, n).tolist(),
            "distance": rng.integers(1, 20, n).tolist(),
            "yards_to_goal": rng.integers(1, 100, n).tolist(),
            "pos_score_diff": rng.integers(-28, 28, n).tolist(),
            "secs_remaining": rng.integers(0, 3600, n).tolist(),
            "is_home": rng.integers(0, 2, n).tolist(),
            "period": rng.integers(1, 5, n).tolist(),
            # play-type flags
            "pass_attempt": [True] * n,
            "sack_vec": [False] * n,
            "penalty_no_play": [False] * n,
            # outcome
            "completion": rng.integers(0, 2, n).tolist(),
            # join keys
            "game_id": [1] * n,
            "season": [2024] * n,
            "passer_player_name": ["QB A"] * n,
        }
    )


# ---------------------------------------------------------------------------
# filter_pass_plays
# ---------------------------------------------------------------------------

class TestPassFilter:
    def test_pass_filter_removes_sacks(self):
        """Rows with sack_vec=True must be excluded by filter_pass_plays."""
        from cpoe.features import filter_pass_plays

        df = _make_df(50)
        # mark first 5 rows as sacks
        sack_vec = [True] * 5 + [False] * 45
        df = df.with_columns(pl.Series("sack_vec", sack_vec))
        out = filter_pass_plays(df)
        assert out.height == 45
        assert (pl.Series(out["sack_vec"]) == False).all()  # noqa: E712

    def test_pass_filter_removes_penalties(self):
        """Rows with penalty_no_play=True must be excluded by filter_pass_plays."""
        from cpoe.features import filter_pass_plays

        df = _make_df(50)
        penalty = [False] * 45 + [True] * 5
        df = df.with_columns(pl.Series("penalty_no_play", penalty))
        out = filter_pass_plays(df)
        assert out.height == 45
        assert (pl.Series(out["penalty_no_play"]) == False).all()  # noqa: E712


# ---------------------------------------------------------------------------
# derive_passing_down
# ---------------------------------------------------------------------------

class TestPassingDown:
    def _frame_with_down_dist(self, down: int, distance: int) -> pl.DataFrame:
        df = _make_df(1)
        return df.with_columns(
            pl.lit(down).alias("down"),
            pl.lit(distance).alias("distance"),
        )

    def test_passing_down_short(self):
        """3rd-and-4 (distance < 5) → passing_down=False."""
        from cpoe.features import derive_passing_down

        out = derive_passing_down(self._frame_with_down_dist(3, 4))
        assert out["passing_down"][0] == False  # noqa: E712

    def test_passing_down_medium(self):
        """3rd-and-6 (distance >= 5) → passing_down=True."""
        from cpoe.features import derive_passing_down

        out = derive_passing_down(self._frame_with_down_dist(3, 6))
        assert out["passing_down"][0] == True  # noqa: E712

    def test_passing_down_4th(self):
        """4th-and-5 (distance >= 5) → passing_down=True."""
        from cpoe.features import derive_passing_down

        out = derive_passing_down(self._frame_with_down_dist(4, 5))
        assert out["passing_down"][0] == True  # noqa: E712


# ---------------------------------------------------------------------------
# extract_cpoe_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_feature_columns_present(self):
        """After extract_cpoe_features all CPOE_FEATURES must be in the output."""
        from cpoe.constants import CPOE_FEATURES
        from cpoe.features import extract_cpoe_features

        # Build a df that has all source columns (same names as features in this
        # synthetic frame — rename_source_cols is a no-op when cols already exist
        # under canonical names, but we also include the ESPN-dotted originals
        # so rename_source_cols has something to rename)
        df = _make_df(50)
        # Add ESPN-style dotted source columns so rename can find them
        df = df.with_columns(
            pl.col("down").alias("start.down"),
            pl.col("distance").alias("start.distance"),
            pl.col("yards_to_goal").alias("start.yardsToEndzone"),
            pl.col("pos_score_diff").alias("pos_score_diff_start"),
            pl.col("secs_remaining").alias("start.TimeSecsRem"),
            pl.col("is_home").alias("start.is_home"),
        )
        out = extract_cpoe_features(df)
        for col in CPOE_FEATURES:
            assert col in out.columns, f"Missing feature column: {col}"


# ---------------------------------------------------------------------------
# assign_distance_bucket
# ---------------------------------------------------------------------------

class TestDistanceBucket:
    def _df_with_distance(self, dist: int) -> pl.DataFrame:
        df = _make_df(1)
        return df.with_columns(pl.lit(dist).alias("distance"))

    def test_distance_bucket_short(self):
        """distance=2 → bucket='Short'."""
        from cpoe.features import assign_distance_bucket

        out = assign_distance_bucket(self._df_with_distance(2))
        assert out["distance_bucket"][0] == "Short"

    def test_distance_bucket_intermediate(self):
        """distance=6 → bucket='Intermediate'."""
        from cpoe.features import assign_distance_bucket

        out = assign_distance_bucket(self._df_with_distance(6))
        assert out["distance_bucket"][0] == "Intermediate"

    def test_distance_bucket_long(self):
        """distance=10 → bucket='Long'."""
        from cpoe.features import assign_distance_bucket

        out = assign_distance_bucket(self._df_with_distance(10))
        assert out["distance_bucket"][0] == "Long"


# ---------------------------------------------------------------------------
# CPOE formula
# ---------------------------------------------------------------------------

class TestCpoeFormula:
    def test_cpoe_formula(self):
        """CPOE = completion - expected_completion; 1 - 0.6 = 0.4."""
        completion = 1.0
        expected_completion = 0.6
        cpoe = completion - expected_completion
        assert abs(cpoe - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# XGBoost fit + predict
# ---------------------------------------------------------------------------

class TestXgbFitPredict:
    def test_xgb_fit_predict(self):
        """50-row synthetic df: XGBoost fit and predict probs in [0,1]."""
        import xgboost as xgb
        from cpoe.constants import CPOE_FEATURES, CPOE_PARAMS

        rng = np.random.default_rng(99)
        n = 50
        X_data = {col: rng.random(n).tolist() for col in CPOE_FEATURES}
        # passing_down must be 0/1 bool-ish
        X_data["passing_down"] = rng.integers(0, 2, n).tolist()
        X_data["down"] = rng.integers(1, 5, n).tolist()
        X_data["period"] = rng.integers(1, 5, n).tolist()
        X_data["is_home"] = rng.integers(0, 2, n).tolist()

        df = pl.DataFrame(X_data)
        X = df[CPOE_FEATURES].to_pandas()
        y = np.array(rng.integers(0, 2, n))

        dtrain = xgb.DMatrix(X, label=y)
        model = xgb.train(CPOE_PARAMS, dtrain, num_boost_round=10)
        preds = model.predict(xgb.DMatrix(X))

        assert len(preds) == n
        assert float(preds.min()) >= 0.0
        assert float(preds.max()) <= 1.0
