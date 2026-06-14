"""Select/rename ESPN final.json pass plays into the CP model input matrix.

Pass-play filter: pass_attempt==True, sack_vec==False, penalty_no_play==False.
All column transformations return polars DataFrames.
"""
from __future__ import annotations

import polars as pl

from . import constants as C


# ---------------------------------------------------------------------------
# Pass-play filter
# ---------------------------------------------------------------------------

def filter_pass_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only genuine pass attempts (exclude sacks and penalty-no-play rows).

    Args:
        df: Full plays DataFrame (must have pass_attempt, sack_vec, penalty_no_play).

    Returns:
        Filtered polars DataFrame.
    """
    return df.filter(
        (pl.col("pass_attempt") == True)      # noqa: E712
        & (pl.col("sack_vec") == False)       # noqa: E712
        & (pl.col("penalty_no_play") == False)  # noqa: E712
    )


# ---------------------------------------------------------------------------
# Feature derivation
# ---------------------------------------------------------------------------

def derive_passing_down(df: pl.DataFrame) -> pl.DataFrame:
    """Add a boolean passing_down column: True when down in {3,4} AND distance >= 5.

    Args:
        df: DataFrame with down and distance columns.

    Returns:
        df with an added passing_down column (Bool).
    """
    return df.with_columns(
        (pl.col("down").is_in([3, 4]) & (pl.col("distance") >= 5))
        .alias("passing_down")
    )


def rename_source_cols(df: pl.DataFrame) -> pl.DataFrame:
    """Rename ESPN source column names (dot-notation) to canonical feature names.

    Uses CPOE_SOURCE_COLS mapping. Source columns that do not exist in df are
    created as null columns so the pipeline never raises on missing data.

    Args:
        df: plays DataFrame with ESPN-flavoured dotted column names.

    Returns:
        DataFrame with canonical feature names (source cols dropped).
    """
    # Build rename + null-fill expressions
    exprs: list[pl.Expr] = []
    for canonical, source in C.CPOE_SOURCE_COLS.items():
        if canonical == "passing_down":
            # Derived later by derive_passing_down; skip here if not present
            if canonical not in df.columns:
                exprs.append(pl.lit(None).cast(pl.Boolean).alias(canonical))
        elif source in df.columns:
            if source != canonical:
                exprs.append(pl.col(source).alias(canonical))
        elif canonical not in df.columns:
            # Create null placeholder — cast to Float32 as a safe default
            exprs.append(pl.lit(None).cast(pl.Float32).alias(canonical))
    if exprs:
        df = df.with_columns(exprs)
    return df


def extract_cpoe_features(df: pl.DataFrame, approach: str = "A") -> pl.DataFrame:
    """Full feature-extraction pipeline: rename → derive_passing_down → select features.

    Args:
        df: plays DataFrame (raw or renamed ESPN columns).
        approach: "A" (8 features) or "B" (9 features with air_yards).

    Returns:
        DataFrame containing exactly the CPOE_FEATURES (or CPOE_FEATURES_B) columns.
    """
    features = C.CPOE_FEATURES if approach == "A" else C.CPOE_FEATURES_B

    # Step 1: rename ESP source columns to canonical names
    df = rename_source_cols(df)

    # Step 2: derive passing_down from down + distance if not already present
    if "passing_down" not in df.columns or df["passing_down"].is_null().all():
        df = derive_passing_down(df)

    # Step 3: select only the feature columns (cast to Float32 for DMatrix)
    select_exprs = []
    for feat in features:
        if feat in df.columns:
            select_exprs.append(pl.col(feat).cast(pl.Float32).alias(feat))
        else:
            select_exprs.append(pl.lit(None).cast(pl.Float32).alias(feat))

    return df.with_columns(select_exprs).select(features)


# ---------------------------------------------------------------------------
# Distance bucket
# ---------------------------------------------------------------------------

def assign_distance_bucket(df: pl.DataFrame) -> pl.DataFrame:
    """Classify start.distance into Short / Intermediate / Long buckets.

    Thresholds (inclusive):
      Short:        distance <= 3
      Intermediate: 4 <= distance <= 8
      Long:         distance >= 9

    Args:
        df: DataFrame with a distance column.

    Returns:
        df with added distance_bucket column (Utf8).
    """
    return df.with_columns(
        pl.when(pl.col("distance") <= C.DISTANCE_BUCKETS["Short"][1])
        .then(pl.lit("Short"))
        .when(pl.col("distance") <= C.DISTANCE_BUCKETS["Intermediate"][1])
        .then(pl.lit("Intermediate"))
        .otherwise(pl.lit("Long"))
        .alias("distance_bucket")
    )


# ---------------------------------------------------------------------------
# Feature matrix builder (polars → pandas for XGBoost DMatrix)
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pl.DataFrame, approach: str = "A"):
    """Build the CP model input matrix from a pass-plays DataFrame.

    Args:
        df: Pass-plays DataFrame (already filtered by filter_pass_plays).
        approach: "A" or "B".

    Returns:
        (X: pd.DataFrame, y: np.ndarray, keys: pd.DataFrame)
        X has columns in CPOE_FEATURES order, cast to Float32.
        y is the binary completion label (0/1 int).
        keys is (game_id, season, passer_player_name) for CPOE aggregation joins.
    """
    features = C.CPOE_FEATURES if approach == "A" else C.CPOE_FEATURES_B
    feat_df = extract_cpoe_features(df, approach=approach)
    X = feat_df.select(features).to_pandas()
    y = df["completion"].cast(pl.Int32).to_numpy()
    key_cols = [c for c in ("game_id", "season", "passer_player_name") if c in df.columns]
    keys = df.select(key_cols).to_pandas()
    return X, y, keys
