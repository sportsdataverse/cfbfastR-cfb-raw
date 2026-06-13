"""Ingest processed play-by-play parquet files for CPOE training.

Expected on-disk layout (cfbfastR-cfb-raw scraper output):

    <season_dir>/
        <game_id>/
            plays.parquet    ← one per game, produced by CFBPlayProcess

``load_season_pass_plays`` walks every ``plays.parquet`` file under
``season_dir``, applies ``extract_pass_features``, and concatenates into
a single DataFrame ready for training or LOSO cross-validation.
"""
from __future__ import annotations

import pathlib

import pandas as pd

from .features import extract_pass_features


def load_season_pass_plays(
    season_dir: pathlib.Path | str,
    *,
    glob: str = "**/plays.parquet",
) -> pd.DataFrame:
    """Load and filter pass plays from all games under ``season_dir``.

    Args:
        season_dir: Root directory to walk.  Typically
            ``<raw_base>/<season>/<season_type>/``.
        glob: Recursive glob pattern for per-game PBP files.

    Returns:
        pandas DataFrame with FEATURE_COLS + TARGET_COL columns.
        Empty (zero rows) DataFrame if no plays files are found.
    """
    season_dir = pathlib.Path(season_dir)
    parts: list[pd.DataFrame] = []

    for plays_path in sorted(season_dir.glob(glob)):
        try:
            raw = pd.read_parquet(plays_path)
        except Exception:
            continue
        feat = extract_pass_features(raw)
        if not feat.empty:
            parts.append(feat)

    if not parts:
        return pd.DataFrame()

    return pd.concat(parts, ignore_index=True)
