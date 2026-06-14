"""Read final.json plays for CP model training.

The CP training frame is simpler than the EP/WP frame in model_training/ingest.py:
no next-score labeling (the outcome is the completion flag already on the play),
no sample weights, no NSH/bad-game-id filtering (handled elsewhere).

We read all plays and let features.filter_pass_plays() do the subset selection.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def build_cp_training_frame(
    final_dir: str | Path,
    seasons: list[int] | None = None,
) -> pl.DataFrame:
    """Read final.json plays from the backfill directory for CP model training.

    Reads all `*.json` files under `final_dir`. Each file is expected to be a
    `CFBPlayProcess` output dict with a `plays` key.

    Args:
        final_dir: Path to cfb/json/final/ directory containing `*.json` files.
        seasons: Optional list of season years to include. When None, all
            available seasons are included.

    Returns:
        polars DataFrame with all play rows (not yet filtered to pass_attempt;
        use features.filter_pass_plays() to get genuine pass attempts).
        Returns an empty DataFrame if no files are found or no plays match.
    """
    frames: list[pl.DataFrame] = []
    final_dir = Path(final_dir)

    for fpath in sorted(final_dir.glob("*.json")):
        with open(fpath, encoding="utf-8") as fh:
            obj = json.load(fh)

        # Season filter
        if seasons is not None and obj.get("season") not in seasons:
            continue

        plays = obj.get("plays") or []
        if not plays:
            continue

        try:
            frame = pl.DataFrame(plays, infer_schema_length=None)
        except Exception:
            # Malformed payload — skip silently
            continue

        # Attach top-level game metadata if absent from individual play rows
        for meta_col, meta_val in (
            ("game_id", obj.get("gameId") or obj.get("game_id")),
            ("season", obj.get("season")),
        ):
            if meta_col not in frame.columns and meta_val is not None:
                frame = frame.with_columns(pl.lit(meta_val).alias(meta_col))

        frames.append(frame)

    if not frames:
        return pl.DataFrame()

    return pl.concat(frames, how="diagonal_relaxed")
