"""Load rush plays from final.json and compute fo_success + is_rush_opp."""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def add_fo_success(df: pl.DataFrame) -> pl.DataFrame:
    """Annotate each rush with first-opportunity success per down tier."""
    return df.with_columns(
        fo_success=pl.when(pl.col("start.down") == 1)
        .then(pl.col("yds_rushed") >= 0.5 * pl.col("start.distance"))
        .when(pl.col("start.down") == 2)
        .then(pl.col("yds_rushed") >= 0.7 * pl.col("start.distance"))
        .otherwise(pl.col("yds_rushed") >= pl.col("start.distance"))
        .cast(pl.Boolean),
    )


def filter_rush_plays(df: pl.DataFrame) -> pl.DataFrame:
    """Keep only individual rusher plays; add fo_success + is_rush_opp."""
    epa_col = "EPA" if "EPA" in df.columns else "epa"
    out = (
        df.filter(pl.col("rush") == True)  # noqa: E712
        .filter(pl.col("pos_team").is_not_null())
        .filter(pl.col(epa_col).is_not_null())
        .filter(pl.col("rusher_player_name").is_not_null())
        .filter(pl.col("rusher_player_name") != "TEAM")
    )
    if epa_col == "EPA":
        out = out.rename({"EPA": "epa"})
    out = add_fo_success(out)
    return out.with_columns(is_rush_opp=(pl.col("yds_rushed") >= 4).cast(pl.Boolean))


def load_rush_plays(final_dir: str | Path, seasons: list[int] | None = None) -> pl.DataFrame:
    """Load rush plays from per-game final.json files in *final_dir*."""
    frames = []
    for path in sorted(Path(final_dir).glob("*.json")):
        raw = json.loads(path.read_text())
        if seasons is not None and raw.get("season") not in seasons:
            continue
        plays = raw.get("plays") or []
        if not plays:
            continue
        frames.append(pl.DataFrame(plays, infer_schema_length=None))
    if not frames:
        return pl.DataFrame()
    return filter_rush_plays(pl.concat(frames, how="diagonal_relaxed"))
