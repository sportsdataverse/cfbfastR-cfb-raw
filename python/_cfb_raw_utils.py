"""Shared helpers for cfbfastR-cfb-raw scrapers."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

from importlib.metadata import version as _pkg_version

# Bump SCHEMA_REV whenever the final-JSON shape or enrichment inputs change in a way
# that should force a reprocess of already-built games.
SCHEMA_REV = 1
try:
    _SDV_VERSION = _pkg_version("sportsdataverse")
except Exception:  # noqa: BLE001
    _SDV_VERSION = "0.0.0"
# NOTE: sportsdataverse exposes no __version__ attribute; use importlib.metadata.
PROCESSING_VERSION = f"{_SDV_VERSION}+{SCHEMA_REV}"


def get_logger(name: str, year: int | str) -> logging.Logger:
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}_{year}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(f"logs/{name}_logfile_{year}.log")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def write_json_atomic(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), default=str)
    os.replace(tmp, path)


def stamp(obj, *, game_id: int, season: int, week=None):
    """Attach self-describing identity. Dicts get keys merged; lists are wrapped."""
    meta = {"game_id": game_id, "season": season, "week": week}
    if isinstance(obj, dict):
        return {**obj, **meta}
    return {**meta, "data": obj}


def most_recent_cfb_season(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    # CFB season rolls over in August; before August belongs to the prior season year.
    return now.year if now.month >= 8 else now.year - 1


def _safe(fn: Callable, *args, logger: logging.Logger | None = None, default=None, **kwargs):
    """Call fn, returning `default` (and logging) on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - intentional broad guard around a single extra
        if logger is not None:
            logger.exception("extra fetch failed: %s%s", getattr(fn, "__name__", fn), args)
        return default


def run_pool(fn: Callable, items: Iterable, *, kind: str = "process",
             workers: int | None = None, desc: str | None = None) -> list:
    items = list(items)
    if not items:
        return []
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    Executor = ProcessPoolExecutor if kind == "process" else ThreadPoolExecutor
    results = []
    try:
        from tqdm import tqdm
    except Exception:  # noqa: BLE001
        tqdm = None
    with Executor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in items}
        it = as_completed(futures)
        if tqdm is not None:
            it = tqdm(it, total=len(futures), desc=desc)
        for fut in it:
            results.append(fut.result())
    return results


def load_schedule_master(path: str = "cfb/cfb_schedule_master.parquet"):
    import pandas as pd
    return pd.read_parquet(path)


def games_for_seasons(master, start: int, end: int) -> list[int]:
    df = master[(master["season"] >= start) & (master["season"] <= end)]
    return df["game_id"].astype(int).unique().tolist()


def filter_undone(games, dir: str = "cfb/json/final", rescrape: bool = False) -> list[int]:
    if rescrape:
        return list(games)
    d = Path(dir)
    return [g for g in games if not (d / f"{g}.json").exists()]


def season_type_from_raw(raw: dict):
    """Best-effort integer season_type from an ESPN summary, or None.
    ESPN places it inconsistently (header.season.type as int or dict, or
    header.competitions[0].type.id)."""
    hdr = raw.get("header", {}) or {}
    st = hdr.get("season", {})
    if isinstance(st, dict):
        val = st.get("type")
        if isinstance(val, dict):
            val = val.get("type") or val.get("id")
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    comps = (hdr.get("competitions") or [{}])
    t = comps[0].get("type", {}) if comps else {}
    if isinstance(t, dict) and t.get("id") is not None:
        try:
            return int(t["id"])
        except (TypeError, ValueError):
            pass
    return None
