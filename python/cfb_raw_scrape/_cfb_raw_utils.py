"""Shared helpers for cfbfastR-cfb-raw scrapers."""

from __future__ import annotations

import json
import logging
import math
import numbers
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Callable, Iterable

# Bump SCHEMA_REV whenever the final-JSON shape or enrichment inputs change in a way
# that should force a reprocess of already-built games.
#   rev 2: odds_override now sourced from the cfb_line_odds multi-book consensus
#          (cfb/odds_consensus.parquet) instead of the ESPN betting aux; EPA/WPA
#          spread inputs change, so every prior 0.0.69+1 final must rebuild.
SCHEMA_REV = 2
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


def json_safe(o):
    """Recursively coerce a value into a STANDARDS-VALID-JSON-serializable form.

    Python's json emits bare ``NaN``/``Infinity`` literals for float nan/inf, which are
    NOT valid JSON — Python can re-read them, but R (jsonlite), JS, Go, etc. reject the
    file. Since the `final` JSON contract is consumed cross-language (the R `-data` repo),
    we map nan/±inf -> null and normalize numpy/Decimal reals to plain float/int here.
    """
    if isinstance(o, dict):
        return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, bool):
        return o
    if isinstance(o, numbers.Integral):
        return int(o)
    if isinstance(o, numbers.Real):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else f
    return o


def write_json_atomic(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Per-writer unique temp: a fixed "{path}.tmp" RACES when two processes write the
    # same game concurrently -- one's os.replace consumes the shared temp, the other's
    # then FileNotFoundErrors. pid+uuid makes each writer's temp private; the final
    # os.replace stays atomic and is harmlessly last-writer-wins on the destination.
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}-{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            # allow_nan=False guarantees a valid-JSON failure rather than a silent NaN literal;
            # json_safe has already removed nan/inf, so this never raises in practice.
            json.dump(
                json_safe(obj), f, separators=(",", ":"), default=str, allow_nan=False
            )
        # On Windows the parallel scraper intermittently hits WinError 32 (PermissionError) here
        # when AV / the NTFS change journal briefly locks the .tmp or destination; os.replace is
        # atomic on Linux and never sees this. Retry a bounded number of times with a short
        # increasing backoff, then re-raise so a genuine permission problem still surfaces.
        _REPLACE_ATTEMPTS = 5
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    raise
                time.sleep(0.1 * (attempt + 1))
    finally:
        # never leave a private temp behind on a failed write
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def write_json_guarded(
    obj, path: str | Path, *, min_ratio: float = 0.5, logger=None
) -> bool:
    """Write ``obj`` unless doing so would replace a substantially larger file.

    ESPN intermittently answers a rescrape with a 5xx-degraded or empty body.
    ``espn_cfb_pbp(raw=True)`` turns that into a near-empty allowlist dict, and
    writing it unconditionally CLOBBERS good banked data. Observed in the 2004
    pilot: 11 games went from 54KB-414KB down to a 251-byte stub, taking 16
    collateral betting/team_box_extra files with them, against a background of
    12x HTTP 429 and 154x 5xx.

    A rescrape must never leave the tree worse than it found it, so a write that
    would shrink an existing file below ``min_ratio`` of its current size is
    refused and the good data is kept. Growth and modest shrinkage (real content
    changes) pass through untouched.

    Returns True if the file was written, False if the write was refused.
    """
    path = Path(path)
    payload = json.dumps(
        json_safe(obj), separators=(",", ":"), default=str, allow_nan=False
    )
    new_bytes = len(payload.encode("utf-8"))
    if path.exists():
        old_bytes = path.stat().st_size
        if old_bytes > 0 and new_bytes < min_ratio * old_bytes:
            if logger is not None:
                logger.warning(
                    "REFUSED degraded write %s: %d -> %d bytes (%.0f%% of existing); keeping banked copy",
                    path,
                    old_bytes,
                    new_bytes,
                    100.0 * new_bytes / old_bytes,
                )
            return False
    write_json_atomic(obj, path)
    return True


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


def _safe(
    fn: Callable, *args, logger: logging.Logger | None = None, default=None, **kwargs
):
    """Call fn, returning `default` (and logging) on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001 - intentional broad guard around a single extra
        if logger is not None:
            logger.exception(
                "extra fetch failed: %s%s", getattr(fn, "__name__", fn), args
            )
        return default


def run_pool(
    fn: Callable,
    items: Iterable,
    *,
    kind: str = "process",
    workers: int | None = None,
    desc: str | None = None,
) -> list:
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
    # spawn, never fork: polars/rayon is already initialised in the parent, and
    # forked workers inherit its locked thread pool -- observed 2026-08-29 as a
    # 0%-CPU pool stuck at 0/948 for 45 min. Thread pools take no context.
    kwargs = {}
    if kind == "process":
        import multiprocessing

        kwargs["mp_context"] = multiprocessing.get_context("spawn")
    with Executor(max_workers=workers, **kwargs) as ex:
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


def status_state(doc: dict) -> str | None:
    """ESPN game state -- ``"pre"`` | ``"in"`` | ``"post"`` -- or None if absent.

    Reads the same place in an ESPN summary and in a banked final, so the scraper's
    pre-game guard and the shell test below cannot drift apart.
    """
    comp = ((doc.get("header") or {}).get("competitions") or [{}])[0] or {}
    return ((comp.get("status") or {}).get("type") or {}).get("state")


def final_is_pregame_shell(path: Path) -> bool:
    """True when a banked final is a PRE-GAME shell rather than a real scrape.

    A summary fetched before kickoff carries no plays, but ``download_game`` used
    to bank it anyway. Because ``filter_undone`` below was existence-only, such a
    shell then counted as a completed scrape and the game was skipped for the rest
    of the season -- while the job stayed green. The 2026-08-02 full-history
    reprocess (a230f2d50) banked 946 of them, one for every unplayed 2026 game.

    The test is deliberately NOT "zero plays". A finished game with no ESPN
    play-by-play source legitimately banks ``count == 0`` (e.g. 242410193), and
    treating those as undone would re-scrape them every single day forever. Only
    zero plays AND an ESPN status still reading ``pre`` is a shell.
    """
    try:
        data = json.loads(path.read_bytes())
        if int(data.get("count") or 0) > 0:
            return False
        return status_state(data) == "pre"
    except Exception:  # noqa: BLE001 - unreadable OR malformed: not a usable scrape
        return True


def filter_undone(
    games, dir: str = "cfb/json/final", rescrape: bool = False
) -> list[int]:
    if rescrape:
        return list(games)
    d = Path(dir)
    # ponytail: this parses each banked final -- measured 22s across a 958-game
    # season, against a scrape that runs for minutes. If that stops being cheap,
    # read only the stamped `count`/`header.status` keys instead of the whole doc.
    out = []
    for g in games:
        f = d / f"{g}.json"
        if not f.exists() or final_is_pregame_shell(f):
            out.append(g)
    return out


def read_ids_file(ids_file: str) -> set[int]:
    """Parse a whitespace-separated game-id list once.

    Split out from ``filter_ids_file`` because the caller loops SEASONS: re-reading
    and re-parsing the same file per season is pure waste on a wide -s/-e range,
    which is exactly what the degraded-retry driver runs.
    """
    return {
        int(tok)
        for tok in Path(ids_file).read_text(encoding="utf-8").split()
        if tok.strip()
    }


def filter_ids_file(games, ids_file: str) -> list[int]:
    """Return only the games named in ``ids_file`` (whitespace-separated ids).

    A targeted recovery pass: `scripts/retry_degraded_games.sh` harvests degraded
    game ids out of the scrape logs and hands them back to be re-fetched.

    Intersected with the season's schedule on purpose -- a stray or mistyped id
    must not send the scraper after a game that is not in this season, and the
    caller loops seasons, so each season takes only its own share of the list.
    """
    return [g for g in games if g in read_ids_file(ids_file)]


def hollow_game_ids(failures_csv: str = "logs/scrape_failures.csv") -> set[int]:
    """Return game IDs recorded as hollow_extras in scrape_failures.csv."""
    import csv as _csv

    p = Path(failures_csv)
    if not p.exists():
        return set()
    with p.open() as f:
        return {
            int(row["game_id"])
            for row in _csv.DictReader(f)
            if row.get("issue") == "hollow_extras"
        }


def filter_hollow(games, failures_csv: str = "logs/scrape_failures.csv") -> list[int]:
    """Return only games flagged as hollow_extras (or no_final) in scrape_failures.csv."""
    import csv as _csv

    p = Path(failures_csv)
    if not p.exists():
        raise FileNotFoundError(
            f"Run scrape_failures.py first to generate {failures_csv}"
        )
    with p.open() as f:
        flagged = {int(row["game_id"]) for row in _csv.DictReader(f)}
    game_set = set(games)
    return [g for g in games if g in flagged and g in game_set]


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
    comps = hdr.get("competitions") or [{}]
    t = comps[0].get("type", {}) if comps else {}
    if isinstance(t, dict) and t.get("id") is not None:
        try:
            return int(t["id"])
        except (TypeError, ValueError):
            pass
    return None


def is_recruiting_class_final(year: int, today=None) -> bool:
    """True once recruiting class `year` can no longer gain recruits.

    Shared by stage 50 (247Sports) and stage 52 (ESPN) so the two producers
    cannot drift into different ideas of when a class is closed.

    A class signs in December of ``year - 1`` (early period) and February of
    ``year`` (traditional), with late additions after. April of the class year
    is the conservative line.

    This exists because a completion MARKER is not finality. Stage 50 treated
    "the manifest exists" as done, so the 2027 class was banked complete on
    2026-08-06 at 4,779 rows -- against signed classes of 5,678-5,952 -- and
    would never have refreshed through its own signing day.
    """
    from datetime import date

    today = today or date.today()
    return (today.year, today.month) >= (year, 4)
