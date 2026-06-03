# cfbfastR-cfb-raw Implementation Plan (Plan 1 of 2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `cfbfastR-cfb-raw` — a Python/uv repo that scrapes ESPN college-football game JSON into per-game `raw` + fully-enriched `final` files (plus standalone aux/extra datasets and a schedule master), supports full 2004→present backfill and incremental daily runs, and can rebuild `final` from on-disk `raw` fully offline.

**Architecture:** Thin Python scrapers call `sportsdataverse` (`CFBPlayProcess` + `espn_cfb_*`) as the SDK boundary; a shared `_cfb_raw_utils.py` provides logging, a ProcessPool runner, atomic writes, schedule-master IO, and a `PROCESSING_VERSION` stamp. Each per-game task banks `raw` first, runs the enrichment pipeline, fetches validated/de-duplicated extra endpoints, persists every aux/extra as standalone season-partitioned JSON, and writes `final` last. A version-gated `reprocess_cfb_json.py` rebuilds `final` from disk with injected odds (no network). Bash orchestrators + GitHub Actions run scrapes on the CFB calendar and fire a `repository_dispatch` to the (future) `-data` repo.

**Tech Stack:** Python 3.11, uv, `sportsdataverse` (local path source for dev / pinned release for CI), pandas, polars, pyarrow, tqdm, pytest; bash; GitHub Actions (`astral-sh/setup-uv`, `peter-evans/repository-dispatch`).

**Spec:** `docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md` (read it first).

**Repo root:** `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfbfastR-cfb-raw` (already `git init`'d, branch `main`, contains only `docs/`).

**sdv-py repo:** `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\sdv-py` (Phase A edits here; branch off `main`).

---

## File Structure

**`cfbfastR-cfb-raw/` (this repo):**
- `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` — uv project config.
- `python/_cfb_raw_utils.py` — shared helpers (logging, pool, atomic write, master IO, version stamp, betting/team-box helpers).
- `python/scrape_cfb_schedules.py` — season → schedules + master.
- `python/scrape_cfb_json.py` — core per-game raw+final+aux+extras.
- `python/reprocess_cfb_json.py` — offline rebuild of final from raw.
- `python/scrape_cfb_rosters.py` / `_participants.py` / `_betting.py` / `_officials.py` / `_power_index.py` — single-dataset refreshers.
- `scripts/daily_cfb_scraper.sh` / `backfill_cfb.sh` / `reprocess_cfb.sh` — orchestration.
- `.github/workflows/scrape_cfb_raw.yml` / `cfbfastR_cfb_data_trigger.yml` — CI.
- `tests/` — pytest suite (unit on helpers; integration on scrapers via fixtures).
- `tests/fixtures/` — saved ESPN JSON fixtures for offline tests.
- `CLAUDE.md`, `README.md`.

**`sdv-py/` (Phase A):**
- `sportsdataverse/cfb/cfb_pbp.py` — allowlist extension, resolved-odds exposure, injected-odds path.
- `tests/cfb/test_cfb_pbp_offline.py` — new offline tests.

---

## Conventions for this plan

- All Python commands run via `uv run` from the repo root.
- Tests are offline by default (use fixtures); live-API tests are gated behind `CFB_LIVE_TESTS=1` and `pytest -m live`.
- Each task is TDD: failing test → run → implement → run → commit.
- Commit messages use Conventional Commits, no AI co-author trailers.

---

# Phase A — sdv-py prerequisites (offline-reprocess correctness)

> These three edits live in the **sdv-py** repo. They make `raw` capture `injuries`/`gameNotes`, and make the spread (an EPA/WPA input) reproducible offline. Without A2/A3, reprocess of 2024+ games is not provably offline (spec §12.2). Do Phase A on a branch in sdv-py and open a PR; the rest of Plan 1 pins that change.

### Task A0: Branch sdv-py

**Files:** none (git only).

- [ ] **Step 1: Create branch**

Run:
```bash
cd /c/Users/saiem/Documents/GitHub-Data/sdv-dev/sdv-py
git checkout main && git pull && git checkout -b feat/cfb-offline-reprocess
```
Expected: `Switched to a new branch 'feat/cfb-offline-reprocess'`

---

### Task A1: Extend raw allowlist with `injuries` + `gameNotes`

**Files:**
- Modify: `sportsdataverse/cfb/cfb_pbp.py:131` (the `incoming_keys_expected` list)
- Test: `tests/cfb/test_cfb_pbp_offline.py`

- [ ] **Step 1: Write the failing test**

Create `tests/cfb/test_cfb_pbp_offline.py`:
```python
import json
from pathlib import Path
from sportsdataverse.cfb.cfb_pbp import CFBPlayProcess

FIX = Path(__file__).parent / "fixtures"


def _load_summary(name="summary_401628455.json"):
    return json.loads((FIX / name).read_text())


def test_raw_allowlist_includes_injuries_and_gamenotes(monkeypatch):
    summary = _load_summary()
    summary["injuries"] = [{"team": {"id": "333"}, "injuries": []}]
    summary["gameNotes"] = [{"type": "note", "headline": "Week 1"}]

    class _Resp:
        def json(self):
            return summary

    monkeypatch.setattr("sportsdataverse.cfb.cfb_pbp.download", lambda *a, **k: _Resp())
    raw = CFBPlayProcess(gameId=401628455, raw=True).espn_cfb_pbp()
    assert "injuries" in raw and raw["injuries"], "injuries dropped by raw allowlist"
    assert "gameNotes" in raw and raw["gameNotes"], "gameNotes dropped by raw allowlist"
```

You must first create the fixture. Run once (live) to capture it, or hand-build a minimal summary:
```bash
cd /c/Users/saiem/Documents/GitHub-Data/sdv-dev/sdv-py
mkdir -p tests/cfb/fixtures
uv run python -c "import json,urllib.request; u='http://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event=401628455'; open('tests/cfb/fixtures/summary_401628455.json','w').write(urllib.request.urlopen(u).read().decode())"
```
Expected: fixture file written (~80–120 KB).

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py::test_raw_allowlist_includes_injuries_and_gamenotes -v`
Expected: FAIL — `KeyError`/assertion: `injuries` not present (current allowlist drops it).

- [ ] **Step 3: Implement — add the two keys to the allowlist**

In `sportsdataverse/cfb/cfb_pbp.py`, the `incoming_keys_expected` list (around line 131) currently ends with `"standings",`. Add the two keys:
```python
        incoming_keys_expected = [
            "boxscore",
            "format",
            "gameInfo",
            "drives",
            "leaders",
            "broadcasts",
            "predictor",
            "pickcenter",
            "againstTheSpread",
            "odds",
            "winprobability",
            "header",
            "scoringPlays",
            "videos",
            "standings",
            "injuries",
            "gameNotes",
        ]
```
`injuries` and `gameNotes` are array-like, so they default to `[]` via the existing `dict_keys_expected` branch (they are not in `dict_keys_expected`, so the `else []` applies). No other change needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py::test_raw_allowlist_includes_injuries_and_gamenotes -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sportsdataverse/cfb/cfb_pbp.py tests/cfb/test_cfb_pbp_offline.py tests/cfb/fixtures/summary_401628455.json
git commit -m "feat(cfb): keep injuries + gameNotes in raw summary allowlist"
```

---

### Task A2: Expose resolved odds + `odds_source`; prefer summary keys

**Files:**
- Modify: `sportsdataverse/cfb/cfb_pbp.py` (`__helper_cfb_pickcenter` ~845; add `self.odds_source`)
- Test: `tests/cfb/test_cfb_pbp_offline.py`

**Context:** `__helper_cfb_pickcenter` already sets `self.gameSpread/overUnder/homeFavorite/gameSpreadAvailable`. It cascades to the live `__helper__espn_cfb_odds_information__` only when the summary `pickcenter` array has ≤1 entries. We add a `self.odds_source` tag recording which path produced the odds, so the scraper can persist it.

- [ ] **Step 1: Write the failing test**

Append to `tests/cfb/test_cfb_pbp_offline.py`:
```python
def test_odds_source_tag_summary_path(monkeypatch):
    summary = _load_summary()
    # force a populated pickcenter (>1) so the summary path is taken
    summary["pickcenter"] = [
        {"provider": {"id": "58"}, "spread": -7.5, "overUnder": 52.5,
         "homeTeamOdds": {"favorite": True}},
        {"provider": {"id": "1002"}, "spread": -7.0, "overUnder": 52.0,
         "homeTeamOdds": {"favorite": True}},
    ]

    class _Resp:
        def json(self):
            return summary

    monkeypatch.setattr("sportsdataverse.cfb.cfb_pbp.download", lambda *a, **k: _Resp())
    proc = CFBPlayProcess(gameId=401628455)
    proc.espn_cfb_pbp()
    assert proc.odds_source == "summary_pickcenter"
    assert proc.gameSpreadAvailable is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py::test_odds_source_tag_summary_path -v`
Expected: FAIL — `AttributeError: 'CFBPlayProcess' object has no attribute 'odds_source'`.

- [ ] **Step 3: Implement — set `self.odds_source` in both branches**

In `__helper_cfb_pickcenter` (cfb_pbp.py ~845), set the tag in each branch. In the `if len(pbp_txt.get("pickcenter", [])) > 1:` branch, after `gameSpreadAvailable = True`, add:
```python
            self.odds_source = "summary_pickcenter"
```
In the `else:` branch, after the `__helper__espn_cfb_odds_information__()` call assigns the tuple, add:
```python
            self.odds_source = "core_odds_api" if gameSpreadAvailable else "default"
```
Also initialize `self.odds_source = None` in `__init__` (near the other `self.gameSpread`-style attributes; if none exist there, add it at the top of `__init__`).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py::test_odds_source_tag_summary_path -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add sportsdataverse/cfb/cfb_pbp.py tests/cfb/test_cfb_pbp_offline.py
git commit -m "feat(cfb): tag odds_source (summary_pickcenter|core_odds_api|default) on CFBPlayProcess"
```

---

### Task A3: Injected-odds path for offline reprocess

**Files:**
- Modify: `sportsdataverse/cfb/cfb_pbp.py` (`__init__` accept `odds_override`; `__helper_cfb_pickcenter` honor it)
- Test: `tests/cfb/test_cfb_pbp_offline.py`

**Context:** On reprocess we must supply the previously-resolved spread so the live core-odds endpoint is never hit and the `(2.5, 55.5, True, False)` default is never inherited.

- [ ] **Step 1: Write the failing test**

Append to `tests/cfb/test_cfb_pbp_offline.py`:
```python
def test_injected_odds_bypasses_network(monkeypatch):
    summary = _load_summary()
    summary["pickcenter"] = []  # force cascade path

    class _Resp:
        def json(self):
            return summary

    def _boom(*a, **k):
        raise AssertionError("network odds endpoint must NOT be called when odds injected")

    monkeypatch.setattr("sportsdataverse.cfb.cfb_pbp.download",
                        lambda url=None, *a, **k: _Resp() if "summary" in (url or "") else _boom())
    proc = CFBPlayProcess(
        gameId=401628455,
        odds_override={"gameSpread": -10.5, "overUnder": 60.0,
                       "homeFavorite": True, "gameSpreadAvailable": True},
    )
    proc.espn_cfb_pbp()
    assert proc.gameSpread == -10.5
    assert proc.overUnder == 60.0
    assert proc.odds_source == "injected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py::test_injected_odds_bypasses_network -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'odds_override'`.

- [ ] **Step 3: Implement — accept and honor `odds_override`**

In `__init__`, add the parameter (keep existing params; append before `**kwargs`):
```python
    def __init__(self, gameId=0, raw=False, path_to_json="/", return_keys=None,
                 odds_override=None, **kwargs):
        ...
        self.odds_override = odds_override
        self.odds_source = None
```
At the **top** of `__helper_cfb_pickcenter`, short-circuit when an override is present:
```python
    def __helper_cfb_pickcenter(self, pbp_txt):
        if self.odds_override is not None:
            o = self.odds_override
            self.gameSpread = o["gameSpread"]
            self.overUnder = o["overUnder"]
            self.homeFavorite = o["homeFavorite"]
            self.gameSpreadAvailable = o["gameSpreadAvailable"]
            self.odds_source = "injected"
            return {
                "gameSpread": self.gameSpread,
                "overUnder": self.overUnder,
                "homeFavorite": self.homeFavorite,
                "gameSpreadAvailable": self.gameSpreadAvailable,
            }
        # ... existing body unchanged ...
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/cfb/test_cfb_pbp_offline.py -v`
Expected: PASS (all four offline tests).

- [ ] **Step 5: Commit + push branch + open PR**

```bash
git add sportsdataverse/cfb/cfb_pbp.py tests/cfb/test_cfb_pbp_offline.py
git commit -m "feat(cfb): accept odds_override to make offline reprocess deterministic"
git push -u origin feat/cfb-offline-reprocess
gh pr create --fill --base main
```
Expected: PR URL printed. (The cfbfastR-cfb-raw `pyproject` pins this via the local path source during dev; CI pins the released version once merged.)

---

# Phase B — cfbfastR-cfb-raw scaffolding

> All remaining tasks are in `c:\Users\saiem\Documents\GitHub-Data\sdv-dev\cfbfastR-dev\cfbfastR-cfb-raw`.

### Task B1: uv project scaffold

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `python/__init__.py` (empty), `tests/__init__.py` (empty), `tests/fixtures/.gitkeep`

- [ ] **Step 1: Create `.python-version`**

```
3.11
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "cfbfastr-cfb-raw"
version = "0.1.0"
description = "Scrapes ESPN college-football game JSON into raw + enriched final per-game files."
requires-python = ">=3.11"
dependencies = [
    "sportsdataverse>=0.0.51",
    "pandas>=2.0",
    "polars>=1.0",
    "pyarrow>=15.0",
    "tqdm>=4.66",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[tool.uv.sources]
sportsdataverse = { path = "../../sdv-py", editable = true }

[tool.pytest.ini_options]
markers = ["live: hits the live ESPN API (gated by CFB_LIVE_TESTS=1)"]
testpaths = ["tests"]
```

- [ ] **Step 3: Create `.gitignore`**

```
.venv/
__pycache__/
*.pyc
.pytest_cache/
/tmp/
*.tmp
.DS_Store
.vscode/
.claude/
```

- [ ] **Step 4: Create empty package markers**

Create `python/__init__.py` (empty), `tests/__init__.py` (empty), and `tests/fixtures/.gitkeep` (empty).

- [ ] **Step 5: Sync and verify the env builds**

Run: `uv sync`
Expected: resolves and installs `sportsdataverse` (editable from `../../sdv-py`) + deps; creates `uv.lock`.

Run: `uv run python -c "import sportsdataverse, pandas, polars, pyarrow, tqdm; print('ok')"`
Expected: `ok`

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore python/__init__.py tests/__init__.py tests/fixtures/.gitkeep
git commit -m "chore: scaffold uv project (pyproject, lock, gitignore)"
```

---

### Task B2: `_cfb_raw_utils.py` — atomic write + season helpers

**Files:**
- Create: `python/_cfb_raw_utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_utils.py`:
```python
import json
from pathlib import Path
import importlib.util

UTILS = Path(__file__).parents[1] / "python" / "_cfb_raw_utils.py"
spec = importlib.util.spec_from_file_location("_cfb_raw_utils", UTILS)
u = importlib.util.module_from_spec(spec)
spec.loader.exec_module(u)


def test_write_json_atomic_creates_dirs_and_no_tmp(tmp_path):
    target = tmp_path / "a" / "b" / "401.json"
    u.write_json_atomic({"id": 401, "x": [1, 2]}, target)
    assert target.exists()
    assert json.loads(target.read_text())["id"] == 401
    assert not list(target.parent.glob("*.tmp")), "temp file left behind"


def test_processing_version_format():
    v = u.PROCESSING_VERSION
    assert "+" in v and v.split("+")[1].isdigit()


def test_stamp_adds_identity():
    out = u.stamp({"k": 1}, game_id=401, season=2024, week=1)
    assert out["game_id"] == 401 and out["season"] == 2024 and out["week"] == 1
    assert out["k"] == 1


def test_stamp_list_wraps_with_meta():
    out = u.stamp([{"a": 1}], game_id=401, season=2024, week=1)
    assert out["game_id"] == 401
    assert out["data"] == [{"a": 1}]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_utils.py -v`
Expected: FAIL — module has no `write_json_atomic`/`PROCESSING_VERSION`/`stamp`.

- [ ] **Step 3: Implement these helpers**

Create `python/_cfb_raw_utils.py`:
```python
"""Shared helpers for cfbfastR-cfb-raw scrapers."""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import sportsdataverse

# Bump SCHEMA_REV whenever the final-JSON shape or enrichment inputs change in a way
# that should force a reprocess of already-built games.
SCHEMA_REV = 1
PROCESSING_VERSION = f"{sportsdataverse.__version__}+{SCHEMA_REV}"


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/_cfb_raw_utils.py tests/test_utils.py
git commit -m "feat(utils): atomic write, identity stamp, season + pool helpers, PROCESSING_VERSION"
```

---

### Task B3: `_cfb_raw_utils.py` — schedule-master IO + incremental filter

**Files:**
- Modify: `python/_cfb_raw_utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_utils.py`:
```python
import pandas as pd


def test_filter_undone_drops_existing(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402, 403], dir=str(final_dir), rescrape=False)
    assert out == [402, 403]


def test_filter_undone_rescrape_keeps_all(tmp_path):
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "401.json").write_text("{}")
    out = u.filter_undone([401, 402], dir=str(final_dir), rescrape=True)
    assert out == [401, 402]


def test_games_for_seasons_filters_range(tmp_path):
    master = tmp_path / "cfb_schedule_master.parquet"
    pd.DataFrame({"game_id": [1, 2, 3], "season": [2003, 2004, 2005]}).to_parquet(master)
    games = u.games_for_seasons(u.load_schedule_master(str(master)), 2004, 2005)
    assert sorted(games) == [2, 3]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_utils.py -k "filter_undone or games_for_seasons" -v`
Expected: FAIL — no such functions.

- [ ] **Step 3: Implement**

Append to `python/_cfb_raw_utils.py`:
```python
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
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_utils.py -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add python/_cfb_raw_utils.py tests/test_utils.py
git commit -m "feat(utils): schedule-master IO + incremental filter_undone"
```

---

# Phase C — scrapers

### Task C1: betting capture (`_capture_betting`)

**Files:**
- Create: `python/cfb_betting.py` (pure functions, importable + testable without network)
- Test: `tests/test_betting.py`

**Context:** `_capture_betting(raw, proc)` builds the normalized `betting` dict from the resolved odds on the processor (`proc.gameSpread/overUnder/homeFavorite/gameSpreadAvailable/odds_source`) plus the raw summary payloads. It must be null-safe and stable-shaped (spec §6.2).

- [ ] **Step 1: Write failing test**

Create `tests/test_betting.py`:
```python
import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "cfb_betting.py"
spec = importlib.util.spec_from_file_location("cfb_betting", P)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class _Proc:
    gameSpread = -7.5
    overUnder = 52.5
    homeFavorite = True
    gameSpreadAvailable = True
    odds_source = "summary_pickcenter"


def test_capture_betting_stable_shape():
    raw = {"pickcenter": [{"spread": -7.5}], "odds": [], "predictor": {},
           "againstTheSpread": []}
    out = b.capture_betting(raw, _Proc(), odds_full=[{"provider": "x"}], propbets=[])
    assert out["game_spread"] == -7.5
    assert out["over_under"] == 52.5
    assert out["home_favorite"] is True
    assert out["home_team_spread"] == -7.5  # favorite -> -abs(spread)
    assert out["game_spread_available"] is True
    assert out["odds_source"] == "summary_pickcenter"
    for k in ("pickcenter", "odds", "predictor", "against_the_spread",
              "odds_core_items", "odds_full", "propbets"):
        assert k in out


def test_capture_betting_handles_missing_keys():
    out = b.capture_betting({}, _Proc(), odds_full=None, propbets=None)
    assert out["pickcenter"] == [] and out["odds"] == [] and out["propbets"] == []
    assert out["predictor"] == {}


def test_home_team_spread_sign_when_away_favorite():
    class _AwayFav(_Proc):
        homeFavorite = False
    out = b.capture_betting({}, _AwayFav(), odds_full=None, propbets=None)
    assert out["home_team_spread"] == 7.5  # away favorite -> +abs(spread)
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_betting.py -v`
Expected: FAIL — no module `cfb_betting`.

- [ ] **Step 3: Implement**

Create `python/cfb_betting.py`:
```python
"""Normalize ESPN betting payloads into a stable, null-safe shape."""
from __future__ import annotations


def capture_betting(raw: dict, proc, *, odds_full=None, propbets=None) -> dict:
    spread = proc.gameSpread
    home_fav = bool(proc.homeFavorite)
    home_team_spread = -abs(spread) if home_fav else abs(spread)
    return {
        # resolved odds (EPA/WPA inputs) — persisted so reprocess injects them
        "game_spread": spread,
        "over_under": proc.overUnder,
        "home_favorite": home_fav,
        "home_team_spread": home_team_spread,
        "game_spread_available": bool(proc.gameSpreadAvailable),
        "odds_source": getattr(proc, "odds_source", None),
        # raw payloads for forensics + re-normalization
        "pickcenter": raw.get("pickcenter") or [],
        "odds": raw.get("odds") or [],
        "predictor": raw.get("predictor") or {},
        "against_the_spread": raw.get("againstTheSpread") or [],
        "odds_core_items": raw.get("odds_core_items") or [],
        "odds_full": odds_full or [],
        "propbets": propbets or [],
    }


def odds_override_from_betting(betting: dict) -> dict:
    """Reconstruct the CFBPlayProcess odds_override from a persisted betting dict."""
    return {
        "gameSpread": betting["game_spread"],
        "overUnder": betting["over_under"],
        "homeFavorite": betting["home_favorite"],
        "gameSpreadAvailable": betting["game_spread_available"],
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_betting.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/cfb_betting.py tests/test_betting.py
git commit -m "feat(betting): stable null-safe betting capture + odds_override reconstruction"
```

---

### Task C2: team_box_extra de-dup from summary

**Files:**
- Create: `python/cfb_team_box_extra.py`
- Test: `tests/test_team_box_extra.py`

**Context:** Spec §6.5 de-dup gate — derive per-team `record`/`linescores`/`statistics`/`leaders` from the summary `header`/`boxscore`/`leaders` when present, returning `None` only if the summary genuinely lacks them (caller then falls back to `event_competitor_*`).

- [ ] **Step 1: Write failing test**

Create `tests/test_team_box_extra.py`:
```python
import importlib.util
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "cfb_team_box_extra.py"
spec = importlib.util.spec_from_file_location("cfb_team_box_extra", P)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def _raw_with_header():
    return {
        "header": {"competitions": [{"competitors": [
            {"team": {"id": "333"}, "record": [{"summary": "1-0"}],
             "linescores": [{"value": 7}, {"value": 14}]},
            {"team": {"id": "99"}, "record": [{"summary": "0-1"}],
             "linescores": [{"value": 3}, {"value": 0}]},
        ]}]},
        "boxscore": {"teams": [
            {"team": {"id": "333"}, "statistics": [{"name": "totalYards", "displayValue": "400"}]},
            {"team": {"id": "99"}, "statistics": [{"name": "totalYards", "displayValue": "250"}]},
        ]},
        "leaders": [{"team": {"id": "333"}, "leaders": []},
                    {"team": {"id": "99"}, "leaders": []}],
    }


def test_team_box_extra_from_summary_present():
    out = m.team_box_extra_from_summary(_raw_with_header(), ["333", "99"])
    assert out is not None
    assert out["333"]["record"] == [{"summary": "1-0"}]
    assert out["333"]["linescores"] == [{"value": 7}, {"value": 14}]
    assert out["333"]["statistics"][0]["displayValue"] == "400"


def test_team_box_extra_returns_none_when_summary_lacks_it():
    out = m.team_box_extra_from_summary({"header": {"competitions": [{}]}}, ["333", "99"])
    assert out is None
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_team_box_extra.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement**

Create `python/cfb_team_box_extra.py`:
```python
"""Derive per-team box 'extra' fields from the summary, per the §6.5 de-dup gate.

Returns None when the summary lacks the data, signalling the caller to fall back
to the event_competitor_* endpoints.
"""
from __future__ import annotations


def _competitors(raw: dict) -> list:
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors")
    return comps or []


def _box_teams(raw: dict) -> list:
    return raw.get("boxscore", {}).get("teams") or []


def _leaders(raw: dict) -> list:
    return raw.get("leaders") or []


def team_box_extra_from_summary(raw: dict, team_ids):
    comps = _competitors(raw)
    if not comps:
        return None
    by_team = {}
    box_by_id = {str(t.get("team", {}).get("id")): t for t in _box_teams(raw)}
    lead_by_id = {str(l.get("team", {}).get("id")): l for l in _leaders(raw)}
    for c in comps:
        tid = str(c.get("team", {}).get("id"))
        by_team[tid] = {
            "record": c.get("record") or [],
            "linescores": c.get("linescores") or [],
            "statistics": (box_by_id.get(tid, {}).get("statistics") or []),
            "leaders": (lead_by_id.get(tid, {}).get("leaders") or []),
        }
    # require at least record/linescores for both requested teams to consider it complete
    for tid in (str(t) for t in team_ids):
        if tid not in by_team or not by_team[tid]["linescores"]:
            return None
    return by_team
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_team_box_extra.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/cfb_team_box_extra.py tests/test_team_box_extra.py
git commit -m "feat(team-box): derive per-team extras from summary (de-dup gate), None-fallback"
```

---

### Task C3: `scrape_cfb_schedules.py`

**Files:**
- Create: `python/scrape_cfb_schedules.py`
- Test: `tests/test_schedules.py`

**Context:** Pulls each season's schedule via `sportsdataverse.cfb.espn_cfb_schedule`, writes per-season parquet/rds-equivalent + appends to the master. (RDS write is R-only; here we write parquet + csv; the `-data` repo or a later task can mirror to `.rds`.)

- [ ] **Step 1: Write failing test (logic unit: master merge)**

Create `tests/test_schedules.py`:
```python
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
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_schedules.py -v`
Expected: FAIL — no `merge_master`.

- [ ] **Step 3: Implement**

Create `python/scrape_cfb_schedules.py`:
```python
"""Scrape CFB schedules per season -> cfb/schedules + master."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import sportsdataverse as sdv

from _cfb_raw_utils import get_logger, most_recent_cfb_season

SCHED_DIR = Path("cfb/schedules")
MASTER = "cfb/cfb_schedule_master.parquet"


def fetch_season(season: int) -> pd.DataFrame:
    # VERIFY LIVE (spec "check validity as you go"): confirm espn_cfb_schedule's season
    # argument name/shape. Per sdv-py cfb_schedule.py the signature is
    # espn_cfb_schedule(dates=..., week=..., season_type=..., groups=..., return_as_pandas=...).
    # `dates` accepts a season year for CFB; if a season pull instead needs a loop over
    # weeks or a YYYY/date form, adjust here. Must yield a `game_id` column.
    df = sdv.cfb.espn_cfb_schedule(dates=season, return_as_pandas=True)
    df["season"] = season
    return df


def merge_master(new: pd.DataFrame, master_path: str = MASTER) -> None:
    p = Path(master_path)
    if p.exists():
        old = pd.read_parquet(p)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    p.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(p, index=False)


def write_season(df: pd.DataFrame, season: int) -> None:
    (SCHED_DIR / "parquet").mkdir(parents=True, exist_ok=True)
    (SCHED_DIR / "csv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(SCHED_DIR / "parquet" / f"cfb_schedule_{season}.parquet", index=False)
    df.to_csv(SCHED_DIR / "csv" / f"cfb_schedule_{season}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    for season in range(args.start_year, end + 1):
        logger = get_logger("cfb_schedules", season)
        try:
            df = fetch_season(season)
            write_season(df, season)
            merge_master(df)
            logger.info("schedules %s: %d games", season, len(df))
        except Exception:
            logger.exception("schedules failed for %s", season)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_schedules.py -v`
Expected: PASS.

- [ ] **Step 5: (live, optional) smoke-test one season**

Run: `CFB_LIVE_TESTS=1 uv run python python/scrape_cfb_schedules.py -s 2024 -e 2024`
Expected: writes `cfb/schedules/parquet/cfb_schedule_2024.parquet` and master; log line with game count. (Skip if offline.)

- [ ] **Step 6: Commit**

```bash
git add python/scrape_cfb_schedules.py tests/test_schedules.py
git commit -m "feat(scrape): season schedules + dedupe-merged schedule master"
```

---

### Task C4: `scrape_cfb_json.py` — core per-game task

**Files:**
- Create: `python/scrape_cfb_json.py`
- Test: `tests/test_scrape_json.py`

**Context:** This is the heart (spec §8.2). The per-game task is `download_game`. We test it with all network + pipeline calls monkeypatched so it's offline and fast; the test asserts the write ordering and that every artifact lands.

- [ ] **Step 1: Write failing test**

Create `tests/test_scrape_json.py`:
```python
import importlib.util
import json
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "scrape_cfb_json.py"
spec = importlib.util.spec_from_file_location("scrape_cfb_json", P)
sj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sj)


class _FakeProc:
    def __init__(self, gameId=0, raw=False, **kw):
        self.gameId = gameId
        self.raw = raw
        self.gameSpread = -7.5
        self.overUnder = 52.5
        self.homeFavorite = True
        self.gameSpreadAvailable = True
        self.odds_source = "summary_pickcenter"

    def espn_cfb_pbp(self):
        if self.raw:
            return {"header": {"competitions": [{"competitors": [
                        {"team": {"id": "333"}, "homeAway": "home"},
                        {"team": {"id": "99"}, "homeAway": "away"}]}]},
                    "injuries": [{"x": 1}], "gameNotes": [{"n": 1}], "pickcenter": []}
        return {}

    def run_processing_pipeline(self):
        return {"plays": [{"id": 1}], "advBoxScore": {}, "drives": {},
                "boxScore": {}, "header": {}, "season": 2024, "week": 1}


def test_download_game_writes_all_artifacts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sj, "CFBPlayProcess", _FakeProc)
    monkeypatch.setattr(sj, "_participants", lambda gid: [{"play_id": 1, "athlete_id": 5}])
    monkeypatch.setattr(sj, "_rosters", lambda gid: [{"athlete_id": 5}])
    monkeypatch.setattr(sj, "_officials", lambda gid: [{"name": "Ref"}])
    monkeypatch.setattr(sj, "_power_index", lambda gid: {"fpi": 1})
    monkeypatch.setattr(sj, "_odds_full", lambda gid: [{"provider": "x"}])
    monkeypatch.setattr(sj, "_propbets", lambda gid: [])

    sj.download_game(401, season=2024, rescrape=True)

    base = Path("cfb")
    assert (base / "json" / "raw" / "401.json").exists()
    final = json.loads((base / "json" / "final" / "401.json").read_text())
    assert final["id"] == 401 and final["season"] == 2024
    assert final["processing_version"]
    assert final["injuries"] == [{"x": 1}]
    assert final["betting"]["game_spread"] == -7.5
    assert final["officials"] == [{"name": "Ref"}]
    for ds in ("rosters", "play_participants", "betting", "officials",
               "power_index", "team_box_extra"):
        assert (base / ds / "json" / "2024" / "401.json").exists(), ds
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_scrape_json.py -v`
Expected: FAIL — no module / no `download_game`.

- [ ] **Step 3: Implement**

Create `python/scrape_cfb_json.py`:
```python
"""Core CFB scraper: per-game raw + enriched final + standalone aux/extras."""
from __future__ import annotations

import argparse

import sportsdataverse as sdv
from sportsdataverse.cfb import CFBPlayProcess

from _cfb_raw_utils import (PROCESSING_VERSION, _safe, filter_undone,
                            games_for_seasons, get_logger, load_schedule_master,
                            most_recent_cfb_season, run_pool, stamp, write_json_atomic)
from cfb_betting import capture_betting
from cfb_team_box_extra import team_box_extra_from_summary

# --- thin sdv-py adapters (monkeypatch points in tests) ---
def _participants(gid):
    return sdv.cfb.espn_cfb_play_participants(game_id=gid, return_as_pandas=True).to_dict("records")

def _rosters(gid):
    return sdv.cfb.espn_cfb_game_rosters(game_id=gid, return_as_pandas=True).to_dict("records")

def _officials(gid):
    return sdv.cfb.espn_cfb_event_officials(event_id=gid)

def _power_index(gid):
    return sdv.cfb.espn_cfb_event_powerindex(event_id=gid)

def _odds_full(gid):
    return sdv.cfb.espn_cfb_event_odds(event_id=gid)

def _propbets(gid):
    return sdv.cfb.espn_cfb_event_propbets(event_id=gid)


def _home_away_ids(raw: dict):
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors") or []
    home = away = None
    for c in comps:
        tid = c.get("team", {}).get("id")
        if c.get("homeAway") == "home":
            home = tid
        elif c.get("homeAway") == "away":
            away = tid
    return home, away


def download_game(game_id: int, season: int, rescrape: bool, logger=None):
    logger = logger or get_logger("cfb_json", season)
    try:
        # 1. bank RAW first
        raw = CFBPlayProcess(gameId=game_id, raw=True).espn_cfb_pbp()
        write_json_atomic(raw, f"cfb/json/raw/{game_id}.json")

        # 2. enrich
        proc = CFBPlayProcess(gameId=game_id)
        proc.espn_cfb_pbp()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)

        # 3. aux (endpoint-backed)
        participants = _safe(_participants, game_id, logger=logger, default=[])
        rosters = _safe(_rosters, game_id, logger=logger, default=[])
        betting = capture_betting(
            raw, proc,
            odds_full=_safe(_odds_full, game_id, logger=logger, default=[]),
            propbets=_safe(_propbets, game_id, logger=logger, default=[]),
        )
        officials = _safe(_officials, game_id, logger=logger, default=[])
        power_index = _safe(_power_index, game_id, logger=logger, default={})

        # 4. team_box_extra: prefer summary (de-dup gate); fall back only if missing
        team_extra = team_box_extra_from_summary(raw, [home_id, away_id]) or {}

        injuries = raw.get("injuries") or []
        game_notes = raw.get("gameNotes") or []
        week = result.get("week")

        # 5. standalone datasets
        standalone = {
            "rosters": rosters, "play_participants": participants, "betting": betting,
            "officials": officials, "power_index": power_index, "team_box_extra": team_extra,
        }
        for name, obj in standalone.items():
            write_json_atomic(stamp(obj, game_id=game_id, season=season, week=week),
                              f"cfb/{name}/json/{season}/{game_id}.json")

        # 6. embed + write FINAL last
        result.update(
            id=game_id, season=season, week=week,
            season_type=result.get("season_type") or raw.get("header", {}).get("season", {}).get("type"),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            play_participants=participants, game_rosters=rosters, betting=betting,
            officials=officials, power_index=power_index, team_box_extra=team_extra,
            injuries=injuries, game_notes=game_notes,
            homeTeamId=home_id, awayTeamId=away_id,
        )
        write_json_atomic(result, f"cfb/json/final/{game_id}.json")
        return "ok"
    except Exception:
        logger.exception("download_game failed: %s", game_id)
        return "error"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="false")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    master = load_schedule_master()
    for season in range(args.start_year, end + 1):
        logger = get_logger("cfb_json", season)
        games = filter_undone(games_for_seasons(master, season, season), rescrape=rescrape)
        logger.info("season %s: %d games to scrape (rescrape=%s)", season, len(games), rescrape)
        run_pool(lambda g, _s=season: download_game(g, _s, rescrape),
                 games, kind="process", desc=f"cfb {season}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_scrape_json.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/scrape_cfb_json.py tests/test_scrape_json.py
git commit -m "feat(scrape): core per-game raw+final+aux+extras task (raw-first, final-last)"
```

---

### Task C5: single-dataset refresher scripts

**Files:**
- Create: `python/scrape_cfb_rosters.py`, `python/scrape_cfb_participants.py`, `python/scrape_cfb_betting.py`, `python/scrape_cfb_officials.py`, `python/scrape_cfb_power_index.py`
- Test: `tests/test_refreshers.py`

**Context:** Each refresher reuses the adapter + utils to (re)write a single standalone dataset for a season range. They share a tiny runner. Tested via one parametrized smoke test that the `main` is importable and the per-game writer lands a file.

- [ ] **Step 1: Write failing test**

Create `tests/test_refreshers.py`:
```python
import importlib.util
import json
from pathlib import Path


def _load(modname):
    P = Path(__file__).parents[1] / "python" / f"{modname}.py"
    spec = importlib.util.spec_from_file_location(modname, P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_officials_refresher_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = _load("scrape_cfb_officials")
    monkeypatch.setattr(m, "_fetch", lambda gid: [{"name": "Ref"}])
    m.write_one(401, 2024)
    out = json.loads((tmp_path / "cfb/officials/json/2024/401.json").read_text())
    assert out["game_id"] == 401 and out["data"] == [{"name": "Ref"}]
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_refreshers.py -v`
Expected: FAIL — no module `scrape_cfb_officials`.

- [ ] **Step 3: Implement (officials shown; the other four are identical with the noted swaps)**

Create `python/scrape_cfb_officials.py`:
```python
"""Refresh the officials dataset for a season range (standalone, outside daily loop)."""
from __future__ import annotations

import argparse

import sportsdataverse as sdv

from _cfb_raw_utils import (filter_undone, games_for_seasons, get_logger,
                            load_schedule_master, most_recent_cfb_season, run_pool,
                            stamp, write_json_atomic)

DATASET = "officials"


def _fetch(gid):
    return sdv.cfb.espn_cfb_event_officials(event_id=gid)


def write_one(game_id: int, season: int) -> None:
    write_json_atomic(stamp(_fetch(game_id), game_id=game_id, season=season),
                      f"cfb/{DATASET}/json/{season}/{game_id}.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=most_recent_cfb_season())
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("-r", "--rescrape", type=str, default="true")
    args = ap.parse_args()
    end = args.end_year or args.start_year
    rescrape = str(args.rescrape).lower() in ("1", "true", "yes")
    master = load_schedule_master()
    for season in range(args.start_year, end + 1):
        logger = get_logger(f"cfb_{DATASET}", season)
        games = filter_undone(games_for_seasons(master, season, season),
                              dir=f"cfb/{DATASET}/json/{season}", rescrape=rescrape)
        logger.info("%s %s: %d games", DATASET, season, len(games))
        run_pool(lambda g, _s=season: write_one(g, _s), games, kind="thread",
                 desc=f"{DATASET} {season}")


if __name__ == "__main__":
    main()
```

Create the other four by copying this file and changing only `DATASET` + `_fetch`:
- `scrape_cfb_power_index.py` → `DATASET = "power_index"`, `_fetch = lambda gid: sdv.cfb.espn_cfb_event_powerindex(event_id=gid)`
- `scrape_cfb_rosters.py` → `DATASET = "rosters"`, `_fetch = lambda gid: sdv.cfb.espn_cfb_game_rosters(game_id=gid, return_as_pandas=True).to_dict("records")`
- `scrape_cfb_participants.py` → `DATASET = "play_participants"`, `_fetch = lambda gid: sdv.cfb.espn_cfb_play_participants(game_id=gid, return_as_pandas=True).to_dict("records")`
- `scrape_cfb_betting.py` → `DATASET = "betting"`, `_fetch = lambda gid: {"odds_full": sdv.cfb.espn_cfb_event_odds(event_id=gid), "propbets": sdv.cfb.espn_cfb_event_propbets(event_id=gid)}` (note: the full normalized betting comes from the core task; this refresher only refreshes the odds_full/propbets payloads).

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_refreshers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/scrape_cfb_officials.py python/scrape_cfb_power_index.py python/scrape_cfb_rosters.py python/scrape_cfb_participants.py python/scrape_cfb_betting.py tests/test_refreshers.py
git commit -m "feat(scrape): single-dataset refreshers (officials/power_index/rosters/participants/betting)"
```

---

# Phase D — reprocess from disk

### Task D1: `reprocess_cfb_json.py`

**Files:**
- Create: `python/reprocess_cfb_json.py`
- Test: `tests/test_reprocess.py`

**Context:** Spec §7. Rebuilds `final` from on-disk `raw` + standalone aux, injecting odds (no network), gated by `processing_version`.

- [ ] **Step 1: Write failing test**

Create `tests/test_reprocess.py`:
```python
import importlib.util
import json
from pathlib import Path

P = Path(__file__).parents[1] / "python" / "reprocess_cfb_json.py"
spec = importlib.util.spec_from_file_location("reprocess_cfb_json", P)
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)


class _FakeProc:
    def __init__(self, gameId=0, path_to_json=None, odds_override=None, **kw):
        self.gameId = gameId
        self.odds_override = odds_override
        self.odds_source = "injected"
        self.gameSpread = odds_override["gameSpread"] if odds_override else -7.5
        self.overUnder = odds_override["overUnder"] if odds_override else 52.5
        self.homeFavorite = True
        self.gameSpreadAvailable = True

    def cfb_pbp_disk(self):
        return {}

    def run_processing_pipeline(self):
        return {"plays": [{"id": 1}], "drives": {}, "advBoxScore": {}, "header": {}, "week": 1}


def _seed(base: Path):
    (base / "cfb/json/raw").mkdir(parents=True)
    (base / "cfb/json/raw/401.json").write_text(json.dumps(
        {"header": {"competitions": [{"competitors": [
            {"team": {"id": "333"}, "homeAway": "home"},
            {"team": {"id": "99"}, "homeAway": "away"}]}]}, "injuries": []}))
    for ds, payload in {
        "betting": {"game_spread": -10.5, "over_under": 60.0, "home_favorite": True,
                    "game_spread_available": True, "game_id": 401, "season": 2024},
        "rosters": {"game_id": 401, "season": 2024, "data": [{"athlete_id": 5}]},
        "play_participants": {"game_id": 401, "season": 2024, "data": [{"play_id": 1}]},
        "officials": {"game_id": 401, "season": 2024, "data": [{"name": "Ref"}]},
        "power_index": {"game_id": 401, "season": 2024, "fpi": 1},
        "team_box_extra": {"game_id": 401, "season": 2024},
    }.items():
        d = base / f"cfb/{ds}/json/2024"
        d.mkdir(parents=True)
        (d / "401.json").write_text(json.dumps(payload))


def test_reprocess_offline_injects_odds(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    monkeypatch.setattr(rp, "CFBPlayProcess", _FakeProc)
    rp.reprocess_game(401, season=2024, refresh_aux=False, force=True)
    final = json.loads((tmp_path / "cfb/json/final/401.json").read_text())
    assert final["betting"]["game_spread"] == -10.5     # injected from disk betting
    assert final["game_rosters"] == [{"athlete_id": 5}]
    assert final["officials"] == [{"name": "Ref"}]
    assert final["processing_version"]


def test_version_gate_skips_current(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _seed(tmp_path)
    monkeypatch.setattr(rp, "CFBPlayProcess", _FakeProc)
    (tmp_path / "cfb/json/final").mkdir(parents=True)
    (tmp_path / "cfb/json/final/401.json").write_text(json.dumps(
        {"processing_version": rp.PROCESSING_VERSION}))
    assert rp.reprocess_game(401, season=2024, refresh_aux=False, force=False) == "skipped"
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_reprocess.py -v`
Expected: FAIL — no module.

- [ ] **Step 3: Implement**

Create `python/reprocess_cfb_json.py`:
```python
"""Rebuild final/{id}.json from on-disk raw + standalone aux, fully offline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from sportsdataverse.cfb import CFBPlayProcess

from _cfb_raw_utils import (PROCESSING_VERSION, get_logger, run_pool, stamp,
                            write_json_atomic)
from cfb_betting import odds_override_from_betting

RAW_DIR = Path("cfb/json/raw")
FINAL_DIR = Path("cfb/json/final")


def _read(path: Path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def _aux(ds: str, season: int, game_id: int):
    return _read(Path(f"cfb/{ds}/json/{season}/{game_id}.json"), {})


def _aux_list(ds: str, season: int, game_id: int):
    obj = _aux(ds, season, game_id)
    return obj.get("data", obj) if isinstance(obj, dict) else obj


def _final_is_current(game_id: int) -> bool:
    f = FINAL_DIR / f"{game_id}.json"
    if not f.exists():
        return False
    return _read(f, {}).get("processing_version") == PROCESSING_VERSION


def _home_away_ids(raw: dict):
    comps = (raw.get("header", {}).get("competitions") or [{}])[0].get("competitors") or []
    home = away = None
    for c in comps:
        if c.get("homeAway") == "home":
            home = c.get("team", {}).get("id")
        elif c.get("homeAway") == "away":
            away = c.get("team", {}).get("id")
    return home, away


def reprocess_game(game_id: int, season: int, refresh_aux: bool, force: bool, logger=None):
    logger = logger or get_logger("cfb_reprocess", season)
    try:
        if not force and _final_is_current(game_id):
            return "skipped"
        raw = _read(RAW_DIR / f"{game_id}.json", None)
        if raw is None:
            logger.warning("no raw on disk for %s", game_id)
            return "missing_raw"

        betting = _aux("betting", season, game_id)
        override = odds_override_from_betting(betting) if betting else None

        proc = CFBPlayProcess(gameId=game_id, path_to_json=str(RAW_DIR),
                              odds_override=override)
        proc.cfb_pbp_disk()
        result = proc.run_processing_pipeline()

        home_id, away_id = _home_away_ids(raw)
        result.update(
            id=game_id, season=season, week=result.get("week"),
            processing_version=PROCESSING_VERSION,
            count=len(result.get("plays") or []),
            betting=betting,
            game_rosters=_aux_list("rosters", season, game_id),
            play_participants=_aux_list("play_participants", season, game_id),
            officials=_aux_list("officials", season, game_id),
            power_index=_aux("power_index", season, game_id),
            team_box_extra=_aux("team_box_extra", season, game_id),
            injuries=raw.get("injuries") or [],
            game_notes=raw.get("gameNotes") or [],
            homeTeamId=home_id, awayTeamId=away_id,
        )
        write_json_atomic(result, str(FINAL_DIR / f"{game_id}.json"))
        return "rebuilt"
    except Exception:
        logger.exception("reprocess failed: %s", game_id)
        return "error"


def _season_of(game_id: int) -> int | None:
    import pandas as pd
    m = pd.read_parquet("cfb/cfb_schedule_master.parquet")
    row = m[m["game_id"] == game_id]
    return int(row["season"].iloc[0]) if len(row) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--start_year", type=int, default=None)
    ap.add_argument("-e", "--end_year", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--refresh-aux", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    master = pd.read_parquet("cfb/cfb_schedule_master.parquet")
    if not args.all:
        start = args.start_year
        end = args.end_year or start
        master = master[(master["season"] >= start) & (master["season"] <= end)]
    pairs = list(master[["game_id", "season"]].itertuples(index=False, name=None))
    # only games with raw on disk
    pairs = [(int(g), int(s)) for g, s in pairs if (RAW_DIR / f"{g}.json").exists()]
    run_pool(lambda gs: reprocess_game(gs[0], gs[1], args.refresh_aux, args.force),
             pairs, kind="process", desc="reprocess")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_reprocess.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add python/reprocess_cfb_json.py tests/test_reprocess.py
git commit -m "feat(reprocess): offline final rebuild from disk raw+aux, odds injection, version gate"
```

---

# Phase E — orchestration + CI + docs

### Task E1: bash orchestrators

**Files:**
- Create: `scripts/daily_cfb_scraper.sh`, `scripts/backfill_cfb.sh`, `scripts/reprocess_cfb.sh`

- [ ] **Step 1: Create `scripts/daily_cfb_scraper.sh`**

```bash
#!/bin/bash
# Scrape raw CFB datasets per season (schedules -> json[+all aux/extras]).
set -uo pipefail

while getopts s:e:r: flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    r) RESCRAPE=${OPTARG};;
  esac
done
RESCRAPE=${RESCRAPE:-false}
mkdir -p logs

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_raw_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    uv run python python/scrape_cfb_schedules.py -s "$i" -e "$i" -r "$RESCRAPE"
    uv run python python/scrape_cfb_json.py      -s "$i" -e "$i" -r "$RESCRAPE"
    git pull >/dev/null
    git add cfb/* >/dev/null 2>&1 || true
    git commit -m "CFB Raw Update (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  git pull --rebase >/dev/null || true
  git add "logs/cfb_raw_logfile_${i}.log"
  git commit -m "CFB Raw log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
done
```

- [ ] **Step 2: Create `scripts/backfill_cfb.sh`**

```bash
#!/bin/bash
# Full historical backfill (default 2004 -> most-recent season).
set -uo pipefail
START_YEAR=${1:-2004}
END_YEAR=${2:-$(uv run python -c "import sys; sys.path.insert(0,'python'); from _cfb_raw_utils import most_recent_cfb_season as m; print(m())")}
bash scripts/daily_cfb_scraper.sh -s "$START_YEAR" -e "$END_YEAR" -r false
```

- [ ] **Step 3: Create `scripts/reprocess_cfb.sh`**

```bash
#!/bin/bash
# Rebuild final/ from on-disk raw for a season range (no re-scrape).
set -uo pipefail
while getopts s:e:f flag; do
  case "${flag}" in
    s) START_YEAR=${OPTARG};;
    e) END_YEAR=${OPTARG};;
    f) FORCE="--force";;
  esac
done
FORCE=${FORCE:-}
mkdir -p logs

for i in $(seq "${START_YEAR}" "${END_YEAR}"); do
  TMPLOG=$(mktemp "/tmp/cfb_reprocess_${i}.XXXXXX.log")
  {
    git pull >/dev/null
    git config --local user.email "action@github.com"
    git config --local user.name "Github Action"
    uv run python python/reprocess_cfb_json.py -s "$i" -e "$i" $FORCE
    git pull >/dev/null
    git add cfb/json/final/* >/dev/null 2>&1 || true
    git commit -m "CFB Reprocess Update (Start: $i End: $i)" || echo "No changes to commit"
    git pull >/dev/null
    git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_reprocess_logfile_${i}.log"
  git pull --rebase >/dev/null || true
  git add "logs/cfb_reprocess_logfile_${i}.log"
  git commit -m "CFB Reprocess log update (Start: $i End: $i)" >/dev/null || true
  git push >/dev/null
  rm -f "$TMPLOG"
done
```

- [ ] **Step 4: Make executable + sanity-check syntax**

Run: `chmod +x scripts/*.sh && bash -n scripts/daily_cfb_scraper.sh && bash -n scripts/backfill_cfb.sh && bash -n scripts/reprocess_cfb.sh && echo OK`
Expected: `OK` (no syntax errors).

- [ ] **Step 5: Commit**

```bash
git add scripts/
git commit -m "feat(scripts): daily scraper, backfill, and reprocess orchestrators"
```

---

### Task E2: GitHub Actions workflows

**Files:**
- Create: `.github/workflows/scrape_cfb_raw.yml`, `.github/workflows/cfbfastR_cfb_data_trigger.yml`

- [ ] **Step 1: Create `.github/workflows/scrape_cfb_raw.yml`**

```yaml
name: Scrape CFB Raw Data

on:
  schedule:
    - cron: '0 9 * 8 *'        # August (preseason)
    - cron: '0 9 * 9-11 *'     # Sept-Nov (regular season)
    - cron: '0 9 1-20 12 *'    # early-mid December (championships/bowls open)
    - cron: '0 9 1-20 1 *'     # early January (bowls + CFP)
  workflow_dispatch:
    inputs:
      start_year:
        description: 'Start year'
        required: false
        type: string
      end_year:
        description: 'End year'
        required: false
        type: string
      rescrape:
        description: 'Rescrape existing games'
        required: false
        type: boolean
        default: false

jobs:
  scrape_cfb_raw:
    runs-on: ubuntu-latest
    env:
      GITHUB_PAT: ${{ secrets.GITHUB_TOKEN }}
      START_YEAR: ${{ inputs.start_year }}
      END_YEAR: ${{ inputs.end_year }}
      RESCRAPE: ${{ inputs.rescrape }}
    steps:
      - uses: actions/checkout@v5

      - name: Set up uv
        uses: astral-sh/setup-uv@v6
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: uv sync --frozen

      - name: Default year inputs if empty
        if: ${{ env.START_YEAR == '' }}
        run: |
          YR=$(uv run python -c "import sys; sys.path.insert(0,'python'); from _cfb_raw_utils import most_recent_cfb_season as m; print(m())")
          echo "START_YEAR=$YR" >> $GITHUB_ENV
          echo "END_YEAR=$YR" >> $GITHUB_ENV

      - name: Default rescrape if empty
        if: ${{ env.RESCRAPE == '' }}
        run: echo "RESCRAPE=false" >> $GITHUB_ENV

      - name: Scrape CFB Raw ${{ env.START_YEAR }}-${{ env.END_YEAR }}
        run: bash scripts/daily_cfb_scraper.sh -s ${{ env.START_YEAR }} -e ${{ env.END_YEAR }} -r ${{ env.RESCRAPE }}
```

> **Note (spec §9):** CI uses `uv sync --frozen` against `uv.lock`. Before first CI run, regenerate the lock without the local path source (pin the released `sportsdataverse`) OR ensure the Phase A PR is merged + published; otherwise `[tool.uv.sources]`'s `../../sdv-py` path won't exist on the runner. Track this in the README "CI prerequisites" note.

- [ ] **Step 2: Create `.github/workflows/cfbfastR_cfb_data_trigger.yml`**

```yaml
name: cfbfastR CFB Data trigger

on: [push, workflow_dispatch]

jobs:
  cfbfastR_cfb_data_trigger:
    runs-on: ubuntu-latest
    steps:
      - name: Trigger cfbfastR CFB Data
        uses: peter-evans/repository-dispatch@v4
        with:
          token: ${{ secrets.SDV_GH_TOKEN }}
          repository: sportsdataverse/cfbfastR-cfb-data
          event-type: daily_cfb_data
          client-payload: |-
            {
              "ref": "refs/heads/main",
              "event_name": "daily_cfb_data",
              "commit_message": ${{ toJSON(github.event.head_commit.message) }}
            }
```

- [ ] **Step 3: Lint YAML**

Run: `uv run python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('yaml ok')"`
Expected: `yaml ok` (add `pyyaml` via `uv add --dev pyyaml` first if missing).

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/
git commit -m "ci: CFB raw scrape workflow (uv) + repository_dispatch trigger to -data"
```

---

### Task E3: README + CLAUDE.md

**Files:**
- Create: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Create `README.md`**

````markdown
# cfbfastR-cfb-raw

Raw + enriched college-football game JSON, scraped from ESPN via `sportsdataverse`.

## What it produces

Per game:
- `cfb/json/raw/{game_id}.json` — ESPN summary (curated allowlist incl. injuries + gameNotes).
- `cfb/json/final/{game_id}.json` — fully enriched (EPA/WPA/QBR plays, advBoxScore) +
  play participants + game rosters + normalized betting + officials + power index (FPI) +
  per-team box extras. Self-describing (`id`/`season`/`week` echoed).

Standalone season-partitioned datasets: `rosters`, `play_participants`, `betting`,
`officials`, `power_index`, `team_box_extra`, plus the `schedules` + `cfb_schedule_master`.

## Usage

```bash
uv sync
# one season, incremental
uv run python python/scrape_cfb_schedules.py -s 2024 -e 2024
uv run python python/scrape_cfb_json.py      -s 2024 -e 2024
# full backfill
bash scripts/backfill_cfb.sh 2004
# rebuild final from raw on disk after a pipeline change (offline)
uv run python python/reprocess_cfb_json.py -s 2024 -e 2024 --force
```

## CI prerequisites

`uv.lock` pins `sportsdataverse`. For local dev `[tool.uv.sources]` points at `../../sdv-py`
(editable). Before CI can `uv sync --frozen`, the Phase A sdv-py changes must be merged +
published (or the lock regenerated to the released version) so the path source isn't needed
on the runner.

## Automation

- `scrape_cfb_raw.yml` — cron over the CFB calendar (Aug→Jan) + manual dispatch.
- On push, `cfbfastR_cfb_data_trigger.yml` fires `repository_dispatch` to
  `sportsdataverse/cfbfastR-cfb-data`, which rectangularizes `final/` into release parquet.

## Reprocess vs. recreate

- **Reprocess** (here, Python): `raw → final`, offline, gated by `processing_version`. Bump
  `SCHEMA_REV` in `python/_cfb_raw_utils.py` to force stale games to rebuild.
- **Recreate** (the `-data` repo, R): `final → parquet`, cheap reshape.

See `docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`.
````

- [ ] **Step 2: Create `CLAUDE.md`**

```markdown
# CLAUDE.md — cfbfastR-cfb-raw

Python/uv scraper for ESPN college-football game JSON. Sibling of `cfbfastR-cfb-data` (R).

## Commands
- `uv sync` — install (editable sdv-py from ../../sdv-py for dev).
- `uv run pytest` — offline test suite. Live tests: `CFB_LIVE_TESTS=1 uv run pytest -m live`.
- `uv run python python/scrape_cfb_json.py -s YYYY -e YYYY -r false` — scrape.
- `uv run python python/reprocess_cfb_json.py -s YYYY -e YYYY --force` — offline rebuild.

## Conventions
- SDK boundary: all ESPN access via `sportsdataverse.cfb` (`CFBPlayProcess`, `espn_cfb_*`).
  Bug fixes go upstream to sdv-py, not here.
- Per-game task order: **raw first**, **final last** (final's existence = completion marker).
- Every aux/extra is persisted standalone AND embedded in final (offline-reprocess source).
- `write_json_atomic` for every write. `_safe()`-wrap every extra endpoint.
- Commit message format is load-bearing: `"CFB Raw Update (Start: YYYY End: YYYY)"` /
  `"CFB Reprocess Update (Start: YYYY End: YYYY)"` — the `-data` trigger greps the years.
- Bump `SCHEMA_REV` when the final shape / enrichment inputs change.
- Never add AI co-author trailers to commits.

## Spec
`docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`
```

- [ ] **Step 3: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: README + CLAUDE.md for cfbfastR-cfb-raw"
```

---

### Task E4: Full suite green + endpoint validation harness (spec §12.8)

**Files:**
- Create: `tests/test_live_endpoints.py` (gated, `-m live`)

**Context:** This encodes the spec §12.8 validity + de-dup gate as a runnable, gated check for one recent + one old game. It does NOT run in CI by default; it's the tool to decide which `event_*` extras survive.

- [ ] **Step 1: Write the gated live test**

Create `tests/test_live_endpoints.py`:
```python
import os
import pytest
import sportsdataverse as sdv

pytestmark = pytest.mark.live

LIVE = os.environ.get("CFB_LIVE_TESTS") == "1"
RECENT = 401628455  # 2024 game
OLD = 242410193     # ~2014 game


@pytest.mark.skipif(not LIVE, reason="set CFB_LIVE_TESTS=1")
@pytest.mark.parametrize("gid", [RECENT, OLD])
@pytest.mark.parametrize("fn_name", [
    "espn_cfb_event_officials",
    "espn_cfb_event_powerindex",
    "espn_cfb_event_odds",
    "espn_cfb_event_propbets",
])
def test_extra_endpoint_validity(gid, fn_name):
    fn = getattr(sdv.cfb, fn_name)
    try:
        out = fn(event_id=gid)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{fn_name}({gid}) raised {e!r} — record validity verdict in spec §12.8")
    # not asserting non-empty (old seasons may be sparse) — print for the de-dup decision
    print(f"\n{fn_name}({gid}) -> type={type(out).__name__} "
          f"len={len(out) if hasattr(out, '__len__') else 'n/a'}")
```

- [ ] **Step 2: Run the offline suite (must be green)**

Run: `uv run pytest -m "not live" -q`
Expected: all tests pass.

- [ ] **Step 3: Run the gated live validity probe (manual, when online)**

Run: `CFB_LIVE_TESTS=1 uv run pytest tests/test_live_endpoints.py -m live -s -v`
Expected: prints type/len for each extra × {recent, old}. **Record the validity + de-dup verdict for each endpoint in spec §12.8** and prune any redundant/invalid extra from `scrape_cfb_json.py` (and its standalone dataset) before backfill.

- [ ] **Step 4: Commit**

```bash
git add tests/test_live_endpoints.py
git commit -m "test: gated live validity/de-dup probe for non-live extra endpoints (§12.8)"
```

---

## Self-review notes (author)

- **Spec coverage:** §6.1 (A1), §6.2/§6.5 embedded keys (C4), §6.3 standalone (C4/C5), §6.4 is `-data` (Plan 2), §6.5 de-dup (C2 + E4), §7 reprocess (D1), §8 utils/task/bash (B2/B3/C4/E1), §9 uv (B1), §10 CI (E2), §11 logging (B2), §12.2 odds (A2/A3), §12.7 allowlist (A1), §12.8 validity/de-dup (E4). §12.1 advBox is a `-data`/sdv-py concern → Plan 2. §12.5 backfill volume + §12.9 pacing: `run_pool` width + `_safe` + retry; **add a `RETRY`/politeness note to C4 if live probing shows rate limits.**
- **Deferred to Plan 2 (`cfbfastR-cfb-data`, R):** all `espn_cfb_0N_*.R` rectangularization, releases, the `-data` workflow, advBox expansion (§12.1).
- **Open dependency:** Phase A PR must merge + publish before CI `uv sync --frozen` works without the local path source (noted in E2/README).
- **`.rds` schedule outputs:** raw repo writes parquet+csv; `.rds` mirrors are produced in the R `-data` repo (Plan 2), avoiding an R dependency here.
```
