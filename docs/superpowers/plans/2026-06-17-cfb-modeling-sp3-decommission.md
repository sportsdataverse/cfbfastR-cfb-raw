# CFB Modeling SP3 — Decommission Modeling from cfb-raw Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the model-training code (EP/WP/QBR, CPOE, fourth-down, RB-eval) from `cfbfastR-cfb-raw` — now living in `cfbfastR-cfb-data` — leaving cfb-raw scraping-only and green, after relocating the one staying-scraper test.

**Architecture:** Three ordered tasks: (1) relocate `test_qbr_scrape` + its fixture OUT of the modeling test tree (it tests the staying `scrape_cfb_qbr`); (2) delete the four modeling packages + their tests/fixtures; (3) drop the modeling-only deps from `pyproject.toml`, re-lock, and verify the scraping suite is green. Verified clean boundary: zero scraper→modeling imports, zero ML-dep usage by scrapers.

**Tech Stack:** Python ≥3.11, uv, pytest. cfb-raw stays on `sportsdataverse`/`pandas`/`polars`/`pyarrow`/`requests`/`tqdm`.

## Global Constraints

- **Commit hygiene (critical):** the working tree has ~50 untracked scraped-data files + logs. Stage ONLY the SP3 paths each task names (the `git rm`/`git mv` targets, `pyproject.toml`, `uv.lock`, the relocated test). **NEVER `git add -A` / `git add .`** — use explicit paths / `git add -u` scoped to named paths.
- Do NOT modify any staying scraper module's logic, the `[tool.pytest.ini_options]`, or `cfbfastR-cfb-data`.
- `xgboost` is dropped as a *direct* dep but remains installed transitively via `sportsdataverse` (the scraper's `CFBPlayProcess` enrichment uses it) — no runtime breakage.
- Run pytest from `python/` is NOT how this repo runs — cfb-raw runs `uv run pytest` from the **repo root** (`testpaths=["tests"]`, `pythonpath=["python"]`). Use repo-root `uv run pytest`.
- Conventional Commits. **NO AI co-author trailers.**

## File Structure

**Relocated:** `tests/model_training/test_qbr_scrape.py` → `tests/test_qbr_scrape.py`; `tests/fixtures/model_training/qbr_endpoint_sample.json` → `tests/fixtures/qbr_endpoint_sample.json`.
**Deleted:** `python/{model_training,cpoe,pregame_wp,rb_eval}/`; `tests/{model_training,cpoe,pregame_wp,rb_eval}/`; `tests/fixtures/{model_training,pregame_wp,rb_eval}/`.
**Modified:** `pyproject.toml` (drop deps + groups), `uv.lock` (re-locked).

---

### Task 1: Relocate `test_qbr_scrape` + its fixture (BEFORE any deletion)

**Files:**
- Move: `tests/model_training/test_qbr_scrape.py` → `tests/test_qbr_scrape.py`
- Move: `tests/fixtures/model_training/qbr_endpoint_sample.json` → `tests/fixtures/qbr_endpoint_sample.json`
- Edit: the relocated `tests/test_qbr_scrape.py` (two path references)

**Interfaces:**
- Consumes: the staying `python/scrape_cfb_qbr.py` (`parse_qbr_payload`). Produces: a root-level scraping test that survives the modeling-tree deletion.

- [ ] **Step 1: Read the current test** to capture the EXACT lines to change

Run: `cat tests/model_training/test_qbr_scrape.py`
Note the two path constructs: the `sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))` line and the fixture path `pathlib.Path(__file__).parent.parent / "fixtures" / "model_training" / "qbr_endpoint_sample.json"` (exact text may differ slightly — use what you see).

- [ ] **Step 2: `git mv` the test + fixture**

```bash
git mv tests/model_training/test_qbr_scrape.py tests/test_qbr_scrape.py
git mv tests/fixtures/model_training/qbr_endpoint_sample.json tests/fixtures/qbr_endpoint_sample.json
```

- [ ] **Step 3: Fix the two path references** in `tests/test_qbr_scrape.py` (now one directory shallower)

The file moved from `tests/model_training/` to `tests/`, so each `__file__`-relative hop loses one level:
- **sys.path:** `parents[2] / "python"` → `parents[1] / "python"` (from `tests/test_qbr_scrape.py`, `parents[1]` is the repo root, which contains `python/`).
- **fixture path:** `(...).parent.parent / "fixtures" / "model_training" / "qbr_endpoint_sample.json"` → `(...).parent / "fixtures" / "qbr_endpoint_sample.json"` (from `tests/test_qbr_scrape.py`, `.parent` is `tests/`; the fixture now lives at `tests/fixtures/qbr_endpoint_sample.json`).

Match the file's actual variable names/spacing when editing.

- [ ] **Step 4: Run the relocated test — verify it passes**

Run: `uv run pytest tests/test_qbr_scrape.py -v`
Expected: PASS (imports `parse_qbr_payload` from `scrape_cfb_qbr`, reads the moved fixture). If it FAILS on import or fixture path, the path edits in Step 3 are wrong — fix and re-run. Do NOT proceed to deletion until this passes.

- [ ] **Step 5: Commit**

```bash
git add tests/test_qbr_scrape.py tests/fixtures/qbr_endpoint_sample.json
git status --short -- tests/model_training/test_qbr_scrape.py tests/fixtures/model_training/qbr_endpoint_sample.json   # should show as deleted (the mv source)
git add tests/model_training/test_qbr_scrape.py tests/fixtures/model_training/qbr_endpoint_sample.json
git commit -m "test(qbr): relocate test_qbr_scrape + fixture out of the modeling test tree"
```

---

### Task 2: Delete the four modeling packages + their tests/fixtures

**Files:**
- Delete: `python/model_training/`, `python/cpoe/`, `python/pregame_wp/`, `python/rb_eval/`
- Delete: `tests/model_training/`, `tests/cpoe/`, `tests/pregame_wp/`, `tests/rb_eval/`
- Delete: `tests/fixtures/model_training/`, `tests/fixtures/pregame_wp/`, `tests/fixtures/rb_eval/`

**Interfaces:**
- Consumes: Task 1 (the QBR test + fixture already relocated, so deleting `tests/model_training/` + `tests/fixtures/model_training/` is safe). Produces: a scraping-only `python/` + `tests/`.

- [ ] **Step 1: Re-verify the boundary is clean** (gate before deleting)

Run: `git grep -nE "(^|[^.])(import|from) (model_training|cpoe|pregame_wp|rb_eval)" -- python tests ':!python/model_training' ':!python/cpoe' ':!python/pregame_wp' ':!python/rb_eval' ':!tests/model_training' ':!tests/cpoe' ':!tests/pregame_wp' ':!tests/rb_eval' || echo "NO external references — safe to delete"`
Expected: `NO external references — safe to delete`. If anything prints, STOP and report — a staying module/test references a modeling package and the deletion would break it.

- [ ] **Step 2: Delete the packages + modeling tests + modeling fixtures**

```bash
git rm -r python/model_training python/cpoe python/pregame_wp python/rb_eval
git rm -r tests/model_training tests/cpoe tests/pregame_wp tests/rb_eval
git rm -r tests/fixtures/model_training tests/fixtures/pregame_wp tests/fixtures/rb_eval
```
(`tests/model_training/test_qbr_scrape.py` + `tests/fixtures/model_training/qbr_endpoint_sample.json` were already removed by Task 1's `git mv`, so these `git rm -r` calls won't find them — that's expected.)

- [ ] **Step 3: Collection sanity — scraping tests only, no errors**

Run: `uv run pytest --collect-only -q`
Expected: collection succeeds with NO errors; the collected set is now ONLY the root scraping tests (`test_betting`, `test_live_endpoints`, `test_refreshers`, `test_reprocess`, `test_schedules`, `test_scrape_json`, `test_team_box_extra`, `test_utils`, `test_qbr_scrape`). No `model_training`/`cpoe`/`pregame_wp`/`rb_eval` test files appear; no `ModuleNotFoundError`.

- [ ] **Step 4: Commit**

```bash
git commit -m "refactor: remove model-training packages + tests from cfb-raw (now in cfb-data)"
```
(The `git rm -r` already staged the deletions; verify with `git status --short | grep -v '^??'` that ONLY deletions of the modeling paths are staged — no untracked data added.)

---

### Task 3: Drop modeling-only deps + re-lock + scraping-suite gate

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated)

**Interfaces:**
- Consumes: Tasks 1–2 (modeling code gone). Produces: a scraping-only dependency set; green scraping suite.

- [ ] **Step 1: Edit `pyproject.toml`** — read it first, then:
  - In `[project].dependencies`: REMOVE `scikit-learn>=1.0` and `xgboost>=2.0`. KEEP `sportsdataverse>=0.0.60`, `pandas>=2.0`, `polars>=1.0`, `pyarrow>=15.0`, `requests>=2.28`, `tqdm>=4.66`.
  - In `[dependency-groups]`: set `dev = ["pytest>=8.0"]` (drop `joblib>=1.3`); DELETE the `figures`, `gam`, and `pregame-wp` group entries entirely.
  - Optionally: if `[project].description` mentions modeling, trim it to the scraping role (leave it if it's already scraping-focused).
  - Leave `[tool.pytest.ini_options]` unchanged.

- [ ] **Step 2: Re-lock + sync**

Run: `uv sync 2>&1 | tail -5`
Expected: resolves cleanly; the modeling deps (xgboost/sklearn/scipy/pygam/plotnine/statsmodels/joblib) are no longer DIRECT deps (xgboost may still appear transitively via `sportsdataverse` — that's fine). `uv.lock` is updated.

- [ ] **Step 3: Scraping-suite green + import smoke + stale-ref grep**

```bash
uv run pytest -m "not integration" -q 2>&1 | tail -3
uv run python -c "import scrape_cfb_json, scrape_cfb_qbr, cfb_betting, reprocess_cfb_json, scrape_cfb_schedules; print('scraper imports ok')"
git grep -nE "model_training|cpoe|pregame_wp|rb_eval" -- python tests || echo "no live modeling refs in python/ or tests/"
```
Expected: pytest green (scraping tests only, integration deselected, 0 errors); `scraper imports ok`; `no live modeling refs in python/ or tests/`. If pytest fails because a scraper needed a dropped dep, STOP + report BLOCKED (the boundary check missed something) — do NOT re-add the dep without surfacing it.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore(deps): drop modeling-only deps from cfb-raw (scraping-only)"
```

---

## Self-Review

**1. Spec coverage:** D1 (remove packages) → T2; D2 (remove tests/fixtures) → T2; D3 (relocate test_qbr) → T1; D4 (drop deps + groups) → T3; D5 (xgboost transitive) → T3 Step 2 expectation; D6 (scope) → Global Constraints + no scraper edits. Verification §5 → T2 Step 3 (collection), T3 Step 3 (suite + smoke + grep). ✓

**2. Placeholder scan:** every step has exact commands. The relocation edits (T1 Step 3) give the exact before/after path transforms with a "match actual variable names" note (adapt-to-file, not a placeholder). No TBD/TODO.

**3. Type/consistency:** the relocation order is correct (T1 `git mv` before T2 `git rm -r` of the now-empty modeling test/fixture dirs); the `git add` calls are scoped to named paths (never `-A`) per the Global Constraint; the suite command matches cfb-raw's root-run convention.

## Notes
- Order matters: **T1 before T2** (relocate the QBR test before deleting `tests/model_training/`).
- The ~50 untracked scraped-data files + the stashed log/uv.lock noise are deliberately untouched; each task's `git add` is path-scoped.
