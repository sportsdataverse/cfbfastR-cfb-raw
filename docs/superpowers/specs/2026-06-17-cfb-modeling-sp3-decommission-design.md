# CFB Modeling Migration — SP3 Design

**Decommission the model-training code from `cfbfastR-cfb-raw` (scraping-only)**

- **Date:** 2026-06-17
- **Status:** Approved design (pre-implementation)
- **Repo:** `cfbfastR-cfb-raw` (the scraper repo)
- **Branch:** `feat/cfb-modeling-sp3-decommission`
- **Context:** SP1 copied the four modeling packages from cfb-raw → `cfbfastR-cfb-data` (PR #3); SP2 added publish + reports there (PR #4). SP3 removes the now-duplicated modeling code from cfb-raw so it is **scraping-only** and still green.

---

## 1. Motivation

The model-training suite (EP/WP/QBR, CPOE, fourth-down, RB-eval) now lives and ships from `cfbfastR-cfb-data`. Leaving the copies in `cfbfastR-cfb-raw` is dead weight + drift risk. SP3 deletes them, leaving cfb-raw focused on its single job: scrape ESPN → commit raw + enriched per-game JSON + the schedule master.

## 2. Locked decisions

| # | Decision | Choice |
|---|---|---|
| D1 | Remove the modeling packages | Delete `python/{model_training (incl. fourth_down), cpoe, pregame_wp, rb_eval}/`. |
| D2 | Remove the modeling tests + fixtures | Delete `tests/{model_training, cpoe, pregame_wp, rb_eval}/` and `tests/fixtures/{model_training, pregame_wp, rb_eval}/`. |
| D3 | Relocate the QBR-scraper test | `tests/model_training/test_qbr_scrape.py` → `tests/test_qbr_scrape.py`; fixture `tests/fixtures/model_training/qbr_endpoint_sample.json` → `tests/fixtures/qbr_endpoint_sample.json`. It tests the **staying** `scrape_cfb_qbr` and must survive. |
| D4 | Drop modeling-only deps | Remove `scikit-learn`, `xgboost`, `joblib`, `plotnine`, `statsmodels`, `pygam`, `scipy` from `pyproject.toml`; delete the `figures`/`gam`/`pregame-wp` dependency-groups; `dev` → `["pytest>=8.0"]`. |
| D5 | xgboost stays transitive | `xgboost` is needed by the scraper's `CFBPlayProcess` enrichment **via `sportsdataverse`** (a kept dep), so cfb-raw's *direct* `xgboost` declaration is redundant — drop it; xgboost remains installed transitively. No runtime breakage. |
| D6 | Scope | cfb-raw scrapers/tests are otherwise unchanged; **cfb-data is untouched.** |

## 3. Verified grounding facts (SP3 exploration)

- **Clean boundary:** ZERO imports of `model_training`/`cpoe`/`pregame_wp`/`rb_eval` by any staying scraper module (grep across `python/` excluding the 4 package dirs → no hits).
- **No ML-dep usage by scrapers:** ZERO direct imports of `xgboost`/`sklearn`/`scipy`/`pygam`/`plotnine`/`statsmodels`/`joblib` by any staying scraper or root scraping test → all 7 are modeling-only + safe to drop.
- **Staying scrapers (12):** `python/{_cfb_raw_utils, cfb_betting, cfb_team_box_extra, reprocess_cfb_json, scrape_cfb_game_rosters, scrape_cfb_json, scrape_cfb_participants, scrape_cfb_power_index, scrape_cfb_qbr, scrape_cfb_schedules, scrape_failures, __init__}.py`.
- **Staying root tests (8):** `tests/test_{betting, live_endpoints, refreshers, reprocess, schedules, scrape_json, team_box_extra, utils}.py` (`live` ones gated by `CFB_LIVE_TESTS=1`) + `tests/conftest.py`.
- **`test_qbr_scrape.py`** (`tests/model_training/test_qbr_scrape.py`): imports `from scrape_cfb_qbr import parse_qbr_payload` (via a `sys.path.insert(parents[2]/"python")`) and reads `tests/fixtures/model_training/qbr_endpoint_sample.json`. The OTHER model_training tests (`test_features_qbr.py`, `test_train_qbr.py`, …) test the modeling code → removed. No other staying-scraper test lives under the modeling test dirs.
- **pyproject `[tool.pytest.ini_options]`** (`markers=["live: …"]`, `testpaths=["tests"]`, `pythonpath=["python"]`) stays unchanged.

## 4. Components (what changes)

- **Delete** the 4 package dirs (D1), the 4 test dirs (D2) — *minus* `test_qbr_scrape.py`, the 3 fixture dirs (D2) — *minus* `qbr_endpoint_sample.json`.
- **Relocate** (D3): `git mv tests/model_training/test_qbr_scrape.py tests/test_qbr_scrape.py`; `git mv tests/fixtures/model_training/qbr_endpoint_sample.json tests/fixtures/qbr_endpoint_sample.json`; then fix the test's fixture path (now `tests/fixtures/qbr_endpoint_sample.json`, a different relative depth) and its `sys.path.insert` (the test now lives at `tests/`, so `parents[1]/"python"` reaches `python/`).
- **Edit** `pyproject.toml` (D4): drop the 7 deps + 3 groups; `dev=["pytest>=8.0"]`; optionally tweak the `[project].description` if it mentions modeling.
- **Re-lock**: `uv sync` so the slimmed env (no modeling deps) is what the scraping suite runs against.

**Commit hygiene:** stage ONLY the SP3 paths (the `git rm`/`git mv`, `pyproject.toml`, `uv.lock`, the relocated test). Do NOT `git add -A` — the working tree has ~50 untracked scraped-data files + logs that must stay out of SP3 commits.

## 5. Verification / acceptance

- `cd python && uv run pytest -m "not integration" -q` is green and now collects ONLY the scraping tests (the 8 root tests + the relocated `test_qbr_scrape`); the modeling tests are gone (collection count drops accordingly), zero import/collection errors.
- `uv sync` resolves cleanly without the dropped deps; `python -c "import scrape_cfb_json, scrape_cfb_qbr, cfb_betting, reprocess_cfb_json"` (a staying-scraper import smoke) succeeds.
- `git grep -nE "model_training|cpoe|pregame_wp|rb_eval"` over `python/` + `tests/` returns no live references (docs/ may mention them historically).
- The relocated `test_qbr_scrape` passes (reads the moved fixture, imports `scrape_cfb_qbr`).

## 6. Non-goals / risks

- **Non-goals:** no scraper-logic changes; no CI changes; cfb-data untouched; the ~50 untracked scraped-data files are not committed by SP3.
- **R1 — test_qbr relocation paths:** the moved test's fixture path + `sys.path` depth change; the plan pins the exact edits (read the file first).
- **R2 — other consumers:** confirm (via grep) no scraping module/test/CLI references the removed packages before deleting (exploration says none; the plan re-verifies as a gate).
- **R3 — uv.lock:** dropping deps re-locks; commit the re-locked `uv.lock` as part of SP3.

## 7. Decision log
1. Decommission scope → remove the 4 packages + their tests/fixtures from cfb-raw; relocate the one staying-scraper test (`test_qbr_scrape`).
2. Deps → drop all 7 modeling-only deps + the 3 groups; `xgboost` stays transitively via `sportsdataverse`.
