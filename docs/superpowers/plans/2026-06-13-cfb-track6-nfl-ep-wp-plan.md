# NFL EP/WP/QBR Model-Training — Implementation Plan (Track 6)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

> **IMPORTANT — CROSS-REPO:** This track's implementation does NOT live in
> `cfbfastR-cfb-raw`. The docs live here (program umbrella); the code lives in
> a separate NFL repo (to be created in Phase 0). All Phase 1+ tasks are
> **conditional on Phase 0's survey findings** — do not implement any training
> code until Phase 0 is complete.

**Goal:** Survey the nflfastR EP/WP/QBR training lineage, confirm the cross-repo
home, assess the nflverse-pbp data source, and produce a scoped Phase 1+ plan
for retraining the three NFL models bundled in sdv-py's `nfl/models/`.

**Architecture:** Mirrors CFB Track 1. Training features come from nflverse-pbp
(confirmed by Phase 0 / UPDATE; raw ESPN is secondary). The only net-new logic
is the outcome label (next score in half), directly analogous to the CFB port.
Two models in scope: EP (**18-feat**, `multi:softprob`, 7-class, 525 rounds),
WP-spread (**12-feat**, `binary:logistic`, 534 rounds). The sdv-py QBR bundle
is a CFB-copy placeholder; there is no nflfastR QBR model — QBR is out of scope
for Track 6. See UPDATE section and spec §12 for confirmed hyperparameters.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-track6-nfl-ep-wp-design.md`

**Commit convention:** conventional commits (`feat(nfl-models):`, `fix(nfl-models):`,
`test(nfl-models):`, etc.). No AI co-author trailers on any commits in this
pipeline — human author only (applies to all commits throughout all phases).

---

## Phase 0 — Survey, repo decision, data-source audit

> **Phase 0 is the gate.** Do not proceed to Phase 1 until all Phase 0 tasks
> are complete and their findings are recorded. Phase 1–8 tasks are scoped
> pending the Phase 0 findings; some may be revised once the recipe is confirmed.

### Task 0.0: Decide and create the implementation repo

This is a **blocking decision** that must be made before any implementation.
The NFL model-training pipeline must NOT land in `cfbfastR-cfb-raw` (CFB-only).

- [ ] **Step 1: Review the three candidate homes** documented in spec §4:
  - Option A: new `nfl-raw` repo (recommended)
  - Option B: existing nflverse-py community repo
  - Option C: `sdv-py` internal `nfl/model_training/` sub-package

- [ ] **Step 2: Make the decision.**

  Recommendation is Option A (new `nfl-raw` repo). Confirm or override.

- [ ] **Step 3: Create the target repo (if Option A).**

  ```bash
  # From the sdv-dev org or personal GitHub
  gh repo create sdv-dev/nfl-raw --private --description "NFL raw scraper + model-training pipeline"
  ```

  Initialize with `pyproject.toml` (uv, Python 3.11+), `uv.lock`, `.gitignore`,
  `python/model_training/` directory.

- [ ] **Step 4: Record the decision.**

  Update the program umbrella
  (`docs/superpowers/specs/2026-06-13-cfb-modeling-suite-program.md`)
  with the confirmed home in Track 6's "Home (proposed)" column.

- [ ] **Step 5: Record the expected canonical commit for the repo creation.**

  ```
  feat(nfl-raw): initialize nfl-raw repo scaffold
  ```

---

### Task 0.1: Confirm the bundled NFL model introspection results

The spec §2.1 records introspection results from running xgboost against the
three bundled NFL models. Reproduce these to confirm they are accurate as a
baseline before any recipe hunting.

**Commands (from `sdv-py` root):**

```bash
uv run python -c "
import xgboost as xgb, json
for name, path in [
    ('ep_model',   'sportsdataverse/nfl/models/ep_model.ubj'),
    ('wp_spread',  'sportsdataverse/nfl/models/wp_spread.ubj'),
    ('qbr_model',  'sportsdataverse/nfl/models/qbr_model.ubj'),
]:
    b = xgb.Booster(); b.load_model(path)
    cfg = json.loads(b.save_config())['learner']
    print(name,
          'feats=', b.num_features(),
          'obj=', cfg['objective']['name'],
          'num_class=', cfg['learner_model_param'].get('num_class','n/a'),
          'trees=', b.num_boosted_rounds(),
          'feat_names=', b.feature_names)
"
```

**Expected output (as established by the survey):**

```
ep_model  feats=8  obj=multi:softprob  num_class=7  trees=525  feat_names=None
wp_spread feats=13 obj=binary:logistic num_class=0  trees=760  feat_names=None
qbr_model feats=6  obj=reg:squarederror num_class=0 trees=45   feat_names=None
```

- [ ] **Step 1:** Run the command. Record whether the output matches the spec.
  If there is a discrepancy, update the spec before proceeding.

- [ ] **Step 2: Verify the NFL `model_vars.py` feature lists.**

  ```bash
  uv run python -c "
  from sportsdataverse.nfl import model_vars as mv
  print('ep_final_names:', mv.ep_final_names)
  print('wp_final_names:', mv.wp_final_names)
  print('qbr_vars:', mv.qbr_vars)
  print('ep_class_to_score:', mv.ep_class_to_score_mapping)
  "
  ```

  Expected: 8 EP features, 13 WP features, 6 QBR features — as documented
  in spec §2.3.

- [ ] **Step 3:** Confirm that `nfl/model_vars.py` feature lists are identical
  to `cfb/model_vars.py`.

  ```bash
  uv run python -c "
  from sportsdataverse.cfb import model_vars as cfb_mv
  from sportsdataverse.nfl import model_vars as nfl_mv
  print('EP match:', cfb_mv.ep_final_names == nfl_mv.ep_final_names)
  print('WP match:', cfb_mv.wp_final_names == nfl_mv.wp_final_names)
  print('QBR match:', cfb_mv.qbr_vars == nfl_mv.qbr_vars)
  print('EP score map match:', cfb_mv.ep_class_to_score_mapping == nfl_mv.ep_class_to_score_mapping)
  "
  ```

  Expected: all `True`. This confirms the NFL and CFB models share an identical
  inference contract — strong evidence of a shared training lineage.

- [ ] **Step 4:** Record findings. Differences (if any) must be documented
  in the spec §2 before Phase 1 begins.

---

### Task 0.2: Source the nflfastR training scripts (the EP/WP recipe)

The nflfastR R package (`https://github.com/nflverse/nflfastR`) is the
authoritative source. The goal is to find the exact EP/WP training scripts —
the NFL analogue of `cfbscrapR-wpa.ipynb` (which was the confirmed CFB WP
recipe).

- [ ] **Step 1: Clone or browse `nflverse/nflfastR` at the tag corresponding
  to the bundled model era.**

  The CFB keepers models date to Dec 2020. If the NFL models are from the same
  era, the relevant nflfastR commit is likely late 2020.

  ```bash
  gh repo clone nflverse/nflfastR /tmp/nflfastR
  cd /tmp/nflfastR
  git log --oneline --since="2020-01-01" --until="2021-06-01" | head -30
  # Look for commits mentioning ep_model, wp_model, model training, xgboost
  git log --oneline --all --grep="model" | head -20
  git log --oneline --all --grep="xgboost" | head -20
  ```

- [ ] **Step 2: Search for training scripts in `data-raw/` and `inst/`.**

  ```bash
  find /tmp/nflfastR -name "*.R" | xargs grep -l "xgboost\|xgb\|train\|nrounds" 2>/dev/null | head -20
  find /tmp/nflfastR -name "*.R" | xargs grep -l "ep_model\|wp_model\|model_file" 2>/dev/null | head -20
  find /tmp/nflfastR -type f -name "*.R" | head -40  # list all R scripts
  ```

- [ ] **Step 3: Also check companion repos.**

  nflfastR has companion repos in the nflverse org:
  - `nflverse/nflfastR-data` (the data releases)
  - `nflverse/nflverse-models` (if it exists)
  - Any `model_training` or `models` subdirectories

  ```bash
  gh repo list nflverse --limit 30 | grep -i "model\|train\|data"
  ```

- [ ] **Step 4: Record the recipe findings.**

  For each training script found, record:

  - Script path and URL
  - XGBoost params: `objective`, `nrounds`, `eta`, `gamma`, `subsample`,
    `colsample_bytree`, `max_depth`, `min_child_weight`
  - Feature set (the column list passed to the DMatrix)
  - Label derivation (how next-score-in-half is computed for NFL)
  - Sample weights (if any)
  - Training data source (enriched PBP parquet vs raw ESPN)
  - Season coverage

- [ ] **Step 5: Cross-check NFL hyperparameters against CFB Track 1 findings.**

  If the NFL recipe uses the same params as CFB (e.g., EP: `eta=0.025, gamma=1,
  subsample=0.8, colsample_bytree=0.8, max_depth=5, min_child_weight=1,
  nrounds=525`), this strongly suggests a shared codebase. Document explicitly.

- [ ] **Step 6: Assess whether the recipe is a "faithful port" or a "reconstruction".**

  - **Faithful port:** the exact nflfastR training script is found, its output
    is the bundled model (verifiable by prediction comparison), and
    implementation follows it directly.
  - **Reconstruction:** the recipe is inferred from architecture + hyperparameter
    guesses (analogous to the CFB QBR situation). Requires an additional
    validation stage.

  Record which category each model (EP, WP, QBR) falls into.

---

### Task 0.3: Audit the nflverse-pbp data source

Determine whether the nflverse-pbp parquet (Option A, spec §6) contains all
the raw game-state columns needed to reconstruct the 8 EP features and 13 WP
features without running `NFLPlayProcess`.

- [ ] **Step 1: Load one season of nflverse-pbp.**

  ```bash
  # From sdv-py root
  uv run python -c "
  from sportsdataverse.nfl import load_nfl_pbp, update_config
  update_config(cache_mode='off')
  df = load_nfl_pbp(seasons=[2024])
  print('rows:', df.shape[0])
  print('cols:', df.shape[1])
  print(df.columns.to_list())
  "
  ```

- [ ] **Step 2: Check for the EP feature source columns.**

  Required columns for `ep_final_names` (8 features):

  | Feature | Source column needed | In nflverse-pbp? |
  |---|---|---|
  | `TimeSecsRem` | `ydstogo`-equivalent or game clock | TBD |
  | `yards_to_goal` | `yardline_100` or equivalent | TBD |
  | `distance` | `ydstogo` | TBD |
  | `down_1..down_4` | `down` (0/1 dummy derivable) | TBD |
  | `pos_score_diff_start` | `score_differential` | TBD |

  ```bash
  uv run python -c "
  from sportsdataverse.nfl import load_nfl_pbp, update_config
  update_config(cache_mode='off')
  df = load_nfl_pbp(seasons=[2024])
  wanted = ['ydstogo', 'yardline_100', 'down', 'score_differential',
            'game_seconds_remaining', 'half_seconds_remaining',
            'quarter_seconds_remaining', 'posteam_score', 'defteam_score',
            'spread_line', 'home_team', 'posteam', 'defteam', 'period']
  found = [c for c in wanted if c in df.columns]
  missing = [c for c in wanted if c not in df.columns]
  print('found:', found)
  print('missing:', missing)
  "
  ```

- [ ] **Step 3: Check for the WP feature source columns.**

  Required for `wp_final_names` (13 features):

  - `spread_time` = derivable from spread + `adj_TimeSecsRem` (if spread is present)
  - `adj_TimeSecsRem` — nflfastR computes this; check if it is in the parquet
  - `ExpScoreDiff_Time_Ratio` — derived from EP output; may NOT be in parquet
    if nflfastR doesn't export it
  - `pos_team_receives_2H_kickoff` — game-situation flag
  - Timeout counts for each team

  ```bash
  uv run python -c "
  from sportsdataverse.nfl import load_nfl_pbp, update_config
  update_config(cache_mode='off')
  df = load_nfl_pbp(seasons=[2024])
  wp_related = ['spread_time', 'adj_TimeSecsRem', 'ExpScoreDiff_Time_Ratio',
                'receive_2h_ko', 'posteam_timeouts_remaining',
                'defteam_timeouts_remaining', 'ep', 'epa', 'wp', 'wpa']
  for c in wp_related:
      print(c, '->', 'PRESENT' if c in df.columns else 'ABSENT')
  "
  ```

- [ ] **Step 4: Check for the label-derivation columns.**

  The training label (next score in half) requires identifying scoring plays
  and the scoring team. Check whether the nflverse-pbp parquet retains the
  raw play-level scoring flags needed for vectorized NSH derivation:

  ```bash
  uv run python -c "
  from sportsdataverse.nfl import load_nfl_pbp, update_config
  update_config(cache_mode='off')
  df = load_nfl_pbp(seasons=[2024])
  label_cols = ['td_prob', 'fg_prob', 'safety_prob', 'posteam',
                'scoring_team', 'next_score_half', 'fixed_drive',
                'game_half', 'td_prob_next_score', 'ep_next_score']
  for c in label_cols:
      print(c, '->', 'PRESENT' if c in df.columns else 'ABSENT')
  "
  ```

- [ ] **Step 5: Record the column-audit verdict.**

  Based on the audit, determine:

  - Can all 8 EP features be reconstructed from nflverse-pbp? (Yes/No/Partial)
  - Can all 13 WP features be reconstructed? (Yes/No/Partial — `ExpScoreDiff_Time_Ratio`
    is the key uncertain one; it depends on `EP_start` which the parquet may carry
    under a different name)
  - Can NSH labels be derived without re-running `NFLPlayProcess`? (Yes/No)
  - **Verdict:** is Option A (nflverse-pbp) sufficient, or does Option B
    (raw ESPN scraper) become necessary?

---

### Task 0.4: Confirm ESPN NFL QBR endpoint availability

The QBR training target is ESPN's raw NFL QBR (same endpoint family as CFB).

- [ ] **Step 1: Test the NFL QBR endpoint.**

  ```bash
  uv run python -c "
  import requests, json
  url = ('https://sports.core.api.espn.com/v2/sports/football/leagues/nfl/'
         'seasons/2024/types/2/weeks/1/qbr/10000?limit=1000')
  r = requests.get(url, timeout=30)
  print('status:', r.status_code)
  data = r.json()
  print('keys:', list(data.keys()))
  print('item count:', len(data.get('items', [])))
  if data.get('items'):
      item = data['items'][0]
      print('first item keys:', list(item.keys()))
  "
  ```

- [ ] **Step 2: Record the endpoint availability and data shape.**

  - Does it return `{items: [...]}` with `athlete.$ref`, `event.$ref`, and
    `splits.categories[0].stats` (QBR values)?
  - Is the format identical to CFB QBR (allowing reuse of `parse_qbr_payload`
    from `cfbfastR-cfb-raw/python/scrape_cfb_qbr.py`)?
  - What is the earliest available NFL QBR season?

---

### Task 0.5: Phase 0 findings record

After completing tasks 0.0–0.4, record all findings in a structured summary.
This is the gate-check before any Phase 1 implementation begins.

- [ ] **Step 1: Write a Phase 0 findings record** in `dev/track6-phase0-findings.md`
  (in the target repo's `dev/` directory, which is gitignored). Contents:

  - Model introspection results (confirm or correct spec §2.1)
  - Recipe sourcing outcome for each model:
    - Script URL + git SHA
    - Exact hyperparameters
    - Feature set confirmed / differences from spec
    - Label derivation approach
    - Training data source confirmed
    - "Faithful port" or "Reconstruction"
  - nflverse-pbp column audit verdict (Option A sufficient?)
  - ESPN NFL QBR endpoint availability
  - Cross-repo home decision (confirmed)
  - Phase 1+ plan revisions needed (if any)

- [ ] **Step 2: Update the spec** `docs/superpowers/specs/2026-06-13-cfb-track6-nfl-ep-wp-design.md`
  with confirmed findings from the survey (filling in the "unknown" gaps in §5).

---

## Phase 1 — Scaffold (CONDITIONAL on Phase 0)

> All Phase 1 tasks are conditional on Phase 0 completion. The specific
> commands reference the target repo created in Task 0.0.

### Task 1.1: Create the package skeleton in the target repo

**Files (in the target NFL repo):**
- `python/model_training/__init__.py`
- `python/model_training/constants.py` — NFL feature lists hard-coded to the
  **confirmed canonical nflfastR contracts** (18-feat EP, 12-feat WP-spread).
  Do NOT import from `sportsdataverse.nfl.model_vars` — those lists reflect the
  CFB-copy architecture (8/13-feat) and would lock in the wrong contracts.
- `tests/model_training/test_package.py`
- `tests/model_training/test_constants.py`

- [ ] **Step 1: Write the failing tests**

  ```python
  # tests/model_training/test_package.py
  def test_package_imports():
      import model_training
      assert hasattr(model_training, "__version__")
  ```

  ```python
  # tests/model_training/test_constants.py
  from model_training import constants as C

  # Canonical nflfastR contracts (from fastrmodels/data-raw/models.R)
  _NFL_EP_FEATURES = [
      "half_seconds_remaining", "yardline_100", "home", "retractable", "dome",
      "outdoors", "ydstogo", "era0", "era1", "era2", "era3", "era4",
      "down1", "down2", "down3", "down4",
      "posteam_timeouts_remaining", "defteam_timeouts_remaining",
  ]  # 18 features
  _NFL_WP_FEATURES = [
      "receive_2h_ko", "spread_time", "home", "half_seconds_remaining",
      "game_seconds_remaining", "diff_time_ratio", "score_differential",
      "down1", "down2", "down3", "down4", "posteam_timeouts_remaining",
  ]  # 12 features (monotone_constraints on spread_time + score_differential)

  def test_ep_features_canonical():
      assert C.EP_FEATURES == _NFL_EP_FEATURES

  def test_wp_features_canonical():
      assert C.WP_FEATURES == _NFL_WP_FEATURES

  def test_ep_score_map():
      assert C.EP_CLASS_TO_SCORE == {0: 7, 1: -7, 2: 3, 3: -3, 4: 2, 5: -2, 6: 0}
  ```

- [ ] **Step 2: Run tests to verify they fail**

  Run: `uv run pytest tests/model_training/ -v`
  Expected: FAIL (`ModuleNotFoundError: No module named 'model_training'`)

- [ ] **Step 3: Create the package**

  ```python
  # python/model_training/__init__.py
  """NFL model-training pipeline (Track 6: EP/WP/QBR play-level models).
  
  Implementation home: [target repo from Task 0.0].
  Spec: cfbfastR-cfb-raw/docs/superpowers/specs/2026-06-13-cfb-track6-nfl-ep-wp-design.md
  """
  from __future__ import annotations
  __version__ = "0.1.0"
  ```

- [ ] **Step 4: Create `constants.py`**

  Mirror the CFB `constants.py` structure (spec §9) but importing from
  `sportsdataverse.nfl.model_vars`. Include:
  - `EP_FEATURES`, `WP_FEATURES`, `QBR_FEATURES` (from `nfl/model_vars.py`)
  - `EP_CLASS_TO_SCORE` (from `nfl/model_vars.py`)
  - `NEXT_SCORE_TO_LABEL` — 7-class label mapping (identical to CFB; the
    scoring classes are the same for NFL)
  - XGBoost param dicts — **confirmed values** from the UPDATE section (no
    longer placeholders). EP: `eta=0.025, gamma=1, subsample=0.8,
    colsample_bytree=0.8, max_depth=5, min_child_weight=1, nrounds=525`.
    WP-spread: `eta=0.05, gamma=.79012017, subsample=0.9224245,
    colsample_bytree=0.6201923, max_depth=3, min_child_weight=5, nrounds=534`
    with `monotone_constraints=(1, 1, 0, ..., 0)` on spread_time +
    score_differential. See UPDATE section for full lists.
  - `NFL_EP_SOURCE`, `NFL_WP_SOURCE` — source column crosswalk from nflverse-pbp
    (or `NFLPlayProcess` output) to model feature names — **PENDING Task 0.3
    column audit**.
  - `BAD_GAME_IDS: set[int] = set()` — initially empty; populate from recipe.

- [ ] **Step 5: Run tests to verify they pass**

  Run: `uv run pytest tests/model_training/ -v`
  Expected: PASS (4 tests if `constants.py` is populated correctly)

- [ ] **Step 6: Commit**

  ```
  feat(nfl-models): scaffold model_training package + NFL constants
  ```

---

### Task 1.2: Add dependencies to the target repo's `pyproject.toml`

- [ ] **Step 1: Add training + figures deps** (same as CFB Track 1):

  ```toml
  [project.dependencies]
  sportsdataverse = ">=0.0.58"   # bundles NFLPlayProcess + nfl/model_vars.py
  xgboost = ">=2.0"
  polars = ">=1.0,<2.0"
  pandas = ">=2.0"
  pyarrow = ">=14.0"
  requests = ">=2.31"
  tqdm = ">=4.0"

  [dependency-groups]
  dev = ["pytest>=8.0"]
  figures = ["plotnine>=0.13", "statsmodels>=0.14", "pillow>=10.0"]
  ```

- [ ] **Step 2: Sync**

  Run: `uv sync --all-groups`
  Expected: resolves with no error.

- [ ] **Step 3: Commit**

  ```
  build(nfl-models): add training + figures deps
  ```

---

### Task 1.3: Vendor the NFL reference models as test fixtures

The three shipped NFL `.ubj` files are the Stage-1 parity targets (analogous to
the CFB keepers models in Track 1). They live in `sdv-py/sportsdataverse/nfl/models/`.

- [ ] **Step 1: Copy the shipped models into the test fixtures directory.**

  ```bash
  mkdir -p tests/fixtures/model_training
  cp <sdv-py>/sportsdataverse/nfl/models/ep_model.ubj   tests/fixtures/model_training/nfl_ep_model_shipped.ubj
  cp <sdv-py>/sportsdataverse/nfl/models/wp_spread.ubj  tests/fixtures/model_training/nfl_wp_spread_shipped.ubj
  cp <sdv-py>/sportsdataverse/nfl/models/qbr_model.ubj  tests/fixtures/model_training/nfl_qbr_model_shipped.ubj
  ```

  Note: these are the shipped inference models; there are no separate
  "R-trained reference" models for NFL (unlike CFB Track 1 which had the
  May-2021 `xgb_*` keepers). The shipped models ARE the Stage-1 parity target.

- [ ] **Step 2: Write a fixture README documenting provenance.**

  ```markdown
  # model_training fixtures

  - `nfl_ep_model_shipped.ubj`  — shipped ep_model.ubj from sdv-py nfl/models/
    (8 feat, multi:softprob, 7-class, 525 trees). Stage-2 parity reference.
  - `nfl_wp_spread_shipped.ubj` — shipped wp_spread.ubj (13 feat, binary:logistic, 760 trees).
  - `nfl_qbr_model_shipped.ubj` — shipped qbr_model.ubj (6 feat, reg:squarederror, 45 trees).

  These are the current sdv-py inference models. Retrained replacements
  must predict within documented tolerance on a held-out season before handoff.
  ```

- [ ] **Step 3: Verify the fixtures load.**

  ```bash
  uv run python -c "
  import xgboost as xgb
  for f in ['nfl_ep_model_shipped', 'nfl_wp_spread_shipped', 'nfl_qbr_model_shipped']:
      b = xgb.Booster()
      b.load_model(f'tests/fixtures/model_training/{f}.ubj')
      print(f, b.num_features(), b.num_boosted_rounds())
  "
  ```

  Expected: `nfl_ep_model_shipped 8 525`, `nfl_wp_spread_shipped 13 760`,
  `nfl_qbr_model_shipped 6 45`.

- [ ] **Step 4: Commit**

  ```
  test(nfl-models): vendor shipped NFL models as parity reference fixtures
  ```

---

## Phase 2 — Data ingest (CONDITIONAL on Phase 0 Task 0.3 verdict)

> The data-source verdict from Task 0.3 determines the ingest approach.
> Two paths are documented below; implement the one chosen.

### If Option A (nflverse-pbp is sufficient):

**Task 2.1A: `nfl_ingest.py` — read nflverse-pbp + derive NSH labels**

This module reads from `load_nfl_pbp()` (or directly from the nflverse-data
parquet URL), applies the label-derivation logic (next score in half), adds
sample weights, and writes `nfl_pbp_full.parquet`.

- [ ] **Step 1: Write failing tests for `clean_nfl_plays` and
  `label_next_score_half_nfl`.**

  Structure: analogous to CFB Track 1 Tasks 2.1 and 1.2. Key differences:
  - NFL play type column names differ from CFB (e.g., `play_type` not `type.text`);
    column names sourced from Task 0.3 column audit.
  - NFL scoring play flags may differ from CFB (`touchdown`, `field_goal_result`,
    `safety`, etc.)
  - OT play detection: nflverse-pbp uses `game_half == "Overtime"` or
    `qtr > 4` depending on nflfastR version.

  Implement these once the column audit is complete.

- [ ] **Step 2: Implement `label_next_score_half_nfl`.**

  Same vectorized backward-fill approach as CFB (`fill_null(strategy="backward")
  .over(["game_id", "game_half"])`) but using nflverse-pbp column names.
  The 7 scoring classes and `NEXT_SCORE_TO_LABEL` mapping are identical to CFB.

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): nfl_ingest — nflverse-pbp clean + next-score-half label
  ```

### If Option B (raw ESPN scraper needed):

**Task 2.1B: `scrape_nfl_json.py` — raw ESPN NFL PBP scraper**

This is a larger task and must be scoped separately. It is a new scraper
analogous to `cfbfastR-cfb-raw/python/scrape_cfb_json.py`. Placeholder only
in this plan — full scope deferred to a sub-spec once Option B is confirmed
as necessary.

---

## Phase 3 — Feature matrices (CONDITIONAL on Phase 0 + Phase 2)

### Task 3.1: `nfl_features.py` — EP/WP/QBR input matrices

Analogous to CFB `features.py`. Select/rename nflverse-pbp (or
`NFLPlayProcess`-output) columns to the exact `ep_final_names`, `wp_final_names`,
and `qbr_vars` contracts.

- [ ] **Step 1: Write failing tests** for `ep_matrix`, `wp_matrix`, `qbr_matrix`.

  Use synthetic data with the confirmed source column names (from Task 0.3 audit).

- [ ] **Step 2: Implement the feature matrices.**

  The implementations are near-copies of CFB `features.py` with NFL-specific
  source column names (from `NFL_EP_SOURCE`, `NFL_WP_SOURCE` in `constants.py`).

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): NFL EP/WP/QBR feature matrices (exact shipped contracts)
  ```

---

## Phase 4 — EP trainer (CONDITIONAL on Phase 0 recipe confirmation)

> Hyperparameters are PENDING Phase 0 Task 0.2. The structure below mirrors
> CFB Track 1 Task 4.1. Do not fill in specific hyperparameter values until
> the nflfastR recipe is confirmed.

### Task 4.1: `nfl_train_ep.py` — EP model trainer

- [ ] **Step 1: Write failing test** verifying `train_ep` produces an
  8-feat `multi:softprob` 7-class model.

  ```python
  def test_train_ep_produces_8feat_7class_softprob():
      model = train_ep(synth_ep_frame(), nrounds=5)
      assert model.num_features() == 8
      assert "multi:softprob" in json.loads(model.save_config())["learner"]["objective"]["name"]
      assert json.loads(model.save_config())["learner"]["learner_model_param"]["num_class"] == "7"
  ```

- [ ] **Step 2: Implement `train_ep`** using `NFL_EP_PARAMS` and `NFL_EP_NROUNDS`
  from `constants.py` (fill these in once Phase 0 Task 0.2 is done).

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): EP trainer (8-feat multi:softprob, nrounds=TBD-from-recipe)
  ```

---

## Phase 5 — WP trainer (CONDITIONAL on Phase 0)

### Task 5.1: `nfl_train_wp.py` — WP-spread model trainer

Same structure as CFB `train_wp.py`. Shipped WP is `binary:logistic`, 13 feat,
760 trees. Hyperparameters pending Phase 0 recipe confirmation.

- [ ] **Step 1: Write failing test** — 13-feat `binary:logistic` output.

- [ ] **Step 2: Implement `train_wp`** with `NFL_WP_SPREAD_PARAMS` and
  `NFL_WP_SPREAD_NROUNDS` from `constants.py`.

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): WP-spread trainer (13-feat binary:logistic, nrounds=TBD)
  ```

---

## Phase 6 — QBR scraper + trainer (CONDITIONAL on Phase 0 Task 0.4)

### Task 6.1: `scrape_nfl_qbr.py` — ESPN NFL QBR scraper

If Task 0.4 confirms the endpoint is available and the format matches CFB,
the CFB `scrape_cfb_qbr.py` can be reused with a league slug change.

- [ ] **Step 1: Implement `scrape_nfl_qbr.py`.**

  Change the league slug from `college-football` to `nfl` in the endpoint URL.
  Verify that `parse_qbr_payload` (from `scrape_cfb_qbr.py`) handles the NFL
  response identically (reuse if so; adapt if not).

- [ ] **Step 2: Commit**

  ```
  feat(nfl-models): ESPN NFL QBR scraper (training target)
  ```

### Task 6.2: `nfl_train_qbr.py` — QBR model trainer

- [ ] **Step 1: Write failing test** — 6-feat `reg:squarederror`.

- [ ] **Step 2: Implement `train_qbr`** with `NFL_QBR_PARAMS` from `constants.py`.

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): QBR trainer (6-feat reg:squarederror)
  ```

---

## Phase 7 — Validation (CONDITIONAL on Phases 4–6)

### Task 7.1: `nfl_validate.py` — parity harness vs shipped models

The parity harness from CFB `validate.py` transfers directly. The NFL reference
models are the shipped `.ubj` from `tests/fixtures/model_training/`.

- [ ] **Step 1: Write failing test** (parity against self = 0 diff).

- [ ] **Step 2: Implement `prediction_parity`, `calibration_table`,
  `weighted_cal_error`** — reuse or import the CFB `validate.py` equivalents
  if the target repo has `cfbfastR-cfb-raw` as a dep.

- [ ] **Step 3: Run parity against the shipped models** on a held-out season
  (e.g., 2024). Document the tolerance achieved.

- [ ] **Step 4: Commit**

  ```
  feat(nfl-models): prediction-parity + LOSO calibration harness
  ```

---

## Phase 8 — Figures (CONDITIONAL on Phase 7)

### Task 8.1: `nfl_figures.py` — calibration plots (plotnine)

Same plotnine styling as CFB `figures.py` (garnet `#500f1b`, Gill Sans MT
fallback, faceted calibration, `y=x` reference, loess smooth). Replace the
cfbfastR hex logo with the nflverse/nflfastR hex.

- [ ] **Step 1: Write failing test** (PNG + CSV emitted).

- [ ] **Step 2: Implement `write_calibration`** — import or copy from CFB
  `figures.py`; swap the logo asset.

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): plotnine calibration figures + data tables
  ```

---

## Phase 9 — CLI

### Task 9.1: `nfl_cli.py` — subcommand dispatch

Same structure as CFB `cli.py`: `ingest | train-ep | train-wp | train-qbr | validate | figures`.

- [ ] **Step 1: Write failing test** (all subcommands present).

- [ ] **Step 2: Implement `build_parser` + `main`.**

- [ ] **Step 3: Commit**

  ```
  feat(nfl-models): CLI subcommand dispatch
  ```

---

## Phase 10 — sdv-py handoff (manual, reviewed)

This phase is documentation + a manual copy into sdv-py; never automated.
Performed once all Phases 1–9 pass the parity gate.

### Task 10.1: Handoff runbook

- [ ] **Step 1: Write `python/model_training/HANDOFF.md`** in the target repo.

  Contents (adapt from CFB Track 1 Task 10.1 runbook):

  ```markdown
  # NFL model handoff to sdv-py (manual, reviewed)

  After full-history training + parity gate passes:

  1. Validate each retrained model vs the shipped one (held-out season 2024):
     `uv run python -m model_training.nfl_cli validate --model <new>.ubj --ref <shipped>.ubj`
  2. Copy under review (open a sdv-py PR; never auto-overwrite):
     - `ep_model.ubj`, `wp_spread.ubj`, `qbr_model.ubj` -> `sportsdataverse/nfl/models/`
  3. Re-run sdv-py's NFL tests; confirm EPA/WPA/QBR on a known game stay within tolerance.
  4. Bump the bundled-model note in sdv-py's CHANGELOG + README.
  ```

- [ ] **Step 2: Commit**

  ```
  docs(nfl-models): sdv-py handoff runbook
  ```

---

## Phase 0 gate checklist

Before any Phase 1 implementation begins, all of the following must be resolved:

- [ ] Target repo created and confirmed (Task 0.0)
- [ ] NFL model introspection results confirmed vs spec (Task 0.1)
- [ ] nflfastR training scripts found (Task 0.2) **OR** explicitly declared
  "not findable" (→ reconstruction path, requires sub-spec)
- [ ] NFL hyperparameters confirmed (Task 0.2)
- [ ] nflverse-pbp column audit completed (Task 0.3)
- [ ] Data source decision made: Option A or Option B (Task 0.3)
- [ ] ESPN NFL QBR endpoint confirmed available (Task 0.4)
- [ ] Phase 0 findings record written (Task 0.5)
- [ ] Spec updated with confirmed findings (Task 0.5)
- [ ] Phase 1+ plan revised as needed based on findings (Task 0.5)

---

## Notes

- All commits use conventional-commits format. No AI co-author trailers.
- All Phase 1+ tasks are structured to follow the TDD pattern from CFB Track 1:
  write a failing test → run it → implement → run to pass → commit.
- The `xgboost` parameter dicts in `constants.py` use the **confirmed
  nflfastR hyperparameters** (see UPDATE section). Phase 0 Task 0.2 is
  resolved — the recipe is `fastrmodels/data-raw/models.R`.
- The file-structure table from the spec §9 (NFL analogue modules) governs
  the one-responsibility-per-file convention. No other files should be created
  without a corresponding spec update.

---

## UPDATE — recipe found; Phase 0 mostly resolved (see spec §12)

`fastrmodels/data-raw/models.R` is the canonical nflfastR build script. **Use these exact
hyperparameters in `constants.py`** (no longer placeholders):

- **NFL EP** — `multi:softprob`, `num_class=7`, `eta=0.025, gamma=1, subsample=0.8,
  colsample_bytree=0.8, max_depth=5, min_child_weight=1`, `nrounds=525`, weight `Total_W_Scaled`,
  `set.seed(2013)`. **18 features** (`half_seconds_remaining, yardline_100, home, retractable, dome,
  outdoors, ydstogo, era0-4, down1-4, posteam_timeouts_remaining, defteam_timeouts_remaining`).
- **NFL WP-spread** — `binary:logistic`, `eta=0.05, gamma=.79012017, subsample=0.9224245,
  colsample_bytree=5/12, max_depth=5, min_child_weight=7`, `nrounds=534`, `monotone_constraints
  "(0,0,0,0,0,1,1,-1,-1,-1,1,-1)"`. **12 features** (`receive_2h_ko, spread_time, home,
  half_seconds_remaining, game_seconds_remaining, Diff_Time_Ratio, score_differential, down, ydstogo,
  yardline_100, posteam_timeouts_remaining, defteam_timeouts_remaining`).
- **NFL WP-naive** — drop `spread_time` (11 feats), `eta=0.2, gamma=0, subsample=0.8,
  colsample_bytree=0.8, max_depth=4, min_child_weight=1`, `nrounds=65`.

**Critical scope change:** sdv-py's bundled `nfl/models/*.ubj` are **CFB-model copies** (8/13/6 feats,
3675/760/45 trees), NOT real NFL models. So this track must ALSO add a phase to **rewire
`NFLPlayProcess`'s feature contract** from the CFB 8/13-feat shape to the NFL 18/12-feat shape — the
retrained `.ubj` cannot drop in without it. Add this as a new high-priority phase before handoff.

**Remaining Phase-0 reads:** nflfastR `R/helper_add_ep_wp.R` (`make_model_mutations`,
`prepare_wp_data`: era buckets, roof one-hot, `spread_time`, `Diff_Time_Ratio`) +
`helper_add_cp_cpoe.R` (`prepare_cp_data`). Data: `nflfastR-data/models/cal_data.rds` (EP) +
`guga31bb/metrics/wp_tuning/cal_data.rds` (WP). Optional adjacent models in the same script: CP
(`binary:logistic`/560 — Track 5's true lineage) and FG (`mgcv::bam` GAM).
