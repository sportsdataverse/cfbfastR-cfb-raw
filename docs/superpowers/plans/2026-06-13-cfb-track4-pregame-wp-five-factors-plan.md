# CFB Modeling Suite — Track 4 (Pregame WP + Five-Factors) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port akeaswaran's pregame win-probability + Five-Factors team-rating pipeline from
`win-prob.ipynb` into `cfbfastR-cfb-raw/python/pregame_wp/`, with a CFBD data ingest path,
fully-tested factor builders, and a reproducible training harness.

**Architecture:** Five-Factors pipeline → per-game box score → `5FRDiff` → 10-tree XGBRegressor
→ normal-CDF WP. Data source is CFBD (not ESPN backfill). NOT a sdv-py bundled model.

**Tech Stack:** Python 3.11+, uv, pandas (training path mirrors the notebook's pandas idioms;
polars for any I/O heavy lifting), xgboost ≥2.0, scipy (normal CDF), pytest.

**Spec:** `docs/superpowers/specs/2026-06-13-cfb-track4-pregame-wp-five-factors-design.md`
(umbrella: `…-cfb-modeling-suite-program.md`).

**Commit convention:** Conventional Commits (`feat(pregame-wp):`, `fix(pregame-wp):`, etc.).
No AI co-author trailers (no `Co-Authored-By:` or similar on any commit in this project).

---

## File structure

```
python/pregame_wp/
    __init__.py
    constants.py         # weights, translate() bounds, success-rate thresholds
    ep_curve.py          # load ep.csv + punt_sr.csv; EqPPP lookup
    play_features.py     # EqPPP, play_successful, play_explosive column derivation
    team_stats.py        # generate_team_{play,drive,turnover,st}_stats
    five_factors.py      # create_*_index + calculate_five_factors_rating
    box_score.py         # calculate_box_score(game_id, year) → DataFrame
    training.py          # build_training_frame + outlier filter + train model
    predict.py           # generate_win_prob + predict_matchup
    talent.py            # calculate_roster_talent + calculate_returning_production
    data_ingest.py       # CFBD CSV load / API fetch helpers
    cli.py               # ingest | build-boxes | train | predict-matchup

python/pregame_wp/assets/
    ep.csv               # vendored from cfb-pbp-analysis/results/ep.csv
    punt_sr.csv          # vendored from cfb-pbp-analysis/results/punt_sr.csv
    fg_sr.csv            # vendored from cfb-pbp-analysis/results/fg_sr.csv

python/pregame_wp/models/
    pgwp_model.ubj       # trained model (committed under review after Phase 5)

tests/pregame_wp/
    test_constants.py
    test_ep_curve.py
    test_play_features.py
    test_team_stats.py
    test_five_factors.py
    test_box_score.py
    test_training.py
    test_predict.py
    test_talent.py
    test_cli.py

tests/fixtures/pregame_wp/
    README.md            # fixture provenance
    ep.csv               # same as assets (test independently)
    punt_sr.csv
    # small synthetic game fixture (game_plays.parquet, game_drives.parquet) for unit tests
```

Tests run with: `uv run pytest tests/pregame_wp/`. All CFBD-live tests are skipped
without `CFB_DATA_API_KEY` in the environment.

---

## PHASE 0 — Extract, verify, and resolve the Five-Factors formulas

**This is a research/spec-completion phase.** The formulas are extracted in the spec but several
ambiguities (OQ-1 through OQ-8) must be resolved and documented as decisions BEFORE any
implementation work in Phases 1–5 begins. Output: a decisions table appended to the spec and a
set of constants locked in `constants.py`.

### Task 0.1: Resolve OQ-5 (punt-return EP copy-paste bug) — the most impactful open question

**Files:** None (decision only; result propagates to `constants.py` + `team_stats.py`)

The issue: `generate_team_st_stats` in the notebook returns `'PuntReturnEqPPP': [punt_eqppp]`
and `'PuntReturnIsoPPP': [punt_isoppp]` (the *punter's* EP, not the *returner's* EP), so the
`PuntEqPPP - PuntReturnEqPPP` term in the field-position formula is always zero. The variables
`punt_ret_eqppp` / `punt_ret_isoppp` are computed but not used.

- [ ] **Step 1: Confirm the bug by reading the field-position formula in the spec (§4.5) and
  tracing the `PuntReturnEqPPP` assignment in `generate_team_st_stats`.**

  Expected: confirmed — the return variable is `punt_eqppp` (offense), not `punt_ret_eqppp`
  (defense).

- [ ] **Step 2: Decide faithfulness vs correctness.**

  Options:
  - **A. Faithful port** — preserve the bug; `PuntEqPPP - PuntReturnEqPPP` = 0 always. The
    field-position factor's punt term contributes nothing; the model was trained with this.
  - **B. Corrected port** — use `punt_ret_eqppp` for `PuntReturnEqPPP` as clearly intended.
    This changes the field-position factor and would require retraining from scratch to match
    the historical notebook's `pgwp_model.model`.

  **Recommended decision:** Option A (faithful port) for Phase 3 (parity target). Document in
  `constants.py` as a known bug. Option B can be explored as an extension in Phase 5.

- [ ] **Step 3: Write the decision into the spec (append to §11 open questions with resolution).**

  Add a "Resolved" column to the OQ table; mark OQ-5 as resolved with the chosen option.

- [ ] **Step 4: Commit**

  ```bash
  git add docs/superpowers/specs/2026-06-13-cfb-track4-pregame-wp-five-factors-design.md
  git commit -m "docs(pregame-wp): resolve OQ-5 punt-return EP bug (faithful port decision)"
  ```

### Task 0.2: Resolve OQ-7 (mu/std normalization) and OQ-3 (AvgEqPPP vs IsoPPP)

**Files:** None (decisions only)

- [ ] **Step 1: Decide mu/std approach (OQ-7).**

  The notebook computes `mu = preds.mean()` and `std = preds.std()` from the 20% test-split
  predictions. Options:
  - **A. Replicate notebook exactly** — random split with `seed=42` (add a seed the notebook
    lacks); store `mu` and `std` as model metadata alongside the `.ubj`.
  - **B. Fix mu=0** — point-differential has zero mean by symmetry (each game has one positive
    and one negative entry). Set `mu = 0.0` and derive `std` from the full training predictions.

  **Recommended decision:** Option B for a cleaner implementation; document that the notebook
  used test-split statistics, which is non-reproducible without a fixed seed.

- [ ] **Step 2: Decide explosiveness factor (OQ-3).**

  The notebook uses `AvgEqPPPDiff` (mean EqPPP across ALL offensive plays). `IsoPPPDiff` (mean
  EqPPP on successful plays only) is also computed and in the `inputs` list but not in the index
  function. Preserve as-is (`AvgEqPPPDiff`) — this is what the model was trained on.

- [ ] **Step 3: Append both resolutions to the spec.**

- [ ] **Step 4: Commit**

  ```bash
  git add docs/superpowers/specs/2026-06-13-cfb-track4-pregame-wp-five-factors-design.md
  git commit -m "docs(pregame-wp): resolve OQ-3 (AvgEqPPP) and OQ-7 (mu/std normalization)"
  ```

### Task 0.3: Vendor lookup tables + scaffold package

**Files:**
- Create: `python/pregame_wp/assets/ep.csv`, `punt_sr.csv`, `fg_sr.csv`
- Create: `python/pregame_wp/__init__.py`
- Create: `tests/pregame_wp/__init__.py`, `tests/fixtures/pregame_wp/README.md`
- Modify: `pyproject.toml` (add `pregame-wp` dep group)

- [ ] **Step 1: Copy lookup tables from cfb-pbp-analysis**

  ```bash
  mkdir -p python/pregame_wp/assets tests/pregame_wp tests/fixtures/pregame_wp
  cp ../cfb-pbp-analysis/results/ep.csv python/pregame_wp/assets/
  cp ../cfb-pbp-analysis/results/punt_sr.csv python/pregame_wp/assets/
  cp ../cfb-pbp-analysis/results/fg_sr.csv python/pregame_wp/assets/
  cp python/pregame_wp/assets/ep.csv tests/fixtures/pregame_wp/
  cp python/pregame_wp/assets/punt_sr.csv tests/fixtures/pregame_wp/
  ```

- [ ] **Step 2: Create package scaffold**

  `python/pregame_wp/__init__.py`:
  ```python
  """CFB pregame win-probability + Five-Factors team ratings (Track 4)."""
  __version__ = "0.1.0"
  ```

- [ ] **Step 3: Add dep group to pyproject.toml**

  ```toml
  [dependency-groups]
  pregame-wp = ["scipy>=1.10", "cfbd>=1.0"]
  ```
  (`xgboost` is already a runtime dep from Track 1; `pandas` / `numpy` are already present.)

- [ ] **Step 4: Sync and verify**

  ```bash
  uv sync --all-groups
  uv run python -c "from pregame_wp import __version__; print(__version__)"
  ```
  Expected: `0.1.0`

- [ ] **Step 5: Write fixture provenance README**

  `tests/fixtures/pregame_wp/README.md`:
  ```markdown
  # pregame_wp test fixtures

  - `ep.csv` — EP curve (101 rows, yardlines 0-100) from cfbscrapR/cfbfastR EP model.
    Source: akeaswaran/cfb-pbp-analysis/results/ep.csv. Used as indexed lookup for EqPPP.
  - `punt_sr.csv` — Expected punt net by yardline (100 rows). Source: same repo.
  - Synthetic game fixtures (when added): hand-constructed plays/drives for factor unit tests.
  ```

- [ ] **Step 6: Commit**

  ```bash
  git add python/pregame_wp/ tests/pregame_wp/ tests/fixtures/pregame_wp/ pyproject.toml uv.lock
  git commit -m "feat(pregame-wp): scaffold package + vendor lookup tables"
  ```

---

## PHASE 1 — `ep_curve.py` + `constants.py`

### Task 1.1: EP curve loader

**Files:**
- Create: `python/pregame_wp/ep_curve.py`
- Test: `tests/pregame_wp/test_ep_curve.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # tests/pregame_wp/test_ep_curve.py
  from pregame_wp.ep_curve import load_ep_curve, load_punt_sr, ep_at, eqppp

  def test_ep_curve_has_101_entries():
      ep = load_ep_curve()
      assert len(ep) == 101  # yardlines 0-100

  def test_ep_at_midfield_is_positive():
      ep = load_ep_curve()
      assert ep_at(ep, 50) > 0  # possession at own 50 is positive EP

  def test_eqppp_10_yard_gain_from_20():
      ep = load_ep_curve()
      val = eqppp(ep, yard_line=20, yards_gained=10)
      # EP should increase from yard_line 20 to 30
      assert val == ep_at(ep, 30) - ep_at(ep, 20)

  def test_eqppp_clamps_at_100():
      ep = load_ep_curve()
      # 90-yard gain from yl=80 should clamp to ep[100] - ep[80]
      val = eqppp(ep, yard_line=80, yards_gained=90)
      assert val == ep_at(ep, 100) - ep_at(ep, 80)

  def test_punt_sr_has_100_entries():
      punt_sr = load_punt_sr()
      assert len(punt_sr) == 100
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/pregame_wp/test_ep_curve.py -v`
  Expected: FAIL (`No module named 'pregame_wp.ep_curve'`)

- [ ] **Step 3: Implement `ep_curve.py`**

  ```python
  # python/pregame_wp/ep_curve.py
  """EP curve + punt success-rate lookup tables (from results/ep.csv + punt_sr.csv)."""
  from __future__ import annotations
  from pathlib import Path
  import pandas as pd

  _ASSETS = Path(__file__).parent / "assets"


  def load_ep_curve() -> list[float]:
      """Return EP values indexed by yardline (ep[yardline], len=101)."""
      df = pd.read_csv(_ASSETS / "ep.csv", encoding="utf-8")
      return df["ep"].tolist()


  def load_punt_sr() -> dict[int, float]:
      """Return {yardline: ExpPuntNet} mapping (yardlines 1-100)."""
      df = pd.read_csv(_ASSETS / "punt_sr.csv", encoding="utf-8")
      return dict(zip(df["Yardline"].astype(int), df["ExpPuntNet"]))


  def ep_at(ep: list[float], yardline: int) -> float:
      return ep[max(0, min(100, int(yardline)))]


  def eqppp(ep: list[float], yard_line: int, yards_gained: int) -> float:
      """EqPPP = EP(yl + yards) - EP(yl), clamped to [0, 100]."""
      return ep_at(ep, yard_line + yards_gained) - ep_at(ep, yard_line)


  def determine_kick_ep(ep: list[float], kick_yardline: int, distance: int, return_yards: int) -> float:
      """Net EP value for a kick play: EP(land) - EP(kick) - EP(return)."""
      return (ep_at(ep, kick_yardline + distance)
              - ep_at(ep, kick_yardline)
              - ep_at(ep, return_yards))
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/pregame_wp/test_ep_curve.py -v`
  Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

  ```bash
  git add python/pregame_wp/ep_curve.py tests/pregame_wp/test_ep_curve.py
  git commit -m "feat(pregame-wp): EP curve + punt SR loader with lookup helpers"
  ```

### Task 1.2: Constants

**Files:**
- Create: `python/pregame_wp/constants.py`
- Test: `tests/pregame_wp/test_constants.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # tests/pregame_wp/test_constants.py
  from pregame_wp import constants as C

  def test_factor_weights_sum_to_one():
      total = C.EFF_WEIGHT + C.EXPL_WEIGHT + C.FIN_DRV_WEIGHT + C.FLD_POS_WEIGHT + C.TRNOVR_WEIGHT
      assert abs(total - 1.0) < 1e-9

  def test_outlier_thresholds_defined():
      assert C.OUTLIER_Z_5FR == 3.2
      assert C.OUTLIER_Z_PTS == 3.0

  def test_success_rate_thresholds():
      # D1=0.5, D2=0.7, D4=1.0
      assert C.SR_DOWN1 == 0.5
      assert C.SR_DOWN2 == 0.7
      assert C.SR_DOWN4 == 1.0

  def test_scoring_opp_threshold():
      # drives reaching (start_yardline + yards) >= 60 are scoring opportunities
      assert C.SCORING_OPP_THRESHOLD == 60
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/pregame_wp/test_constants.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `constants.py`**

  ```python
  # python/pregame_wp/constants.py
  """All numeric constants for the Five-Factors system.

  Bug note (OQ-5): PuntReturnEqPPP is assigned punt_eqppp (not punt_ret_eqppp)
  in the notebook's generate_team_st_stats, making PuntEqPPP - PuntReturnEqPPP = 0
  always. This faithful port preserves that behavior. See Phase 0 decision.

  Mu/std note (OQ-7): WP conversion uses mu=0.0 (point-diff is symmetric around
  zero) and std derived from full-training-set predictions. The notebook used
  test-split prediction statistics, which is non-reproducible without a fixed seed.
  """
  from __future__ import annotations

  # --- 5FR weights ---
  EFF_WEIGHT = 0.35
  EXPL_WEIGHT = 0.30
  FIN_DRV_WEIGHT = 0.15
  FLD_POS_WEIGHT = 0.10
  TRNOVR_WEIGHT = 0.10

  # --- translate() domains (inMin, inMax, outMin, outMax) ---
  EFF_DOMAIN = (-1.0, 1.0, 0.0, 10.0)         # OffSRDiff -> 0-10
  # Expl domain derived at runtime from pbp_data.EqPPP.min/max
  FIN_DRV_PPD_DOMAIN = (-7.0, 7.0, 0.0, 3.5)
  FIN_DRV_RATE_DOMAIN = (-1.0, 1.0, 0.0, 4.0)
  FIN_DRV_SR_DOMAIN = (-1.0, 1.0, 0.0, 2.5)
  FLD_POS_QUANT_DOMAIN = (-10.0, 10.0, 0.0, 10.0)
  TRNOVR_LUCK_DOMAIN = (-5.0, 5.0, 0.0, 3.0)
  TRNOVR_SACK_DOMAIN = (-1.0, 1.0, 0.0, 3.0)
  TRNOVR_HAVOC_DOMAIN = (-1.0, 1.0, 0.0, 4.0)

  # --- field position sub-factor weights (used in the quant formula) ---
  FP_SR_WEIGHT = 0.37
  FP_TO_WEIGHT = 0.21
  FP_KICK_WEIGHT = 0.22
  FP_PUNT_WEIGHT = 0.20

  # --- success rate thresholds (down-specific) ---
  SR_DOWN1 = 0.5
  SR_DOWN2 = 0.7
  SR_DOWN4 = 1.0
  EXPLOSIVE_THRESHOLD = 15  # yards for play_explosive

  # --- scoring opportunity threshold ---
  SCORING_OPP_THRESHOLD = 60  # start_yardline + yards >= 60

  # --- expected TO formula weights ---
  EXP_TO_INT_WEIGHT = 0.22
  EXP_TO_FUM_WEIGHT = 0.49

  # --- kickoff thresholds ---
  KICKOFF_NET_SUCCESS = 40   # yards
  KICKOFF_RETURN_SUCCESS = 24  # yards
  TOUCHBACK_RETURN_YARDS = 25
  PUNT_TOUCHBACK_RETURN_YARDS = 20

  # --- training / outlier ---
  OUTLIER_Z_5FR = 3.2
  OUTLIER_Z_PTS = 3.0
  TRAIN_SPLIT = 0.80
  XGB_N_ESTIMATORS = 10
  XGB_SEED = 123

  # --- WP normalization (OQ-7 resolution: mu=0, std from full training preds) ---
  WP_MU = 0.0  # symmetric by construction; override with full-training-set std at train time

  # --- recruiting talent floor (2nd percentile of FBS ratings) ---
  TALENT_FCS_PERCENTILE = 0.02
  RETURNING_PROD_FLOOR_PERCENTILE = 0.02
  PRESEASON_WEEKS = 4  # weeks <= 4 trigger returning-production adjustment

  # --- HFA adjustments ---
  HFA_NORMAL = 2.5
  HFA_COVID = 1.0
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/pregame_wp/test_constants.py -v`
  Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

  ```bash
  git add python/pregame_wp/constants.py tests/pregame_wp/test_constants.py
  git commit -m "feat(pregame-wp): factor constants + translate() bounds + training params"
  ```

---

## PHASE 2 — `play_features.py` (per-play derived columns)

### Task 2.1: play_successful and play_explosive

**Files:**
- Create: `python/pregame_wp/play_features.py`
- Test: `tests/pregame_wp/test_play_features.py`

- [ ] **Step 1: Write the failing test**

  ```python
  # tests/pregame_wp/test_play_features.py
  import pandas as pd
  from pregame_wp.play_features import add_play_features

  ST_TYPES = ["Kickoff", "Punt", "Field Goal Good"]
  BAD_TYPES = ["Interception", "Sack", "Fumble Recovery (Opponent)"]

  def _play(play_type, down, distance, yards):
      return {"play_type": play_type, "down": down, "distance": distance, "yards_gained": yards,
              "yard_line": 20}

  def test_down1_50pct_is_successful():
      df = pd.DataFrame([_play("Rush", 1, 10, 5)])  # 5 >= 0.5*10
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_successful"].iloc[0] == True

  def test_down1_below_50pct_is_not_successful():
      df = pd.DataFrame([_play("Rush", 1, 10, 4)])  # 4 < 5
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_successful"].iloc[0] == False

  def test_down2_70pct_is_successful():
      df = pd.DataFrame([_play("Rush", 2, 10, 7)])  # 7 >= 0.7*10
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_successful"].iloc[0] == True

  def test_down3_not_successful_by_default():
      # 3rd down: default is False regardless of yards (not in np.select conditions)
      df = pd.DataFrame([_play("Rush", 3, 5, 100)])
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_successful"].iloc[0] == False

  def test_explosive_15_yards():
      df = pd.DataFrame([_play("Rush", 1, 10, 15)])
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_explosive"].iloc[0] == True

  def test_bad_type_not_explosive():
      df = pd.DataFrame([_play("Interception", 1, 10, 15)])
      out = add_play_features(df, [], ST_TYPES, BAD_TYPES)
      assert out["play_explosive"].iloc[0] == False

  def test_eqppp_computed_for_off_play():
      from pregame_wp.ep_curve import load_ep_curve
      ep = load_ep_curve()
      df = pd.DataFrame([_play("Rush", 1, 10, 10)])  # yl=20, gain=10 -> ep[30]-ep[20]
      out = add_play_features(df, ep, ST_TYPES, BAD_TYPES)
      assert abs(out["EqPPP"].iloc[0] - (ep[30] - ep[20])) < 1e-9
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/pregame_wp/test_play_features.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `play_features.py`**

  Per spec §4.1: `play_successful` uses `np.select` with the five conditions from cell 22.
  The **3rd down condition is intentionally absent** (default = False).

  ```python
  # python/pregame_wp/play_features.py
  """Per-play derived columns: EqPPP, play_successful, play_explosive.

  Faithful port of win-prob.ipynb cells 20 and 22.

  Note on play_successful (OQ-2): 3rd-down plays default to False unless they convert
  (which would be recorded as a 1st-down play in the next sequence). This matches
  the notebook's np.select conditions exactly.
  """
  from __future__ import annotations
  import numpy as np
  import pandas as pd

  from .constants import SR_DOWN1, SR_DOWN2, SR_DOWN4, EXPLOSIVE_THRESHOLD
  from .ep_curve import eqppp as _eqppp


  def add_play_features(
      df: pd.DataFrame,
      ep_data: list[float],
      st_types: list[str],
      bad_types: list[str],
  ) -> pd.DataFrame:
      df = df.copy()
      # --- play_explosive ---
      df["play_explosive"] = np.select(
          [
              df["play_type"].isin(bad_types),
              df["play_type"].isin(st_types),
              df["yards_gained"] >= EXPLOSIVE_THRESHOLD,
          ],
          [False, False, True],
          default=False,
      )
      # --- play_successful ---
      df["play_successful"] = np.select(
          [
              df["play_type"].isin(bad_types),
              df["play_type"].isin(st_types),
              (df["down"] == 1) & (df["yards_gained"] >= SR_DOWN1 * df["distance"]),
              (df["down"] == 2) & (df["yards_gained"] >= SR_DOWN2 * df["distance"]),
              (df["down"] >= 4) & (df["yards_gained"] >= SR_DOWN4 * df["distance"]),
          ],
          [False, False, True, True, True],
          default=False,
      )
      # --- EqPPP (zero for ST plays; uses ep_data list if provided) ---
      if ep_data:
          df["EqPPP"] = df.apply(
              lambda x: 0.0
              if x["play_type"] in st_types
              else _eqppp(ep_data, int(x["yard_line"]), int(x["yards_gained"])),
              axis=1,
          )
      return df
  ```

- [ ] **Step 4: Run test to verify it passes**

  Run: `uv run pytest tests/pregame_wp/test_play_features.py -v`
  Expected: PASS (7 tests)

- [ ] **Step 5: Commit**

  ```bash
  git add python/pregame_wp/play_features.py tests/pregame_wp/test_play_features.py
  git commit -m "feat(pregame-wp): play_successful / play_explosive / EqPPP column derivation"
  ```

---

## PHASE 3 — `team_stats.py` (per-team game box statistics)

> **Dependency:** Phases 0 + 2 must be complete. The exact `PuntReturnEqPPP` behavior is
> controlled by the Phase 0 OQ-5 decision — implement per the resolved decision.

### Task 3.1: play-level team stats (OffSR, OffER, IsoPPP, AvgEqPPP)

**Files:**
- Create: `python/pregame_wp/team_stats.py`
- Test: `tests/pregame_wp/test_team_stats.py`

- [ ] **Step 1: Write the failing test (synthetic play frame)**

  Create a minimal game fixture (5 offensive plays, 2 ST plays) and assert the expected stat
  values by hand.

  ```python
  # tests/pregame_wp/test_team_stats.py
  import pandas as pd
  import numpy as np
  from pregame_wp.team_stats import generate_team_play_stats

  OFF_TYPES = ["Rush", "Pass Reception", "Pass Incompletion", "Rushing Touchdown"]
  ST_TYPES = ["Kickoff", "Punt"]

  def _make_plays():
      return pd.DataFrame([
          # Rush D1 gain 6 of 10 (successful), EqPPP=0.5, not explosive
          {"play_type":"Rush","offense":"A","defense":"B","down":1,"distance":10,
           "yards_gained":6,"EqPPP":0.5,"play_successful":True,"play_explosive":False},
          # Rush D2 gain 3 of 7 (not successful: 3 < 0.7*7=4.9), EqPPP=0.2
          {"play_type":"Rush","offense":"A","defense":"B","down":2,"distance":7,
           "yards_gained":3,"EqPPP":0.2,"play_successful":False,"play_explosive":False},
          # Pass D1 gain 15 (explosive, successful), EqPPP=1.2
          {"play_type":"Pass Reception","offense":"A","defense":"B","down":1,"distance":5,
           "yards_gained":15,"EqPPP":1.2,"play_successful":True,"play_explosive":True},
          # Kickoff (ST - excluded from OffSR/OffER), EqPPP=0.0
          {"play_type":"Kickoff","offense":"A","defense":"B","down":0,"distance":0,
           "yards_gained":60,"EqPPP":0.0,"play_successful":False,"play_explosive":False},
          # Rush D3 gain 0 (not successful by default), EqPPP=-0.1
          {"play_type":"Rush","offense":"A","defense":"B","down":3,"distance":5,
           "yards_gained":0,"EqPPP":-0.1,"play_successful":False,"play_explosive":False},
      ])


  def test_off_sr_excludes_st():
      df = _make_plays()
      result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
      # 4 off plays (exc. Kickoff): 2 successful / 4 = 0.50
      assert abs(result["OffSR"].iloc[0] - 0.50) < 1e-9

  def test_off_er_explosive_rate():
      df = _make_plays()
      result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
      # 1 explosive out of 4 = 0.25
      assert abs(result["OffER"].iloc[0] - 0.25) < 1e-9

  def test_avg_eqppp():
      df = _make_plays()
      result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
      # mean of [0.5, 0.2, 1.2, -0.1] = 0.45
      assert abs(result["AvgEqPPP"].iloc[0] - 0.45) < 1e-9

  def test_iso_ppp_only_successful():
      df = _make_plays()
      result = generate_team_play_stats(df, "A", OFF_TYPES, ST_TYPES)
      # successful plays: EqPPP [0.5, 1.2], mean = 0.85
      assert abs(result["IsoPPP"].iloc[0] - 0.85) < 1e-9
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/pregame_wp/test_team_stats.py::test_off_sr_excludes_st -v`
  Expected: FAIL

- [ ] **Step 3: Implement `generate_team_play_stats`**

  Port of `generate_team_play_stats` from cell 24 — as per spec §4.2.

  The implementation is a direct port of the notebook function; details omitted here but
  **must implement per Phase 0's confirmed formula for `OffSR` (spec §4.2) and `IsoPPP`
  (spec §4.3)**. Use `pandas` matching the notebook idiom.

- [ ] **Step 4: Run all team_stats play-level tests**

  Run: `uv run pytest tests/pregame_wp/test_team_stats.py -k "off_sr or off_er or avg_eq or iso_ppp" -v`
  Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

  ```bash
  git add python/pregame_wp/team_stats.py tests/pregame_wp/test_team_stats.py
  git commit -m "feat(pregame-wp): generate_team_play_stats (OffSR, OffER, IsoPPP, AvgEqPPP)"
  ```

### Task 3.2: drive-level team stats (OppRate, OppEff, OppPPD, OppSR, FP)

**Files:**
- Modify: `python/pregame_wp/team_stats.py`, `tests/pregame_wp/test_team_stats.py`

- [ ] **Step 1: Write failing tests for `generate_team_drive_stats`**

  Hand-build 3 drives: 2 that reach `start_yardline + yards >= 60` (scoring opps), 1 that
  doesn't. Assert `OppRate`, `OppEff`, `OppPPD`, `FP`.

- [ ] **Step 2: Implement `generate_team_drive_stats`**

  Port of cell 24 `generate_team_drive_stats`. Use `SCORING_OPP_THRESHOLD = 60` from constants.
  **No formula ambiguities here** — all formulas are explicit in the spec (§4.4).

- [ ] **Step 3: Run tests and commit**

  ```bash
  git add python/pregame_wp/team_stats.py tests/pregame_wp/test_team_stats.py
  git commit -m "feat(pregame-wp): generate_team_drive_stats (OppRate, OppEff, OppPPD, FP)"
  ```

### Task 3.3: turnover stats (ExpTO, ActualTO, HavocRate, SackRate)

**Files:**
- Modify: `python/pregame_wp/team_stats.py`, `tests/pregame_wp/test_team_stats.py`

- [ ] **Step 1: Write failing tests for `generate_team_turnover_stats`**

  Build plays with `"Pass Incompletion"` + `"broken up"` in text, `"Interception"`, and
  `"Fumble Recovery (Opponent)"`. Assert ExpTO formula: `0.22*(PDs+INTs) + 0.49*fums`.

- [ ] **Step 2: Implement per spec §4.7**

  Port cell 24 `generate_team_turnover_stats` + `calculate_havoc_rate` + `calculate_sack_rate`.

- [ ] **Step 3: Run tests and commit**

  ```bash
  git add python/pregame_wp/team_stats.py tests/pregame_wp/test_team_stats.py
  git commit -m "feat(pregame-wp): turnover stats (ExpTO, ActualTO, HavocRate, SackRate)"
  ```

### Task 3.4: special-teams stats (kick/punt SR + EP, faithful OQ-5 behavior)

**Files:**
- Modify: `python/pregame_wp/team_stats.py`, `tests/pregame_wp/test_team_stats.py`

> **Note:** Implement `generate_team_st_stats` per the Phase 0 OQ-5 decision (faithful port).
> The `PuntReturnEqPPP` column in the returned DataFrame uses `punt_eqppp` (not `punt_ret_eqppp`)
> matching the notebook's bug. Document this with a comment in code.

- [ ] **Step 1: Write failing tests for `generate_team_st_stats`**

  Build synthetic kick plays with regex-extractable `play_text` (e.g., `"kickoff for 65 yards"`).
  Assert `KickoffSR`, `KickoffEqPPP`, and verify `PuntReturnEqPPP == PuntEqPPP` (the known bug
  is tested explicitly so future developers see it's intentional).

- [ ] **Step 2: Implement per spec §4.6**

  Port `generate_team_st_stats`, `determine_kick_ep`, `determine_kick_return`,
  `determine_punt_return`, `is_punt_successful` from cells 22 and 24.

- [ ] **Step 3: Run tests and commit**

  ```bash
  git add python/pregame_wp/team_stats.py tests/pregame_wp/test_team_stats.py
  git commit -m "feat(pregame-wp): ST stats (kick/punt SR + EP; faithful OQ-5 port)"
  ```

---

## PHASE 4 — `five_factors.py` (factor index functions + 5FR)

### Task 4.1: the five factor index functions

**Files:**
- Create: `python/pregame_wp/five_factors.py`
- Test: `tests/pregame_wp/test_five_factors.py`

- [ ] **Step 1: Write the failing test — translate() helper first**

  ```python
  # tests/pregame_wp/test_five_factors.py
  from pregame_wp.five_factors import translate, create_eff_index, create_expl_index
  from pregame_wp.five_factors import create_finish_drive_index, create_fp_index
  from pregame_wp.five_factors import create_turnover_index, calculate_five_factors_rating

  def test_translate_midpoint():
      # midpoint of [-1,1] should map to midpoint of [0,10] = 5
      assert translate(0.0, -1.0, 1.0, 0.0, 10.0) == 5.0

  def test_translate_min_max():
      assert translate(-1.0, -1.0, 1.0, 0.0, 10.0) == 0.0
      assert translate(1.0, -1.0, 1.0, 0.0, 10.0) == 10.0

  def test_eff_index_from_diff():
      row = type("R", (), {"OffSRDiff": 0.0})()
      assert create_eff_index(row) == 5.0

  def test_finish_drive_index_all_zero():
      row = type("R", (), {"OppPPDDiff": 0.0, "OppRateDiff": 0.0, "OppSRDiff": 0.0})()
      assert create_finish_drive_index(row) == (3.5/2 + 4.0/2 + 2.5/2)  # midpoints = 5.0

  def test_5fr_weights_sum():
      # If all indices = 5.0, 5FR = 5.0 (weighted sum with weights summing to 1)
      import pandas as pd
      row = pd.Series({
          "OffSRDiff": 0.0, "AvgEqPPPDiff": 0.0,
          "OppPPDDiff": 0.0, "OppRateDiff": 0.0, "OppSRDiff": 0.0,
          "ActualTODiff": 0.0, "Plays": 40,
          "KickoffEqPPP": 0.0, "KickoffReturnEqPPP": 0.0,
          "PuntEqPPP": 0.0, "PuntReturnEqPPP": 0.0,
          "ExpTO": 1.0, "ActualTO": 1.0,
          "SackRateDiff": 0.0, "HavocRateDiff": 0.0,
          "_eq_ppp_min": -2.0, "_eq_ppp_max": 2.0,
      })
      ffr = calculate_five_factors_rating(row)
      assert abs(ffr - 5.0) < 0.5  # won't be exactly 5.0 because field_pos uses quant=0 -> translate(0,-10,10,0,10)=5
  ```

- [ ] **Step 2: Run test to verify it fails**

  Run: `uv run pytest tests/pregame_wp/test_five_factors.py -v`
  Expected: FAIL

- [ ] **Step 3: Implement `five_factors.py` per spec §4.2–4.8**

  Port `translate`, `create_eff_index`, `create_expl_index`, `create_fp_index`,
  `create_finish_drive_index`, `create_turnover_index`, `calculate_five_factors_rating`.

  The `create_expl_index` function requires the global EqPPP min/max from the training PBP
  (`pbp_data.EqPPP.min()` / `pbp_data.EqPPP.max()`). In the notebook this is a global.
  In the port, pass it as a parameter (or embed in the row as `_eq_ppp_min` / `_eq_ppp_max`
  as done in the test above). **This is the only factor that requires a global data stat.**

  The `create_turnover_index` uses the per-team `ExpTO - ActualTO` (not a diff) per spec §4.7.

- [ ] **Step 4: Run tests**

  Run: `uv run pytest tests/pregame_wp/test_five_factors.py -v`
  Expected: PASS (5+ tests)

- [ ] **Step 5: Commit**

  ```bash
  git add python/pregame_wp/five_factors.py tests/pregame_wp/test_five_factors.py
  git commit -m "feat(pregame-wp): five-factor index functions + 5FR composite"
  ```

---

## PHASE 5 — `box_score.py`, `training.py`, `predict.py`, `talent.py`, `cli.py`

> **Dependency:** Phases 1–4 complete. These phases depend on CFBD data being available locally
> (via pre-staged CSVs or a CFBD API key). Tests in this phase use synthetic fixtures where
> possible and are skipped with `@pytest.mark.skipif(not CFBD_DATA_AVAILABLE)` for live-data
> steps.

### Task 5.1: `box_score.py` — full game box (port of cell 24 `calculate_box_score`)

**Files:**
- Create: `python/pregame_wp/box_score.py`
- Test: `tests/pregame_wp/test_box_score.py`

- [ ] **Step 1: Write failing test (synthetic fixture)**

  Build a minimal synthetic `(games, drives, pbp)` triple with known stats, call
  `calculate_box_score_from_frames`, and assert that 5FR and 5FRDiff have the right signs
  (the team with higher OffSR gets higher 5FR).

- [ ] **Step 2: Implement per spec §4.1 and cell 24**

  Implement `calculate_box_score_from_frames(game_data, game_drives, game_pbp, ep_data, punt_sr,
  eq_ppp_global_min, eq_ppp_global_max)` — takes pre-loaded data frames; the outer
  `calculate_box_score(game_id, year, ...)` wrapper loads from the data cache.

- [ ] **Step 3: Run tests and commit**

  ```bash
  git add python/pregame_wp/box_score.py tests/pregame_wp/test_box_score.py
  git commit -m "feat(pregame-wp): calculate_box_score (full game 5FR box)"
  ```

### Task 5.2: `training.py` — build stored_game_boxes, outlier filter, train

**Files:**
- Create: `python/pregame_wp/training.py`
- Test: `tests/pregame_wp/test_training.py`

- [ ] **Step 1: Write the failing test (synthetic box scores)**

  ```python
  # tests/pregame_wp/test_training.py
  import numpy as np
  import pandas as pd
  from pregame_wp.training import filter_outliers, train_pgwp_model

  def test_filter_outliers_removes_extreme_rows():
      rng = np.random.default_rng(0)
      df = pd.DataFrame({"5FRDiff": rng.normal(0, 2, 200), "PtsDiff": rng.normal(0, 14, 200)})
      # Add one extreme outlier
      df.loc[0, "5FRDiff"] = 100.0
      filtered = filter_outliers(df)
      assert len(filtered) < len(df)
      assert filtered["5FRDiff"].max() < 100.0

  def test_train_pgwp_model_returns_model_and_stats():
      rng = np.random.default_rng(1)
      df = pd.DataFrame({"5FRDiff": rng.normal(0, 2, 500), "PtsDiff": rng.normal(0, 14, 500)})
      model, mu, std = train_pgwp_model(df)
      assert model.n_estimators == 10
      assert mu == 0.0   # per OQ-7 resolution
      assert std > 0
  ```

- [ ] **Step 2: Implement per spec §5.1–5.2**

  ```python
  # python/pregame_wp/training.py
  # Implement filter_outliers (zscore thresholds from constants), train_pgwp_model
  # (XGBRegressor, n_estimators=10, seed=123). Per OQ-7: mu=0.0, std = std of
  # full training set predictions (not test split).
  ```

- [ ] **Step 3: Run tests and commit**

  ```bash
  git add python/pregame_wp/training.py tests/pregame_wp/test_training.py
  git commit -m "feat(pregame-wp): outlier filter + XGBRegressor training (1-feat, 10 trees)"
  ```

### Task 5.3: `talent.py`, `predict.py`, `cli.py`

These three modules implement the inference surface. All CFBD-live tests are skipped unless
`CFB_DATA_API_KEY` is set and CFBD data is locally staged.

- [ ] **Step 1: `talent.py` — implement `calculate_roster_talent` + `calculate_returning_production`
  per spec §6. Unit test with synthetic recruiting data (rolling 4-year mean, FCS floor).**

  ```bash
  git commit -m "feat(pregame-wp): roster talent + returning production helpers"
  ```

- [ ] **Step 2: `predict.py` — implement `generate_win_prob(game_id, year)` + `predict_matchup(...)`.
  Unit test `generate_win_prob` with a synthetic model + mu/std. Test that WP = 0.5 when
  predicted MOV = mu = 0.**

  ```bash
  git commit -m "feat(pregame-wp): generate_win_prob + predict_matchup inference path"
  ```

- [ ] **Step 3: `cli.py` — subcommands `ingest | build-boxes | train | predict-matchup`.
  Test that all subcommands parse; data-dependent tests skipped.**

  ```bash
  git commit -m "feat(pregame-wp): CLI (ingest / build-boxes / train / predict-matchup)"
  ```

---

## PHASE 6 — data ingest + integration smoke test (CFBD-gated)

### Task 6.1: CFBD data ingest

**Files:**
- Create: `python/pregame_wp/data_ingest.py`
- Test: `tests/pregame_wp/test_data_ingest.py`

These tests require `CFB_DATA_API_KEY` and are always skipped in CI.

- [ ] **Step 1: Implement `data_ingest.py`**

  ```python
  # python/pregame_wp/data_ingest.py
  # Fetches games/drives/pbp/recruiting/returning_production from CFBD API
  # for a given year range, caches as CSV or parquet in data/cfbd/{endpoint}/{year}.csv.
  # Reads from disk if already present (idempotent).
  ```

- [ ] **Step 2: Write a smoke test (skipped without API key)**

  ```python
  import os, pytest
  @pytest.mark.skipif(not os.environ.get("CFB_DATA_API_KEY"), reason="no CFBD key")
  def test_fetch_games_2019():
      from pregame_wp.data_ingest import fetch_games
      gm = fetch_games(2019)
      assert len(gm) > 500  # 2019 had ~900 FBS games
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add python/pregame_wp/data_ingest.py tests/pregame_wp/test_data_ingest.py
  git commit -m "feat(pregame-wp): CFBD data ingest (games / drives / pbp / recruiting / prod)"
  ```

### Task 6.2: End-to-end smoke with pre-staged data (CFBD-gated)

- [ ] **Step 1: Pre-stage a minimal CFBD dataset (one season, e.g. 2019) locally.**

- [ ] **Step 2: Write an end-to-end smoke test**

  ```python
  @pytest.mark.skipif(not CFBD_DATA_AVAILABLE, reason="no staged CFBD data")
  def test_end_to_end_2019(tmp_path):
      # ingest -> build_box_scores -> filter_outliers -> train_pgwp_model -> generate_win_prob
      # ...
      model, mu, std = train_pgwp_model(stored_game_boxes)
      wp, mov = generate_win_prob(401110865, 2019)  # 2019 Iron Bowl
      assert 0.0 < wp < 1.0
  ```

- [ ] **Step 3: Commit**

  ```bash
  git add tests/pregame_wp/test_e2e.py
  git commit -m "test(pregame-wp): end-to-end smoke (CFBD-gated, 2019 season)"
  ```

### Task 6.3: Commit trained model under review

Once the E2E smoke passes on a full dataset (2012–2020 or latest available):

- [ ] **Step 1: Train the model on the full dataset and save as UBJ**

  ```bash
  uv run python -m pregame_wp.cli train --seasons 2012:2020 --out python/pregame_wp/models/pgwp_model.ubj
  ```

- [ ] **Step 2: Verify model introspection matches spec (1 feature, 10 trees)**

  ```bash
  uv run python -c "
  import xgboost as xgb
  m = xgb.Booster(); m.load_model('python/pregame_wp/models/pgwp_model.ubj')
  print('feats:', m.num_features(), 'trees:', m.num_boosted_rounds())
  "
  ```
  Expected: `feats: 1  trees: 10`

- [ ] **Step 3: Commit (manual review, not automated)**

  ```bash
  git add python/pregame_wp/models/pgwp_model.ubj
  git commit -m "feat(pregame-wp): trained pgwp_model.ubj (1-feat, 10-tree XGBRegressor, 2012-2020)"
  ```

---

## Stage gating note

Unlike Track 1, Track 4 has no "Stage 1 faithful replica vs Stage 2 parity" distinction —
there is only one target model (the notebook's `pgwp_model.model`). The gate is:

- **Offline (Phases 0–4):** all factor unit tests pass on synthetic fixtures without CFBD data.
- **Integration (Phase 5–6, CFBD-gated):** E2E smoke test produces sensible WP values on
  known historical games; `predict_matchup("LSU", "Clemson", 2019, week=-1, games_to_consider=4)`
  returns WP ≈ 0.65–0.75 for LSU (consistent with the notebook's output at cell 83).

Run the full offline suite at any time:
```bash
uv run pytest tests/pregame_wp/ -v
```
All CFBD-live tests skip without staged data; all factor unit tests pass unconditionally.

---

## Open questions requiring Phase 0 resolution before implementation

| OQ | Blocks | Resolution in |
|---|---|---|
| OQ-5 (punt-return EP bug) | Phase 3 Task 3.4 | Task 0.1 |
| OQ-7 (mu/std normalization) | Phase 5 Task 5.2 | Task 0.2 |
| OQ-3 (AvgEqPPP vs IsoPPP) | Phase 4 | Task 0.2 (accept as-is) |

OQ-1 (ep.csv drift), OQ-2 (3rd-down success), OQ-4 (FP formula mixing), OQ-6 (TO formula
mixing), and OQ-8 (SoS double-counting) are low-impact and can be documented in code comments
without blocking implementation.
