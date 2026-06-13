# CFB Track 4 — Pregame Win Probability + Five-Factors Team Ratings: Design Spec

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repo:** `cfbfastR-cfb-raw` — new `python/pregame_wp/` package
- **Source of truth (notebook):** `../cfb-pbp-analysis/win-prob.ipynb` (akeaswaran/cfb-pbp-analysis)
- **Program:** **Track 4** of the CFB Modeling Suite (see `2026-06-13-cfb-modeling-suite-program.md`).

---

## 1. Goal

Port and productionize akeaswaran's pregame win-probability system from `win-prob.ipynb` into a
Python-native pipeline in `cfbfastR-cfb-raw`. The pipeline:

1. Builds per-team, per-game **Five-Factors Ratings (5FR)** from CFBD drives, games, and PBP —
   Bill Connelly's five factors (efficiency, explosiveness, field position, finishing drives,
   turnovers), extended with special-teams sub-factors and EP-curve lookups.
2. Computes **`5FRDiff = home_5FR − away_5FR`** as the single feature for the pregame model.
3. Trains a trivial **`XGBRegressor` (1 feat, 10 trees, `reg:squarederror`)** predicting game
   point differential (`PtsDiff`) from `5FRDiff`.
4. Converts point-differential predictions to win probability via a **normal CDF of the z-scored
   prediction**: `WP = Φ((pred − μ) / σ)` where `μ, σ` come from the training-set predictions.
5. Exposes a **`predict_matchup`** path that accepts season-averaged 5FR ratings (optionally
   SoS-adjusted, returning-production-adjusted, and HFA-adjusted) as the pregame input.

This is a **standalone system**: the model is NOT bundled into sdv-py (it is CFBD-sourced, not
ESPN-sourced, and operates at team-game grain rather than play grain). It lives in
`cfbfastR-cfb-raw/python/pregame_wp/` and is primarily an analysis/research artifact.

---

## 2. What the investigation established

### 2.1 The model is trivial

Introspecting `pgwp_model.model` (saved at Cell 96):

- `XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)`
- **Feature:** `5FRDiff` (scalar)
- **Target:** `PtsDiff` (signed point differential, home team's perspective)
- Trained on 80 % of outlier-filtered rows; test-set `preds` statistics (`mu`, `std`) are used
  at inference time as the z-score baseline.
- WP conversion: `z = (pred − mu) / std; WP = scipy.stats.norm.cdf(z)` — the normal CDF
  interprets "how many standard deviations above zero is the predicted point margin?" as a win
  probability.

This 10-tree model is not a production-grade pregame WP model; it is a demonstration that 5FR
explains point margin. **The real engineering is the Five-Factors pipeline.**

### 2.2 Five-Factors overview

The five factors and their weights in `calculate_five_factors_rating`:

```
5FR = 0.35 × Eff + 0.30 × Expl + 0.15 × FinDrv + 0.10 × FldPos + 0.10 × Trnovr
```

Each factor is computed from game-level box stats and then expressed as a scaled index (0–10
range, approximately). Differences (`XDiff = home_X − away_X`) flow into the scaling functions.
The full list of intermediate statistics computed per team per game is documented in §4.

---

## 3. Data sources and access implications

### 3.1 CFBD API (not ESPN)

**This track is entirely CFBD-sourced.** Unlike Tracks 1–3 (which read from the ESPN backfill's
`final.json`), Track 4 requires:

| CFBD endpoint | Purpose |
|---|---|
| `/drives` (year-by-year) | Drive stats: offense/defense, yards, scoring, start/end yardlines, drive result |
| `/games` (year-by-year) | Game metadata: home/away teams, final scores, season, week |
| `/plays` (year-by-year PBP) | Play-level: play type, yards gained, down, distance, yard line, play text |
| `/recruiting` (year-by-year) | Recruit ratings by committed school, used for `calculate_roster_talent` |
| `/returning_production` (year-by-year) | Overall returning-production percentage per team/year |

**The notebook reads pre-downloaded CSVs** (cell 4: `retrieveCfbDataFile(endpoint, year)`) from
`data/{endpoint}/{year}.csv`; there is no live-API call in the main training path. A CFBD data
ingest script is needed to populate those CSVs (or their parquet equivalents).

**CFBD API key is required** for bulk historical pulls. The `cfbd` Python client or direct HTTP
to `api.collegefootballdata.com` both work; the notebook used the client but pre-staged data
locally. This is a distinct data-access path from the ESPN backfill used by Tracks 1–3.

### 3.2 Auxiliary lookup tables (pre-computed, not re-derived)

The notebook loads two pre-computed lookups that the EP-curve and punt-SR computations depend on:

| File | Shape | Content |
|---|---|---|
| `results/ep.csv` | 101 rows × 2 cols (`yardline`, `ep`) | Expected points at each yard line 0–100, derived from cfbscrapR EP model. `ep_data` is loaded as a Python list indexed `ep_data[yardline]`. |
| `results/punt_sr.csv` | 100 rows × 2 cols (`Yardline`, `ExpPuntNet`) | Expected net punt yards from each yardline 1–100; used to classify a punt as "successful". |
| `results/fg_sr.csv` | 48 rows × 3 cols (`Distance`, `Accuracy`, `ExpFGValue`) | Expected field-goal accuracy and value by distance; **loaded but not consumed in the core 5FR computation**. |

These tables are pre-built from earlier notebook work (cfbscrapR EP model + historical punt/FG
distributions). They must be vendored into the repo.

**Open question OQ-1:** Are these lookup tables stable across retraining cycles? `ep.csv` is the
EP curve from the shipped cfbscrapR/cfbfastR EP model — it should match `ep_data` from sdv-py's
`ep_model.ubj` output at inference. A drift check should be added once the Track 1 EP model is
retrained.

---

## 4. Five-Factors computation (extracted from the notebook)

### 4.1 Input data preparation

**Games** (cell 11–12): join `game_id` field across years; clean `id`/`game_id` column;
add `drive_pts = max(off_score_delta, def_score_delta)` per drive; merge drives onto games
to get `home_team`/`away_team` on the drive record.

**Drive yardline alignment** (cell 13): away-team drives have their yardlines inverted:
`start_yardline = 100 - start_yardline` and `end_yardline = 100 - end_yardline` so that
all yardlines are from the *offense's* perspective (distance to own end zone). After inversion,
`scoring_opps = drives where (start_yardline + yards) >= 60` — drives that reached opponent's
40 yard line.

**PBP** (cell 14): rename `id_play → play_id`, `offense_play → offense`, `defense_play →
defense`; drop spread/over-under/clock columns from the per-play CFBD output.

**Play type filtering** (cell 18): define `st_types` (special-teams plays excluded from
scrimmage stats) and `ignore_types` (timeouts, end-of-half, penalties, etc.); define
`off_play_types = all types not in st_types or ignore_types`. Turnovers/fumbles/sacks have
`yards_gained` zeroed out (they don't contribute EPA-style credit to the ball-carrier).

**`EqPPP`** (cell 20): Expected Points per Play — the marginal EP change for a play:
```
EqPPP = ep_data[min(100, max(0, yard_line + yards_gained))] - ep_data[min(100, max(0, yard_line))]
```
Zero for special-teams plays. Uses `ep.csv` as the lookup table.

**`play_successful`** (cell 22):
```
True  if play_type in bad_types (INT/fumble/sack) → False (success = no turnover)
True  if down == 1  AND yards_gained >= 0.5 × distance
True  if down == 2  AND yards_gained >= 0.7 × distance
True  if down >= 4  AND yards_gained >= 1.0 × distance
False otherwise (down == 3 defaults to False — **see OQ-2**)
```

**Open question OQ-2:** The `np.select` in cell 22 has no explicit 3rd-down condition. On
3rd down (non-bad-type, non-ST), the default is `False`. This appears intentional — 3rd down
success is implicitly captured by the drive continuing (if they make it on 3rd they get a new
set of downs). But this means 3rd-down plays can never be `play_successful = True` unless they
convert (which would show up as a 1st-down play in the next sequence). This is consistent with
standard S&P+ success-rate definitions where 3rd down success = conversion. **Confirm this
intent before porting.**

**`play_explosive`** (cell 22): `True if yards_gained >= 15 AND play_type not in bad_types or st_types`, else `False`.

### 4.2 Factor 1 — Efficiency (Eff, 35 % weight)

**Intermediate stat:** `OffSR` = Offensive Success Rate

```python
OffSR = count(off_play_types where play_successful == True) / count(off_play_types)
```

Only plays in `off_play_types` are included; `st_types` and `ignore_types` are excluded.

**Index:**
```python
Eff = translate(OffSRDiff, -1, 1, 0, 10)
```
where `OffSRDiff = home_OffSR − away_OffSR`.

`translate(v, inMin, inMax, outMin, outMax)` is linear interpolation:
`outMin + ((v − inMin) / (inMax − inMin)) × (outMax − outMin)`.

**`OppSR`** (also in `inputs` list) = success rate *within scoring opportunities* (drives where
`start_yardline + yards >= 60`), computed per drive. This appears in `create_finish_drive_index`
as `OppSRDiff`, not separately in the efficiency factor.

### 4.3 Factor 2 — Explosiveness (Expl, 30 % weight)

**Intermediate stats:**
- `AvgEqPPP` = mean EqPPP across all offensive plays (including zero values).
- `IsoPPP` = mean EqPPP restricted to *successful plays only*
  (`EqPPP.mean()` of rows where `play_successful == True`).
- `OffER` = Explosive Rate = `count(play_explosive == True) / count(off_play_types)`.

**Index (uses `AvgEqPPPDiff`):**
```python
Expl = translate(AvgEqPPPDiff, ep_data_min - ep_data_max, ep_data_max - ep_data_min, 0, 10)
```

The bounds are derived from the global `pbp_data.EqPPP.min()` and `.max()` — meaning the
translate() domain is `[EqPPP_min - EqPPP_max, EqPPP_max - EqPPP_min]`.

**Open question OQ-3:** The explosiveness factor uses `AvgEqPPPDiff` (mean EqPPP differential),
NOT `IsoPPPDiff` or `OffERDiff`. The notebook plots `IsoPPP` as an input in the correlation
analysis (cell 31) and includes it in the `inputs` list, but `create_expl_index` explicitly uses
`AvgEqPPPDiff`. `IsoPPP` appears only in the Finishing Drives sub-stats and the correlation
comparison plots, not in any of the five `create_*_index` functions. This is potentially
surprising given the Bill Connelly framing of explosiveness as IsoPPP; document clearly.

### 4.4 Factor 3 — Finishing Drives (FinDrv, 15 % weight)

**Scoring opportunities** = drives where `start_yardline + yards >= 60` (reached or started
inside the opponent's 40-yard line).

**Intermediate stats:**
- `OppRate` = scoring-opportunity rate = `len(scoring_opps) / len(all_drives)`
- `OppEff` = efficiency within scoring opps = `scoring(scoring_opps) / len(scoring_opps)` (bool
  `scoring` field from CFBD drive result)
- `OppPPD` = points per scoring opportunity = `sum(drive_pts on scoring_opps) / len(scoring_opps)`
- `OppSR` = success rate within scoring opps (per cell 22's `calculate_success_in_scoring_opps`)

**Index (three translated sub-components, NOT separately scaled):**
```python
FinDrv = translate(OppPPDDiff, -7, 7, 0, 3.5) \
       + translate(OppRateDiff, -1, 1, 0, 4.0) \
       + translate(OppSRDiff,   -1, 1, 0, 2.5)
```

Note that `FinDrv` sums three sub-components (not 0–10 scaled overall); the maximum theoretical
value is `3.5 + 4.0 + 2.5 = 10.0`.

### 4.5 Factor 4 — Field Position (FldPos, 10 % weight)

**Intermediate stats:**
- `FP` = average drive start yardline (offense's perspective, post-inversion) =
  `mean(start_yardline)` across all drives for that team
- `ActualTO` = actual turnovers (INT + fumble-recovery-opponent)
- Kickoff special teams: `KickoffSR`, `KickoffEqPPP`, `KickoffReturnSR`,
  `KickoffReturnEqPPP` (see §4.6)
- Punt special teams: `PuntEqPPP`, `PuntReturnEqPPP` (see §4.6)

**Index (a weighted combination of five sub-factors, THEN translated):**
```python
quant = (OffSRDiff * 0.37) \
      + (ActualTODiff / Plays) * 0.21 \
      + (KickoffEqPPP - KickoffReturnEqPPP) * 0.22 \
      + (PuntEqPPP - PuntReturnEqPPP) * 0.20
FldPos = translate(quant, -10, 10, 0, 10)
```

Note: `OffSRDiff` appears again here (efficiency bleeds into field position); the kick-EP terms
are raw per-team values (not diffs, since team 1's kickoff EP *is* team 2's kickoff-return
context within the same game). `ActualTODiff / Plays` normalizes by the team's play count.

**Open question OQ-4:** The field-position formula uses the *game-level* `KickoffEqPPP` and
`PuntEqPPP` from `generate_team_st_stats` — these are per-*game* averages of the EP lookup, not
season-level. In the `predict_matchup` path, the season-level 5FR is the *average* over all
game-level box scores, so the EP deltas are implicitly season-averaged. This is coherent but
means the "field position" factor at inference time captures season-average special-teams EP net,
not a snapshot of the current game situation.

### 4.6 Special-teams sub-statistics

Both kickoff and punt sub-stats are computed from regex-parsed play texts (cells 24). These are
brittle text parsers and **not** sourced from structured fields.

**Kickoff stats** (regex: `'kickoff for (\d+) ya*r*ds'` on `play_text`):

| Stat | Formula |
|---|---|
| `KickoffSR` | Fraction of kickoffs by this team with `Net >= 40` yards (coverage success) |
| `KickoffReturnSR` | Fraction of kickoff returns against this team with `Return >= 24` yards (return success) |
| `KickoffEqPPP` | Mean of `determine_kick_ep(yardline, distance, return)` for this team's kickoffs |
| `KickoffIsoPPP` | Same but restricted to `Net >= 40` kickoffs |
| `KickoffReturnEqPPP` | Mean kick EP when this team is the *defense* (kickoff opponent) |
| `KickoffReturnIsoPPP` | Restricted to `Net >= 40` from the *defense's* perspective |

`determine_kick_ep(kick_yardline, distance, return_yards)`:
```python
ep_data[max(0, min(100, int(kick_yardline + distance)))]
- ep_data[max(0, min(100, kick_yardline))]
- ep_data[max(0, min(100, int(return_yards)))]
```

Touchbacks are set to 25 yards of return.

**Punt stats** (regex: `'punt for (\d+) ya*r*ds'` on `play_text`):

| Stat | Formula |
|---|---|
| `PuntSR` | Fraction of punts by this team where `Net >= ExpPuntNet[Yardline]` (from `punt_sr.csv`) |
| `PuntReturnSR` | Fraction of punts *against* this team where `Net < ExpPuntNet[Yardline]` (favorable return) |
| `PuntEqPPP` | Mean kick EP for this team's punts |
| `PuntIsoPPP` | Mean kick EP for this team's successful punts only |
| `PuntReturnEqPPP` | Mean kick EP when this team is the punt returner |
| `PuntReturnIsoPPP` | Mean kick EP for successful punt returns against this team |

The `EP` for punts uses the same `determine_kick_ep` function as kickoffs.

**Open question OQ-5:** There is a copy-paste bug in `generate_team_st_stats` for punt return
stats: `'PuntReturnEqPPP': [punt_eqppp]` and `'PuntReturnIsoPPP': [punt_isoppp]` both assign the
*punt* (offense) EP rather than the *punt return* (defense) EP. The variables `punt_ret_eqppp`
and `punt_ret_isoppp` are computed but not referenced in the return DataFrame. This means the
field-position formula's `PuntReturnEqPPP` is identical to `PuntEqPPP`, making the
`PuntEqPPP - PuntReturnEqPPP` term always zero. **Decide whether to preserve this bug (faithful
port) or fix it (correct calculation). Resolution required in Phase 0.**

### 4.7 Factor 5 — Turnovers (Trnovr, 10 % weight)

**Intermediate stats:**

`ExpTO` (expected turnovers, from `generate_team_turnover_stats`):
```python
team_pds  = pass incompletions with "broken up" in play_text, offense == team
team_ints = interception-type plays, offense == team
fum_plays = fumble-type plays (both teams)
ExpTO = 0.22 × (len(team_pds) + len(team_ints)) + 0.49 × len(fum_plays)
```

`ActualTO`:
```python
ActualTO = len(team_ints where offense == team)
         + len(fum_plays where offense == team AND play_type == "Fumble Recovery (Opponent)")
```

`HavocRate` = defensive havoc / disruptive plays rate:
```python
HavocRate = count(defense == team AND (broken-up pass OR fumble OR sack OR INT OR negative yards))
          / count(defense == team AND play_type in off_play_types)
```

`SackRate`:
```python
SackRate = count(defense == team AND play_type == "Sack") / count(defense == team AND off_play_types)
```

**Index (three sub-components, NOT separately translated to 0–10):**
```python
Trnovr = translate(ExpTO - ActualTO,  -5, 5, 0, 3.0) \
       + translate(SackRateDiff,       -1, 1, 0, 3.0) \
       + translate(HavocRateDiff,      -1, 1, 0, 4.0)
```

Note that `ExpTO - ActualTO` is *per-team* (not a diff between teams). The commented-out
alternative uses `ExpTODiff - ActualTODiff` (cross-team diff). The active formula uses
raw per-team "turnover luck" = how much better/worse the team did versus expectation. The
`SackRateDiff` and `HavocRateDiff` are team differences (defensive rates).

**Open question OQ-6:** The active formula mixes apples and oranges: `ExpTO - ActualTO`
is an absolute count (raw game turnovers) while `SackRateDiff` and `HavocRateDiff` are
rate differentials. In the notebook context they're computed *per game*, so the per-game
turnover count is small (0–5 range), while the rate diffs are in (−1, 1). The translate()
bounds absorb this. Confirm this is intentional and not a leftover from an earlier iteration.

### 4.8 5FR composite

```python
5FR = 0.35 × Eff + 0.30 × Expl + 0.15 × FinDrv + 0.10 × FldPos + 0.10 × Trnovr
5FRDiff = home_5FR - away_5FR  (per game, computed after both teams' 5FR are ready)
```

### 4.9 Season aggregation for predict_matchup

`predict_matchup` (cell 59) computes the *season average* of each game's 5FR for a team:
```python
team_avg_ffr = grouped_by_year[(team, year)][: week]['5FR'].tail(games_to_consider).mean()
```

That is, it takes the last `games_to_consider` (default 4) game-level 5FR values up to a given
week and averages them. Before the season's games are available (`week == 0`), it falls back to
the *prior season's* ratings.

**SoS adjustment:** if opponent's SoS (strength of schedule, measured as mean 5FR of opponents)
is lower, the weaker-SoS team's average 5FR is multiplied by `(team_sos / opponent_sos)`.
Two additional SoS adjustments layer on: P5/G5 subdivision adjustment and conference SoS
adjustment (skipped if either team is an FBS Independent).

**Returning production adjustment** (weeks 1–4 only): if one team's `returning_production ×
roster_talent` is lower, that team's 5FR is scaled down by `team_talent / opponent_talent`.
`calculate_roster_talent` is the 4-year rolling average CFBD recruiting rating for the school;
FCS teams use the 2nd percentile of FBS ratings as a floor.

**HFA adjustment:** +2.5 points to the projected MOV (or +1.0 for COVID bubble games).

---

## 5. Model training recipe

### 5.1 Outlier filtering

```python
z_5fr = |zscore(5FRDiff)|  # per-game
z_pts = |zscore(PtsDiff)|
basis = rows where z_5fr < 3.2 AND z_pts < 3.0
train_data = 80% random sample of basis (msk = np.random.rand(len(basis)) < 0.80)
test_data  = remaining 20%
```

The z-score threshold for 5FRDiff is 3.2 (slightly looser), while PtsDiff uses 3.0.

### 5.2 Model

```python
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10, seed=123)
model.fit(train_data['5FRDiff'][:, np.newaxis], train_data['PtsDiff'][:, np.newaxis])
preds = model.predict(test_data['5FRDiff'][:, np.newaxis])
mu = preds.mean()
std = preds.std()
```

`mu` and `std` are statistics of the *test-set predictions*, used as the z-score baseline for
WP conversion. This is non-standard: the baseline floats with the test split.

**Open question OQ-7:** Using test-set prediction statistics (`preds.mean()`, `preds.std()`) as
the z-score normalization for WP conversion is fragile — it depends on the 80/20 random split
and will change on retrain unless a fixed seed is used end-to-end (and even then, it changes if
data is added). A more stable approach would fix `mu=0` (the model is predicting point diff,
which is symmetric around zero) and derive `std` from the full training-set predictions or from
a fixed historical distribution. This is an architectural decision for Phase 0.

### 5.3 WP conversion

```python
z = (model.predict([[5FRDiff]]) - mu) / std
WP = scipy.stats.norm.cdf(z)
```

---

## 6. Recruiting and returning-production inputs

These are used in `predict_matchup` (pregame prediction path), not in the `calculate_box_score`
training-data generation path.

**Recruiting** (cell 9): CFBD `/recruiting` endpoint, years 2007–2021. `calculate_roster_talent(team, year)` = mean `rating` for recruits `committedTo == team` in years `(year-3)` through `year` (4-year rolling window). FCS teams use 2nd-percentile FBS rating as a floor.

**Returning production** (cell 10): CFBD `/returning_production` endpoint, years 2015–2021. `calculate_returning_production(team, yr)` = `overall` field from the team-year row, or 2nd-percentile FBS value if missing. This is the CFBD-aggregated returning-production percentage (fraction of production returning from last year).

Both are used only in weeks 1–4 of `predict_matchup`, where in-season game data is sparse.

---

## 7. Module architecture — `python/pregame_wp/`

| Module | Responsibility |
|---|---|
| `__init__.py` | Package marker + version |
| `constants.py` | Factor weights, translate() bounds, outlier thresholds, success-rate definitions |
| `data_ingest.py` | CFBD data fetch/cache: games, drives, PBP, recruiting, returning-production (years 2012–present). Reads pre-staged CSVs or pulls via CFBD API client |
| `ep_curve.py` | Load and expose `ep.csv` and `punt_sr.csv` lookup tables; validate shape (101 rows / 100 rows) |
| `play_features.py` | `EqPPP`, `play_successful`, `play_explosive` — per-play derived columns (from cells 20, 22) |
| `team_stats.py` | `generate_team_play_stats`, `generate_team_drive_stats`, `generate_team_turnover_stats`, `generate_team_st_stats` — per-team box stats for a single game |
| `five_factors.py` | `create_eff_index`, `create_expl_index`, `create_fp_index`, `create_finish_drive_index`, `create_turnover_index`, `calculate_five_factors_rating` — the five factor index functions |
| `box_score.py` | `calculate_box_score(game_id, year)` — assembles the full game box score including 5FR and 5FRDiff (port of cell 24) |
| `training.py` | `build_training_frame()` — loop over game_ids, build stored_game_boxes; outlier filter; 80/20 split; train XGBRegressor; return model + (mu, std) |
| `predict.py` | `generate_win_prob(game_id, year)` + `predict_matchup(team1, team2, year, ...)` — inference path (ports cells 45 and 59) |
| `talent.py` | `calculate_roster_talent(team, year)` + `calculate_returning_production(team, yr)` — recruiting/production adjustments |
| `cli.py` | Subcommands: `ingest | build-boxes | train | predict-matchup` |

---

## 8. Data grain and scope

- **Grain:** team-game (two rows per game, one per team).
- **Season coverage:** 2012–2020 (notebook's training range). Extension to 2021+ requires
  CFBD data availability for `/drives`, `/plays`, `/games` for those years.
- **FBS-only:** only games where both `home_team` and `away_team` are in the FBS team list are
  included in training. FCS teams are handled in `predict_matchup` via FCS floor defaults.
- **No sdv-py bundled model.** This pipeline does not produce a `.ubj` that ships in sdv-py.
  The output model is local to `cfbfastR-cfb-raw/python/pregame_wp/models/`.

---

## 9. Dependencies

Add to `cfbfastR-cfb-raw/pyproject.toml` (in a new `pregame_wp` optional group):

```toml
[dependency-groups]
pregame-wp = ["xgboost>=2.0", "scipy>=1.10", "cfbd>=1.0"]
```

`pandas`, `numpy`, `polars`, `pyarrow` are already present. `scipy` is needed for `stats.norm.cdf`.
`cfbd` (Python CFBD API client) is needed for data ingest — or raw `requests` with CFBD token.

---

## 10. Validation

Because the model is trivial and the training data is not bundled with the repo, validation is:

1. **Factor formula unit tests** — hand-computed fixtures for each of the five index functions.
2. **Box score regression** — `calculate_box_score(401013183, 2018)` produces known output
   (GT vs UVA 2018); if CFBD data is available, assert 5FR values within tolerance.
3. **WP sanity** — `generate_win_prob(401013183, 2018)` should return WP > 0.5 for the
   actual winner.
4. **predict_matchup smoke** — `predict_matchup("LSU", "Clemson", 2019, week=-1)` should
   return a WP close to the historical notebook output (near 0.7 for LSU; see cell 83 output).

---

## 11. Open questions summary

| ID | Question | Impact | Resolution path |
|---|---|---|---|
| **OQ-1** | Is `ep.csv` stable / aligned with the Track 1 retrained EP model? | Medium — EqPPP values shift if EP curve changes | Compare once Track 1 EP model is retrained; update `ep.csv` if needed |
| **OQ-2** | Is 3rd-down `play_successful = False` intentional? | Low — consistent with S&P+ but surprising | Confirm against Bill Connelly's SR definition |
| **OQ-3** | Explosiveness uses `AvgEqPPPDiff`, not `IsoPPPDiff` — is this intentional? | Medium — contradicts the Connelly framing | Accept as-is or switch to IsoPPPDiff; document either way |
| **OQ-4** | Field-position factor mixes `OffSRDiff` (scrimmage) with special-teams EP; is this the intended formula? | Low — the formula is explicit; just odd | Document as-is |
| **OQ-5** | **Punt-return EP copy-paste bug:** `PuntReturnEqPPP` uses punt (offense) EP, not return EP — making `PuntEqPPP - PuntReturnEqPPP = 0` always | **High** — the field-position factor's punt sub-term is silently zero | Phase 0 decision: faithful port (preserve bug) vs corrected port |
| **OQ-6** | Turnover formula mixes absolute count (`ExpTO - ActualTO`) with rate diffs; intentional? | Low | Document as-is; the translate() bounds absorb the magnitude difference |
| **OQ-7** | `mu/std` derived from test-set predictions is fragile; should use a fixed distribution | Medium — WP calibration shifts on retrain | Phase 0 decision: fix `mu=0, std=historical` or replicate notebook's behavior |
| **OQ-8** | `predict_matchup` SoS adjustments multiply *both* teams' 5FR by the ratio — is this double-counting when `team1_sos > team2_sos`? Only the weaker-SoS team is penalized per branch; confirm the logic is not accidentally applied twice | Low | Trace the code path in Phase 0 |

---

## 12. Risks

- **CFBD data availability.** The notebook used CFBD CSV exports from 2012–2020. CFBD's
  API schema and data availability for older seasons may differ from the current API. A data
  ingest step is needed before any training can proceed.
- **Text-parsing brittleness.** Kickoff and punt stats rely on regex against `play_text`
  (`'kickoff for (\d+) ya*r*ds'`, `'punt for (\d+) ya*r*ds'`). CFBD play text format may have
  changed across years. These parsers need validation across the full date range.
- **Model trivialness.** The 10-tree XGBoost is not production-grade for pregame WP. It is a
  research artifact. Any use of WP outputs should be framed as a demonstration, not a betting or
  decision tool.
- **No CFBD API key in CI.** Live CFBD fetches require an API key; all tests that require data
  must either use pre-staged fixtures or be gated with `@pytest.mark.skipif`.

---

## 13. Non-goals

- No sdv-py integration — this model does not ship in the library.
- No real-time / in-season inference — the system is a batch offline tool.
- No play-level WP (that is Track 1's WP models).
- No automatic model update; trained model is committed to the repo under review.
