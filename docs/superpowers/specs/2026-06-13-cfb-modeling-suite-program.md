# CFB Modeling Suite — Program Overview

- **Date:** 2026-06-13
- **Author:** Saiem Gilani
- **Status:** Active program (umbrella; tracks specced individually)
- **Goal:** Retrain/port the full college-football model suite to a Python-native pipeline, sourced
  from the CFB backfill, decoupled from the R toolchain. Each track is its own spec → plan → build
  cycle; this doc is the committed scope + sequencing.

## Why a program (not one spec)

Folding every modeling effort surveyed (across `gp-cfb-raw-keepers`, `akeaswaran/cfb_qbr`,
`akeaswaran/cfb-pbp-analysis`) into scope yields **eight models spanning three data grains
(play / season / team-game), four data sources (ESPN backfill, cfbfastR-data, CFBD, StatsBomb AMF),
and two+ repos.** A single spec would be un-plannable and un-reviewable. Each track below is scoped,
specced, planned, and built independently; cross-cutting conventions (below) are shared.

## Tracks (all in scope)

| # | Track | Models | Grain / source | Recipe | Home (proposed) | Status |
|---|---|---|---|---|---|---|
| **1** | **Play-level shipped models** | EP (8-feat), WP-spread (13-feat), **WP-naive (12-feat)**, QBR (6-feat) | play / ESPN backfill | EP=keepers `02`; WP=`cfbscrapR-wpa.ipynb` ✅; QBR=reconstruct + `cfb_qbr` GAM ancestor | `cfbfastR-cfb-raw/python/model_training/` | **Spec ready** → plan next |
| **2** | **Fourth-down** | yards-gained (5-feat `multi:softprob`, 76-class) + decision logic | play / cfbfastR + CFBD lines | `fourth-downs.ipynb` (cfb4th / Jason Lee lineage) ✅ | `cfb4th`-aligned (TBD) | Recipe known; needs spec |
| **3** | **RB-eval (DAKOTA)** | `s(epa_per_play)+s(success)` GAM (xREPA) | season / cfbfastR-data | `rb_eval_model.R` ✅ | cfb-raw metrics or own | Recipe known; needs spec |
| **4** | **Pregame WP + Five Factors** | 1-feat regressor + Five-Factors team ratings | team-game / CFBD + recruiting + returning-production | `win-prob.ipynb` (model trivial; feature-eng heavy) ⚠️ | TBD | Needs spec (large feature-eng) |
| **5** | **CPOE** | completion-prob `binary:logistic` (560 trees) | play / **StatsBomb AMF** (not CFB) | `cpoe_model.R` — **must re-base onto CFB pbp** ⚠️ | TBD | Needs data-feasibility check then spec |
| **6** | **NFL EP/WP** | NFL EP/WP (sdv-py `nfl/models/*`) | play / nflverse | not surveyed (nflfastR lineage) ❌ | **separate repo** (CFB-raw is CFB-only) | Needs survey + own spec |

## Sequencing

1. **Track 1 first** — highest value (the three shipped CFB models + naive), fully understood,
   recipes identified. Proceed to implementation plan now.
2. **Tracks 2 & 3 next** — recipes fully known; small, self-contained ports.
3. **Track 4** — recipe partial; the Five-Factors team-rating pipeline is the real work.
4. **Track 5** — gated by a feasibility check: does CFB pbp carry the completion features the
   StatsBomb-trained CPOE needs? Spec only after that's answered.
5. **Track 6** — cross-repo; survey nflfastR's EP/WP training first; lives outside `cfbfastR-cfb-raw`.

## Cross-cutting conventions (shared by all tracks)

- **Producer == consumer:** where a model's features are computed by `CFBPlayProcess` (Tracks 1, and
  partly 2), the trainer reuses those exact functions → train/inference parity by construction.
- **Figures:** plotnine, bespoke styling (garnet `#500f1b`, Gill Sans MT + fallback, cfbfastR hex),
  + calibration data tables. (See Track 1 spec §8.)
- **Validation:** prediction-parity vs reference/shipped artifacts; LOSO calibration as data; sanity
  fixtures from `cfb-pbp-analysis` where applicable (cfbscrapR-lineage — ballpark, not exact).
- **Model handoff to sdv-py:** retrained `.ubj` copied in **manually, under review** — never
  auto-overwritten by a pipeline.
- **No new heavy deps beyond:** `xgboost`, `plotnine`/`statsmodels`/`pillow` (figures), `pygam`
  (GAM tracks 3 + the QBR ancestor).

## Track specs

- Track 1: `2026-06-13-cfb-ep-wp-model-training-port-design.md`
- Tracks 2–6: to be written when each is reached.
