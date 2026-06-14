# cfbfastR-cfb-raw

Raw + enriched college-football game JSON, scraped from ESPN via `sportsdataverse`.

## What it produces

Per game:
- `cfb/json/raw/{game_id}.json` — ESPN summary (curated allowlist incl. injuries + gameNotes).
- `cfb/json/final/{game_id}.json` — fully enriched (EPA/WPA/QBR plays, advBoxScore) +
  play participants + game rosters + normalized betting + power index (FPI, recent seasons) +
  per-team box extras (derived from the summary). Self-describing (`id`/`season`/`week` echoed).

Standalone datasets, each a flat `cfb/{dataset}/json/{game_id}.json` folder (no season
subdirectories — ESPN game ids are globally unique): `game_rosters`, `play_participants`,
`betting`, `power_index`, `team_box_extra`, plus the `schedules` + `cfb_schedule_master`.

> **Not collected (probe §12.8, 2026-06-03):** ESPN does not expose CFB **officials**
> (neither the summary nor the core officials endpoint returns data) and **propbets**
> 404s for CFB — both dropped. **FPI (`power_index`) and full `event_odds`** only return
> data for recent seasons, so they are season-gated (`EXTRAS_MIN_SEASON = 2015`). The four
> per-team `event_competitor_*` calls are redundant with the summary and derived from it
> (no extra requests). Net ~5 GETs/game.

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

## Dependencies / local dev

`uv.lock` pins `sportsdataverse>=0.0.52` (the offline-reprocess release — sportsdataverse-py
PR #91) from PyPI, so CI's `uv sync --frozen` works on a clean runner. For local
co-development against an unreleased `sdv-py`, run `uv pip install -e ../../sdv-py` after
`uv sync` (do not add a `[tool.uv.sources]` path source — it would break CI, which has no
sibling checkout).

## Automation

- `scrape_cfb_raw.yml` — cron over the CFB calendar (Aug→Jan) + manual dispatch.
- On push, `cfbfastR_cfb_data_trigger.yml` fires `repository_dispatch` to
  `sportsdataverse/cfbfastR-cfb-data`, which rectangularizes `final/` into release parquet.

## Reprocess vs. recreate

- **Reprocess** (here, Python): `raw → final`, offline, gated by `processing_version`. Bump
  `SCHEMA_REV` in `python/_cfb_raw_utils.py` to force stale games to rebuild.
- **Recreate** (the `-data` repo, R): `final → parquet`, cheap reshape.

See `docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`.

## Model training suite

Native Python reimplementation of the CFB model training pipeline (cfbfastR reference).
All packages live under `python/` and emit `.ubj` XGBoost boosters compatible with
`sportsdataverse/cfb/models/`.

| Track | Package | Algorithm | Target |
|---|---|---|---|
| T1 | `model_training` | XGBoost `reg:squarederror` / `binary:logistic` | EP / WP-spread / WP-naive / QBR |
| T2 | `model_training/fourth_down` | XGBoost `multi:softprob` (76 classes) | Yards-gained distribution on 3rd/4th downs |
| T3 | `rb_eval` | pygam `LinearGAM(s(0)+s(1))` | xREPA (expected rushing EPA) |
| T4 | `pregame_wp` | XGBoost `XGBRegressor` + five-factors | Pre-game win probability |
| T5 | `cpoe` | XGBoost `binary:logistic` | Completion probability / CPOE |

```bash
# Train a single model (example — T5 CPOE)
uv run python -m cpoe train \
    --input-parquet data/cfb_passes.parquet \
    --output-model models/cp_model.ubj

# Run leave-one-season-out calibration
uv run python -m cpoe loso \
    --input-parquet data/cfb_passes.parquet \
    --output-csv cal/cpoe_loso.csv

# Figures (requires figures dep group)
uv sync --group figures
uv run python -m cpoe figures \
    --results cal/cpoe_loso.csv --output-dir figures/cpoe
```

Optional dependency groups:

| Group | Install | Required by |
|---|---|---|
| `figures` | `uv sync --group figures` | T1/T2/T4/T5 calibration plots (plotnine) |
| `gam` | `uv sync --group gam` | T3 rb_eval training (pygam) |

See `python/model_training/HANDOFF.md` for the sdv-py integration checklist.
