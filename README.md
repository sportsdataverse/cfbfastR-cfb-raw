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
# resolve this repo's interpreter once (never `uv run` for a long scrape --
# it re-syncs the env mid-run). CFB_PY overrides.
source scripts/_venv.sh

"$PY" python/espn_cfb_02_schedules_scrape.py -s 2024 -e 2024
"$PY" python/espn_cfb_04_pbp_scrape.py      -s 2024 -e 2024
# full backfill
bash scripts/backfill_cfb.sh 2004
# rebuild final from raw on disk after a pipeline change (offline)
uv run python python/espn_cfb_60_reprocess.py -s 2024 -e 2024 --force
# recruit classes (247). Idempotent: a signed class is immutable, so complete
# years are skipped and only the current cycle fetches. Floor is 2002 --
# ratings collapse before then (2001: 52% rated on page 1, 0% by page 4).
bash scripts/50_scrape_recruits.sh              # current cycle
bash scripts/50_scrape_recruits.sh 2002 2026    # cold backfill
```

### Pushing a bulk rebuild

`scripts/chunked_push.sh` commits and pushes a large `final/` rebuild in
season-sized chunks. A single ~20k-file commit produces a pack GitHub refuses
(`bad line length` over HTTP/2, `RPC failed` over HTTP/1.1). `reprocess_cfb.sh`
already commits per season; this restores that shape for an after-the-fact bulk
rebuild — a model retrain that touches every game, say — where the work is
already in the worktree and there is no per-season loop to hang commits off.

```bash
bash scripts/chunked_push.sh
```

Do not run two of these (or any two git jobs) against this repo concurrently.

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

Manual recovery drivers (not wired into CI; reach for them around a full rescrape):

- `scripts/push_completed_seasons.sh` — watches the rescrape checkpoint and commits +
  pushes each season the moment it's verified (idempotent via `logs/pushed_seasons.txt`);
  run alongside `rescrape_cfb_full.sh`, or `ONESHOT=1` to push what's ready and exit.
- `scripts/retry_degraded_games.sh` — re-fetches the games the write guard skipped as
  degraded (transient ESPN 5xx); run after the main rescrape finishes (`DRY_RUN=1` lists
  them without fetching).

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

## Automation & status

<!-- BEGIN GENERATED: status -->

| workflow | schedule | last run |
|---|---|---|
| [![cfbfastR_cfb_data_trigger.yml](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/cfbfastR_cfb_data_trigger.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/cfbfastR_cfb_data_trigger.yml) | on push / dispatch | 2026-09-01 |
| [![orphan_scripts.yml](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/orphan_scripts.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/orphan_scripts.yml) | on push / dispatch | 2026-09-01 |
| [![scrape_cfb_raw.yml](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/scrape_cfb_raw.yml/badge.svg)](https://github.com/sportsdataverse/cfbfastR-cfb-raw/actions/workflows/scrape_cfb_raw.yml) | on dispatch | 2026-08-29 |

<!-- END GENERATED: status -->

## Reports & explainers

<!-- BEGIN GENERATED: reports -->

| Report | What it is | Last updated |
|---|---|---|
| [ESPN CFB rosters — collection strategy, column union, gotchas](docs/ESPN_ROSTERS.md) | explainer | 2026-08-29 |

<!-- END GENERATED: reports -->
