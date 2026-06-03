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

`uv.lock` pins `sportsdataverse` (`>=0.0.52`, the offline-reprocess release — sportsdataverse-py
PR #91). For local dev `[tool.uv.sources]` points at `../../sdv-py` (editable). Before CI can
`uv sync --frozen`, sportsdataverse 0.0.52 must be published to PyPI (so the editable path
source is no longer required on the runner) — or the workflow must also check out `sdv-py`.

## Automation

- `scrape_cfb_raw.yml` — cron over the CFB calendar (Aug→Jan) + manual dispatch.
- On push, `cfbfastR_cfb_data_trigger.yml` fires `repository_dispatch` to
  `sportsdataverse/cfbfastR-cfb-data`, which rectangularizes `final/` into release parquet.

## Reprocess vs. recreate

- **Reprocess** (here, Python): `raw → final`, offline, gated by `processing_version`. Bump
  `SCHEMA_REV` in `python/_cfb_raw_utils.py` to force stale games to rebuild.
- **Recreate** (the `-data` repo, R): `final → parquet`, cheap reshape.

See `docs/superpowers/specs/2026-06-03-cfbfastR-cfb-raw-consolidation-design.md`.
