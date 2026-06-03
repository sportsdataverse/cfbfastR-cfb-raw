# cfbfastR CFB JSON Consolidation — Design Spec

- **Date:** 2026-06-03
- **Author:** Saiem Gilani
- **Status:** Draft (pending review)
- **Target repos:** `cfbfastR-cfb-raw` (new, Python/uv), `cfbfastR-cfb-data` (new, R)

## 1. Goal

Re-think CFB JSON scraping/consolidation as a modern, two-repo pipeline that mirrors the
sibling ecosystems (`hoopR-nba-raw/-data`, `wehoop-wnba-raw/-data`,
`fastRhockey-nhl-raw/-data`). Each college-football game is stored as two
single-game JSON files:

- **`raw`** — verbatim ESPN summary payload, exactly as the API returns it.
- **`final`** — fully enriched output of the `cfb_pbp` pipeline (EPA/WPA/QBR plays,
  advanced box score) plus play participants, game rosters, and normalized betting —
  self-contained per game.

The system must support (a) full historical backfill (2002→present), (b) incremental
in-season daily runs, and critically (c) **rebuilding `final` from `raw` already on disk**
when processing logic changes — no re-scrape, no full reshape.

## 2. Decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Relationship to legacy `cfbfastR-raw`/`-data` | **Clean-slate replacement.** New `cfbfastR-cfb-raw` + `cfbfastR-cfb-data` follow the `{pkg}-{league}-raw/-data` naming. Legacy repos untouched, retired later (out of scope). |
| 2 | Scraper language | **Python**, packaged as a **uv** project. |
| 3 | `-data` rectangularization language | **R** — match sibling `-data` repos, reuse cfbfastR `pbp_output_schema.R` + `piggyback` releases. |
| 4 | Final JSON enrichment depth | **Full** `run_processing_pipeline()` (EPA/WPA/QBR + advBoxScore) + `play_participants` + `game_rosters` + normalized `betting`. |
| 5 | Datasets produced by `-data` | `play_by_play`, ESPN `team_box`/`player_box`, advanced box (`adv_box`), `play_participants`, `drives`, `rosters`, `betting`, `schedules`. |
| 6 | Concurrency / scope | **ProcessPool** (CPU-bound XGBoost bypasses GIL) + full backfill via `-s/-e` year args + incremental daily. |
| 7 | Rosters & betting placement | **Both** embedded in `final/{id}.json` **and** written as standalone per-game JSON datasets. |
| 8 | Reprocess-from-disk | New first-class capability gated by a `processing_version` stamp (see §7). |

## 3. SDK boundary

All ESPN fetching + enrichment lives in **sdv-py** (`sportsdataverse`), not in the
scraper. The scraper stays thin; bug fixes go upstream. Key entry points:

- `CFBPlayProcess(gameId, raw=True).espn_cfb_pbp()` → verbatim ESPN summary dict.
- `CFBPlayProcess(gameId).espn_cfb_pbp(); .run_processing_pipeline()` → enriched plays
  + `advBoxScore` + drive data (15-step pipeline; loads `ep_model.ubj`,
  `wp_spread.ubj`, `qbr_model.ubj` at import).
- `CFBPlayProcess(gameId, path_to_json=...).cfb_pbp_disk()` → load raw **from disk**
  (offline reprocess).
- `espn_cfb_play_participants(game_id)` → per-play participant mapping.
- `espn_cfb_game_rosters(game_id)` → game roster.
- `espn_cfb_schedule(...)` → weekly/season schedule.

> **Note:** cfbfastR R package is currently on branch `refactor/pbp-epa-wpa-modular`;
> the sdv-py CFB enrichment engine is the live contract this repo pins.

## 4. Repository architecture

```
cfbfastR-cfb-raw  (Python/uv)
  scrape ESPN -> cfb/json/{raw,final}/{game_id}.json
              + schedule master + standalone rosters/betting JSON
  on push: peter-evans/repository-dispatch (event-type daily_cfb_data) ─┐
                                                                        ▼
cfbfastR-cfb-data (R)
  read final JSON (raw.githubusercontent.com/.../cfb/json/final/{id}.json)
  -> rectangularize -> parquet/csv/rds  -> piggyback release to sportsdataverse-data
```

- **raw repo** owns scraping + enrichment; the `final` JSON is the contract.
- **data repo** owns rectangularization + releases; reuses `pbp_output_schema.R`.

## 5. Directory layout

### 5.1 `cfbfastR-cfb-raw/`

```
python/
  _cfb_raw_utils.py        # logging, ProcessPool runner, manifest reader, on-disk filter,
                           #   atomic JSON write, season helpers, PROCESSING_VERSION
  scrape_cfb_schedules.py  # season -> cfb/schedules/{parquet,rds,csv} + master
  scrape_cfb_json.py       # game_id -> cfb/json/{raw,final}/{id}.json   (core, scrape+enrich)
  reprocess_cfb_json.py    # raw-on-disk -> final/{id}.json              (offline rebuild)
  scrape_cfb_rosters.py    # game_id -> cfb/rosters/json/{season}/{id}.json
  scrape_cfb_betting.py    # game_id -> cfb/betting/json/{season}/{id}.json
scripts/
  daily_cfb_scraper.sh     # season-loop orchestrator (schedules->json->rosters->betting)
  backfill_cfb.sh          # wrapper: full 2002->present range
  reprocess_cfb.sh         # season-loop, reprocess only, commits "CFB Reprocess Update (...)"
cfb/
  json/raw/{game_id}.json
  json/final/{game_id}.json
  schedules/{parquet,rds,csv}/cfb_schedule_{year}.*
  rosters/json/{season}/{game_id}.json
  betting/json/{season}/{game_id}.json
  cfb_schedule_master.{parquet,rds}
logs/{scraper}_logfile_{year}.log
.github/workflows/
  scrape_cfb_raw.yml             # cron (Aug->Jan) + workflow_dispatch
  cfbfastR_cfb_data_trigger.yml  # on push -> repository_dispatch to -data
pyproject.toml                   # uv project
uv.lock
.python-version
CLAUDE.md  README.md  .gitignore  .Rbuildignore-equivalent (.gitattributes)
```

### 5.2 `cfbfastR-cfb-data/`

```
R/
  espn_cfb_01_pbp_creation.R           # final JSON -> play_by_play_{year}.parquet/csv.gz
  espn_cfb_02_team_box_creation.R      # ESPN team box
  espn_cfb_03_player_box_creation.R    # ESPN player box
  espn_cfb_04_adv_box_creation.R       # advBoxScore (adv team + adv player) -- SEE §9 open item
  espn_cfb_05_play_participants_creation.R
  espn_cfb_06_drives_creation.R
  espn_cfb_07_rosters_creation.R
  espn_cfb_08_betting_creation.R       # odds/lines/pickcenter/predictor/ATS
  espn_cfb_09_schedules_creation.R
  upload_releases.R                    # piggyback -> sportsdataverse-data
  run_summary.R                        # $GITHUB_STEP_SUMMARY
scripts/daily_cfb_R_processor.sh
cfb/
  pbp/{parquet,csv}/play_by_play_{year}.{parquet,csv.gz}
  team_box/  player_box/  adv_box/  play_participants/  drives/  rosters/  betting/  schedules/
    -> {parquet,rds}/{dataset}_{year}.*
  cfb_{dataset}_in_data_repo.{parquet,rds}     # master indices
logs/{processor}_logfile_{year}.log
.github/workflows/daily_cfb.yml          # repository_dispatch + cron + workflow_dispatch
CLAUDE.md  README.md  DESCRIPTION-or-renv (R deps)
```

Output `cfb/` is committed to each repo; GitHub doubles as store + CDN. Flat per-season
files (no nested `game_id/` dirs) keep `arrow::open_dataset` + per-season binds fast.

## 6. Data contract: raw vs. final JSON

### 6.1 `cfb/json/raw/{game_id}.json`

Verbatim ESPN summary payload, no transformation. Written **first**, before enrichment,
for crash resilience and offline reprocess. Top-level keys as ESPN returns them:
`boxscore, drives, header, gameInfo, leaders, scoringPlays, winprobability, odds,
pickcenter, predictor, againstTheSpread, broadcasts, videos, standings, format`.
(`drives` is sometimes absent/empty upstream.)

### 6.2 `cfb/json/final/{game_id}.json`

```jsonc
{
  "id": 401628455,
  "season": 2024, "week": 1, "season_type": 2,
  "processing_version": "0.0.51+1",          // §7 version gate
  "count": 185,
  "plays": [ { /* 150+ enriched cols: EPA, wpa, *_player_name, down/dist, yardage, drive.* */ } ],
  "play_participants": [ { "play_id": ..., "athlete_id": ..., "role": ... } ],
  "advBoxScore": { "teams": {}, "players": {} },   // SEE §9 open item
  "boxScore":    { /* raw ESPN team/player box passthrough */ },
  "game_rosters": [ { "game_id":..., "season":..., "team_id":..., "athlete_id":..., "name":..., "position":..., "jersey":... } ],
  "betting": {
      "home_team_spread": -7.5, "over_under": 52.5,
      "game_spread": -7.5, "game_spread_available": true,
      "odds": [], "pickcenter": [], "predictor": {}, "against_the_spread": []
  },
  "drives": { /* raw ESPN drives passthrough */ },
  "scoringPlays": [], "winprobability": [], "leaders": [],
  "header": {}, "gameInfo": {}, "standings": {},
  "homeTeamId": 333, "awayTeamId": 99, "timeouts": {},
  "broadcasts": [], "videos": []
}
```

- Top-level `id/season/week/season_type` are **echoed** (self-describing returns).
- `betting` is **normalized in Python** (stable shape, null-safe defaults) rather than
  passed through raw — insulates the R binder from ESPN's `odds`/`pickcenter`/ATS drift.
- `drives` stays a **raw passthrough**; the enriched `plays` already carry `drive.*`
  columns, so the drives dataset in `-data` is a light re-normalization.

### 6.3 Standalone aux JSON

`cfb/rosters/json/{season}/{id}.json` carries the `game_rosters` array;
`cfb/betting/json/{season}/{id}.json` carries the `betting` object. Each echoes
`game_id`/`season`/`week` for self-describing joins. Same data as embedded in `final`.

## 7. Reprocess-from-disk (rebuild `final` from `raw`)

Two rebuild axes, two costs:

1. **Reprocess** = `raw JSON -> final JSON` (Python, ~5-10s/game XGBoost) — **expensive**;
   this section. Done from disk when `CFBPlayProcess` logic changes.
2. **Recreate** = `final JSON -> parquet` (R, in `-data`) — cheap reshape, already covered
   by re-running the `espn_cfb_0N_*.R` scripts.

### 7.1 `python/reprocess_cfb_json.py`

- Args: `-s/-e` season range (or `--all`), `--force`, `--refresh-aux`.
- Iterates `cfb/json/raw/*.json` on disk (NOT the schedule master, NOT ESPN).
- Per game:
  - `CFBPlayProcess(gameId, path_to_json="cfb/json/raw").cfb_pbp_disk()` → load raw from disk.
  - `run_processing_pipeline()` → rebuild `plays` / `advBoxScore` / `drives`.
  - `betting` re-normalized from disk raw (deterministic, offline).
  - `game_rosters` from `cfb/rosters/json/{season}/{id}.json` on disk (offline); only
    re-fetch via `espn_cfb_game_rosters()` if `--refresh-aux`.
  - `play_participants` carried forward from existing `final/{id}.json` (offline); only
    re-fetch via `espn_cfb_play_participants()` if `--refresh-aux`.
  - Stamp `processing_version`; atomic write `final/{id}.json`.

| `final` field | reprocess source | network? |
|---|---|---|
| `plays`, `advBoxScore`, `drives` | re-run pipeline on disk raw | no |
| `betting` | re-normalize disk raw | no |
| `game_rosters` | disk rosters JSON | no (yes if `--refresh-aux`) |
| `play_participants` | carry forward existing final | no (yes if `--refresh-aux`) |
| `header`/`gameInfo`/`scoringPlays`/… | passthrough disk raw | no |

### 7.2 Version gate

`_cfb_raw_utils.py` defines `PROCESSING_VERSION = f"{sportsdataverse.__version__}+{SCHEMA_REV}"`
and stamps it into every `final` JSON. Reprocess **skips** games whose
`final.processing_version` already matches the current value — so after an enrichment
change you bump `SCHEMA_REV` (or sdv-py ships a new version) and only stale games rebuild.
`--force` rebuilds everything; reprocess is idempotent and resumable (re-run after a crash
skips already-current games for free).

### 7.3 Trigger reuse

`scripts/reprocess_cfb.sh` uses the same season-loop + temp-log + git-commit idiom, runs
only `reprocess_cfb_json.py`, and commits `"CFB Reprocess Update (Start: $i End: $i)"`.
The `Start:/End:` tokens are preserved, so the existing push trigger fires and `-data`
recreates parquet for exactly those seasons — no special wiring.

## 8. Python scraper internals + orchestration

### 8.1 `_cfb_raw_utils.py`

- `get_logger(name, year)` — `FileHandler(logs/{name}_logfile_{year}.log)` + `StreamHandler`
  (both the GH Actions log and the tracked logfile capture output).
- `run_pool(fn, items, *, kind="process"|"thread", workers, desc)` — tqdm-wrapped pool.
  ProcessPool for enrichment (`scrape_cfb_json`, `reprocess_cfb_json`); ThreadPool for
  IO-only (`scrape_cfb_schedules`, rosters, betting).
- `load_schedule_master()`, `games_for_seasons(master, start, end)`.
- `filter_undone(games, dir, rescrape=False)` — drop games whose `final/{id}.json` exists.
- `write_json_atomic(obj, path)` — write `{id}.json.tmp` then `os.replace`.
- `most_recent_cfb_season()` — Aug rollover heuristic / sdv-py helper.
- `PROCESSING_VERSION`, `SCHEMA_REV`.

### 8.2 `scrape_cfb_json.py` per-game task (raw-first ordering)

```python
def download_game(game_id, rescrape, logger):
    raw = CFBPlayProcess(gameId=game_id, raw=True).espn_cfb_pbp()
    write_json_atomic(raw, f"cfb/json/raw/{game_id}.json")        # bank raw first
    proc = CFBPlayProcess(gameId=game_id)
    proc.espn_cfb_pbp(); result = proc.run_processing_pipeline()  # ~5-10s, XGBoost
    result["play_participants"] = espn_cfb_play_participants(game_id=game_id).to_dicts()
    result["game_rosters"]      = espn_cfb_game_rosters(game_id=game_id).to_dicts()
    result["betting"]           = _normalize_betting(raw)
    result.update(id=game_id, season=..., week=..., season_type=...,
                  processing_version=PROCESSING_VERSION)
    write_json_atomic(result, f"cfb/json/final/{game_id}.json")
```

Each task is wrapped so a bad game logs `logger.exception(...)` and the pool continues.
Raw is banked before the risky enrichment, so a mid-enrichment crash still leaves raw on
disk for reprocess.

### 8.3 `scripts/daily_cfb_scraper.sh`

Season loop with the sibling temp-log-then-commit git idiom:

```bash
for i in $(seq "$START" "$END"); do
  TMPLOG=$(mktemp /tmp/cfb_raw_${i}.XXXXXX.log)
  { git pull >/dev/null; git config --local user.email "action@github.com"; git config --local user.name "Github Action"
    uv run python python/scrape_cfb_schedules.py -s $i -e $i -r $RESCRAPE   # schedules FIRST
    uv run python python/scrape_cfb_json.py      -s $i -e $i -r $RESCRAPE
    uv run python python/scrape_cfb_rosters.py   -s $i -e $i -r $RESCRAPE
    uv run python python/scrape_cfb_betting.py   -s $i -e $i -r $RESCRAPE
    git pull >/dev/null; git add cfb/* ; git commit -m "CFB Raw Update (Start: $i End: $i)" || echo "No changes"
    git pull >/dev/null; git push >/dev/null
  } 2>&1 | tee "$TMPLOG"
  cp "$TMPLOG" "logs/cfb_raw_logfile_${i}.log"
  git pull --rebase >/dev/null || true; git add logs/cfb_raw_logfile_${i}.log
  git commit -m "CFB Raw log update (Start: $i End: $i)" >/dev/null || true; git push >/dev/null; rm -f "$TMPLOG"
done
```

- **Schedules run first** — every other scraper reads `cfb_schedule_master.parquet`.
- **Commit message is load-bearing** — `"CFB Raw Update (Start: YYYY End: YYYY)"` is the
  only channel carrying the season range across the repo boundary.

## 9. uv project configuration

- `pyproject.toml` + `uv.lock` + `.python-version` (replaces `requirements.txt`).
- Deps: `sportsdataverse>=0.0.51`, `pandas`, `polars`, `pyarrow`, `tqdm`.
- `[tool.uv.sources]` path source for local dev (`sportsdataverse = { path = "../../sdv-py", editable = true }`
  — relative to repo root: `cfbfastR-dev/cfbfastR-cfb-raw` → `sdv-dev/sdv-py`),
  while `uv.lock` pins the published release for reproducible CI.
- All script invocations use `uv run python ...` so the locked env is used everywhere.

## 10. GitHub Actions

### 10.1 `cfbfastR-cfb-raw/.github/workflows/scrape_cfb_raw.yml`

- Triggers: `schedule` (CFB calendar windows — Aug preseason → Jan CFP, UTC) +
  `workflow_dispatch` (`start_year`/`end_year`/`rescrape` inputs).
- Steps: `actions/checkout` → `astral-sh/setup-uv` → `uv sync --frozen` → empty-input
  fallback to `most_recent_cfb_season()` → `bash scripts/daily_cfb_scraper.sh`.
- Token: `secrets.GITHUB_TOKEN` (same-repo commits).

### 10.2 `cfbfastR-cfb-raw/.github/workflows/cfbfastR_cfb_data_trigger.yml`

- On push → `peter-evans/repository-dispatch@v4`, `token: secrets.SDV_GH_TOKEN`
  (cross-repo PAT), `repository: sportsdataverse/cfbfastR-cfb-data`,
  `event-type: daily_cfb_data`, client payload includes the commit message.

### 10.3 `cfbfastR-cfb-data/.github/workflows/daily_cfb.yml`

- Triggers: `repository_dispatch: [daily_cfb_data]` + `schedule` (offset +2h after raw) +
  `workflow_dispatch`.
- Extract years from commit message: `grep -oP 'Start:\s*\K[0-9]{4}'` / `'End:\s*\K[0-9]{4}'`.
- Steps: `r-lib/actions/setup-r` + deps (`sportsdataverse/cfbfastR`, `ropensci/piggyback`)
  → `bash scripts/daily_cfb_R_processor.sh` (runs `espn_cfb_0N_*.R` sequentially,
  non-fatal per script, exit code propagated) → `upload_releases.R` → `run_summary.R`.

## 11. Logging

- Per-season files: `logs/{scraper}_logfile_{year}.log` (raw) /
  `logs/{processor}_logfile_{year}.log` (data).
- Written to `/tmp` during the script, copied to tracked `logs/` after git ops, committed
  separately (avoids conflicts inside the pull/commit/push block).
- Python `get_logger` dual-sinks to file + stdout; GH Actions captures stdout.
- Use `::error ::` / `::warning ::` workflow annotations for failures.

## 12. Open work items / risks

1. **advBoxScore expansion (flagged).** The `create_box_score()` output (team + player
   advanced stats) likely needs flattening/expansion refinement in
   `espn_cfb_04_adv_box_creation.R`, and possibly upstream tweaks in sdv-py's
   `create_box_score()`. Treat as its own sub-task when reached; do not assume the current
   shape is release-ready.
2. **`espn_cfb_play_participants` / `espn_cfb_game_rosters` network dependence.** If these
   require auxiliary ESPN `$ref` resolution beyond the summary payload, the offline
   reprocess relies on carry-forward (participants) / disk (rosters). `--refresh-aux` is
   the escape hatch. Confirm during implementation whether they parse the summary alone.
3. **`drives` upstream absence.** ESPN sometimes omits `drives`; the drives dataset and the
   `drive.*` play columns must degrade gracefully (empty, not error).
4. **Backfill volume.** 2002→present × thousands of games × ~5-10s enrichment is a long
   run; ProcessPool + `filter_undone` + `processing_version` make it resumable. Initial
   backfill likely run in season chunks rather than one job.
5. **cfbfastR `pbp_output_schema.R` reuse.** `-data` PBP creation should conform to the
   ~150-column canonical schema; verify alignment with the modular-refactor branch.

## 13. Out of scope

- Retiring/migrating legacy `cfbfastR-raw` / `cfbfastR-data` (later).
- Changes to the sdv-py enrichment engine beyond what's needed for advBoxScore (§12.1).
- KenPom/other non-ESPN sources.
