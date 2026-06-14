# Track 5 CPOE — Phase 0 Feasibility Analysis

## Source model features (`cpoe_model.R`, StatsBomb AMF)

The R original trains on StatsBomb `tb12_events_dataset_{start}_{end}.csv` joined to
`tb12_plays_dataset_{start}_{end}.csv`. The five model features are:

| Feature | Present in ESPN final.json? |
|---|---|
| event_pass_air_yards | NO |
| play_target_separation | NO |
| event_pass_target_x | NO |
| event_pass_target_y | NO |
| play_qb_pressure | NO |
| endline_receiver_dist (derived: 110 - target_x) | NO |
| sideline_receiver_dist (derived: min(y, 53.33-y)) | NO |

## ESPN final.json inspection (game: 401628455, 2024 season)

Inspection of `cfb/json/final/401628455.json` (169 plays, 65 pass plays, 36 completions).

Pass plays found: 65. Completions: 36. All five StatsBomb features: None on every play.
`yds_receiving`: null on 33/36 completions (~91%). `statYardage` = total play yards, not
throw distance; not a usable air-yards proxy.

## Verdict

The StatsBomb-trained CPOE cannot be ported to CFB ESPN data. Approach A (8-feature
game-state model) is the primary path. CFBD air-yards (Approach B) requires Task 0.2 investigation.

## Available features for Approach A (Reduced Game-State Model)

| Feature | Column | Always populated? |
|---|---|---|
| Down | start.down | YES |
| Distance to 1st | start.distance | YES |
| Yards to goal | start.yardsToEndzone | YES |
| Score diff | pos_score_diff_start | YES |
| Seconds remaining | start.TimeSecsRem | YES |
| Is home | start.is_home | YES |
| Period | period | YES |
| Passing down flag | passing_down | YES |

## CFBD Air Yards (Task 0.2 result)

CFBD air_yards fill rate: **0%** (0/123 pass plays across 5 sampled game-seasons,
2020–2024 regular season week 1). Fields checked: `air_yards`, `yards_to_sticks`,
`pass_length`, `passLength` — all absent from the CFBD `/plays` API response.

The CFBD `/plays` endpoint key set (2021–2024) is:
`away, clock, defense, defenseConference, defenseScore, defenseTimeouts, distance,
down, driveId, driveNumber, gameId, home, id, offense, offenseConference,
offenseScore, offenseTimeouts, period, playNumber, playText, playType, ppa, scoring,
wallclock, yardline, yardsGained, yardsToGoal`

**Verdict: Approach B INFEASIBLE.** CFBD does not provide air_yards on individual plays
through any inspected field name. Phase 3 Task 3.2 is SKIPPED. Approach A (8-feature
game-state model) is the only viable path.

## True CP lineage note

`cpoe_model.R` is the nflfastR CP model recipe applied to StatsBomb AMF data — same
hyperparameters (`binary:logistic, eta=0.025, gamma=5, subsample=0.8, colsample_bytree=0.8,
max_depth=4, min_child_weight=6, nrounds=560`). The StatsBomb dependency is the *data
overlay*, not the recipe. The canonical CFB feature set for Approach A is derived from
nflfastR's `prepare_cp_data()` mapped to ESPN `final.json` columns.
