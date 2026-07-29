# CFB athlete_id ↔ `*_player_name` join coverage, 2004–2025

Measured 2026-07-28 against the committed `cfbfastR-cfb-raw` tree (18,640 games)
plus live probes of ESPN Core v2 / CDN.

## 1. The cliff is **2014**, and it is structural

`espn_cfb_play_participants` reads Core v2
`/events/{g}/competitions/{g}/plays`. Live probe, 3 games/season:

| era | play items returned | plays carrying `participants[]` | distinct participant types |
|---|---|---|---|
| 2004–2013 | 150–240 (full PBP) | **0** | 0 |
| 2014–2025 | 140–220 | 125–198 (~90%) | 11–16 |

ESPN serves the play stream for 2004–2013 but ships **no `participants[]`
array at all**. A rescrape cannot change this — the 8,003 pre-2014 games will
remain 0-participant. Confirmed live, not inferred from the cached tree.

Also note: the game *summary* payload carries no `participants[]` in **any**
era (2023 included). Participants only ever come from Core v2.

## 2. What we get pre-2014 instead: a roster-backed name→id join

`CFBPlayProcess.__attach_player_ids` (cfb_pbp.py:6673) matches each
regex-extracted `{type}_player_name` against `espn_cfb_game_rosters` for that
game — team-aware exact match on `_norm_player_name`, global-unique fallback,
ambiguous names dropped. `game_rosters` **is** populated for every season back
to 2004 (~60–73 athletes/game pre-2014 vs ~206–245 post-2014; the older ones
are the dressed/statted subset, not the full squad).

So the direction inverts: 2014+ is authoritative id→name; pre-2014 is
best-effort name→id.

## 3. Measured `pct_id_given_name` (share of populated name cells that also carry an id)

25 games/season, 11 player-column families (passer/rusher/receiver/sack/int/
punter/fg_kicker/kickoff/fumble/kick-return/punt-return).

| season | % id given name | note |
|---|---|---|
| 2004 | **54.4** | passer-regex `(ABBR)` defect — see §4.1 |
| 2005 | 89.2 | |
| 2006 | 93.2 | |
| 2007 | 90.3 | |
| 2008 | 90.9 | |
| 2009 | 91.3 | |
| 2010 | 89.1 | |
| 2011 | 87.5 | |
| 2012 | 89.4 | |
| 2013 | 86.4 | last pre-participants season |
| 2014 | 89.5 | 58 empty rosters |
| 2015 | 97.9 | |
| 2016 | 98.5 | |
| 2017 | 94.6 | |
| 2018 | **74.2** | 376/898 games have EMPTY game_rosters — see §4.2 |
| 2019 | 98.3 | |
| 2020 | 98.0 | (sample dodged the 142 empty-roster games) |
| 2021 | 94.8 | |
| 2022 | 91.0 | |
| 2023 | 98.5 | |
| 2024 | 98.4 | |
| 2025 | 93.8 | |

**Headline: pre-2014 lands at ~86–93%, vs ~94–98.5% in the participants era.**
2004 is a separate, fixable defect, not an era characteristic.

## 4. Classified causes of the misses

### 4.1 — 2004: passer regex keeps the team abbreviation (worth ~+35pp on 2004)

2004 ESPN text is `Player Name (TEAM) verb ...`. The rusher path strips it, the
passer path does not:

```
'Bryan Randall (VT) pass incomplete to the right side.'  -> passer_player_name = 'Bryan Randall (VT)'   ✗
'Cedric Humes (VT) rushed left side for no gain.'        -> rusher_player_name = 'Cedric Humes'         ✓
```

`_norm_player_name` strips non-alphanumerics, so `'Matt Ryan (BC)'` normalizes
to `matt ryan bc` and can never match roster `matt ryan`. Passer cells are the
largest single name family, which is the whole 54% → ~90% gap.

Top 2004 unmatched names are all of this shape: `Walter Washington (TEM)` (68),
`Shawn Bell (BU)` (57), `Matt Ryan (BC)` (51), `Omarr Conner (MSU)` (49)…

### 4.2 — 2018 + 2020: empty `game_rosters` (rescrape-fixable)

Full scan, all 18,640 games:

| season | games | game_rosters empty | missing | % bad |
|---|---|---|---|---|
| 2018 | 898 | **376** | 0 | **41.9** |
| 2020 | 706 | 142 | 14 | 22.1 |
| 2014 | 873 | 58 | 0 | 6.6 |
| 2004 | 712 | 37 | 0 | 5.2 |
| 2017 | 890 | 35 | 0 | 3.9 |
| 2022 | 904 | 32 | 0 | 3.5 |
| all others | — | ≤20 each | ≤2 | ≤2.7 |
| **TOTAL** | 18,640 | **829** | 16 | 4.5 |

The 2018 unmatched names are ordinary starting QBs (Sam Hartman, Gardner
Minshew II, Jarrett Stidham, Peyton Ramsey…) — they would match fine if the
roster were present. This is a scrape hole, not a data-availability limit.

`play_participants`: 8,186 empty, of which 8,003 are the structural pre-2014
set. Only **183** are genuine post-2014 holes (2020 alone: 124).

### 4.3 — 2005–2013: regex bleed into the captured name (worth a few pp)

Pass-direction words and missing-space concatenations survive into the name:
`'Dominique Davis screen'`, `'Russell Wilson sideline'`, `'Russell Wilson deep out'`,
`'Raynard Hornetackled by'`, `'Dwayne Harristackled by'`, `'N/A.'`, `'Team'`.
`_PLAYER_NAME_GARBAGE` (cfb_pbp.py:68) does not currently list
`screen|sideline|middle|crossing|deep|out`.

### 4.4 — 2015+: at ceiling

Modern-era misses are ~entirely the legitimate `'TEAM'` sentinel (126 of ~130 in
a 2023 sample) plus single-occurrence regex truncation tails
(`'o the FLA  Graham Mertz'`). Nothing material left.

## 5. What to do, ranked by measured payoff

| # | Action | Where | Est. gain |
|---|---|---|---|
| 1 | Rescrape `game_rosters` for the 845 empty/missing games | cfb-raw | 2018 +24pp, 2020 recovers, ~+1pp global |
| 2 | Strip trailing ` (ABBR)` in the passer capture **and** in `_norm_player_name` | cfb_pbp.py | 2004 +~35pp |
| 3 | Extend `_PLAYER_NAME_GARBAGE` with pass-direction stopwords; fix `Xtackled by` concatenation | cfb_pbp.py:68 | 2005–2013 +2–4pp |
| 4 | Add join fallback tiers below exact: first-initial+last, then unique-last-name | cfb_pbp.py:6723 | 2005–2013 +2–5pp (measured: initial 1.0–3.9pp, last 0.9–2.2pp) |
| 5 | Union all of a team's game rosters across a season into the pre-2014 candidate pool | cfb_pbp.py | small; raises ambiguity risk — tier strictly below exact |

Actions 2–4 are pure reprocess (no new fetches). Action 1 needs the scrape.

**Realistic post-fix targets:** 2004 ≈ 90%, 2005–2013 ≈ 91–95%, 2014+ ≈ 96–99%.
The residual pre-2014 gap is players who never appear on the ~62-name game
roster (walk-ons, special-teams-only, and the genuinely-unstatted) — those have
no ESPN athlete_id reachable from a 2004–2013 payload at all.
