# pregame_wp test fixtures

Static lookup tables used as offline fixtures for Track 4 unit tests.
These are NOT live API captures — they are derived from the notebook
`cfbfastR-dev/cfbfastR-cfb-raw/docs/pregame-wp-model-training-port-spec`.

## Files

| File | Source | Rows | Notes |
|------|--------|------|-------|
| `ep.csv` | `python/pregame_wp/assets/ep.csv` | 101 | EP curve, yardlines 0–100 |
| `punt_sr.csv` | `python/pregame_wp/assets/punt_sr.csv` | ~40 | Expected punt net yardage by ball position |

Captured: 2026-06-13
