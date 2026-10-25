# Undercurrent / M3 refinement backlog (ideas only — Justin, 2026-10-25)

Baseline B is FROZEN for the whole Undercurrent test (config/m3_baseline_frozen.json). Nothing here touches the live
test. Each idea becomes a *challenger*: pre-registered in config/registry.yaml, fitted on training seasons only,
screened on 2024-25, then scored only on Undercurrent dates AFTER the verdict or on a fresh forward window
(alpha = 0.05 / 2^k for the k-th challenger). Ideas are never chosen by looking at Undercurrent results.

| # | Idea | Why it might help | Source / status |
|---|------|-------------------|-----------------|
| 1 | Vacated minutes from Out teammates | biggest known line-lag spot | m3-spec addition 1; screening approved (exploratory) |
| 2 | Vacated usage (FGA + 0.44 FTA + TOV) | role changes when a scorer sits | addition 2 |
| 3 | Blowout minutes (starters x spread) | starters sit 4th quarters in mismatches | addition 3 |
| 4 | Pace (game total vs league mean) | more possessions, more points | addition 4 |
| 5 | Rest (back-to-back, 3-in-4) | fatigue / load management | addition 5 |
| 6 | Shot profile vs opponent zones | matchup efficiency | addition 6 |
| 7 | Calibration layer (shrink model toward market) | week-1 model flagged 75% of props = overconfident | new; would need its own registration |
| 8 | Early-season prior from last season, decaying | the week-1 signal used last season's history | new; tests the "stale lines" story directly |
| 9 | Minutes restrictions / return-from-injury flags | first games back are capped | new; needs a timestamped source |
| 10 | Sizing grades (EV tiers, fractional Kelly) | only if higher-EV picks win more | post-verdict pilot only |
