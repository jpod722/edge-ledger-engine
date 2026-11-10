# Early-tip games lose their decision snapshot when the monitor hands off (draft for review)

Status: diagnosis only. No code, workflow or test rule is changed by this note.

## What happens

On days with an early tip (decision before 5:30 PM ET), one game's props are excluded as
`captured_after_decision` even though the monitor **did** take that game's decision snapshot on time.

1. Run A takes the early game's `decision` poll about 5 minutes before its decision time.
2. Forecasts are only written once every remaining game has a first snapshot
   (`Monitor.run`: `all(e["id"] in self.first_payload ...)`). The evening games' decisions (5:30 PM ET)
   come after run A's job limit, so run A never writes forecasts. It hands off (`relay`, `pending_closes`)
   and ends.
3. Run B starts with an empty in-memory `first_payload`. It reads the log and correctly does not repeat the
   early game's decision poll (`done_dec`), but it does not reload that snapshot either. The first payload it
   holds for the early game is the `close` poll about 6 minutes before tip.
4. Run B's forecasts use that close snapshot, so `decision_at` is about 84 minutes after the decision and
   `m3_forward` excludes all of that game's props.

The exclusion itself is correct (no look-ahead enters H-U1). The loss is avoidable.

## Evidence (data branch, monitor logs and `m1/forecasts`)

| Date | Game | Decision (ET) | Decision poll (UTC) | Run A ends (UTC) | Forecast `decision_at` (UTC) | Props lost |
|---|---|---|---|---|---|---|
| Oct 31 | POR @ UTA | 1:40 PM | 17:35 | 18:28 | 19:04 (close poll) | 14 |
| Nov 1 | BOS @ ORL | 2:10 PM | 19:05 | 19:54 | 20:34 (close poll) | 12 |
| Nov 7 | DEN @ IND | 3:40 PM | 20:35 | 21:30 | 22:04 (close poll) | 12 |
| Nov 8 | PHI @ CHI | 2:10 PM | 19:05 | 19:38 | 20:34 (close poll) | 13 |

Example: Nov 8, `raw/2026-11-08/190514_basketball_nba_event_a5d2d9c6_ecb98adf.json.gz` holds the on-time
decision snapshot, with DraftKings `player_points` last updated at 19:04:27Z. The forecast rows for that game
carry `decision_at` 20:34:00Z, the close poll.

There are 52 props in total, the four games above plus one Oct 20 prop. This repeats on most weekends
and was not covered by `d88dadc` (relay) or `6f3db76` (runtime cap).

## The other late props (94, not this bug)

They have 5:30 PM ET decisions and were captured 2 to 4 hours late, and 79 of the 94 are in games whose other
props were captured on time. Later runs append a second forecast batch built from close snapshots
(`m1_live.write_and_push` appends; `m3_forward` keeps the first row per player and game). Props first
listed by DraftKings after the decision therefore enter only with a late `decision_at`. They are correctly
excluded and not a scheduling problem. This is not yet confirmed prop by prop against the raw decision
snapshots.

## Proposed fix (for review, not implemented)

At the start of a run, rebuild `first_payload` for every event whose `decision` poll is already in today's
log, by loading the raw snapshot named in that poll's `ref`, with its original `retrieved_at`. A run that
writes forecasts then uses the true decision snapshot, whichever run took it.

- Operational only: the decision time, eligibility rules and exclusion of late captures are unchanged.
- It applies from the date it ships. Past nights are not rebuilt.
- Justified on operational grounds (snapshots taken on time were not used), not on how those props performed.
- Test: a two-run simulation in which run A takes an early decision poll and stops, and run B must write
  forecasts with run A's `retrieved_at`.
