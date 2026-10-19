# edge-ledger-engine

Deterministic pricing engine for the Edge Ledger +EV system (NFL, NBA). It fetches odds directly
from The Odds API, stores every raw response immutably, computes fair prices, EV, stakes and CLV in
tested Python, and writes a file ledger that a reporting agent copies verbatim into the dashboard.
The agent never produces a number.

## Pipeline

1. **Adapters** (`client.py`): direct HTTPS to The Odds API and ESPN. Credits are read from response headers.
2. **Raw store** (`data/raw/`, `data` branch): gzipped response bytes + `manifest.jsonl` with request params
   (key redacted), `retrieved_at`, HTTP status, SHA-256 and credit counts. Never overwritten.
3. **Pricing** (`normalize.py`, `fair.py`, `ladder.py`, `odds.py`): exact-contract pairing, power devig
   (multiplicative and additive logged alongside), weighted sharp consensus, alt-ladder distributions with
   isotonic fit, push mass at whole numbers, and no interpolation across key numbers.
4. **Eligibility and risk** (`evaluate.py`): market-specific staleness, source agreement, single-source and
   prop-source minimums, uncertainty haircut, execution drag, quarter Kelly with game/team/player exposure caps.
5. **Ledger** (`ledger.py`, `data/ledger`, `data/runs`, `data/sync`): every candidate (including rejected ones,
   with reasons), the exact source quotes and timestamps, engine and config versions, and raw file refs.

## Evaluation

- **Primary CLV**: Pinnacle's close for the exact contract, devigged. Reported as probability points
  (close fair minus break-even), log-odds, and EV at the close.
- **Secondary CLV**: independent consensus without Pinnacle (Kalshi + LowVig). Never substituted silently.
- **Realized ROI**: tracked separately. Real bets logged from slips get CLV at the price actually taken.
- Close timestamps: requested time, API retrieval time and Pinnacle quote age are all stored.

## Validation (pre-registered in `config/rules.yaml`)

- Paper mode until the 2025 NFL replay's sealed holdout (weeks 13–18) shows bet-tier Pinnacle-close CLV with a
  95% cluster-bootstrap lower bound above zero, then 4 live paper weeks.
- Replay re-prices each flag 10 minutes later at the same venue (execution drag, miss rate) and reports
  thresholds of 2/3/4/5%.
- Learning is frozen to monitoring until the evaluation period ends; any later rule change needs n ≥ 200,
  95% intervals, Holm correction and clustering by game/date.

## Run

```
pip install -r requirements.txt
python -m pytest -q
ODDS_API_KEY=... python -m edge_engine.cli --data data scan --sport NBA --scan nba-evening --window today_et --derivatives --props
python -m edge_engine.cli --data data close --sport NBA --back 30 --ahead 240
python -m edge_engine.cli --data data grade
python scripts/backtest_all.py --data data && python -m edge_engine.cli --data data backtest-report
```

GitHub Actions (`.github/workflows/engine.yml`) runs these on `workflow_dispatch` from the scheduler, runs the
test suite first (fail closed), and commits results to the `data` branch.
