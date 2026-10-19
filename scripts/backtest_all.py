"""Run every un-replayed week of the 2025 NFL season, committing after each week."""
import argparse
import subprocess
from datetime import date, timedelta
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--max-weeks", type=int, default=18)
a = ap.parse_args()
done = {int(p.stem) for p in (Path(a.data) / "backtest" / "weeks").glob("*.json")}
start = date(2025, 9, 4)
ran = 0
for w in range(1, 19):
    if w in done or ran >= a.max_weeks:
        continue
    d = start + timedelta(days=7 * (w - 1))
    subprocess.run(["python", "-m", "edge_engine.cli", "--data", a.data, "backtest-week", "--start", d.isoformat()], check=True)
    subprocess.run("cd {0} && git add -A && (git commit -q -m 'backtest week {1}' || true) && "
                   "(git push -q origin data || true)".format(a.data, w), shell=True, check=True)
    ran += 1
