"""L1 pilot plan (free inputs only): halftime triggers and controls for 2025-26, and the snapshot request times.

    python -m edge_engine.l1_plan --pbp PBP_2025-26.jsonl.gz --odds ESPN_ODDS.json --box BOX.csv.gz --out PLAN.json

Rules (config/prereg/l1-spec.yaml): favorite = ESPN's first non-live line, |spread| >= 6.5. Expected margin at the half
= 0.5 x spread. Trigger: favorite margin at end of Q2 <= 0.25 x spread. Control: margin >= 0.5 x spread.
Entry window for a game: [end of Q2, first Q3 play). A request at time R returns the latest snapshot <= R, which lies in
[R - 5 min, R] on the 5-minute grid; so R must be in [end + 5 min, Q3 start) for the snapshot to fall in the window.
Request times are chosen by interval stabbing over these windows (fewest pulls), from timing alone, never prices.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from datetime import datetime, timedelta

import pandas as pd

PILOT_FROM, PILOT_TO = "2025-10-21", "2025-11-30"
SEED = 20261024


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def favorite(items):
    for i in items or []:
        if "live" in (i.get("provider") or "").lower():
            continue
        det = (i.get("details") or "").strip()
        if not det or det.upper() in ("EVEN", "PK"):
            return None, 0.0, i.get("provider")
        abbr, _, pts = det.rpartition(" ")
        try:
            return abbr, abs(float(pts)), i.get("provider")
        except ValueError:
            return None, None, i.get("provider")
    return None, None, None


def build(pbp_path, odds_path, box_path):
    odds = json.load(open(odds_path))
    box = pd.read_csv(box_path, usecols=["game_id", "team", "home"]).drop_duplicates(["game_id", "team"])
    teams = {str(g): {("home" if h else "away"): t for t, h in zip(d.team, d.home)} for g, d in box.groupby("game_id")}
    rows = []
    for line in gzip.open(pbp_path, "rt"):
        r = json.loads(line)
        gid = r["game_id"]
        fav, spread, prov = favorite(odds.get(gid))
        tm = teams.get(gid, {})
        end2 = q3 = None
        h2 = a2 = None
        for p in r.get("plays", []):
            per = (p.get("period") or {}).get("number")
            if per == 2 and (p.get("type") or {}).get("text") == "End Period" and p.get("wallclock"):
                end2, h2, a2 = ts(p["wallclock"]), p.get("homeScore"), p.get("awayScore")
            if per == 3 and p.get("wallclock") and q3 is None and (p.get("type") or {}).get("text") not in ("End Period",):
                q3 = ts(p["wallclock"])
        row = {"game_id": gid, "home": tm.get("home"), "away": tm.get("away"), "fav": fav, "spread": spread,
               "line_provider": prov, "end_q2": end2.isoformat() if end2 else None, "q3_start": q3.isoformat() if q3 else None}
        if fav and spread is not None and spread >= 6.5 and end2 and q3 and h2 is not None and fav in (tm.get("home"), tm.get("away")):
            margin = (h2 - a2) if fav == tm.get("home") else (a2 - h2)
            row["fav_margin_half"] = margin
            row["group"] = "trigger" if margin <= 0.25 * spread else ("control" if margin >= 0.5 * spread else "neither")
        else:
            row["group"] = "excluded"
        rows.append(row)
    return pd.DataFrame(rows)


def stab(windows):
    """Fewest request times R such that every window [lo, hi) contains one; windows as (lo, hi, key)."""
    picks, assign = [], {}
    for lo, hi, key in sorted(windows, key=lambda w: w[1]):
        if picks and lo <= picks[-1] < hi:
            assign[key] = picks[-1]
            continue
        r = hi - timedelta(seconds=30)            # latest safe request inside the window
        if r < lo:
            assign[key] = None
            continue
        picks.append(r)
        assign[key] = r
    return picks, assign


def plan(df: pd.DataFrame, n=50):
    d = df.copy()
    d["date_et"] = pd.to_datetime(d.end_q2, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    pilot = d[(d.date_et >= PILOT_FROM) & (d.date_et <= PILOT_TO)]
    rng = random.Random(SEED)
    trig = sorted(pilot[pilot.group == "trigger"].game_id)
    ctrl = sorted(pilot[pilot.group == "control"].game_id)
    pick_t = sorted(rng.sample(trig, min(n, len(trig))))
    pick_c = sorted(rng.sample(ctrl, min(n, len(ctrl))))
    # Entry = FIRST snapshot at or after the end of Q2 (reviewer condition). Snapshots sit on a 5-minute grid observed
    # at :x0:37 / :x5:37 UTC (feasibility check, Dec 28 2025). Each game's request is its first grid point G >= end of
    # Q2; games sharing G share one pull. The runner verifies per game: snapshot >= end AND previous snapshot < end
    # (so it is the first), and snapshot < first Q3 play. Any failure is recorded, never replaced.
    OFF = timedelta(seconds=37)

    def first_grid(t):
        base = t - OFF
        k = -(-int(base.timestamp()) // 300)
        return datetime.fromtimestamp(k * 300, tz=base.tzinfo) + OFF

    sub = d[d.game_id.isin(pick_t + pick_c)].copy()
    sub["request_dt"] = [first_grid(ts(e)) for e in sub.end_q2]
    t_req = sorted(set(sub[sub.game_id.isin(pick_t)].request_dt))
    spread_reqs = set(random.Random(SEED + 1).sample(t_req, min(20, len(t_req))))
    budget = 1000 - 10 * len(t_req) - 10 * len(spread_reqs)
    c_only = sorted(set(sub[sub.game_id.isin(pick_c)].request_dt) - set(t_req))
    kept_c = c_only[: max(budget // 10, 0)]
    dropped = [g for g, r in zip(sub.game_id, sub.request_dt) if g in pick_c and r not in t_req and r not in kept_c]
    chosen = sub[~sub.game_id.isin(dropped)].copy()
    chosen["role"] = ["trigger" if g in pick_t else "control" for g in chosen.game_id]
    chosen["request_at"] = [r.isoformat() for r in chosen.request_dt]
    chosen["short_window_attempt"] = False
    chosen = chosen.drop(columns=["request_dt"])
    short = []
    picks = sorted(spread_reqs)
    picks_ml = sorted((set(t_req) - spread_reqs) | set(kept_c))
    counts = {s: {g: int(n_) for g, n_ in d[d.date_et.str.startswith(s[:4])].groupby("group").size().items()} for s in ("2025",)}
    season_counts = d.groupby("group").size().to_dict()
    return chosen, picks, {"season_2025_26_counts": {k: int(v) for k, v in season_counts.items()},
                           "pilot_window": [PILOT_FROM, PILOT_TO],
                           "pilot_pool": {"trigger": len(trig), "control": len(ctrl)},
                           "chosen": {"trigger": len(pick_t), "control": int((chosen.role == "control").sum())},
                           "controls_dropped_for_cap": len(dropped),
                           "requests_ml_spread": len(picks), "requests_ml_only": len(picks_ml),
                           "est_credits": 20 * len(picks) + 10 * len(picks_ml),
                           "short_window_attempts": len(short)}, picks_ml


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbp", required=True)
    ap.add_argument("--odds", required=True)
    ap.add_argument("--box", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = build(a.pbp, a.odds, a.box)
    chosen, picks, summary, picks_ml = plan(df)
    json.dump({"summary": summary, "requests": [p.isoformat() for p in picks], "requests_ml_only": [p.isoformat() for p in picks_ml],
               "games": chosen.to_dict(orient="records")}, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1))
