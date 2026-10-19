"""Frozen scorers and the morning Scoreboard (written and committed before the first eligible night).

Tracks
- Gate A (M1 and M1.1 vs the market, forward): every eligible logged forecast that resolves. Blend
  logit(p) = logit(mkt) + w (logit(model) - logit(mkt)), ridge lambda from config, expanding window over
  earlier game dates, first `burn_in_game_dates` dates unscored. Pass: >= min_forecasts over >= min_game_dates
  and the 95% cluster-bootstrap (by game date) lower bound of (market log loss - blend log loss) > 0.
  Gate B (CLV of blend EV >= 3% candidates vs the leave-venue-out consensus close) is scored only after
  Gate A passes; its scorer is added and frozen before then.
- K1 (Kalshi paper maker): through-only fills before min(tip, injury cancel), CLV vs the last two-sided
  Kalshi mid within 10 minutes before tip. Pass: >= 300 fills over >= 40 dates, CLV lower bound > 0 and
  M1.1 beats the naive mid poster (per-date CLV dollars, lower bound > 0).
- N1 (injury reaction): data collection only until its analysis plan is pre-registered.
Results stay SEALED (only progress counts shown) until a track's pre-registered sample is complete.

Run: python -m edge_engine.scoring --data data   ->  data/scoreboard.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

from .m1_live import TEAM_ABBR

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent


def ts(s):
    return datetime.fromisoformat(str(s).replace("Z", "+00:00")) if s else None


def et_date(s):
    return ts(s).astimezone(ET).date().isoformat()


def jsonl(paths):
    out = []
    for p in sorted(paths):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_lower(values: np.ndarray, clusters: np.ndarray, n=2000, seed=29) -> dict:
    s = pd.Series(values).groupby(clusters).apply(np.array).values
    if len(s) < 2:
        return {"mean": float(np.mean(values)) if len(values) else None, "lo": None, "hi": None}
    rng = random.Random(seed)
    m = sorted(float(np.concatenate([s[rng.randrange(len(s))] for _ in s]).mean()) for _ in range(n))
    return {"mean": round(float(np.mean(values)), 6), "lo": round(m[int(0.025 * n)], 6), "hi": round(m[int(0.975 * n) - 1], 6)}


# ---------------- raw store helpers ----------------

class Raw:
    def __init__(self, data: str):
        self.root = Path(data)
        man = self.root / "raw" / "manifest.jsonl"
        self.rows = [json.loads(l) for l in man.read_text().splitlines()] if man.exists() else []

    def load(self, ref: str):
        return json.loads(gzip.open(self.root / ref).read())

    def remaining(self):
        for r in reversed(self.rows):
            if r.get("credits_remaining") not in (None, ""):
                return int(float(r["credits_remaining"]))
        return None

    def credits_since(self, since: datetime) -> int:
        return sum(int(r.get("credits_last") or 0) for r in self.rows if ts(r["retrieved_at"]) and ts(r["retrieved_at"]) >= since)


# ---------------- Gate A ----------------

def outcomes(data: str, season: str) -> dict:
    p = Path(data) / "m1" / "boxscores" / f"{season}.csv.gz"
    if not p.exists():
        return {}
    b = pd.read_csv(p)
    b["d"] = pd.to_datetime(b["date"], utc=True).dt.tz_convert(ET).dt.date.astype(str)
    return {(str(r.player_id), r.d): (None if (r.dnp == 1 or not r.minutes or r.minutes <= 0) else float(r.pts))
            for r in b.itertuples()}


def fit_w(lm, lmod, y, lam):
    f = lambda w: float(ll(1 / (1 + np.exp(-(lm + w * (lmod - lm)))), y).sum() + lam * w * w)
    return float(minimize_scalar(f, bounds=(-1, 2), method="bounded").x)


def gate_a(data: str, cfg: dict, raw: Raw, monitor_logs: list[dict]) -> dict:
    m = cfg["m1"]
    recs = jsonl((Path(data) / "m1" / "forecasts").glob("*.jsonl"))
    seen, rows = set(), []
    for r in recs:                                   # first decision per contract only
        key = (r["event_id"], r.get("player"), r.get("line"), r.get("venue"))
        if key in seen:
            continue
        seen.add(key)
        rows.append(r)
    elig = [r for r in rows if r.get("eligible")]
    out = outcomes(data, cfg["monitor"]["season"])
    res = []
    for r in elig:
        pts = out.get((str(r.get("player_id")), et_date(r["commence_time"])), "missing")
        if pts == "missing" or pts is None or float(pts) == float(r["line"]):
            continue                                 # unplayed yet, DNP (void) or push
        res.append({**r, "y": float(pts > float(r["line"])), "gdate": et_date(r["commence_time"])})
    dates = sorted({r["gdate"] for r in res})
    g = m["gateA"]
    status = {"logged": len(rows), "eligible": len(elig), "resolved": len(res), "game_dates": len(dates),
              "need_forecasts": g["min_forecasts"], "need_game_dates": g["min_game_dates"]}
    complete = len(res) >= g["min_forecasts"] and len(dates) >= g["min_game_dates"]
    status["complete"] = complete
    if not complete or not res:
        status["results"] = "sealed"
        return status
    df = pd.DataFrame(res)
    burn = set(dates[: m["blend"]["burn_in_game_dates"]])
    verdicts = {}
    for name, col in (("M1", "m1_p_over"), ("M1.1", "m1_1_p_over")):
        pb = np.full(len(df), np.nan)
        for d in dates:
            if d in burn:
                continue
            tr, te = df.gdate < d, df.gdate == d
            w = fit_w(logit(df.market_p_over[tr].values), logit(df[col][tr].values), df.y[tr].values, m["blend"]["ridge_lambda"])
            lm, lo = logit(df.market_p_over[te].values), logit(df[col][te].values)
            pb[te.values] = 1 / (1 + np.exp(-(lm + w * (lo - lm))))
        sc = ~np.isnan(pb)
        diff = ll(df.market_p_over.values[sc], df.y.values[sc]) - ll(pb[sc], df.y.values[sc])
        b = boot_lower(diff, df.gdate.values[sc])
        verdicts[name] = {"scored": int(sc.sum()), "market_logloss": round(float(ll(df.market_p_over.values[sc], df.y.values[sc]).mean()), 5),
                          "blend_logloss": round(float(ll(pb[sc], df.y.values[sc]).mean()), 5),
                          "model_only_logloss": round(float(ll(df[col].values[sc], df.y.values[sc]).mean()), 5),
                          "gain": b, "pass": bool(b["lo"] is not None and b["lo"] > 0)}
    status["results"] = verdicts
    return status


# ---------------- K1 ----------------

def kalshi_index(raw: Raw):
    trades, books = {}, defaultdict(list)
    for r in raw.rows:
        if r.get("kind") != "kalshi" or r.get("status") != 200:
            continue
        path = r["path"]
        if "/markets/trades?ticker=" in path:
            t = path.split("ticker=")[1].split("&")[0]
            trades[t] = r["file"]                     # latest fetch wins
        elif "/markets?series_ticker=" in path:
            try:
                for mk in raw.load(r["file"]).get("markets", []):
                    yb, ya = mk.get("yes_bid_dollars"), mk.get("yes_ask_dollars")
                    books[mk.get("ticker")].append((ts(r["retrieved_at"]), float(yb) if yb else None, float(ya) if ya else None))
            except (OSError, ValueError):
                continue
    return trades, books


def cancel_times(monitor_logs: list[dict]) -> dict:
    out = defaultdict(list)
    for e in monitor_logs:
        if e.get("kind") == "injury_change":
            ab = TEAM_ABBR.get(e.get("team"))
            if ab:
                out[ab].append(ts(e["first_seen"]) + timedelta(seconds=60))
    return out


def k1_score(data: str, cfg: dict, raw: Raw, monitor_logs: list[dict]) -> dict:
    from .k1 import event_teams
    k = cfg["k1"]
    quotes = [q for q in jsonl((Path(data) / "k1" / "quotes").glob("*.jsonl")) if q.get("eligible")]
    trades_idx, books = kalshi_index(raw)
    cancels = cancel_times(monitor_logs)
    fills = {"m1_1": [], "m1": [], "naive": []}
    for q in quotes:
        dec, tip = ts(q["decision_at"]), ts(q["tip"])
        teams = event_teams(q["event_ticker"])
        stop = min([tip] + [c for t in teams for c in cancels.get(t, []) if c > dec])
        tr = []
        if q["ticker"] in trades_idx:
            tr = [t for t in raw.load(trades_idx[q["ticker"]]).get("trades", []) if dec < ts(t["created_time"]) < stop]
        snaps = [s for s in books.get(q["ticker"], []) if tip - timedelta(minutes=10) <= s[0] <= tip and s[1] and s[2] and s[1] < s[2]]
        mid = (snaps[-1][1] + snaps[-1][2]) / 2 if snaps else None
        for model in fills:
            o = q.get(f"orders_{model}")
            if not o:
                continue
            for side, bid_key, placed_key, price_key in (("yes", "yes_bid", "yes_placed", "yes_price_dollars"),
                                                         ("no", "no_bid", "no_placed", "no_price_dollars")):
                if not o.get(placed_key):
                    continue
                b = o[bid_key]
                through = sum(float(t["count_fp"]) for t in tr if float(t[price_key]) < b)
                at_price = sum(float(t["count_fp"]) for t in tr if abs(float(t[price_key]) - b) < 1e-9)
                n = min(k["contracts"], through)
                if n <= 0:
                    continue
                fee = math.ceil(k["fees"]["maker_rate"] * n * b * (1 - b) * 100) / 100
                clv = None if mid is None else ((mid - b) if side == "yes" else ((1 - mid) - b))
                fills[model].append({"date": et_date(q["tip"]), "ticker": q["ticker"], "side": side, "price": b, "contracts": n,
                                     "clv": clv, "fee_per_contract": fee / n, "at_price_volume": at_price})
    g = k["gate"]
    f = fills["m1_1"]
    dates = sorted({x["date"] for x in f})
    status = {"rungs_logged": len(quotes), "fills": len(f), "fill_dates": len(dates), "need_fills": g["min_fills"],
              "need_dates": g["min_game_dates"]}
    status["complete"] = len(f) >= g["min_fills"] and len(dates) >= g["min_game_dates"]
    if not status["complete"]:
        status["results"] = "sealed"
        return status
    def clv_stats(fl):
        c = [x for x in fl if x["clv"] is not None]
        return boot_lower(np.array([x["clv"] for x in c]), np.array([x["date"] for x in c])), len(fl) - len(c)
    main, missing = clv_stats(f)
    per_date = lambda fl: pd.Series({d: sum((x["clv"] or 0) * x["contracts"] for x in fl if x["date"] == d) for d in dates})
    vs = per_date(f) - per_date(fills["naive"]).reindex(dates).fillna(0)
    vsb = boot_lower(vs.values, np.array(dates))
    status["results"] = {"m1_1_clv_cents": {k2: (None if v is None else round(v * 100, 3)) for k2, v in main.items()},
                         "missing_close": missing, "m1_1_minus_naive_per_date_dollars": vsb,
                         "m1_clv": clv_stats(fills["m1"])[0], "naive_clv": clv_stats(fills["naive"])[0],
                         "pass": bool(main["lo"] is not None and main["lo"] > 0 and vsb["lo"] is not None and vsb["lo"] > 0)}
    return status


# ---------------- health, N1, scoreboard ----------------

def health(data: str, cfg: dict, raw: Raw, now: datetime) -> dict:
    logs = sorted((Path(data) / "monitor").glob("*.jsonl"))
    out = {"monitor_last_night": None, "status": "grey", "problems": []}
    if logs:
        last = jsonl([logs[-1]])
        kinds = defaultdict(int)
        for e in last:
            kinds[e["kind"]] += 1
        games = next((e.get("n") for e in last if e["kind"] == "games"), None)
        out["monitor_last_night"] = {"date": logs[-1].stem, "games": games, "polls": kinds["poll"],
                                     "injury_changes": kinds["injury_change"], "kalshi_snapshots": kinds["kalshi_snapshot"],
                                     "finished": kinds["done"] > 0}
        errs = {k: kinds[k] for k in ("poll_error", "credit_guard", "box_ingest_error", "forecast_error", "kalshi_error", "fetch_error") if kinds[k]}
        if games:
            if not kinds["done"]:
                out["problems"].append("monitor did not finish")
            for k in ("credit_guard", "box_ingest_error", "forecast_error"):
                if k in errs:
                    out["problems"].append(f"{k} x{errs[k]}")
            if kinds["kalshi_error"] > 3 or kinds["poll_error"] > 5:
                out["problems"].append("repeated fetch errors")
        out["errors"] = errs
        age_h = (now - ts(last[-1]["at"])).total_seconds() / 3600 if last else 99
        if age_h > 36:
            out["problems"].append(f"no monitor activity for {age_h:.0f} h")
        out["status"] = "red" if out["problems"] else "green"
    snaps_24h = sum(1 for r in raw.rows if r.get("kind") == "availability" and ts(r["retrieved_at"]) and ts(r["retrieved_at"]) > now - timedelta(hours=24))
    out["injury_archive_snapshots_24h"] = snaps_24h
    if snaps_24h == 0:
        out["problems"].append("injury archive idle 24 h")
        out["status"] = "red"
    return out


def run(data: str, cfg: dict, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    raw = Raw(data)
    logs = jsonl((Path(data) / "monitor").glob("*.jsonl"))
    a = gate_a(data, cfg, raw, logs)
    k = k1_score(data, cfg, raw, logs)
    n1 = {"injury_changes": sum(1 for e in logs if e.get("kind") == "injury_change"),
          "reaction_polls": sum(1 for e in logs if e.get("kind") == "poll" and "inj+" in str(e.get("reason"))),
          "note": "collecting; the analysis plan is written and frozen before any of this data is analysed"}
    h = health(data, cfg, raw, now)
    rem = raw.remaining()
    burn7 = raw.credits_since(now - timedelta(days=7))
    def track_status(t):
        if t.get("complete"):
            r = t.get("results", {})
            ok = any(v.get("pass") for v in r.values() if isinstance(v, dict)) if "pass" not in r else r["pass"]
            return "pass" if ok else "fail"
        return "collecting"
    sb = {"generated_at": now.isoformat(), "paper_only": True,
          "health": h,
          "credits": {"remaining": rem, "used_last_7_days": burn7, "floor": cfg["d1"]["min_remaining_credits"],
                      "reset": "Nov 18"},
          "tracks": {
              "gate_a": {"title": "Does the points model beat the market?", "state": track_status(a), **a},
              "k1": {"title": "Do Kalshi resting orders beat Kalshi's close?", "state": track_status(k), **k},
              "n1": {"title": "Are books slow after injury news?", "state": "collecting", **n1}},
          "rules": "Results stay sealed until each test's sample is complete. Nothing here is a bet."}
    p = Path(data) / "scoreboard.json"
    p.write_text(json.dumps(sb, indent=1, default=str))
    print(json.dumps({"health": h["status"], "gate_a": sb["tracks"]["gate_a"]["state"], "k1": sb["tracks"]["k1"]["state"],
                      "credits": rem}))
    return sb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    run(a.data, yaml.safe_load(open(ROOT / "config" / "rules.yaml")))
