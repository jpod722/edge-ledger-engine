"""L1 confirmatory run (plan: config/prereg/l1-confirmatory-plan.json, spec: config/prereg/l1-spec.yaml).

pull:    one historical moneyline snapshot per planned request (hard cap 6,100 credits), then per game the frozen
         entry/freshness checks and the DK + Pinnacle favorite prices at the verified snapshot -> data/l1/confirm_quotes.json
analyze: the pre-registered primary test, run once on the quotes file -> data/l1/confirm_result.json
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from zoneinfo import ZoneInfo

from .m1_live import TEAM_ABBR

SK = "basketball_nba"
ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "config" / "prereg" / "l1-confirmatory-plan.json"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def pull(api, data: str, books: str):
    plan = json.loads(PLAN.read_text())
    snaps, errors = {}, []
    for r in plan["requests"]:
        try:
            d, _, ref = api.historical_odds(SK, r.replace("+00:00", "Z"), "h2h", books)
            snaps[r] = {"snapshot": d.get("timestamp"), "previous": d.get("previous_timestamp"), "ref": ref, "data": d.get("data", [])}
        except Exception as e:  # noqa: BLE001  (credit cap or API error: stop, never expand)
            errors.append({"request": r, "error": str(e)[:200]})
            break
    out = []
    for g in plan["games"]:
        row = {k: g[k] for k in ("game_id", "season", "role", "fav", "home", "away", "spread", "end_q2")}
        s = snaps.get(g["request_at"])
        if s is None:
            out.append({**row, "status": "not_pulled"})
            continue
        S, P, end, q3 = ts(s["snapshot"]), ts(s["previous"]), ts(g["end_q2"]), ts(g["q3_start"])
        row.update(snapshot=s["snapshot"], delay_min=round((S - end).total_seconds() / 60, 2),
                   in_window=bool(S >= end and (P is None or P < end) and S < q3))
        ev = next((e for e in s["data"] if {TEAM_ABBR.get(e["home_team"]), TEAM_ABBR.get(e["away_team"])} == {g["home"], g["away"]}
                   and abs(ts(e["commence_time"]) - end) < timedelta(hours=3)), None)
        fav_name = None
        if ev:
            fav_name = ev["home_team"] if TEAM_ABBR.get(ev["home_team"]) == g["fav"] else ev["away_team"]
        q = {}
        for b in (ev or {}).get("bookmakers", []):
            if b["key"] not in ("draftkings", "pinnacle"):
                continue
            m = next((m for m in b.get("markets", []) if m["key"] == "h2h"), None)
            if not m:
                continue
            prices = {o["name"]: o["price"] for o in m.get("outcomes", [])}
            q[b["key"]] = {"last_update": m.get("last_update"), "fav": prices.get(fav_name),
                           "dog": next((v for k, v in prices.items() if k != fav_name), None)}
        fresh = lambda k: bool(k in q and ts(q[k]["last_update"]) and ts(q[k]["last_update"]) >= end
                               and ts(q[k]["last_update"]) >= S - timedelta(minutes=2))
        row.update(event_found=ev is not None, dk=q.get("draftkings"), pin=q.get("pinnacle"),
                   dk_fresh=fresh("draftkings"), pin_fresh=fresh("pinnacle"))
        row["scored"] = bool(row["in_window"] and row["dk_fresh"] and row["pin_fresh"]
                             and q["draftkings"]["fav"] and q["pinnacle"]["fav"] and q["pinnacle"]["dog"])
        out.append(row)
    p = Path(data) / "l1" / "confirm_quotes.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"credits_used": api.used, "requests_made": len(snaps), "errors": errors, "games": out}, indent=1))
    print(json.dumps({"credits_used": api.used, "requests_made": len(snaps), "errors": errors,
                      "scored": {r: sum(1 for x in out if x["role"] == r and x.get("scored")) for r in ("trigger", "control")}}))


def edge(row):
    pf, pd_ = 1 / row["pin"]["fav"], 1 / row["pin"]["dog"]
    p = pf / (pf + pd_)                       # Pinnacle no-vig (multiplicative)
    return p * row["dk"]["fav"] - 1, p


def boot_lb(vals, dates, n=5000, seed=31, alpha=0.05):
    vals, dates = np.asarray(vals, float), np.asarray(dates)
    u, inv = np.unique(dates, return_inverse=True)
    s, c = np.bincount(inv, weights=vals), np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), (n, len(u)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return float(vals.mean()), float(np.quantile(m, alpha)), m


def analyze(data: str, outcomes: dict):
    q = json.loads((Path(data) / "l1" / "confirm_quotes.json").read_text())
    rows = [r for r in q["games"] if r.get("scored")]
    for r in rows:
        r["E"], r["p_pin"] = edge(r)
        r["date"] = ts(r["end_q2"]).astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
        r["fav_won"] = outcomes.get(r["game_id"])
    T = [r for r in rows if r["role"] == "trigger"]
    C = [r for r in rows if r["role"] == "control"]
    mT, lbT, bT = boot_lb([r["E"] for r in T], [r["date"] for r in T])
    mC, _, _ = boot_lb([r["E"] for r in C], [r["date"] for r in C])
    # difference: resample trigger and control dates independently
    _, _, bC = boot_lb([r["E"] for r in C], [r["date"] for r in C], seed=32)
    diff_lb = float(np.quantile(bT - bC, 0.05))
    n_dates = len({r["date"] for r in T})
    enough = len(T) >= 150 and n_dates >= 60
    res = {"scored_triggers": len(T), "trigger_dates": n_dates, "scored_controls": len(C),
           "primary": {"mean_E_trigger": mT, "lb95_E_trigger": lbT, "mean_E_control": mC,
                       "diff_trigger_minus_control": mT - mC, "lb95_diff": diff_lb},
           "pass": bool(enough and lbT > 0 and diff_lb > 0), "sample_minimum_met": enough}
    sec = {}
    for season in sorted({r["season"] for r in T}):
        sub = [r["E"] for r in T if r["season"] == season]
        sec[f"mean_E_{season}"] = float(np.mean(sub)) if sub else None
    won = [r for r in T if r["fav_won"] is not None]
    if won:
        sec["fav_win_rate_trigger"] = float(np.mean([r["fav_won"] for r in won]))
        sec["pinnacle_p_trigger"] = float(np.mean([r["p_pin"] for r in won]))
        sec["roi_at_dk_trigger"] = float(np.mean([(r["dk"]["fav"] - 1) if r["fav_won"] else -1 for r in won]))
    for lo, hi, lab in ((6.5, 9.5, "spread_6.5-9.5"), (10, 99, "spread_10+")):
        sub = [r["E"] for r in T if lo <= r["spread"] <= hi]
        sec[f"mean_E_{lab}"] = float(np.mean(sub)) if sub else None
    sec["median_delay_min"] = float(np.median([r["delay_min"] for r in T])) if T else None
    res["secondary"] = sec
    (Path(data) / "l1" / "confirm_result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return res
