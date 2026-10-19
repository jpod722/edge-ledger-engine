"""Pre-run probe for the derivatives replay: measures real historical credit cost, snapshot cadence
and per-book coverage on a couple of 2025 NFL games before any large spend.

Nothing here prices or selects a bet. Output: data/probe/<run_id>.json.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from .backtest import et, iso_z
from .client import OddsAPI, RawStore, api_key, utcnow
from .normalize import parse_ts

PROBE_MARKETS = "alternate_spreads,alternate_totals,team_totals,alternate_team_totals,spreads_h1,totals_h1,h2h_h1"
PROBE_BOOKS = "pinnacle,lowvig,betonlineag,kalshi,draftkings,fanduel,betmgm,williamhill_us,espnbet,betrivers"


def coverage(payload: dict) -> dict:
    """Per market, per book: outcomes, distinct points, two-sided rungs, and quote age at the snapshot."""
    snap = parse_ts(payload.get("timestamp"))
    ev = payload.get("data") or {}
    out: dict = {}
    for b in ev.get("bookmakers", []):
        for m in b.get("markets", []):
            pts: dict = {}
            for o in m.get("outcomes", []):
                key = (o.get("description"), abs(o["point"]) if "spreads" in m["key"] and o.get("point") is not None
                       else o.get("point"))
                pts.setdefault(key, set()).add(o["name"])
            lu = parse_ts(m.get("last_update") or b.get("last_update"))
            out.setdefault(m["key"], {})[b["key"]] = {
                "outcomes": len(m.get("outcomes", [])),
                "points": len(pts),
                "two_sided_rungs": sum(1 for s in pts.values() if len(s) >= 2),
                "age_s": round((snap - lu).total_seconds()) if snap and lu else None,
            }
    return out


def run(a, cfg) -> dict:
    store = RawStore(a.data)
    api = OddsAPI(api_key(), store, a.max_credits)
    sk = "americanfootball_nfl"
    d = __import__("datetime").date.fromisoformat(a.date)
    t0 = et(d, 9, 48)
    calls = []

    def log(kind, payload, ref, requested):
        calls.append({"kind": kind, "requested": iso_z(requested), "snapshot": payload.get("timestamp"),
                      "previous": payload.get("previous_timestamp"), "next": payload.get("next_timestamp"),
                      "credits": api.store_last, "ref": ref})

    # wrap _get so each call's billed credits (from the response header) is recorded
    orig = api._get

    def _get(path, params, label, est):
        before = api.used
        res = orig(path, params, label, est)
        api.store_last = api.used - before
        return res
    api._get = _get

    evs, _, ref = api._get(f"/historical/sports/{sk}/events", {"date": iso_z(t0), "dateFormat": "iso"},
                           f"{sk}_hist_events", 1)
    log("events", evs, ref, t0)
    games = sorted(evs.get("data", []), key=lambda e: e["commence_time"])
    games = [g for g in games if parse_ts(g["commence_time"]).astimezone(t0.tzinfo).date() == d]
    picks = []
    for want in (13, 16):  # first 1 PM game, first late-afternoon game
        g = next((g for g in games if parse_ts(g["commence_time"]).astimezone(t0.tzinfo).hour >= want), None)
        if g and g not in picks:
            picks.append(g)
    results = []
    for g in picks:
        p, _, ref = api._get(f"/historical/sports/{sk}/events/{g['id']}/odds",
                             {"date": iso_z(t0), "markets": PROBE_MARKETS, "bookmakers": PROBE_BOOKS,
                              "oddsFormat": "decimal", "dateFormat": "iso"},
                             f"{sk}_hist_event_{g['id'][:8]}", 10 * len(PROBE_MARKETS.split(",")))
        log("event_odds", p, ref, t0)
        results.append({"event_id": g["id"], "game": f"{g['away_team']} @ {g['home_team']}",
                        "commence_time": g["commence_time"],
                        "minutes_to_start": round((parse_ts(g["commence_time"]) - t0).total_seconds() / 60),
                        "coverage": coverage(p)})
    # snapshot cadence at short delays: one market only, first game
    lat = []
    if picks:
        g = picks[0]
        for dm in a.delays:
            t = t0 + timedelta(minutes=dm)
            p, _, ref = api._get(f"/historical/sports/{sk}/events/{g['id']}/odds",
                                 {"date": iso_z(t), "markets": "alternate_spreads", "bookmakers": PROBE_BOOKS,
                                  "oddsFormat": "decimal", "dateFormat": "iso"},
                                 f"{sk}_hist_event_{g['id'][:8]}_d{dm}", 10)
            log(f"delay_{dm}m", p, ref, t)
            lat.append({"delay_min": dm, "requested": iso_z(t), "snapshot": p.get("timestamp"),
                        "coverage": coverage(p).get("alternate_spreads", {})})
    out = {"run_id": f"{utcnow().strftime('%Y%m%d-%H%M%S')}-probe", "date": a.date, "decision_at": iso_z(t0),
           "markets": PROBE_MARKETS, "bookmakers": PROBE_BOOKS, "credits_total": api.used,
           "calls": calls, "events": results, "latency": lat}
    path = Path(a.data) / "probe" / f"{out['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps({"credits_total": api.used, "calls": [(c["kind"], c["credits"], c["snapshot"]) for c in calls]}))
    return out
