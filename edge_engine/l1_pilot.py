"""L1 data-quality pilot (reviewer-approved Oct 24, hard cap 1,000 credits). Pulls the planned historical snapshots and
reports DATA QUALITY ONLY: coverage, entry delay, book presence, quote freshness, exact-contract spread matching.
Prices, price edges, ROI and outcomes are never computed or printed here (sealed). Raw payloads are archived."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

from .m1_live import TEAM_ABBR

SK = "basketball_nba"
PLAN = Path(__file__).resolve().parent.parent / "config" / "prereg" / "l1-pilot-plan.json"


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def run(api, data: str, books: str):
    plan = json.loads(PLAN.read_text())
    reqs = [(r, "h2h,spreads") for r in plan["requests"]] + [(r, "h2h") for r in plan["requests_ml_only"]]
    snaps, errors = {}, []
    for r, mk in sorted(reqs):
        try:
            d, _, ref = api.historical_odds(SK, r.replace("+00:00", "Z"), mk, books)
            snaps[r] = {"markets": mk, "snapshot": d.get("timestamp"), "previous": d.get("previous_timestamp"),
                        "ref": ref, "data": d.get("data", [])}
        except Exception as e:  # noqa: BLE001  (credit guard or API error: stop, never expand)
            errors.append({"request": r, "error": str(e)[:200]})
            break
    rows = []
    for g in plan["games"]:
        row = {"game_id": g["game_id"], "role": g["role"], "short_window_attempt": g.get("short_window_attempt", False)}
        s = snaps.get(g["request_at"]) if g.get("request_at") else None
        if s is None:
            row["status"] = "not_pulled"
            rows.append(row)
            continue
        S, end, q3 = ts(s["snapshot"]), ts(g["end_q2"]), ts(g["q3_start"])
        row["delay_min"] = round((S - end).total_seconds() / 60, 2) if S else None
        P = ts(s.get("previous"))
        row["is_first_after_end"] = bool(S and S >= end and (P is None or P < end))
        row["before_q3"] = bool(S and S < q3)
        row["in_window"] = bool(row["is_first_after_end"] and row["before_q3"])
        ev = next((e for e in s["data"] if {TEAM_ABBR.get(e["home_team"]), TEAM_ABBR.get(e["away_team"])} == {g["home"], g["away"]}
                   and abs(ts(e["commence_time"]) - end) < timedelta(hours=3)), None)
        row["event_found"] = ev is not None
        books_ = {b["key"]: {m["key"]: m for m in b.get("markets", [])} for b in (ev or {}).get("bookmakers", [])}
        for bk in ("draftkings", "pinnacle"):
            m = books_.get(bk, {}).get("h2h")
            row[f"{bk}_h2h"] = m is not None
            lu = ts(m.get("last_update")) if m else None
            row[f"{bk}_fresh"] = bool(lu and S and lu >= end and lu >= S - timedelta(minutes=2))
        if "spreads" in s["markets"]:
            pts = {}
            for bk in ("draftkings", "pinnacle"):
                m = books_.get(bk, {}).get("spreads")
                pts[bk] = sorted(o.get("point") for o in m["outcomes"]) if m else None
            row["spread_pulled"] = True
            row["spread_exact_match"] = bool(pts["draftkings"] and pts["draftkings"] == pts["pinnacle"])
        row["scored"] = bool(row["in_window"] and row["event_found"] and row["draftkings_h2h"] and row["pinnacle_h2h"]
                             and row["draftkings_fresh"] and row["pinnacle_fresh"])
        rows.append(row)
    def share(role, key):
        sub = [x for x in rows if x["role"] == role and key in x]
        return {"n": len(sub), "yes": sum(bool(x[key]) for x in sub)}
    delays = sorted(x["delay_min"] for x in rows if x.get("delay_min") is not None and x.get("in_window"))
    report = {"credits_used": api.used, "requests_made": len(snaps), "errors": errors,
              "by_role": {role: {k: share(role, k) for k in ("is_first_after_end", "before_q3", "in_window", "event_found", "draftkings_h2h", "pinnacle_h2h",
                                                             "draftkings_fresh", "pinnacle_fresh", "scored")}
                          for role in ("trigger", "control")},
              "short_window_attempts_in_window": share("trigger", "in_window") if False else
                  {"n": sum(x["short_window_attempt"] for x in rows),
                   "in_window": sum(bool(x.get("in_window")) for x in rows if x["short_window_attempt"])},
              "spread_exact_match": {"pulled": sum(1 for x in rows if x.get("spread_pulled")),
                                     "matched": sum(1 for x in rows if x.get("spread_exact_match"))},
              "entry_delay_min": {"median": delays[len(delays) // 2] if delays else None,
                                  "p90": delays[int(len(delays) * 0.9)] if delays else None, "max": delays[-1] if delays else None},
              "sealed": "price edges, ROI and outcomes not computed"}
    out = Path(data) / "l1" / "pilot_dq.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"report": report, "games": rows}, indent=1))
    print(json.dumps(report, indent=1))
