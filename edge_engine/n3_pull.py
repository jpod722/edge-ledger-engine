"""N3 tranche 1: historical player_receptions quotes at decision (kickoff - 90 min), 2025 Weeks 10-14.

Plan: config/prereg/n3-tranche1-plan.json. Reviewer approval 2026-10-24 (one-time exception to reset_path_rule,
recorded in config/prereg/n3-spec.yaml amendments_2026-10-24). Staged by week; before each stage the projected
Nov 18 balance must stay >= 3,000 using the lean monitor budget; after Week 10 a coverage check fails closed.
No retries. Prices are stored only; nothing is scored here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from .client import CreditGuard

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "config" / "prereg" / "n3-tranche1-plan.json"
SK = "americanfootball_nfl"
RESET = datetime(2026, 11, 18, tzinfo=timezone.utc)
FLOOR, NIGHTLY = 3000, 32
COVERAGE = {"matched_min": 0.90, "pinnacle_min": 0.50}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def projected(remaining, still_planned, now):
    days = max((RESET - now).days + 1, 0)
    return remaining - still_planned - NIGHTLY * days


def pull(api, data: str, books: str, now=None):
    plan = json.loads(PLAN.read_text())
    out_p = Path(data) / "n3" / "tranche1_quotes.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out = {"plan": str(PLAN.relative_to(ROOT)), "weeks": [], "stopped": None}
    left = plan["credits"]["total"]
    now = now or datetime.now(timezone.utc)

    def save():
        out["credits_used"] = api.used
        out["remaining_after"] = api.remaining
        out_p.write_text(json.dumps(out, indent=1))

    for wk in plan["weeks"]:
        need = 1 + 10 * len(wk["games"])
        p = projected(api.remaining, left, now)
        if api.remaining is None or p < FLOOR:
            out["stopped"] = f"week {wk['week']}: projected Nov 18 balance {p} < {FLOOR}"
            break
        rec = {"week": wk["week"], "games": []}
        out["weeks"].append(rec)
        try:
            evs, _, ref = api._get(f"/historical/sports/{SK}/events", {"date": wk["events_at"], "dateFormat": "iso"},
                                   f"{SK}_hist_events", 1)
        except (CreditGuard, RuntimeError) as err:
            out["stopped"] = f"week {wk['week']} events: {str(err)[:200]}"
            break
        left -= 1
        rec["events_ref"] = ref
        idx = {(e["home_team"], e["away_team"]): e for e in evs.get("data", [])}
        for g in wk["games"]:
            row = {**g}
            e = idx.get((g["home"], g["away"]))
            if e is None or abs((ts(e["commence_time"]) - ts(g["kickoff"])).total_seconds()) > 3600:
                row["status"] = "unmatched"          # recorded, never searched for again
                rec["games"].append(row)
                left -= 10
                continue
            row["event_id"] = e["id"]
            try:
                payload, _, ref2 = api.historical_event_odds(SK, e["id"], g["decision_at"], plan["market"], books, kind="n3")
            except CreditGuard as err:
                row["status"] = "credit_guard"
                rec["games"].append(row)
                out["stopped"] = str(err)[:200]
                save()
                return out
            except RuntimeError as err:
                row["status"], row["error"] = "error", str(err)[:200]
                rec["games"].append(row)
                left -= 10
                continue
            left -= 10
            d = payload.get("data", {})
            row.update(status="ok", ref=ref2, snapshot=payload.get("timestamp"))
            row["books"] = {}
            for b in d.get("bookmakers", []):
                for m in b.get("markets", []):
                    if m.get("key") == plan["market"]:
                        row["books"][b["key"]] = {"last_update": m.get("last_update"), "n_outcomes": len(m.get("outcomes", []))}
            rec["games"].append(row)
        g_ok = [g for g in rec["games"] if g["status"] == "ok"]
        rec["matched_share"] = len(g_ok) / max(len(wk["games"]), 1)
        rec["pinnacle_share"] = sum("pinnacle" in g["books"] for g in g_ok) / max(len(wk["games"]), 1)
        rec["dk_share"] = sum("draftkings" in g["books"] for g in g_ok) / max(len(wk["games"]), 1)
        save()
        if wk["week"] == plan["weeks"][0]["week"] and (rec["matched_share"] < COVERAGE["matched_min"]
                                                        or rec["pinnacle_share"] < COVERAGE["pinnacle_min"]):
            out["stopped"] = f"coverage check failed after week {wk['week']}: {rec['matched_share']:.2f} matched, {rec['pinnacle_share']:.2f} Pinnacle"
            break
    save()
    return out
