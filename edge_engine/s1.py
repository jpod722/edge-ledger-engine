"""Research track S1: stale lines. Pre-registered 2026-10-19 and approved by the reviewer.

Question: when Pinnacle's devigged fair on an exact main-line contract moves >= 1.5 probability points in a
side's favour between consecutive 5-minute snapshots, and DraftKings/FanDuel leave that side's price
unchanged, does the stale venue price beat Pinnacle's close?

- Data: 2025 Sundays (weeks in config s1.weeks), every consecutive historical snapshot 11:00-12:55 ET,
  chained by next_timestamp; spreads and totals; 1 PM ET games only.
- Decision at snapshot k: snapshot timestamp + capture_window_s. Quotes stamped later are dropped.
- Candidate = trigger AND the existing research rules at snapshot k (2% with two sharp sources, 3% single).
  A contract counts once per Sunday (first trigger).
- Execution: availability at k+1 (+5) and k+2 (+10) from the same chain. Opportunity rate over all
  candidates and conditional CLV are reported separately; a vanished quote is unavailable, not a loss.
- Close: the last snapshot at or before 12:55; gap to kickoff must be in (0, max_close_gap_min].
- Reported separately: raw triggers, triggers passing research rules, and those with a valid close.
- Comparison: research candidates at the same snapshots that did not follow a Pinnacle move.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime, timedelta
from pathlib import Path

from .backtest import ET, SEASON_START, bt_cfg, et, iso_z
from .client import CreditGuard, OddsAPI, RawStore, api_key, espn_get, last_remaining_from_manifest, utcnow
from .close import close_candidate, metrics
from .d1 import _reason_key, close_gap_ok, cluster_boot, no_lookahead, opportunity
from .evaluate import price_quotes, to_dict
from .grade import ESPN, find_game, outcome, parse_scoreboard, units
from .normalize import pairs_from, parse_ts, quotes_from
from .odds import devig

SK = "americanfootball_nfl"


def s1_cfg(cfg: dict) -> dict:
    c = bt_cfg(cfg)  # venues DraftKings/FanDuel; sharps Pinnacle 0.67 + LowVig 0.33, as in the main replay
    c["bet_groups"] = []
    return c


def pin_fair(quotes, method="power") -> dict:
    """Pinnacle devigged fair per exact contract: (event, family, description, side, point) -> (p, last_update)."""
    out = {}
    for pr in pairs_from([q for q in quotes if q.book == "pinnacle" and q.group == "main"]):
        pa, pb = devig(pr.a.dec, pr.b.dec, method)
        for q, p in ((pr.a, pa), (pr.b, pb)):
            out[(q.event_id, q.family, q.description, q.side, q.point)] = (p, q.last_update)
    return out


def venue_book(quotes, venues) -> dict:
    return {(q.event_id, q.family, q.description, q.side, q.point, q.book): q
            for q in quotes if q.book in venues and q.group == "main"}


def triggers(prev_q, cur_q, venues, min_move) -> list[dict]:
    """Exact contracts where Pinnacle moved >= min_move toward a side the venue left unchanged."""
    fp, fc = pin_fair(prev_q), pin_fair(cur_q)
    vp, vc = venue_book(prev_q, venues), venue_book(cur_q, venues)
    out = []
    for key, q in vc.items():
        ck = key[:5]
        if ck not in fp or ck not in fc:
            continue
        move = fc[ck][0] - fp[ck][0]
        if move < min_move:
            continue
        old = vp.get(key)
        if old is None or old.dec != q.dec:
            continue
        v_lu, p_lu = parse_ts(q.last_update), parse_ts(fc[ck][1])
        if v_lu is None or p_lu is None or not v_lu < p_lu:
            continue
        out.append({"key": key, "move": round(move, 6), "pin_prev": round(fp[ck][0], 6), "pin_now": round(fc[ck][0], 6),
                    "venue_last_update": q.last_update, "pin_last_update": fc[ck][1]})
    return out


def chain(api: OddsAPI, start: datetime, end: datetime, books: str, max_snaps: int):
    """Consecutive historical snapshots from the one at/before start through the last at/before end."""
    snaps, t = [], start
    while len(snaps) < max_snaps:
        p, ret, ref = api.historical_odds(SK, iso_z(t), "spreads,totals", books)
        ts = parse_ts(p.get("timestamp"))
        if ts is None or ts > end or (snaps and ts <= snaps[-1]["ts"]):
            break
        snaps.append({"ts": ts, "payload": p, "ref": ref, "retrieved": ret})
        nxt = parse_ts(p.get("next_timestamp"))
        if nxt is None or nxt > end:
            break
        t = nxt
    return snaps


def run(a, cfg):
    s = cfg["s1"]
    store = RawStore(a.data)
    root = Path(a.data) / "s1"
    for sub in ("picks", "sundays"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    c = s1_cfg(cfg)
    api = OddsAPI(api_key(), store, s["cap_credits"], min_remaining=s["min_remaining_credits"],
                  remaining=last_remaining_from_manifest(store))
    books = ",".join(s["books"])
    for week in s["weeks"]:
        if (root / "sundays" / f"{week:02d}.json").exists():
            continue
        sun = SEASON_START + timedelta(days=7 * (week - 1) + 3)
        start, end = et(sun, *map(int, s["start_et"].split(":"))), et(sun, *map(int, s["end_et"].split(":")))
        used0, stopped = api.used, None
        try:
            snaps = chain(api, start, end, books, s["max_snapshots"])
        except CreditGuard as g:
            snaps, stopped = [], str(g)
        if stopped or not snaps:
            print(json.dumps({"week": week, "stopped": stopped, "snapshots": len(snaps)}))
            break
        board = parse_scoreboard(espn_get(store, f"{ESPN['NFL']}/scoreboard?dates={sun:%Y%m%d}", f"espn_NFL_{sun:%Y%m%d}"))
        lo, hi = et(sun, *map(int, s["kickoff_from_et"].split(":"))), et(sun, *map(int, s["kickoff_to_et"].split(":")))
        cap = timedelta(seconds=s["capture_window_s"])
        qs = []
        for sn in snaps:
            evs = [e for e in sn["payload"].get("data", []) if lo <= parse_ts(e["commence_time"]) <= hi]
            q, dropped = no_lookahead(quotes_from(evs, "NFL"), sn["ts"] + cap)
            qs.append(q)
            sn["dropped"] = dropped
        raw_trig, picks, comparison = 0, {}, {}
        for k in range(1, len(snaps)):
            dec_at = snaps[k]["ts"] + cap
            trig = {t["key"]: t for t in triggers(qs[k - 1], qs[k], s["venues"], s["min_move"])}
            raw_trig += len(trig)
            cands = price_quotes(qs[k], c, dec_at, scan=f"s1-w{week}-k{k}", api_retrieved_at=snaps[k]["retrieved"],
                                 raw_refs=[snaps[k]["ref"]])
            for x in cands:
                if x.tier != "research" or x.group != "main":
                    continue
                key = (x.event_id, x.family, x.description, x.side, x.point, x.venue)
                d = to_dict(x)
                d.update({"week": week, "k": k, "snapshot_at": snaps[k]["ts"].isoformat(),
                          "threshold": cfg["research_min_ev"] if len(x.sources) >= 2 else cfg["single_source_min_ev"]})
                if key in trig:
                    if x.contract in picks:
                        continue
                    d["trigger"] = trig[key]
                    d["track"] = "triggered"
                    picks[x.contract] = (d, k)
                elif x.contract not in comparison and x.contract not in picks:
                    d["track"] = "comparison"
                    comparison[x.contract] = (d, k)
        last = snaps[-1]
        allp = list(picks.values()) + list(comparison.values())
        for d, k in allp:
            for step, label in ((1, "5"), (2, "10")):
                if k + step < len(snaps):
                    idx = venue_book(qs[k + step], [d["venue"]])
                    q = idx.get((d["event_id"], d["family"], d["description"], d["side"], d["point"], d["venue"]))
                    o = opportunity(d, q, d["threshold"])
                    o["snapshot"] = snaps[k + step]["ts"].isoformat()
                else:
                    o = {"available": None, "eligible": None, "reason": "chain_ended"}
                d.setdefault("exec", {})[label] = o
            kick = parse_ts(d["commence_time"])
            ok, gap = close_gap_ok(kick, last["ts"], s["max_close_gap_min"])
            out = close_candidate(d, quotes_from(last["payload"], "NFL"), c, last["ts"], last["retrieved"], [last["ref"]])
            out.update({"close_snapshot": last["ts"].isoformat(), "close_gap_min": gap, "close_gap_ok": ok})
            if not ok:
                out.update({"close_pinnacle_p": None, "clv_pinnacle": metrics(None, 1, 0),
                            "close_pinnacle_reason": f"close_gap_{gap}_min"})
            d.update(out)
            for o in d["exec"].values():
                o["clv_pinnacle"] = metrics(d["close_pinnacle_p"], o["price_dec"], d.get("push_prob", 0)) \
                    if o.get("available") and d["close_pinnacle_p"] is not None else None
            g = find_game(board, d["game"])
            d["result"] = (outcome(d, g) if g and g["completed"] else None) or "void"
            d["units"] = units(d["result"], d["effective_dec"])
            (root / "picks" / f"w{week:02d}-{d['track']}-{d['id']}.json").write_text(json.dumps(d, indent=1, default=str))
        meta = {"week": week, "sunday": sun.isoformat(), "snapshots": len(snaps),
                "first": snaps[0]["ts"].isoformat(), "last": last["ts"].isoformat(),
                "gaps_s": sorted({int((snaps[i]["ts"] - snaps[i - 1]["ts"]).total_seconds()) for i in range(1, len(snaps))}),
                "dropped_lookahead": sum(sn["dropped"] for sn in snaps), "raw_triggers": raw_trig,
                "triggered_research": len(picks), "comparison_research": len(comparison),
                "credits": api.used - used0, "remaining_after": api.remaining, "ran_at": utcnow().isoformat()}
        (root / "sundays" / f"{week:02d}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))


def report(a, cfg):
    s = cfg["s1"]
    root = Path(a.data) / "s1"
    metas = [json.loads(p.read_text()) for p in sorted((root / "sundays").glob("*.json"))]
    rows = [json.loads(p.read_text()) for p in sorted((root / "picks").glob("*.json"))]
    done = sorted(m["week"] for m in metas)
    complete = all(w in done for w in s["weeks"])
    summ = {"track": "S1 stale lines (research tier)", "weeks_done": done, "complete": complete,
            "credits": sum(m["credits"] for m in metas), "remaining_after": metas[-1]["remaining_after"] if metas else None,
            "snapshots": sum(m["snapshots"] for m in metas), "snapshot_gaps_s": sorted({g for m in metas for g in m["gaps_s"]}),
            "dropped_lookahead": sum(m["dropped_lookahead"] for m in metas),
            "raw_triggers": sum(m["raw_triggers"] for m in metas),
            "triggered_research": sum(m["triggered_research"] for m in metas),
            "comparison_research": sum(m["comparison_research"] for m in metas),
            "generated_at": utcnow().isoformat()}
    if not complete:
        summ["results"] = "sealed until all S1 Sundays have run"
    else:
        res = {}
        for track in ("triggered", "comparison"):
            rs = [r for r in rows if r["track"] == track]
            closed = [r for r in rs if r.get("close_pinnacle_p") is not None]
            x = {"n": len(rs), "with_valid_close": len(closed),
                 "clv_decision": cluster_boot(rs, lambda r: (r.get("clv_pinnacle") or {}).get("prob")),
                 "clv_decision_logodds": cluster_boot(rs, lambda r: (r.get("clv_pinnacle") or {}).get("logodds")),
                 "missing_close_reasons": {}}
            for r in rs:
                if r.get("close_pinnacle_p") is None:
                    k = _reason_key(str(r.get("close_pinnacle_reason")))
                    x["missing_close_reasons"][k] = x["missing_close_reasons"].get(k, 0) + 1
            for lab in ("5", "10"):
                ex = [r for r in rs if (r.get("exec") or {}).get(lab, {}).get("available") is not None]
                x[f"available_{lab}"] = round(sum(r["exec"][lab]["available"] for r in ex) / len(rs), 3) if rs else None
                x[f"opportunity_{lab}"] = round(sum(bool(r["exec"][lab]["eligible"]) for r in ex) / len(rs), 3) if rs else None
                x[f"clv_exec_{lab}"] = cluster_boot([r for r in ex if r["exec"][lab].get("clv_pinnacle")],
                                                    lambda r, lab=lab: r["exec"][lab]["clv_pinnacle"]["prob"])
            graded = [r for r in rs if r["result"] in ("win", "loss", "push")]
            x["units"] = round(sum(r["units"] for r in graded), 3)
            x["roi"] = round(x["units"] / len(graded), 4) if graded else None
            res[track] = x
        t = res["triggered"]
        clv = t["clv_decision"]
        if t["n"] < s["min_triggered"] or (clv["mean"] is not None and clv["hi"] is not None and clv["mean"] <= 0 and clv["hi"] < 0):
            verdict = "stop"
        elif clv["n"] < s["min_closed_for_verdict"] or clv["lo"] is None or clv["lo"] <= 0 < clv["hi"]:
            verdict = "inconclusive"
        elif clv["lo"] > 0 and (t["opportunity_5"] or 0) >= s["min_opportunity_5"]:
            verdict = "continue: build live paper monitor"
        else:
            verdict = "inconclusive"
        summ["results"], summ["verdict"] = res, verdict
    (root / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("weeks_done", "credits", "raw_triggers", "triggered_research",
                                          "comparison_research")} | {"verdict": summ.get("verdict")}))
    return summ
