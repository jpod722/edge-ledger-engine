"""Research track D1: walk-forward replay of NFL derivatives (alt spreads/totals, team totals, first halves).

Pre-registered in config/rules.yaml (d1:) and the pre-run package. Rules are frozen; nothing is tuned here.
- Pinnacle is the only historical sharp source (Kalshi and LowVig are absent), so there is no
  multi-source disagreement test. Every candidate is single-source.
- Decision: the latest snapshot at or before the scan time. Staleness is measured from the snapshot's
  timestamp; any quote updated after the scan time is dropped (no look-ahead).
- Execution: the candidate's market is re-requested at +10 and +30 minutes. Two statistics are kept
  apart: the opportunity rate over ALL candidates (a vanished quote counts as not executable, never as a
  loss) and CLV conditional on the quote still being available.
- Close: a snapshot targeted 5 minutes before kickoff. Its real gap to kickoff is stored; a gap outside
  (0, max_close_gap_min] makes the benchmark missing, not a valid close.
- CLV, survival and ROI stay sealed until every configured week has run (or the track is finalized).
"""
from __future__ import annotations

import copy
import gzip
import json
import random
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from .backtest import ET, SEASON_START, et, iso_z
from .client import CreditGuard, OddsAPI, RawStore, api_key, espn_get, last_remaining_from_manifest, utcnow
from .close import close_candidate, metrics
from .evaluate import price_quotes, to_dict
from .grade import ESPN, find_game, outcome, parse_scoreboard, units
from .normalize import FAMILY, parse_ts, quotes_from
from .odds import ev_no_push_prob

SK = "americanfootball_nfl"
FAMILY_MARKETS = {
    "spreads": "spreads,alternate_spreads",
    "totals": "totals,alternate_totals",
    "team_totals": "team_totals,alternate_team_totals",
    "spreads_h1": "spreads_h1",
    "totals_h1": "totals_h1",
}
REPORT_FAMILY = {"spreads": "alt_spreads", "totals": "alt_totals", "team_totals": "team_totals",
                 "spreads_h1": "first_half", "totals_h1": "first_half"}


def d1_cfg(cfg: dict) -> dict:
    d = cfg["d1"]
    c = copy.deepcopy(cfg)
    c["bettable_books"] = list(d["venues"])
    c["accounts"] = list(d["venues"])
    c["sharp_books"] = list(d["sources"])
    c["weights"] = {b: 1.0 for b in d["sources"]}
    c["bet_groups"] = []          # research tier only
    return c


def week_sunday(week: int) -> date:
    return SEASON_START + timedelta(days=7 * (week - 1) + 3)


def no_lookahead(quotes, scan_at: datetime):
    """Drop any quote stamped after the scan time; it could not have been seen."""
    keep, dropped = [], 0
    for q in quotes:
        lu = parse_ts(q.last_update)
        if lu is not None and lu > scan_at:
            dropped += 1
            continue
        keep.append(q)
    return keep, dropped


def stored_main_snapshot(store: RawStore, when: datetime):
    """Reuse the main-market bulk snapshot already bought by the main replay at this exact time."""
    if not store.manifest.exists():
        return None
    want = iso_z(when)
    for line in store.manifest.read_text().splitlines():
        m = json.loads(line)
        if m.get("path") == f"/historical/sports/{SK}/odds" and m.get("params", {}).get("date") == want \
                and m.get("status") == 200:
            return json.loads(gzip.open(store.root / m["file"]).read()), m["file"]
    return None


def opportunity(d: dict, q, thr: float) -> dict:
    """Execution outcome at one delay for one candidate (fair price frozen at decision time)."""
    if q is None:
        return {"available": False, "price_dec": None, "ev_at_orig_fair": None, "eligible": False}
    ev = ev_no_push_prob(d["fair_prob"], q.dec, d.get("push_prob", 0.0))
    return {"available": True, "price_dec": q.dec, "ev_at_orig_fair": round(ev, 6), "eligible": ev >= thr}


def close_gap_ok(kick: datetime, snap: datetime | None, max_gap_min: float) -> tuple[bool, float | None]:
    if snap is None:
        return False, None
    gap = (kick - snap).total_seconds() / 60
    return (0 < gap <= max_gap_min), round(gap, 2)


def _index(quotes):
    return {(q.event_id, q.family, q.description, q.side, q.point, q.book): q for q in quotes}


def run_weeks(a, cfg):
    d = cfg["d1"]
    store = RawStore(a.data)
    root = Path(a.data) / "d1"
    for sub in ("picks", "weeks", "coverage"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    c = d1_cfg(cfg)
    rem = last_remaining_from_manifest(store)
    api = OddsAPI(api_key(), store, d["stage_cap_credits"], min_remaining=d["min_remaining_credits"], remaining=rem)
    books = ",".join(d["venues"] + d["sources"])
    markets = d["markets"]
    for week in a.weeks:
        if (root / "weeks" / f"{week:02d}.json").exists():
            print(f"week {week} already done")
            continue
        sun = week_sunday(week)
        used0 = api.used
        board = parse_scoreboard(espn_get(store, f"{ESPN['NFL']}/scoreboard?dates={sun:%Y%m%d}", f"espn_NFL_{sun:%Y%m%d}"))
        flagged: dict[str, dict] = {}
        cover_rows = []
        stopped = None
        try:
            for name, hh, mm, window_min, news in [(s["name"], *map(int, s["at_et"].split(":")), s.get("within_min"),
                                                    s.get("news", False)) for s in d["scans"]]:
                scan_at = et(sun, hh, mm)
                main = stored_main_snapshot(store, scan_at)
                if main is None:
                    before = api.used
                    p, _, ref = api.historical_odds(SK, iso_z(scan_at), "h2h,spreads,totals", "pinnacle," + ",".join(d["venues"]))
                    main = (p, ref)
                    api.by_label["main"] = api.by_label.get("main", 0) + api.used - before
                payload, main_ref = main
                evs = []
                for e in payload.get("data", []):
                    t = parse_ts(e["commence_time"])
                    if t <= scan_at or t.astimezone(ET).date() != sun:
                        continue
                    if window_min and (t - scan_at) > timedelta(minutes=window_min):
                        continue
                    evs.append(e)
                for e in evs:
                    ep, retrieved, ref = api.historical_event_odds(SK, e["id"], iso_z(scan_at), markets, books, "decision")
                    snap = parse_ts(ep.get("timestamp")) or scan_at
                    q_all = quotes_from([e], "NFL") + quotes_from(ep, "NFL")
                    q_all, dropped = no_lookahead(q_all, scan_at)
                    cands = price_quotes(q_all, c, snap, scan=f"d1-w{week}-{name}", news_window=news,
                                         api_retrieved_at=retrieved, raw_refs=[main_ref, ref])
                    fams = {}
                    for q in q_all:
                        fams.setdefault(q.family, set()).add(q.book)
                    row = {"week": week, "scan": name, "event_id": e["id"], "game": f"{e['away_team']} @ {e['home_team']}",
                           "snapshot": ep.get("timestamp"), "dropped_lookahead": dropped,
                           "families": {f: sorted(b) for f, b in fams.items()}, "rejections": {}, "candidates": 0}
                    for x in cands:
                        if x.group not in ("alt", "deriv"):
                            continue
                        if x.tier == "rejected":
                            hard = [_reason_key(r) for r in x.reasons if r != "single_source"]
                            if x.fair_prob is None and not hard:
                                hard = ["no_pinnacle_price"]
                            for k in hard or ["below_threshold"]:
                                row["rejections"][k] = row["rejections"].get(k, 0) + 1
                            continue
                        if x.id in flagged:
                            continue
                        dd = to_dict(x)
                        dd.update({"week": week, "scan_name": name, "scan_at": scan_at.isoformat(),
                                   "snapshot_at": ep.get("timestamp"), "report_family": REPORT_FAMILY[x.family],
                                   "threshold": cfg["single_source_min_ev"] if len(x.sources) < 2 else cfg["research_min_ev"]})
                        flagged[x.id] = dd
                        row["candidates"] += 1
                    cover_rows.append(row)
            # execution re-pricing: one request per (game, family) per delay, venues only
            groups: dict = {}
            for dd in flagged.values():
                groups.setdefault((dd["event_id"], dd["family"], dd["snapshot_at"]), []).append(dd)
            for (eid, fam, snap_s), ds in groups.items():
                snap = parse_ts(snap_s)
                for dm in d["delays_min"]:
                    t = snap + timedelta(minutes=dm)
                    p, _, ref = api.historical_event_odds(SK, eid, iso_z(t), FAMILY_MARKETS[fam], ",".join(d["venues"]), f"delay_{dm}")
                    idx = _index(quotes_from(p, "NFL"))
                    actual = parse_ts(p.get("timestamp"))
                    for dd in ds:
                        q = idx.get((dd["event_id"], dd["family"], dd["description"], dd["side"], dd["point"], dd["venue"]))
                        o = opportunity(dd, q, dd["threshold"])
                        o.update({"requested": iso_z(t), "snapshot": p.get("timestamp"),
                                  "actual_delay_min": round((actual - snap).total_seconds() / 60, 2) if actual else None,
                                  "ref": ref})
                        dd.setdefault("exec", {})[str(dm)] = o
            # closes: one request per (game, family), Pinnacle only, targeted before kickoff
            cg: dict = {}
            for dd in flagged.values():
                cg.setdefault((dd["event_id"], dd["family"], dd["commence_time"]), []).append(dd)
            for (eid, fam, kick_s), ds in cg.items():
                kick = parse_ts(kick_s)
                t = kick - timedelta(minutes=d["close_target_min"])
                p, ret, ref = api.historical_event_odds(SK, eid, iso_z(t), FAMILY_MARKETS[fam], ",".join(d["sources"]), "close")
                snap = parse_ts(p.get("timestamp"))
                ok, gap = close_gap_ok(kick, snap, d["max_close_gap_min"])
                cq = quotes_from(p, "NFL")
                for dd in ds:
                    out = close_candidate(dd, cq, c, snap or t, ret, [ref])
                    out.update({"close_snapshot": p.get("timestamp"), "close_gap_min": gap, "close_gap_ok": ok})
                    if not ok:
                        out.update({"close_pinnacle_p": None, "clv_pinnacle": metrics(None, 1, 0),
                                    "close_pinnacle_reason": f"close_gap_{gap}_min_outside_0_{d['max_close_gap_min']}"})
                    dd.update(out)
                    for dm, o in dd.get("exec", {}).items():
                        o["clv_pinnacle"] = metrics(dd["close_pinnacle_p"], o["price_dec"], dd.get("push_prob", 0)) \
                            if o["available"] and dd["close_pinnacle_p"] is not None else None
        except CreditGuard as g:
            stopped = str(g)
        for dd in flagged.values():
            g = find_game(board, dd["game"])
            res = outcome(dd, g) if g and g["completed"] else None
            dd["result"] = res or "void"
            dd["units"] = units(dd["result"], dd["effective_dec"])
            (root / "picks" / f"{dd['id']}.json").write_text(json.dumps(dd, indent=1, default=str))
        (root / "coverage" / f"{week:02d}.json").write_text(json.dumps(cover_rows, indent=1))
        meta = {"week": week, "sunday": sun.isoformat(), "candidates": len(flagged), "credits": api.used - used0,
                "credits_by_kind": dict(api.by_label), "remaining_after": api.remaining, "stopped": stopped,
                "complete": stopped is None, "ran_at": utcnow().isoformat()}
        if stopped is None:
            (root / "weeks" / f"{week:02d}.json").write_text(json.dumps(meta, indent=1))
        print(json.dumps(meta))
        api.by_label = {}
        if stopped:
            break


def _reason_key(r: str) -> str:
    """Collapse reasons with embedded numbers (e.g. 'pinnacle_stale_52s') to a stable key."""
    parts = [p for p in r.split("_") if not any(ch.isdigit() for ch in p)]
    return "_".join(parts) or r


# ---------------- reporting ----------------

def cluster_boot(rows, key, cluster="event_id", n_boot=2000, seed=11):
    vals = [(r[cluster], key(r)) for r in rows if key(r) is not None]
    if len(vals) < 5:
        return {"n": len(vals), "mean": statistics.fmean(v for _, v in vals) if vals else None,
                "lo": None, "hi": None, "p_le_0": None}
    by: dict = {}
    for cl, v in vals:
        by.setdefault(cl, []).append(v)
    cls = list(by.values())
    rng = random.Random(seed)
    boots = sorted(statistics.fmean([v for cl in [rng.choice(cls) for _ in cls] for v in cl]) for _ in range(n_boot))
    return {"n": len(vals), "mean": round(statistics.fmean(v for _, v in vals), 6),
            "lo": round(boots[int(0.025 * n_boot)], 6), "hi": round(boots[int(0.975 * n_boot) - 1], 6),
            "p_le_0": round(sum(b <= 0 for b in boots) / n_boot, 4)}


def holm(pvals: dict, alpha: float) -> dict:
    """Holm step-down; returns {key: rejected_null}."""
    order = sorted((p, k) for k, p in pvals.items() if p is not None)
    out = {k: False for k in pvals}
    m = len(order)
    for i, (p, k) in enumerate(order):
        if p <= alpha / (m - i):
            out[k] = True
        else:
            break
    return out


def verdicts(fams: dict, d: dict) -> dict:
    v = {}
    holm_ok = holm({f: (x["clv_exec_10"] or {}).get("p_le_0") for f, x in fams.items()}, d["holm_alpha_one_sided"])
    for f, x in fams.items():
        clv = x["clv_decision"]
        n = clv["n"]
        if n < d["min_closed_for_verdict"] or clv["lo"] is None:
            v[f] = "inconclusive"
        elif clv["mean"] <= 0 and clv["hi"] < 0:
            v[f] = "stop"
        elif clv["lo"] > 0 and (x["available_10"] or 0) >= d["min_survival_10"] \
                and (x["method_sensitive_share"] or 0) < d["max_method_sensitive"]:
            e10 = x["clv_exec_10"] or {}
            v[f] = "expand" if (e10.get("lo") or -1) > 0 and holm_ok.get(f) else "continue"
        else:
            v[f] = "inconclusive"
    return v


def report(a, cfg):
    d = cfg["d1"]
    root = Path(a.data) / "d1"
    rows = [json.loads(p.read_text()) for p in sorted((root / "picks").glob("*.json"))]
    weeks = sorted(int(p.stem) for p in (root / "weeks").glob("*.json"))
    cover = [r for p in sorted((root / "coverage").glob("*.json")) for r in json.loads(p.read_text())]
    metas = [json.loads((root / "weeks" / f"{w:02d}.json").read_text()) for w in weeks]
    all_weeks = d["stage1_weeks"] + d["stage2_weeks"]
    reveal = all(w in weeks for w in all_weeks) or getattr(a, "final", False)
    # coverage and cost: always visible
    fam_cov: dict = {}
    for r in cover:
        for fam in ("spreads", "totals", "team_totals", "spreads_h1", "totals_h1"):
            rf = REPORT_FAMILY[fam] + ("" if fam not in ("spreads_h1", "totals_h1") else "_" + fam.split("_")[0])
            x = fam_cov.setdefault(rf, {"game_snapshots": 0, "pinnacle_present": 0, "venue_present": 0})
            x["game_snapshots"] += 1
            books = set(r["families"].get(fam, []))
            x["pinnacle_present"] += "pinnacle" in books
            x["venue_present"] += bool(books & set(d["venues"]))
    for x in fam_cov.values():
        x["pinnacle_share"] = round(x["pinnacle_present"] / x["game_snapshots"], 3) if x["game_snapshots"] else None
    rej: dict = {}
    for r in cover:
        for k, n in r["rejections"].items():
            rej[k] = rej.get(k, 0) + n
    by_kind: dict = {}
    for m in metas:
        for k, n in m["credits_by_kind"].items():
            by_kind[k] = by_kind.get(k, 0) + n
    n_snap = len(cover)
    credits = sum(m["credits"] for m in metas)
    cand_by_fam: dict = {}
    for r in rows:
        cand_by_fam[r["report_family"]] = cand_by_fam.get(r["report_family"], 0) + 1
    summ = {
        "track": "D1 derivatives replay (research tier, Pinnacle-only)", "weeks_done": weeks,
        "stage1_complete": all(w in weeks for w in d["stage1_weeks"]), "complete": reveal,
        "credits": credits, "credits_by_kind": by_kind,
        "credits_per_game_snapshot_decision": round(by_kind.get("decision", 0) / n_snap, 2) if n_snap else None,
        "forecast_per_game_snapshot": d["forecast_per_game_snapshot"],
        "remaining_after": metas[-1]["remaining_after"] if metas else None,
        "game_snapshots": n_snap, "dropped_lookahead_quotes": sum(r["dropped_lookahead"] for r in cover),
        "coverage": fam_cov, "candidates": len(rows), "candidates_by_family": cand_by_fam,
        "rejections": dict(sorted(rej.items(), key=lambda kv: -kv[1])),
        "credits_per_candidate": round(credits / len(rows), 1) if rows else None,
        "generated_at": utcnow().isoformat(),
    }
    gate = []
    if summ["stage1_complete"]:
        for f, x in fam_cov.items():
            if x["pinnacle_share"] is not None and x["pinnacle_share"] < d["min_pinnacle_coverage"]:
                gate.append(f"drop {f}: Pinnacle coverage {x['pinnacle_share']:.0%} < {d['min_pinnacle_coverage']:.0%}")
        if summ["credits_per_game_snapshot_decision"] and \
                summ["credits_per_game_snapshot_decision"] > 1.25 * d["forecast_per_game_snapshot"]:
            gate.append("stop: cost per game-snapshot above 1.25x forecast")
        if len(rows) < d["min_stage1_candidates"]:
            gate.append(f"pause: {len(rows)} candidates < {d['min_stage1_candidates']}")
        if summ["remaining_after"] is not None and summ["remaining_after"] < d["min_remaining_credits"]:
            gate.append("stop: account below the live-pipeline floor")
        summ["stage_gate"] = gate or ["proceed to Stage 2"]
    if reveal:
        fams = {}
        for f in sorted({r["report_family"] for r in rows}):
            rs = [r for r in rows if r["report_family"] == f]
            ex = lambda dm: [r for r in rs if (r.get("exec") or {}).get(dm)]
            x = {"n": len(rs),
                 "close_coverage": round(sum(r.get("close_pinnacle_p") is not None for r in rs) / len(rs), 3),
                 "clv_decision": cluster_boot(rs, lambda r: (r.get("clv_pinnacle") or {}).get("prob")),
                 "clv_decision_logodds": cluster_boot(rs, lambda r: (r.get("clv_pinnacle") or {}).get("logodds")),
                 "method_sensitive_share": round(sum(bool(r.get("method_sensitive")) for r in rs) / len(rs), 3)}
            for dm in map(str, d["delays_min"]):
                e = ex(dm)
                x[f"available_{dm}"] = round(sum(r["exec"][dm]["available"] for r in e) / len(rs), 3) if rs else None
                x[f"opportunity_{dm}"] = round(sum(r["exec"][dm]["eligible"] for r in e) / len(rs), 3) if rs else None
                x[f"clv_exec_{dm}"] = cluster_boot([r for r in e if r["exec"][dm].get("clv_pinnacle")],
                                                   lambda r, dm=dm: r["exec"][dm]["clv_pinnacle"]["prob"])
            graded = [r for r in rs if r["result"] in ("win", "loss", "push")]
            x["units"] = round(sum(r["units"] for r in graded), 3)
            x["roi"] = round(x["units"] / len(graded), 4) if graded else None
            x["missing_close_reasons"] = {}
            for r in rs:
                if r.get("close_pinnacle_p") is None:
                    k = _reason_key(str(r.get("close_pinnacle_reason")))
                    x["missing_close_reasons"][k] = x["missing_close_reasons"].get(k, 0) + 1
            fams[f] = x
        summ["results"] = fams
        summ["verdicts"] = verdicts(fams, d)
        summ["thresholds"] = {f"{t:.0%}": sum((r.get("ev_quoted") or 0) >= t for r in rows) for t in (0.03, 0.04, 0.05)}
    else:
        summ["results"] = "sealed until all D1 weeks have run (CLV, survival and ROI)"
    (root / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("weeks_done", "credits", "candidates", "candidates_by_family")} |
                     {"stage_gate": summ.get("stage_gate")}, default=str))
    return summ
