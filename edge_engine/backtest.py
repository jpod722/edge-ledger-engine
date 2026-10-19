"""Walk-forward replay of the 2025 NFL regular season on historical snapshots.

Rules are frozen (no tuning inside the replay). Weeks 13-18 are a sealed holdout:
their results are stored but not summarized until every week has run.
Each flag is re-priced 10 minutes later at the same venue to measure execution drag.
"""
from __future__ import annotations

import copy
import json
import random
import statistics
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .client import OddsAPI, RawStore, api_key, espn_get, utcnow
from .close import close_candidate, metrics
from .evaluate import price_quotes, size_stakes, to_dict
from .grade import ESPN, find_game, outcome, parse_scoreboard, units
from .normalize import parse_ts, quotes_from

ET = ZoneInfo("America/New_York")
SEASON_START = date(2025, 9, 4)
BOOKS = "draftkings,fanduel,pinnacle,lowvig"


def bt_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["bettable_books"] = ["draftkings", "fanduel"]
    c["accounts"] = ["draftkings", "fanduel"]
    c["sharp_books"] = ["pinnacle", "lowvig"]
    c["weights"] = {"pinnacle": 0.67, "lowvig": 0.33}
    return c


def iso_z(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def et(d: date, hh: int, mm: int) -> datetime:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET)


def schedule(thu: date) -> list[tuple[str, datetime, str]]:
    sun, mon = thu + timedelta(days=3), thu + timedelta(days=4)
    return [("thu", et(thu, 14, 48), "week"), ("sun", et(sun, 9, 48), "week"),
            ("news1", et(sun, 11, 40), "within:120"), ("news2", et(sun, 14, 45), "within:120"),
            ("mon", et(mon, 17, 48), "today")]


def run_week(a, cfg):
    store = RawStore(a.data)
    root = Path(a.data) / "backtest"
    (root / "picks").mkdir(parents=True, exist_ok=True)
    (root / "weeks").mkdir(parents=True, exist_ok=True)
    thu = date.fromisoformat(a.start)
    week = (thu - SEASON_START).days // 7 + 1
    holdout = week in cfg["validation"]["backtest_holdout_weeks"]
    c = bt_cfg(cfg)
    api = OddsAPI(api_key(), store, 600)
    week_end = et(thu + timedelta(days=5), 6, 0)
    games = []
    for i in range(5):
        day = (thu + timedelta(days=i)).strftime("%Y%m%d")
        games += parse_scoreboard(espn_get(store, f"{ESPN['NFL']}/scoreboard?dates={day}", f"espn_NFL_{day}"))
    flagged: dict[str, dict] = {}
    for name, when, window in schedule(thu):
        payload, retrieved, ref = api.historical_odds("americanfootball_nfl", iso_z(when), "h2h,spreads,totals", BOOKS)
        snap_at = parse_ts(payload.get("timestamp")) or when
        evs = []
        for e in payload.get("data", []):
            t = parse_ts(e["commence_time"])
            if not (snap_at < t < week_end):
                continue
            if window.startswith("within:") and t - snap_at > timedelta(minutes=int(window.split(":")[1])):
                continue
            if window == "today" and t.astimezone(ET).date() != when.date():
                continue
            evs.append(e)
        cands = price_quotes(quotes_from(evs, "NFL"), c, snap_at, scan=f"bt-w{week}-{name}",
                             news_window=name.startswith("news"), api_retrieved_at=retrieved, raw_refs=[ref])
        cands = size_stakes(cands, c)
        new = [to_dict(x) for x in cands if x.tier in ("bet", "research") and x.id not in flagged]
        if not new:
            continue
        # latency replay: the same venue's price for the exact contract 10 minutes later
        later = when + timedelta(minutes=a.latency_min)
        lp, _, lref = api.historical_odds("americanfootball_nfl", iso_z(later), "h2h,spreads,totals", BOOKS)
        lq = {(q.event_id, q.family, q.side, q.point, q.book): q for q in quotes_from(lp, "NFL")}
        for d in new:
            q = lq.get((d["event_id"], d["family"], d["side"], d["point"], d["venue"]))
            d["exec_dec"] = q.dec if q else None
            d["exec_status"] = "filled" if q else "missed"
            d["exec_ref"] = lref
            d.update({"week": week, "holdout": holdout, "snapshot_at": snap_at.isoformat()})
            flagged[d["id"]] = d
    # closes: one snapshot five minutes before each distinct kickoff among flagged games
    kicks = sorted({d["commence_time"] for d in flagged.values()})
    for k in kicks:
        t = parse_ts(k) - timedelta(minutes=cfg["clv"]["close_minutes_before"])
        cp, cret, cref = api.historical_odds("americanfootball_nfl", iso_z(t),
                                             "h2h,spreads,totals", BOOKS)
        cq = quotes_from(cp, "NFL")
        for d in flagged.values():
            if d["commence_time"] != k:
                continue
            d.update(close_candidate(d, cq, c, t, cret, [cref]))
            if d.get("exec_dec") and d.get("close_pinnacle_p") is not None:
                d["clv_pinnacle_exec"] = metrics(d["close_pinnacle_p"], d["exec_dec"], d.get("push_prob", 0))
    for d in flagged.values():
        g = find_game(games, d["game"])
        res = outcome(d, g) if g and g["completed"] else None
        d["result"] = res or "void"
        d["units"] = units(d["result"], d["effective_dec"])
        (root / "picks" / f"{d['id']}.json").write_text(json.dumps(d, indent=1, default=str))
    (root / "weeks" / f"{week:02d}.json").write_text(json.dumps(
        {"week": week, "start": a.start, "holdout": holdout, "picks": len(flagged), "credits": api.used,
         "ran_at": utcnow().isoformat()}, indent=1))
    print(json.dumps({"week": week, "picks": len(flagged), "credits": api.used, "holdout": holdout}))


def cluster_ci(rows: list[dict], key, cluster="event_id", n_boot=2000, seed=7) -> tuple:
    vals = [(r[cluster], key(r)) for r in rows if key(r) is not None]
    if len(vals) < 5:
        return (None, None, None, len(vals))
    by: dict = {}
    for c, v in vals:
        by.setdefault(c, []).append(v)
    clusters = list(by.values())
    mean = statistics.fmean(v for _, v in vals)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = [rng.choice(clusters) for _ in clusters]
        flat = [v for cl in s for v in cl]
        boots.append(statistics.fmean(flat))
    boots.sort()
    return (mean, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1], len(vals))


def summarize(rows: list[dict], thresholds: list[float]) -> dict:
    out = {}
    for thr in thresholds:
        for tier_key, evk in (("research", "ev_quoted"), ("bet", "ev_adj_net")):
            sel = [r for r in rows if (r.get(evk) or -1) >= thr and (tier_key == "research" or r["tier"] == "bet")]
            pin = cluster_ci(sel, lambda r: (r.get("clv_pinnacle") or {}).get("prob"))
            ind = cluster_ci(sel, lambda r: (r.get("clv_indep") or {}).get("prob"))
            exe = cluster_ci(sel, lambda r: (r.get("clv_pinnacle_exec") or {}).get("prob"))
            graded = [r for r in sel if r.get("result") in ("win", "loss", "push")]
            u = sum(r["units"] for r in graded)
            out[f"{tier_key}@{thr:.0%}"] = {
                "n": len(sel), "pin_clv_prob": pin, "indep_clv_prob": ind, "exec_clv_prob": exe,
                "beat_close_pct": (sum(1 for r in sel if ((r.get("clv_pinnacle") or {}).get("prob") or 0) > 0) / len(sel)) if sel else None,
                "missed_pct": (sum(1 for r in sel if r.get("exec_status") == "missed") / len(sel)) if sel else None,
                "units": round(u, 3), "roi": round(u / len(graded), 4) if graded else None,
            }
    return out


def report(a, cfg):
    root = Path(a.data) / "backtest"
    rows = [json.loads(p.read_text()) for p in (root / "picks").glob("*.json")]
    weeks = sorted(int(p.stem) for p in (root / "weeks").glob("*.json"))
    hold = set(cfg["validation"]["backtest_holdout_weeks"])
    done = all(w in weeks for w in range(1, 19))
    thr = cfg["validation"]["thresholds_reported"]
    summ = {"weeks_done": len(weeks), "complete": done,
            "credits_used": sum(json.loads((root / "weeks" / f"{w:02d}.json").read_text())["credits"] for w in weeks),
            "train": summarize([r for r in rows if not r.get("holdout")], thr),
            "holdout": summarize([r for r in rows if r.get("holdout")], thr) if done else "sealed until all 18 weeks run",
            "generated_at": utcnow().isoformat()}
    (root / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("weeks_done", "complete", "credits_used")}))
