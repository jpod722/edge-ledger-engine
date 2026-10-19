"""Command line: scan, close, grade, backtest-week, backtest-report.

Every number written to the ledger is computed here from stored raw payloads.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from . import ENGINE_VERSION
from .client import CreditGuard, OddsAPI, RawStore, api_key, espn_get, utcnow
from .close import close_candidate, metrics
from .evaluate import price_quotes, size_stakes, to_dict
from .grade import ESPN, find_game, outcome, parse_scoreboard, player_stat, units
from .ledger import Ledger
from .normalize import parse_ts, quotes_from

ET = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parent.parent


def load_cfg(path: str | None = None) -> dict:
    return yaml.safe_load(open(path or ROOT / "config" / "rules.yaml"))


def in_window(commence: str, window: str, now: datetime) -> bool:
    t = parse_ts(commence)
    if t <= now:
        return False
    if window == "all":
        return True
    if window == "today_et":
        return t.astimezone(ET).date() == now.astimezone(ET).date()
    if window.startswith("within:"):
        return t - now <= timedelta(minutes=int(window.split(":")[1]))
    if window.startswith("after_et:"):
        hh, mm = map(int, window.split(":", 1)[1].split(":"))
        lt = t.astimezone(ET)
        return lt.date() == now.astimezone(ET).date() and (lt.hour, lt.minute) >= (hh, mm)
    raise ValueError(f"unknown window {window}")


def fmt_american(dec: float) -> str:
    a = (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)
    return f"{a:+.0f}"


def report(run_id: str, sport: str, cands: list[dict], cfg: dict, credits: int) -> str:
    bets = [c for c in cands if c["tier"] == "bet"]
    research = [c for c in cands if c["tier"] == "research"]
    rejected = [c for c in cands if c["tier"] == "rejected"]
    lab = "PAPER — not validated, do not bet" if cfg["mode"] == "paper" else "LIVE"
    lines = [f"**{sport} scan · {run_id}** · {lab}", ""]
    if bets:
        lines += ["| Bet | Venue | Price | Fair | EV (haircut, net) | Stake |", "|---|---|---|---|---|---|"]
        for c in sorted(bets, key=lambda c: -c["ev_adj_net"]):
            pt = "" if c["point"] is None else f" {c['point']:+g}" if "spreads" in c["family"] else f" {c['point']:g}"
            who = f"{c['description']} " if c["description"] else ""
            acct = "" if c["has_account"] else " (no acct)"
            lines.append(f"| {who}{c['side']}{pt} · {c['game']} | {c['venue']}{acct} | {fmt_american(c['price_dec'])} | "
                         f"{c['fair_prob']:.1%} | {c['ev_adj_net']:+.1%} | ${c['stake_usd']:.0f} |")
    else:
        lines.append("No bet-tier plays.")
    lines += ["", f"Research candidates logged: {len(research)} · rejected: {len(rejected)} · credits: {credits}"]
    return "\n".join(lines)


def cmd_scan(a, cfg):
    now = utcnow()
    store, led = RawStore(a.data), Ledger(a.data)
    month_used = store.credits_this_month(now, cfg["odds_api"].get("cycle_reset_day", 1))
    api = OddsAPI(api_key(), store, cfg["odds_api"]["max_credits_per_run"])
    sk = cfg["sports"][a.sport]
    oa = cfg["odds_api"]
    run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-{a.scan}"
    try:
        payload, retrieved, _ = api.odds(sk, "h2h,spreads,totals", oa["bulk_bookmakers"])
        events = [e for e in payload if in_window(e["commence_time"], a.window, now)]
        quotes = quotes_from(events, a.sport)
        expand = month_used < oa["monthly_soft_cap"]
        if expand and (a.derivatives or a.props):
            mk = ",".join(filter(None, [oa["derivative_markets"][a.sport] if a.derivatives else "",
                                        oa["prop_markets"][a.sport] if a.props else ""]))
            for e in events:
                ep, _, _ = api.event_odds(sk, e["id"], mk, oa["event_bookmakers"])
                quotes += quotes_from(ep, a.sport)
    except CreditGuard as g:
        quotes = quotes if "quotes" in locals() else []
        retrieved = locals().get("retrieved", now.isoformat())
        print(f"credit guard: {g}")
    decision_at = utcnow()
    cands = price_quotes(quotes, cfg, decision_at, scan=a.scan, news_window=a.news,
                         api_retrieved_at=retrieved, raw_refs=api.refs)
    open_expo = {}
    for p in led.all_picks():
        if p.get("tier") == "bet" and not p.get("result"):
            open_expo["game:" + p["event_id"]] = open_expo.get("game:" + p["event_id"], 0) + p.get("stake_usd", 0)
    cands = size_stakes(cands, cfg, open_expo)
    ds = [to_dict(c) for c in cands]
    for c in ds:
        if c["tier"] in ("bet", "research"):
            led.upsert_pick(c, decision_at.isoformat())
    summary = {"run_id": run_id, "mode": "scan", "sport": a.sport, "scan": a.scan, "window": a.window,
               "at": decision_at.isoformat(), "credits_est": api.used, "credits_month_before": month_used,
               "events": len({q.event_id for q in quotes}), "quotes": len(quotes),
               "bet": sum(c["tier"] == "bet" for c in ds), "research": sum(c["tier"] == "research" for c in ds),
               "rejected": sum(c["tier"] == "rejected" for c in ds), "engine_version": ENGINE_VERSION,
               "config_version": cfg["version"], "paper": cfg["mode"] == "paper"}
    led.write_run(run_id, summary, ds, report(run_id, a.sport, ds, cfg, api.used))
    print(json.dumps(summary))


def cmd_close(a, cfg):
    now = utcnow()
    store, led = RawStore(a.data), Ledger(a.data)
    lo, hi = now - timedelta(minutes=a.back), now + timedelta(minutes=a.ahead)
    todo = [p for p in led.all_picks() if p["sport"] == a.sport and not p.get("close_requested_at")
            and lo <= parse_ts(p["commence_time"]) <= hi]
    run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-close-{a.sport.lower()}"
    if not todo:
        print("nothing to close")
        return
    api = OddsAPI(api_key(), store, cfg["odds_api"]["max_credits_per_run"])
    sk, oa = cfg["sports"][a.sport], cfg["odds_api"]
    payload, retrieved, _ = api.odds(sk, "h2h,spreads,totals", oa["bulk_bookmakers"])
    quotes = quotes_from(payload, a.sport)
    need: dict[str, set] = {}
    for p in todo:
        if p["group"] != "main":
            need.setdefault(p["event_id"], set()).add(p["family"] if p["group"] == "prop" else
                                                      {"spreads": "alternate_spreads", "totals": "alternate_totals"}.get(p["family"], p["family"]))
    for eid, mks in need.items():
        try:
            ep, _, _ = api.event_odds(sk, eid, ",".join(sorted(mks)), oa["event_bookmakers"])
            quotes += quotes_from(ep, a.sport)
        except CreditGuard:
            break
    asked = utcnow()
    bets = {}
    for b in led.all_bets():
        bets.setdefault(b.get("pick_id"), []).append(b)
    n = 0
    for p in todo:
        out = close_candidate(p, quotes, cfg, asked, retrieved, list(api.refs))
        # real execution: CLV at the price Justin actually got, against the same Pinnacle close
        real = []
        for b in bets.get(p["id"], []):
            m = metrics(out["close_pinnacle_p"], float(b["price_dec"]), p.get("push_prob", 0))
            real.append({"bet_id": b["id"], "book": b["book"], "price_dec": b["price_dec"],
                         "placed_at": b.get("placed_at"), "stake_usd": b.get("stake_usd"), "clv_pinnacle": m})
        if real:
            out["real_bets"] = real
        led.update_pick(p["id"], out)
        n += 1
    led.write_run(run_id, {"run_id": run_id, "mode": "close", "sport": a.sport, "at": asked.isoformat(),
                           "credits_est": api.used, "closed": n}, [], f"Closed {n} {a.sport} picks.")
    print(f"closed {n}")


def cmd_grade(a, cfg):
    now = utcnow()
    store, led = RawStore(a.data), Ledger(a.data)
    open_ = [p for p in led.all_picks() if not p.get("result") and parse_ts(p["commence_time"]) < now - timedelta(hours=4)]
    boards: dict = {}
    n = 0
    for p in open_:
        day = parse_ts(p["commence_time"]).astimezone(ET).strftime("%Y%m%d")
        key = (p["sport"], day)
        if key not in boards:
            boards[key] = parse_scoreboard(espn_get(store, f"{ESPN[p['sport']]}/scoreboard?dates={day}", f"espn_{p['sport']}_{day}"))
        g = find_game(boards[key], p["game"])
        res = None
        if g and g["completed"]:
            stat = None
            if p["family"].startswith("player_"):
                summ = espn_get(store, f"{ESPN[p['sport']]}/summary?event={g['espn_id']}", f"espn_box_{g['espn_id']}")
                stat = player_stat(summ, p["description"], p["family"])
            res = outcome(p, g, stat)
        age_days = (now - parse_ts(p["commence_time"])).days
        if res is None and age_days >= 3:
            res = "void"
        if res:
            led.update_pick(p["id"], {"result": res, "units": units(res, p["effective_dec"]), "graded_at": now.isoformat()})
            n += 1
    run_id = f"{now.strftime('%Y%m%d-%H%M%S')}-grade"
    led.write_run(run_id, {"run_id": run_id, "mode": "grade", "at": now.isoformat(), "credits_est": 0, "graded": n},
                  [], f"Graded {n} picks.")
    print(f"graded {n}")


def main(argv=None):
    ap = argparse.ArgumentParser("edge-engine")
    ap.add_argument("--data", default="data")
    ap.add_argument("--config")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan")
    s.add_argument("--sport", required=True, choices=["NFL", "NBA"])
    s.add_argument("--window", default="all")
    s.add_argument("--scan", required=True)
    s.add_argument("--news", action="store_true")
    s.add_argument("--derivatives", action="store_true")
    s.add_argument("--props", action="store_true")
    c = sub.add_parser("close")
    c.add_argument("--sport", required=True, choices=["NFL", "NBA"])
    c.add_argument("--back", type=int, default=30)
    c.add_argument("--ahead", type=int, default=60)
    sub.add_parser("grade")
    b = sub.add_parser("backtest-week")
    b.add_argument("--start", required=True)
    b.add_argument("--latency-min", type=int, default=10)
    sub.add_parser("backtest-report")
    dr = sub.add_parser("d1-run")
    dr.add_argument("--weeks", type=int, nargs="+", required=True)
    dp = sub.add_parser("d1-report")
    dp.add_argument("--final", action="store_true")
    m2p = sub.add_parser("m2")
    m2p.add_argument("--cache", default="/tmp/nfl")
    sub.add_parser("s1-run")
    sub.add_parser("s1-report")
    pr = sub.add_parser("probe")
    pr.add_argument("--date", default="2025-11-09")
    pr.add_argument("--delays", type=int, nargs="*", default=[2, 10, 30])
    pr.add_argument("--max-credits", type=int, default=200)
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    if a.cmd == "scan":
        cmd_scan(a, cfg)
    elif a.cmd == "close":
        cmd_close(a, cfg)
    elif a.cmd == "grade":
        cmd_grade(a, cfg)
    elif a.cmd == "backtest-week":
        from .backtest import run_week
        run_week(a, cfg)
    elif a.cmd == "backtest-report":
        from .backtest import report as bt_report
        bt_report(a, cfg)
    elif a.cmd == "d1-run":
        from .d1 import run_weeks
        run_weeks(a, cfg)
    elif a.cmd == "d1-report":
        from .d1 import report as d1_report
        d1_report(a, cfg)
    elif a.cmd == "m2":
        from .m2 import run as m2_run
        m2_run(a, cfg)
    elif a.cmd == "s1-run":
        from .s1 import run as s1_run
        s1_run(a, cfg)
    elif a.cmd == "s1-report":
        from .s1 import report as s1_report
        s1_report(a, cfg)
    elif a.cmd == "probe":
        from .probe import run as probe_run
        probe_run(a, cfg)


if __name__ == "__main__":
    main()
