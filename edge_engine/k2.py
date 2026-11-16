"""K2 — in-game NO on Kalshi KXNBAPTS: historical replay (config/prereg/k2-replay.yaml). Paper only; reads the K2 cache.

    python -m edge_engine.k2_pull all --cache ~/k2-cache          # fetch (public endpoints, cached outside the repo)
    python -m edge_engine.k2 --cache ~/k2-cache --data data        # replay -> data/k2/

Works from quotes (minute candlesticks), not from other people's trades; in-game is defined by ESPN play-by-play
(actual tip = first play's wallclock); one position per player per checkpoint. Every mechanic is fixed in the config
file before the run. The PRIMARY rule (taker NO at halftime) on 2026-27 with the tradable filter decides the stop rule.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .k2_pull import ET, espn_code, load, norm_name, ts

C = 20                                   # contracts per position
TAKER, MAKER_SENS = 0.07, 0.0175         # fee rates (KXNBAPTS 'quadratic', multiplier 1; maker sensitivity only)
BAND_LO, BAND_HI = 0.10, 0.90
CHECK_LAG_S, TRADABLE_S, MID_AFTER_S = 60, 60, 300
PERIOD_MIN, PERIOD_MAX = 15 * 60, 75 * 60     # wallclock sanity: a quarter outside 15-75 min means broken stamps
BOOT, SEED = 10000, 20261116
QUOTE_LOOKBACK_S = 3 * 3600             # quote window starts at scheduled tip - 3 h (same for both seasons)
CHECKPOINTS = {1: "q1", 2: "half", 3: "q3"}
SEASON_SPLIT = "2026-08-01"


def season(d: str) -> str:
    return "2026-27" if d >= SEASON_SPLIT else "2025-26"


# ---------------------------------------------------------------- fees
def fee(price, contracts, rate: float = TAKER):
    """Kalshi quadratic fee for ONE order: cent-ceiling of rate x C x P x (1 - P) (fee-rounding doc: per-fill fee to
    $0.000001, balance aligned to $0.01, order-level accumulator -> the order pays the cent-ceiling). Vectorised."""
    raw = np.round(rate * np.asarray(contracts, float) * np.asarray(price, float) * (1 - np.asarray(price, float)) * 100, 6)
    out = np.ceil(raw) / 100
    return float(out) if np.ndim(out) == 0 else out


# ---------------------------------------------------------------- quotes, game state
class Book:
    """Minute candles [end_ts, yes_bid, yes_ask, vol] for one market. A candle exists only when something changed, so
    the quote at t is the close of the last candle with end_ts <= t."""
    def __init__(self, candles):
        self.t = [c[0] for c in candles]
        self.c = candles

    def at(self, t: float):
        i = bisect.bisect_right(self.t, t) - 1
        if i < 0:
            return None
        return self.c[i][1], self.c[i][2]


def no_quote(q):
    """(no_ask, no_bid) from a (yes_bid, yes_ask) close; None where that side is empty."""
    if q is None:
        return None, None
    yb, ya = q
    ask = round(1 - yb, 4) if yb is not None and yb > 0 else None
    bid = round(1 - ya, 4) if ya is not None and 0 < ya < 1 else None
    return ask, bid


class Points:
    """Player points so far from ESPN scoring plays [wallclock, athlete_id, value, period]."""
    def __init__(self, scoring):
        by = defaultdict(list)
        for wc, aid, v, _ in scoring:
            by[aid].append((ts(wc).timestamp(), v))
        self.by = {}
        for aid, xs in by.items():
            xs.sort()
            self.by[aid] = ([x[0] for x in xs], np.cumsum([x[1] for x in xs]).tolist())

    def at(self, aid: str, t: float) -> int:
        if aid not in self.by:
            return 0
        tt, cum = self.by[aid]
        i = bisect.bisect_right(tt, t) - 1
        return int(cum[i]) if i >= 0 else 0

    def final(self, aid: str) -> int:
        return int(self.by[aid][1][-1]) if aid in self.by else 0


def select(cands: list[dict]):
    """One strike per player: NO ask in [0.10, 0.90], closest to 0.50, ties -> lower floor_strike."""
    ok = [c for c in cands if c["no_ask"] is not None and BAND_LO - 1e-9 <= c["no_ask"] <= BAND_HI + 1e-9]
    if not ok:
        return None
    return min(ok, key=lambda c: (round(abs(c["no_ask"] - 0.5), 6), c["floor_strike"]))


def tradable(trades, t: float, no_ask: float, window: float = TRADABLE_S) -> bool:
    """A trade within +/- window s of t at a NO price <= the quoted NO ask (NO price = 1 - yes_price)."""
    return any(abs(x[0] - t) <= window and (1 - x[1]) <= no_ask + 1e-9 for x in trades)


def maker_fill(trades, t_place: float, t_cancel: float, no_bid: float):
    """First trade strictly after placement and before cancel with NO price strictly below the resting NO bid."""
    for x in sorted(trades):
        if t_place < x[0] < t_cancel and (1 - x[1]) < no_bid - 1e-9:
            return x[0]
    return None


def pace_band(pts: int, period: int, strike: float) -> str:
    r = (pts / (period / 4)) / (strike + 0.5)
    return "<0.75" if r < 0.75 else "0.75-1.00" if r < 1.0 else "1.00-1.25" if r < 1.25 else ">=1.25"


def price_band(p: float) -> str:
    return "0.10-0.30" if p < 0.30 else "0.30-0.50" if p < 0.50 else "0.50-0.70" if p < 0.70 else "0.70-0.90"


# ---------------------------------------------------------------- statistics
def boot(df: pd.DataFrame, seed: int = SEED, n: int = BOOT):
    """ROI = sum(pnl)/sum(cost); 90% interval from resampling whole ET dates."""
    if not len(df):
        return None, None, None
    g = df.groupby("date")[["pnl", "cost"]].sum()
    P, Cc = g.pnl.values, g.cost.values
    roi = float(P.sum() / Cc.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    r = P[idx].sum(1) / Cc[idx].sum(1)
    return roi, float(np.quantile(r, 0.05)), float(np.quantile(r, 0.95))


def summarize(df: pd.DataFrame) -> dict:
    roi, lo, hi = boot(df)
    nights = df.groupby("date").pnl.sum() if len(df) else pd.Series(dtype=float)
    by_m = {m: {"bets": int(len(x)), "roi": float(x.pnl.sum() / x.cost.sum())}
            for m, x in df.groupby(df.date.str[:7])} if len(df) else {}
    return {"bets": int(len(df)), "nights": int(len(nights)), "roi": roi, "ci90": [lo, hi],
            "share_profitable_nights": float((nights > 0).mean()) if len(nights) else None, "by_month": by_m}


# ---------------------------------------------------------------- per-game replay
def _player_map(roster: dict, teams: set[str]):
    by = defaultdict(list)
    for aid, (name, abbr) in roster.items():
        if abbr in teams and name:
            by[norm_name(name)].append(aid)
    return by


def replay_game(g: dict, mk: list[dict], tape) -> tuple[list[dict], dict]:
    """Bets for one game. g: reduced ESPN game + event fields; mk: its markets; tape(ticker) -> (candles, trades)."""
    stats = defaultdict(int)
    pe = {int(k): ts(v).timestamp() for k, v in (g["period_end"] or {}).items()}
    good, prev = set(), ts(g["actual_tip"]).timestamp()     # checkpoint k usable only if quarters 1..k look sane
    for k in (1, 2, 3):
        if k not in pe or not PERIOD_MIN <= pe[k] - prev <= PERIOD_MAX:
            break
        good.add(k)
        prev = pe[k]
    end = ts(g["game_end"]).timestamp()
    pts = Points(g["scoring"])
    pmap = _player_map(g["roster"], {espn_code(g["away"]), espn_code(g["home"])})
    by_player = defaultdict(list)
    for m in mk:
        by_player[m["player"]].append(m)
    bets = []
    for player, ms in by_player.items():
        ids = pmap.get(norm_name(player), [])
        if len(ids) != 1:
            stats["player_unmatched" if not ids else "player_ambiguous"] += 1
            continue
        aid = ids[0]
        books, trades = {}, {}
        for m in ms:
            cnd, trd = tape(m["ticker"])
            books[m["ticker"]], trades[m["ticker"]] = Book(cnd), trd
            # mechanics check: ESPN play-by-play final vs Kalshi settlement
            stats["settle_checked"] += 1
            stats["settle_agree"] += int((pts.final(aid) > m["floor_strike"]) == (m["result"] == "yes"))
        for per, name in CHECKPOINTS.items():
            if per not in good:
                stats[f"bad_wallclock_{name}"] += 1
                continue
            t = pe[per] + CHECK_LAG_S
            t_next = pe[per + 1] + CHECK_LAG_S if per + 1 in pe and per < 3 else end
            p_now = pts.at(aid, t)
            cands = []
            for m in ms:
                if ts(m["open_time"]).timestamp() > t or ts(m["close_time"]).timestamp() <= t:
                    continue
                q = books[m["ticker"]].at(t)
                ask, bid = no_quote(q)
                decided = p_now > m["floor_strike"]
                if decided:
                    stats["decided_by_pbp"] += 1
                    stats["decided_kalshi_agrees"] += int(q is not None and q[0] is not None and q[0] >= 0.95)
                    continue
                if ask is None:
                    continue
                cands.append({**m, "no_ask": ask, "no_bid": bid})
            s = select(cands)
            stats[f"player_checkpoints_{name}"] += 1
            if s is None:
                stats[f"no_eligible_strike_{name}"] += 1
                continue
            trd = trades[s["ticker"]]
            win = s["result"] == "no"
            row = {"date": g["date"], "season": season(g["date"]), "game_id": g["game_id"], "event_ticker": g["event_ticker"],
                   "player": player, "checkpoint": name, "ticker": s["ticker"], "floor_strike": s["floor_strike"],
                   "pts_so_far": p_now, "pace_band": pace_band(p_now, per, s["floor_strike"]), "result": s["result"],
                   "n_strikes_eligible": len(cands)}
            a = s["no_ask"]
            f = fee(a, C)
            bets.append({**row, "rule": "taker", "price": a, "price_band": price_band(a), "fee": f, "cost": C * a + f,
                         "pnl": C * win - (C * a + f), "tradable": tradable(trd, t, a)})
            b = s["no_bid"]
            if b is not None and b >= 0.01:
                ft = maker_fill(trd, t, t_next, b)
                mrow = {**row, "rule": "maker", "price": b, "price_band": price_band(max(b, BAND_LO)), "filled": ft is not None}
                if ft is not None:
                    fm = fee(b, C, MAKER_SENS)
                    mid = books[s["ticker"]].at(ft + MID_AFTER_S)
                    na, nb = no_quote(mid)
                    mrow.update(fee=0.0, cost=C * b, pnl=C * win - C * b, fee_sens=fm, pnl_sens=C * win - (C * b + fm),
                                mid5=(na + nb) / 2 if na is not None and nb is not None else None,
                                fill_delay_s=ft - t)
                bets.append(mrow)
    return bets, dict(stats)


# ---------------------------------------------------------------- background reproduction (trades that happened)
POSITIONS = ["yes_taker", "maker_vs_yes_buyers", "no_taker", "maker_vs_no_buyers"]


def background_rows(trades, result: str, start: float, end: float, maker_rate: float):
    """Per-position (n, cost, pnl) over in-game trades with yes_price in [0.10, 0.90], each capped at 100 contracts."""
    out = {p: [0, 0.0, 0.0] for p in POSITIONS}
    if not trades:
        return out
    T = np.asarray([x[:3] for x in trades], float)
    side = np.asarray([x[3] for x in trades])
    k = (T[:, 0] >= start) & (T[:, 0] <= end) & (T[:, 1] >= BAND_LO - 1e-9) & (T[:, 1] <= BAND_HI + 1e-9)
    y, c, sd = T[k, 1], np.minimum(T[k, 2], 100), side[k]
    yes = result == "yes"
    legs = {"yes_taker": (sd == "yes", y, TAKER, yes), "maker_vs_yes_buyers": (sd == "yes", 1 - y, maker_rate, not yes),
            "no_taker": (sd == "no", 1 - y, TAKER, not yes), "maker_vs_no_buyers": (sd == "no", y, maker_rate, yes)}
    for p, (msk, price, rate, wins) in legs.items():
        if msk.any():
            cost = c[msk] * price[msk] + (fee(price[msk], c[msk], rate) if rate else 0)
            out[p] = [int(msk.sum()), float(np.sum(cost)), float(np.sum(c[msk] * wins) - np.sum(cost))]
    return out


# ---------------------------------------------------------------- driver
def run(cache: str, data: str) -> dict:
    cache = Path(cache)
    M = pd.read_csv(cache / "markets.csv.gz")
    E = pd.read_csv(cache / "event_games.csv.gz", dtype={"game_id": str})
    G = pd.read_csv(cache / "espn_games.csv.gz", dtype={"game_id": str}).set_index("game_id")
    M = M[(M.date >= "2025-11-18") & (M.date <= "2026-11-15")]
    counts = {"markets_in_window": int(len(M)), "markets_scalar_excluded": int((M.result == "scalar").sum()),
              "events": int(E.event_ticker.nunique()), "events_unmatched_espn": int(E.game_id.isna().sum())}
    # archived markets are labelled "structured", recent ones "greater"; both are "points > floor_strike" (title "N+")
    M = M[M.result.isin(["yes", "no"]) & M.strike_type.isin(["greater", "structured"]) & M.floor_strike.notna()]
    eg = dict(zip(E.event_ticker, E.game_id))
    M = M[M.event_ticker.map(lambda e: isinstance(eg.get(e), str))].copy()
    M["game_id"] = M.event_ticker.map(eg)

    def tape(t, sched=None):
        """Candles inside the fixed quote window only (>= scheduled tip - 3 h), whichever endpoint fetched them."""
        cp, tp = cache / "candles" / f"{t}.json.gz", cache / "trades" / f"{t}.json.gz"
        c = load(cp) if cp.exists() else []
        if sched is not None:
            c = [x for x in c if x[0] >= sched - QUOTE_LOOKBACK_S]
        return c, (load(tp) if tp.exists() else [])

    bets, stats, bg, delays = [], defaultdict(int), [], []
    no_pbp = []
    for gid, mk in M.groupby("game_id"):
        gp = cache / "espn" / "games" / f"{gid}.json.gz"
        sched = ts(G.loc[gid, "sched_tip"]).timestamp()
        d = mk.date.iloc[0]
        rg = load(gp) if gp.exists() else None
        has_pbp = bool(rg and rg.get("actual_tip") and rg.get("n_wallclock", 0) > 0 and "2" in (rg.get("period_end") or {}))
        recs = mk.to_dict("records")
        for m in recs:                                         # background table, trades that happened
            _, trd = tape(m["ticker"])
            close = ts(m["close_time"]).timestamp()
            defs = [("sched_all", sched, close)]
            if has_pbp:
                gend = ts(rg["game_end"]).timestamp()
                defs += [("sched_pbp", sched, gend), ("actual_pbp", ts(rg["actual_tip"]).timestamp(), gend)]
            for name, s0, s1 in defs:
                for mr in (MAKER_SENS, 0.0):
                    r = background_rows(trd, m["result"], s0, s1, mr)
                    for p, (n, cost, pnl) in r.items():
                        if n:
                            bg.append((d, name, mr, p, n, cost, pnl))
            if has_pbp and trd:                               # pre-tip leakage under the scheduled-tip definition
                at = ts(rg["actual_tip"]).timestamp()
                for x in trd:
                    if x[3] == "yes" and sched <= x[0] and BAND_LO <= x[1] <= BAND_HI:
                        stats["yes_buy_contracts_after_sched"] += min(x[2], 100)
                        stats["yes_buy_contracts_first10_after_sched"] += min(x[2], 100) * (x[0] < sched + 600)
                        stats["yes_buy_contracts_before_actual_tip"] += min(x[2], 100) * (x[0] < at)
        if not has_pbp:
            no_pbp.append(gid)
            continue
        delays.append((d, (ts(rg["actual_tip"]).timestamp() - sched) / 60))
        g = {**rg, "date": d, "game_id": gid, "event_ticker": mk.event_ticker.iloc[0], "away": mk.away.iloc[0],
             "home": mk.home.iloc[0]}
        b, st = replay_game(g, recs, lambda t, s0=sched: tape(t, s0))
        bets += b
        for k, v in st.items():
            stats[f"{season(d)}:{k}"] += v
    B = pd.DataFrame(bets)
    BG = pd.DataFrame(bg, columns=["date", "tipdef", "maker_rate", "position", "n", "cost", "pnl"])
    D = pd.DataFrame(delays, columns=["date", "delay_min"])
    counts.update(games=int(M.game_id.nunique()), games_without_pbp=len(no_pbp), games_without_pbp_ids=no_pbp)
    return report(B, BG, D, counts, dict(stats), Path(data) / "k2")


def _fmt(s: dict) -> str:
    if not s or s["roi"] is None:
        return "| 0 | - | - | - |"
    return (f"| {s['bets']} | {100 * s['roi']:+.1f}% | ({100 * s['ci90'][0]:+.1f}, {100 * s['ci90'][1]:+.1f}) | "
            f"{100 * s['share_profitable_nights']:.0f}% |")


def report(B: pd.DataFrame, BG: pd.DataFrame, D: pd.DataFrame, counts: dict, stats: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    res = {"generated_at": datetime.now(ET).isoformat(), "config": "config/prereg/k2-replay.yaml", "counts": counts,
           "mechanics": stats, "tip_delay_min": {s: {"median": float(x.delay_min.median()), "p90": float(x.delay_min.quantile(0.9)),
                                                    "max": float(x.delay_min.max()), "games": int(len(x))}
                                                for s, x in D.groupby(D.date.map(season))}}
    # step 3: background table, scheduled vs actual tips
    bgt = {}
    for (tipdef, mr, pos), x in BG.groupby(["tipdef", "maker_rate", "position"]):
        for sn, y in x.groupby(x.date.map(season)):
            bgt[f"{tipdef}|{mr}|{pos}|{sn}"] = summarize(y.rename(columns={}))
    res["background"] = bgt
    nob = BG[(BG.position == "no_taker") & (BG.maker_rate == MAKER_SENS)]
    res["background_no_taker_by_month"] = {td: {m: float(y.pnl.sum() / y.cost.sum()) for m, y in x.groupby(x.date.str[:7])}
                                           for td, x in nob.groupby("tipdef")}
    # steps 4 + 6: quote replay
    rep = {}
    T = B[B.rule == "taker"]
    for sn in ("2025-26", "2026-27"):
        for cp in CHECKPOINTS.values():
            x = T[(T.season == sn) & (T.checkpoint == cp)]
            rep[f"taker|{cp}|{sn}|all_quotes"] = summarize(x)
            rep[f"taker|{cp}|{sn}|tradable"] = summarize(x[x.tradable])
            mk = B[(B.rule == "maker") & (B.season == sn) & (B.checkpoint == cp)]
            f = mk[mk.filled.astype("boolean").fillna(False).astype(bool)] if len(mk) else mk
            s = summarize(f)
            s.update(orders=int(len(mk)), fill_rate=float(len(f) / len(mk)) if len(mk) else None,
                     mid5_minus_fill_cents=float(100 * (f.mid5 - f.price).mean()) if len(f) else None,
                     roi_maker_fee_sensitivity=float(f.pnl_sens.sum() / (f.cost + f.fee_sens).sum()) if len(f) else None)
            rep[f"maker|{cp}|{sn}"] = s
    res["replay"] = rep
    expl = {}
    for sn in ("2025-26", "2026-27"):
        x = T[(T.season == sn) & T.tradable]
        for col in ("price_band", "pace_band"):
            for (cp, v), y in x.groupby(["checkpoint", col]):
                expl[f"{col}|{cp}|{v}|{sn}"] = summarize(y)
    res["exploratory"] = expl
    p = rep["taker|half|2026-27|tradable"]
    res["stop_rule"] = {"primary": "taker NO at halftime, 2026-27, tradable filter", "roi": p["roi"], "ci90": p["ci90"],
                        "bets": p["bets"], "passes": bool(p["roi"] is not None and p["roi"] > 0)}
    B.to_csv(out / "bets.csv.gz", index=False)
    BG.groupby(["date", "tipdef", "maker_rate", "position"], as_index=False)[["n", "cost", "pnl"]].sum().to_csv(
        out / "background_by_date.csv.gz", index=False)
    (out / "result.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "report.md").write_text(markdown(res))
    return res


def markdown(res: dict) -> str:
    L = ["# K2 replay (config/prereg/k2-replay.yaml)", "",
         f"Stop rule: PRIMARY ROI {100 * (res['stop_rule']['roi'] or 0):+.1f}% "
         f"(90% {100 * (res['stop_rule']['ci90'][0] or 0):+.1f}, {100 * (res['stop_rule']['ci90'][1] or 0):+.1f}), "
         f"{res['stop_rule']['bets']} bets -> {'PASS' if res['stop_rule']['passes'] else 'STOP'}", ""]
    L += ["## Step 3: background table (trades that happened)", "",
          "| position | season | tips | maker fee | trades | ROI | 90% CI | profitable nights |", "|---|---|---|---|---|---|---|---|"]
    for k, s in sorted(res["background"].items()):
        td, mr, pos, sn = k.split("|")
        if pos.startswith("maker") or float(mr) > 0:
            L.append(f"| {pos} | {sn} | {td} | {mr} " + _fmt(s))
    L += ["", "## Steps 4 + 6: quote replay", "", "| rule | checkpoint | season | quotes | bets | ROI | 90% CI | profitable nights |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in res["replay"].items():
        parts = k.split("|")
        L.append(f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3] if len(parts) > 3 else 'fills'} " + _fmt(s))
    L += ["", "## By month (taker, tradable)", ""]
    for k, s in res["replay"].items():
        if k.startswith("taker") and k.endswith("tradable"):
            L.append(f"- {k}: " + ", ".join(f"{m} {100 * v['roi']:+.1f}% ({v['bets']})" for m, v in s["by_month"].items()))
    L += ["", "## Exploratory (taker, tradable)", "", "| cut | checkpoint | band | season | bets | ROI | 90% CI | profitable nights |",
          "|---|---|---|---|---|---|---|---|"]
    for k, s in res["exploratory"].items():
        a, b, c, d = k.split("|")
        L.append(f"| {a} | {b} | {c} | {d} " + _fmt(s))
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / "k2-cache"))
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    r = run(a.cache, a.data)
    print(json.dumps({"stop_rule": r["stop_rule"], "counts": {k: v for k, v in r["counts"].items() if k != "games_without_pbp_ids"}},
                     indent=1, default=float))
