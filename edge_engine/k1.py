"""K1: paper maker test on Kalshi NBA player-points ladders (approved 2026-10-19). Logging side.

At the decision time (the monitor's first Kalshi snapshot of the night, with box scores frozen through
yesterday), every eligible rung gets:
- fair P(points > floor_strike) from frozen M1 and M1.1 (same code and constants as Gate A);
- the simulated resting orders: YES bid floor(fair - 3c), NO bid floor((1 - fair) - 3c), 10 contracts,
  not placed if they would cross the current ask;
- the naive comparison orders from mid +/- 3c on the identical rung set;
- top of book with sizes (depth diagnostic).
Fills (through-only), the 60 s cancel on injury news, the closing mid and settlement are scored later
from the archived tape and snapshots by the frozen rules in config k1.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from . import m1
from .m1_live import TEAM_ABBR, apply_cal, norm_name, p_over_push, pmf

KALSHI_TO_ESPN = {"SAS": "SA", "NYK": "NY", "GSW": "GS", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX",
                  "BRK": "BKN", "CHO": "CHA"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def event_teams(event_ticker: str) -> set[str]:
    code = event_ticker.split("-")[1][7:] if "-" in event_ticker else ""
    if len(code) != 6:
        return set()
    return {KALSHI_TO_ESPN.get(code[:3], code[:3]), KALSHI_TO_ESPN.get(code[3:], code[3:])}


def cents_floor(x: float) -> float:
    return math.floor(round(x * 100, 6)) / 100


def orders(fair: float, yes_bid: float | None, yes_ask: float | None, offset: float) -> dict:
    yb = cents_floor(fair - offset)
    nb = cents_floor((1 - fair) - offset)
    no_ask = None if yes_bid is None else round(1 - yes_bid, 4)
    return {"yes_bid": yb, "yes_placed": bool(0 < yb < 1 and (yes_ask is None or yb < yes_ask)),
            "no_bid": nb, "no_placed": bool(0 < nb < 1 and (no_ask is None or nb < no_ask))}


def quotes(box: pd.DataFrame, markets: list[dict], decision_at: str, games: list[dict], cfg: dict, frozen: dict) -> list[dict]:
    k, c = cfg["k1"], frozen["constants"]
    app = box[(box.dnp == 0) & (box.minutes > 0) & box.pts.notna()].copy()
    app["date"] = pd.to_datetime(app["date"], utc=True)
    by_name = {}
    for pid, g in app.sort_values("date").groupby("player_id"):
        by_name.setdefault(norm_name(g.iloc[-1].player), []).append((pid, g.iloc[-1].team))
    tips = {}
    for e in games:
        tips[frozenset({TEAM_ABBR.get(e["home_team"]), TEAM_ABBR.get(e["away_team"])})] = e["commence_time"]
    out, cache, seen = [], {}, set()
    for mk in markets:
        if mk.get("ticker") in seen:
            continue
        seen.add(mk.get("ticker"))
        sub = mk.get("yes_sub_title") or mk.get("no_sub_title") or ""
        player = sub.split(":")[0].strip()
        strike = _f(mk.get("floor_strike"))
        teams = event_teams(mk.get("event_ticker", ""))
        yb, ya = _f(mk.get("yes_bid_dollars")), _f(mk.get("yes_ask_dollars"))
        rec = {"ticker": mk.get("ticker"), "event_ticker": mk.get("event_ticker"), "player": player, "floor_strike": strike,
               "strike_type": mk.get("strike_type"), "decision_at": decision_at, "yes_bid": yb, "yes_ask": ya,
               "yes_bid_size": _f(mk.get("yes_bid_size_fp")), "yes_ask_size": _f(mk.get("yes_ask_size_fp")),
               "open_interest": _f(mk.get("open_interest_fp")), "eligible": False, "reason": None}
        tip = tips.get(frozenset(teams))
        cand = [(pid, tm) for pid, tm in by_name.get(norm_name(player), []) if tm in teams]
        if mk.get("strike_type") != "greater" or strike is None:
            rec["reason"] = "unsupported_strike"
        elif tip is None:
            rec["reason"] = "no_matching_game"
        elif len(cand) != 1:
            rec["reason"] = "no_unique_player_match" if not cand else "ambiguous_player"
        if rec["reason"]:
            out.append(rec)
            continue
        pid = cand[0][0]
        if pid not in cache:
            hist = app[(app.player_id == pid) & (app.date < pd.Timestamp(tip))]
            if len(hist) < m1.MIN_APPS:
                cache[pid] = ("prior", len(hist))
            else:
                pseudo = hist.iloc[[-1]].copy()
                pseudo["date"], pseudo["pts"], pseudo["minutes"], pseudo["game_id"] = pd.Timestamp(tip), 0.0, 1.0, "tonight"
                f = m1.features(pd.concat([hist, pseudo]).sort_values("date").reset_index(drop=True))
                f = f[f.game_id == "tonight"]
                _, med = m1.m1_logscore(f.assign(pts=0), c)
                cache[pid] = ("ok", pmf(f, c), float(med[0]), f.role.iloc[0])
        got = cache[pid]
        if got[0] == "prior":
            rec["reason"] = f"prior_appearances_{got[1]}_lt_{m1.MIN_APPS}"
            out.append(rec)
            continue
        P, med, role = got[1], got[2], got[3]
        rec.update({"proj_median_minutes": med, "role": role, "tip": tip})
        if med < cfg["m1"]["min_projected_minutes"]:
            rec["reason"] = "projected_minutes_lt_12"
            out.append(rec)
            continue
        over, _ = p_over_push(P, np.array([strike]))
        f_m1 = float(over[0])
        f_m11 = float(apply_cal(np.array([f_m1]), frozen["m1_1_isotonic"])[0])
        rec.update({"fair_m1": f_m1, "fair_m1_1": f_m11,
                    "orders_m1_1": orders(f_m11, yb, ya, k["offset"]), "orders_m1": orders(f_m1, yb, ya, k["offset"])})
        if yb is not None and ya is not None and 0 < yb < ya < 1:
            rec["orders_naive"] = orders((yb + ya) / 2, yb, ya, k["offset"])
        rec["eligible"], rec["reason"] = True, "ok"
        out.append(rec)
    return out


def write_and_push(data: str, day: str, recs: list[dict]) -> str:
    p = Path(data) / "k1" / "quotes" / f"{day}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        for r in recs:
            fh.write(json.dumps(r, default=float) + "\n")
    subprocess.run(f"cd {data} && git add k1 && (git commit -q -m 'K1 quotes {day}' || true) && "
                   "for i in 1 2 3 4 5; do git push -q origin data && break; git pull -q --rebase origin data; done",
                   shell=True, check=False)
    return str(p)
