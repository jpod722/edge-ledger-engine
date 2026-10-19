"""M1 / M1.1 frozen forward forecasting for Gate A (signed off 2026-10-19).

- M1: the Gate 0 model exactly (edge_engine.m1, engine 6925544), constants fitted on 2023-24.
- M1.1: M1 + one isotonic map of P(over), fitted only on 2023-24 (pooled over lines at the stand-in
  centre floor(rolling-10 avg) + 0.5 + j, j = -4..4, for the Gate 0 universe). Nothing else differs.
Both are frozen into config/m1_frozen.json by `freeze`, committed before the first forward forecast.

Every eligible DraftKings/FanDuel player_points contract at the decision snapshot is logged with both
models' probabilities and the leave-the-venue-out market consensus, so Gate A can score the full set.
"""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import m1
from .odds import devig

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "m1_frozen.json"
KMAX = 80
TEAM_ABBR = {"Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
             "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
             "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
             "Los Angeles Clippers": "LAC", "LA Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
             "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
             "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
             "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
             "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH"}


def norm_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def pmf(rows: pd.DataFrame, c: dict) -> np.ndarray:
    """P(points = k), k = 0..KMAX, from the frozen M1 code path."""
    out = np.zeros((len(rows), KMAX + 1))
    for k in range(KMAX + 1):
        lp, _ = m1.m1_logscore(rows.assign(pts=k), c)
        out[:, k] = np.exp(lp)
    return out / out.sum(1, keepdims=True)


def p_over_push(P: np.ndarray, lines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    k = np.arange(KMAX + 1)[None, :]
    over = (P * (k > lines[:, None])).sum(1)
    push = (P * (k == lines[:, None])).sum(1)
    return over, push


def pav_weighted(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Increasing isotonic fit (stack-based pool-adjacent-violators, O(n))."""
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnt.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnt.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2)); wts.append(w1 + w2); cnt.append(c1 + c2)
    return np.repeat(vals, cnt)


def apply_cal(p: np.ndarray, knots: dict) -> np.ndarray:
    return np.interp(p, knots["x"], knots["y"])


def freeze(data: str, out: Path = FROZEN) -> dict:
    df = m1.load(data)
    f = m1.features(df)
    c = m1.fit_constants(f)
    t = f[f.season == m1.FIT_SEASON].copy()
    _, med = m1.m1_logscore(t, c)
    t = t[med >= m1.MIN_PROJ_MIN].copy()
    P = pmf(t, c)
    xs, ys = [], []
    centre = np.floor(t.mean_pts10.values) + 0.5
    for j in range(-4, 5):
        L = np.maximum(centre + j, 0.5)
        over, _ = p_over_push(P, L)
        xs.append(over)
        ys.append((t.pts.values > L).astype(float))
    x, y = np.concatenate(xs), np.concatenate(ys)
    o = np.argsort(x, kind="stable")
    xo = x[o]
    fit = pav_weighted(y[o], np.ones(len(o)))
    # compress to knots at the end of each constant block
    keep = np.r_[np.diff(fit) != 0, True]
    knots = {"x": [0.0] + [float(v) for v in xo[keep]] + [1.0],
             "y": [float(fit[0])] + [float(v) for v in fit[keep]] + [float(fit[-1])]}
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "model": "M1 (engine 6925544) and M1.1 = M1 + isotonic P(over)",
              "fit_season": m1.FIT_SEASON, "calibration_points": int(len(x)), "calibration_lines": "floor(rolling-10)+0.5+j, j=-4..4",
              "constants": c, "m1_1_isotonic": knots}
    out.write_text(json.dumps(frozen, indent=1, default=float))
    print(json.dumps({"calibration_points": len(x), "knots": len(knots["x"])}))
    return frozen


def consensus(quotes: list[dict], venue: str, books: list[str], min_books: int):
    ps = []
    for b in books:
        if b == venue:
            continue
        q = [x for x in quotes if x["book"] == b]
        o = next((x for x in q if x["side"] == "Over"), None)
        u = next((x for x in q if x["side"] == "Under"), None)
        if o and u:
            ps.append((b, devig(o["dec"], u["dec"], "power")[0]))
    if len(ps) < min_books:
        return None, [b for b, _ in ps]
    return float(np.mean([p for _, p in ps])), [b for b, _ in ps]


def forecasts(box: pd.DataFrame, payloads: list[tuple[dict, str]], cfg: dict, frozen: dict) -> list[dict]:
    """box: current-season player-game rows (appearances and DNPs) for games before tonight.
    payloads: [(event payload, retrieved_at)] from the decision snapshot."""
    c = frozen["constants"]
    m = cfg["m1"]
    app = box[(box.dnp == 0) & (box.minutes > 0) & box.pts.notna()].copy()
    app["date"] = pd.to_datetime(app["date"], utc=True)
    by_name = {}
    for pid, g in app.sort_values("date").groupby("player_id"):
        last = g.iloc[-1]
        by_name.setdefault(norm_name(last.player), []).append((pid, last.team))
    out = []
    for ev, retrieved in payloads:
        teams = {TEAM_ABBR.get(ev.get("home_team")), TEAM_ABBR.get(ev.get("away_team"))}
        tip = pd.Timestamp(ev["commence_time"])
        qs = []
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != m["market"]:
                    continue
                for o in mk.get("outcomes", []):
                    qs.append({"book": bk["key"], "player": o.get("description", ""), "side": o.get("name"),
                               "line": o.get("point"), "dec": o.get("price"), "last_update": mk.get("last_update")})
        groups = {}
        for q in qs:
            groups.setdefault((norm_name(q["player"]), q["line"]), []).append(q)
        rows = []
        for (pname, line), gq in groups.items():
            for venue in m["venues"]:
                vo = next((x for x in gq if x["book"] == venue and x["side"] == "Over"), None)
                vu = next((x for x in gq if x["book"] == venue and x["side"] == "Under"), None)
                if not (vo and vu):
                    continue
                rec = {"event_id": ev["id"], "commence_time": ev["commence_time"], "decision_at": retrieved,
                       "player": vo["player"], "line": line, "venue": venue, "over_dec": vo["dec"], "under_dec": vu["dec"],
                       "venue_last_update": vo["last_update"], "eligible": False, "reason": None}
                cand = [(pid, tm) for pid, tm in by_name.get(pname, []) if tm in teams]
                if len(cand) != 1:
                    rec["reason"] = "no_unique_player_match" if not cand else "ambiguous_player"
                    out.append(rec)
                    continue
                pid = cand[0][0]
                rec["player_id"], rec["team"] = str(pid), cand[0][1]
                hist = app[(app.player_id == pid) & (app.date < tip)]
                if len(hist) < m1.MIN_APPS:
                    rec["reason"] = f"prior_appearances_{len(hist)}_lt_{m1.MIN_APPS}"
                    out.append(rec)
                    continue
                pseudo = hist.iloc[[-1]].copy()
                pseudo["date"], pseudo["pts"], pseudo["minutes"] = tip, 0.0, 1.0
                pseudo["game_id"] = "tonight"
                f = m1.features(pd.concat([hist, pseudo]).sort_values("date").reset_index(drop=True))
                f = f[f.game_id == "tonight"]
                rows.append((rec, f, gq))
        for rec, f, gq in rows:
            P = pmf(f, c)
            _, med = m1.m1_logscore(f.assign(pts=0), c)
            over, push = p_over_push(P, np.array([float(rec["line"])]))
            rec["proj_median_minutes"] = float(med[0])
            rec["role"] = f.role.iloc[0]
            if med[0] < m["min_projected_minutes"]:
                rec["reason"] = "projected_minutes_lt_12"
                out.append(rec)
                continue
            p_act = float(over[0] / max(1 - push[0], 1e-9))
            mkt, used = consensus(gq, rec["venue"], m["consensus_books"], m["consensus_min_books"])
            rec.update({"p_push": float(push[0]), "m1_p_over": p_act,
                        "m1_1_p_over": float(apply_cal(np.array([p_act]), frozen["m1_1_isotonic"])[0]),
                        "market_p_over": mkt, "market_books": used})
            if mkt is None:
                rec["reason"] = f"consensus_books_{len(used)}_lt_{m['consensus_min_books']}"
            else:
                rec["eligible"], rec["reason"] = True, "ok"
            out.append(rec)
    return out


def write_and_push(data: str, day: str, recs: list[dict]) -> str:
    p = Path(data) / "m1" / "forecasts" / f"{day}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        for r in recs:
            f.write(json.dumps(r, default=float) + "\n")
    cmd = (f"cd {data} && git add m1/forecasts && (git commit -q -m 'M1/M1.1 forecasts {day}' || true) && "
           "for i in 1 2 3 4 5; do git push -q origin data && break; git pull -q --rebase origin data; done")
    subprocess.run(cmd, shell=True, check=False)
    return str(p)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze"])
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    freeze(a.data)
