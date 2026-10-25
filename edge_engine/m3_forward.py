"""M3 prospective paper log + H-U1 progress (config/prereg/m3-forward.yaml). Free: reads what the monitor captured.

    python -m edge_engine.m3_forward --data data

Every captured DraftKings player_points prop for 2026-27 is logged with an eligibility reason. Eligible props get the
frozen Baseline B probability; settled ones get outcomes. H-U1 is computed ONCE, at the first date on which the
minimums (>= 150 settled model-selected unders over >= 30 dates) hold; until then only counts are reported.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import nbinom, norm

from .injury_pdf import norm_name, parse
from .m3_data import header_time

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "m3_baseline_frozen.json"
ET = ZoneInfo("America/New_York")
SEASON = "2026-27"
EV_MIN, MIN_PICKS, MIN_DATES = 0.04, 150, 30
BOOT, SEED = 10000, 20261024


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def decision(tip):
    t = tip.astimezone(ET)
    return min(t.replace(hour=17, minute=30, second=0, microsecond=0), t - timedelta(minutes=90))


def reports(data: str):
    """Captured official PDFs this season: (printed time, path in raw store)."""
    out = []
    for f in sorted(glob.glob(str(Path(data) / "monitor" / "2026-*.jsonl"))):
        for line in open(f):
            r = json.loads(line)
            if r.get("kind") == "official_report" and r.get("ref"):
                out.append(r["ref"])
    return sorted(set(out))


class Reports:
    def __init__(self, data):
        self.data, self.cache = data, {}
        self.items = []
        for ref in reports(data):
            p = Path(data) / ref
            if not p.exists():
                continue
            with tempfile.NamedTemporaryFile(suffix=".pdf") as t:
                t.write(gzip.open(p).read())
                t.flush()
                ht = header_time(Path(t.name))
                if ht is not None:
                    self.items.append((ht, ref))
        self.items.sort()

    def out_set(self, cutoff, game_day):
        """Normalized names listed Out for games on game_day, from the latest report printed <= cutoff."""
        cand = [x for x in self.items if x[0] <= cutoff and x[0].date() == game_day]
        if not cand:
            return None, set()
        ht, ref = cand[-1]
        if ref not in self.cache:
            with tempfile.NamedTemporaryFile(suffix=".pdf") as t:
                t.write(gzip.open(Path(self.data) / ref).read())
                t.flush()
                res = parse(t.name)
            self.cache[ref] = res
        res = self.cache[ref]
        want = game_day.strftime("%m/%d/%Y")
        outs = {norm_name(r["player"]) for r in res["rows"] if r["status"] == "Out" and r["date"] == want}
        return ref, outs


KX_FEE_RATE, KX_MULT, KX_C = 0.07, 1.0, 10      # KXNBAPTS fee_type quadratic, fee_multiplier 1 (series API, 2026-10-24)


def kx_fee(price: float, contracts: int = KX_C) -> float:
    """Kalshi taker fee for one order: round up to the next cent of M x 0.07 x C x P x (1 - P)."""
    import math
    return math.ceil(round(KX_MULT * KX_FEE_RATE * contracts * price * (1 - price) * 100, 6)) / 100


class Kalshi:
    """KXNBAPTS market pages the monitor captured (free). Index: (player key, ET game date, floor_strike) ->
    list of (retrieved_at, market)."""
    def __init__(self, data):
        self.idx = {}
        man = Path(data) / "raw" / "manifest.jsonl"
        if not man.exists():
            return
        for line in open(man):
            if "series_ticker=KXNBAPTS&status=open" not in line:
                continue
            r = json.loads(line)
            if r.get("status") != 200 or not r.get("file"):
                continue
            fp = Path(data) / r["file"]
            if not fp.exists():
                continue
            at = ts(r["retrieved_at"])
            try:
                mk = json.load(gzip.open(fp)).get("markets", [])
            except Exception:  # noqa: BLE001
                continue
            for m in mk:
                sub = m.get("yes_sub_title") or ""
                if ":" not in sub or m.get("floor_strike") is None or not m.get("occurrence_datetime"):
                    continue
                name = sub.rsplit(":", 1)[0]
                day = ts(m["occurrence_datetime"]).astimezone(ET).date()
                self.idx.setdefault((norm_name(name), day, float(m["floor_strike"])), []).append((at, m))
        self.by_pd = {}
        for (k, d, st), v in self.idx.items():
            self.by_pd.setdefault((k, d), {})[(k, d, st)] = v

    def rungs(self, key, day, dec):
        """Thresholds listed for this player-game in any capture at or before the decision."""
        return sorted(st for (k, d, st), v in self.by_pd.get((key, day), {}).items() if any(at <= dec for at, _ in v))

    def quote(self, key, day, strike, dec):
        """Last captured quote at or before the decision; (age_min, market) or (None, reason)."""
        c = [x for x in self.idx.get((key, day, strike), []) if x[0] <= dec]
        if not c:
            return None, "kx_no_rung"
        at, m = max(c, key=lambda x: x[0])
        age = (dec - at).total_seconds() / 60
        if age > 30:
            return None, "kx_stale"
        yb, ya = float(m.get("yes_bid_dollars") or 0), float(m.get("yes_ask_dollars") or 0)
        if not (0 < yb < ya < 1):
            return None, "kx_not_two_sided"
        return age, m


def kx_arm(row: dict, kx: Kalshi, key: str, day, dec, p_over_at):
    """H-U1-KX fields for one eligible DK prop (amendment 2026-10-24b, before any qualifying observation): the rung is
    the listed Kalshi threshold (floor_strike) numerically nearest the DK main line, ties to the lower rung, chosen from
    the rungs listed at or before the decision and before any Kalshi price is read. The model prices that exact
    threshold: p_no = 1 - P(points > strike). A missing/stale/one-sided selected rung is excluded, never substituted."""
    L = float(row["line"])
    rungs = kx.rungs(key, day, dec)
    if not rungs:
        return {"kx_reason": "kx_no_ladder"}
    strike = min(rungs, key=lambda x: (abs(x - L), x))
    out = {"kx_strike": strike, "kx_strike_gap": strike - L}
    age, m = kx.quote(key, day, strike, dec)
    if age is None:
        out["kx_reason"] = m
        return out
    no_ask = round(1 - float(m["yes_bid_dollars"]), 4)
    depth = float(m.get("yes_bid_size_fp") or 0)          # top-of-book NO size = YES bid size
    fee = kx_fee(no_ask)
    cost = KX_C * no_ask + fee                            # 10-contract paper order at the NO ask, fee on the order
    p_no = 1 - p_over_at(strike)
    out.update(kx_eligible=True, kx_ticker=m.get("ticker"), kx_quote_age_min=round(age, 2), kx_no_ask=no_ask,
               kx_no_bid=round(1 - float(m["yes_ask_dollars"]), 4), kx_depth_no=depth, kx_fee_order=fee,
               kx_cost=cost, kx_p_no=p_no, kx_ev=p_no * KX_C / cost - 1)
    out["kx_pick"] = out["kx_ev"] >= EV_MIN
    return out


def build(data: str):
    fz = json.loads(FROZEN.read_text())
    C, P = fz["constants"], fz["priors"]
    bp = Path(data) / "m1" / "boxscores" / f"{SEASON}.csv.gz"
    B = pd.read_csv(bp)
    B = B[B.player.notna()].copy()
    B["tip"] = pd.to_datetime(B.date, utc=True)
    B["key"] = B.player.astype(str).map(norm_name)
    B["played"] = (B.dnp.fillna(0).astype(int) == 0) & (B.minutes.fillna(0) > 0)
    B["starter"] = B.starter.fillna(0).astype(int).astype(bool)
    B["d"] = B.tip.dt.tz_convert(ET).dt.strftime("%Y-%m-%d")
    pl = B[B.played].sort_values("tip")
    by_key = {k: g for k, g in pl.groupby("key")}
    box_day = {(k, d): g for (k, d), g in B.groupby(["key", "d"])}
    rep = Reports(data)
    kx = Kalshi(data)
    rows, seen = [], set()
    grid = np.arange(0, 49)
    for f in sorted(glob.glob(str(Path(data) / "m1" / "forecasts" / "2026-*.jsonl"))):
        for line in open(f):
            q = json.loads(line)
            if q.get("venue") != "draftkings":
                continue
            tip = ts(q["commence_time"])
            day = tip.astimezone(ET).date()
            key = norm_name(q["player"])
            uid = (key, q["event_id"])
            if uid in seen:
                continue
            seen.add(uid)
            dec = decision(tip)
            cap = ts(q["decision_at"])
            row = {"date": str(day), "event_id": q["event_id"], "player": q["player"], "line": q["line"],
                   "over_dec": q["over_dec"], "under_dec": q["under_dec"], "captured_at": q["decision_at"],
                   "venue_last_update": q.get("venue_last_update"), "decision": dec.isoformat()}
            reason = None
            if cap > dec:
                reason = "captured_after_decision"
            elif q.get("venue_last_update") and ts(q["venue_last_update"]) < dec - timedelta(minutes=30):
                reason = "stale_quote"
            hist = by_key.get(key)
            hist = hist[hist.tip < cap] if hist is not None else None
            if reason is None and (hist is None or len(hist) < 5):
                reason = f"prior_games_{0 if hist is None else len(hist)}_lt_5"
            m_hat = r_hat = None
            if reason is None:
                i = len(hist)
                w = 0.5 ** ((i - np.arange(i)) / C["h"])
                role = "starter" if hist.starter.values[-5:].sum() >= 3 else "bench"
                mins, pts = hist.minutes.values.astype(float), hist.pts.values.astype(float)
                m_hat = (w @ mins + C["n0_minutes"] * P["min"][role]) / (w.sum() + C["n0_minutes"])
                r_hat = (w @ pts + C["n0_rate"] * P["rate"][role] * P["mbar"]) / (w @ mins + C["n0_rate"] * P["mbar"])
                if m_hat < 12:
                    reason = "projected_minutes_lt_12"
                else:
                    ref, outs = rep.out_set(dec - timedelta(minutes=15), day)
                    row["report_ref"] = ref
                    if ref is None:
                        row["report_missing"] = True        # spec: everyone treated as not listed; flagged
                    if key in outs:
                        reason = "listed_out"
            if reason is None:
                s = C["sigma"][role]
                pm = norm.pdf(grid, m_hat, s)
                pm /= pm.sum()
                L = q["line"]
                k = C["k"]
                mu = np.maximum(r_hat * grid, 1e-6)
                p_over = float(pm @ nbinom.sf(np.floor(L), k, k / (k + mu)))
                p_push = float(pm @ nbinom.pmf(L, k, k / (k + mu))) if float(L).is_integer() else 0.0
                p_over_np = p_over / max(1 - p_push, 1e-9)
                row.update(eligible=True, m_hat=m_hat, r_hat=r_hat, p_over=p_over_np,
                           ev_under=(1 - p_over_np) * q["under_dec"] - 1, ev_over=p_over_np * q["over_dec"] - 1)
                row["model_under_pick"] = row["ev_under"] >= EV_MIN
                p_at = lambda S_, pm=pm, mu=mu, k=k: float(pm @ nbinom.sf(np.floor(S_), k, k / (k + mu)))  # noqa: E731
                row.update(kx_arm(row, kx, key, day, dec, p_at))
                act = box_day.get((key, str(day)))
                if act is not None and len(act):
                    a = act.iloc[0]
                    if not a.played:
                        row["settle"] = "void_dnp"
                    elif float(L).is_integer() and a.pts == L:
                        row["settle"] = "void_push"
                    else:
                        row.update(settle="settled", pts=float(a.pts), over=int(a.pts > L))
            else:
                row.update(eligible=False, reason=reason)
            rows.append(row)
    return pd.DataFrame(rows)


def h_u1(S: pd.DataFrame, pick_col="model_under_pick"):
    """Date-cluster bootstrap of ROI(model picks) - ROI(all eligible). Called only once minimums hold. Expects ret_u."""
    S = S.copy()
    S["model_under_pick"] = S[pick_col]
    dates = np.array(sorted(S.date.unique()))
    g = {d: x for d, x in S.groupby("date")}
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(BOOT):
        pick = rng.choice(dates, len(dates), replace=True)
        X = pd.concat([g[d] for d in pick])
        mu = X[X.model_under_pick]
        if len(mu):
            diffs.append(mu.ret_u.mean() - X.ret_u.mean())
    mu = S[S.model_under_pick]
    obs = mu.ret_u.mean() - S.ret_u.mean()
    lb = float(np.quantile(diffs, 0.05))
    best = mu.groupby("date").ret_u.sum().idxmax()
    ex = S[S.date != best]
    return {"roi_model_unders": float(mu.ret_u.mean()), "roi_blind_unders": float(S.ret_u.mean()), "diff": float(obs),
            "lb95_one_sided": lb, "pass": lb > 0,
            "diagnostics": {"players": int(mu.player.nunique()), "games": int(mu.event_id.nunique()), "dates": int(mu.date.nunique()),
                            "diff_excl_best_date": float(ex[ex.model_under_pick].ret_u.mean() - ex.ret_u.mean()),
                            "best_date": best}}


def run(data: str):
    D = build(data)
    for c in ("eligible", "settle", "model_under_pick", "reason", "date", "over", "under_dec", "kx_eligible", "kx_pick",
              "kx_cost", "kx_reason", "player", "event_id"):
        if c not in D:
            D[c] = None
    D["eligible"] = D.eligible.fillna(False).astype(bool)
    D["model_under_pick"] = D.model_under_pick.fillna(False).astype(bool)
    out = Path(data) / "m3" / "forward"
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "props.csv", index=False)
    S = D[D.eligible & (D.settle == "settled")].copy()
    S["ret_u"] = (1 - S.over) * S.under_dec - 1
    picks = S[S.model_under_pick]
    n, nd = int(len(picks)), int(picks.date.nunique()) if len(picks) else 0
    status = {"generated_at": datetime.now(ET).isoformat(), "captured_dk_props": int(len(D)),
              "eligible": int(D.eligible.sum()) if len(D) else 0,
              "exclusions": D[~D.eligible].reason.value_counts().to_dict(),
              "settled_eligible": int(len(S)), "model_under_picks_settled": n, "pick_dates": nd,
              "minimums": {"picks": MIN_PICKS, "dates": MIN_DATES}, "result": None}
    res_p = out / "h_u1_result.json"
    if res_p.exists():
        status["result"] = json.loads(res_p.read_text())          # evaluated once; never recomputed
    elif n >= MIN_PICKS and nd >= MIN_DATES:
        # the evaluation uses data through the FIRST date on which both minimums hold, even if this job ran later
        c = picks.groupby("date").size().sort_index()
        cum, first = 0, None
        for i, (d, k) in enumerate(c.items()):
            cum += k
            if cum >= MIN_PICKS and i + 1 >= MIN_DATES:
                first = d
                break
        S = S[S.date <= first]
        r = h_u1(S)
        r["evaluated_on"] = str(S.date.max())
        res_p.write_text(json.dumps(r, indent=1, default=float))
        status["result"] = r
    # ---- H-U1-KX (secondary replication; reviewer Oct 24). Kalshi DNP = "fair price" settlement -> excluded like a void
    for c in ("kx_eligible", "kx_pick"):
        if c not in D:
            D[c] = False
        D[c] = D[c].fillna(False).astype(bool)
    K = S[S.get("kx_eligible", False) == True].copy() if "kx_eligible" in S else S.iloc[0:0].copy()  # noqa: E712
    if len(K):
        K["kx_pick"] = K.kx_pick.fillna(False).astype(bool)
        K["ret_u"] = (1 - K.over) * KX_C / K.kx_cost - 1
    kp = K[K.kx_pick] if len(K) else K
    kn, knd = int(len(kp)), int(kp.date.nunique()) if len(kp) else 0
    status["kx"] = {"eligible_settled": int(len(K)), "picks_settled": kn, "pick_dates": knd,
                    "exclusions": D[D.eligible].get("kx_reason", pd.Series(dtype=str)).value_counts().to_dict(), "result": None}
    kres = out / "h_u1_kx_result.json"
    if kres.exists():
        status["kx"]["result"] = json.loads(kres.read_text())
    elif kn >= MIN_PICKS and knd >= MIN_DATES:
        c = kp.groupby("date").size().sort_index()
        cum, first = 0, None
        for i, (d, k) in enumerate(c.items()):
            cum += k
            if cum >= MIN_PICKS and i + 1 >= MIN_DATES:
                first = d
                break
        Kc = K[K.date <= first]
        r = h_u1(Kc, "kx_pick")
        kpk = Kc[Kc.kx_pick]
        r["track_b_liquidity_proxy"] = {"share_depth_lt_10": float((kpk.kx_depth_no < KX_C).mean()),
                                        "median_spread": float((kpk.kx_no_ask - kpk.kx_no_bid).median()),
                                        "roi_depth_ge_10_only": float(kpk[kpk.kx_depth_no >= KX_C].ret_u.mean()) if (kpk.kx_depth_no >= KX_C).any() else None,
                                        "overlap_with_dk_picks": float(kpk.model_under_pick.mean()),
                                        "note": "top-of-book only; not a full executable simulation"}
        r["evaluated_on"] = str(first)
        kres.write_text(json.dumps(r, indent=1, default=float))
        status["kx"]["result"] = r
    (out / "status.json").write_text(json.dumps(status, indent=1, default=str))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    print(json.dumps(run(a.data), indent=1, default=str))
