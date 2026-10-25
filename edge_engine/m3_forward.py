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


def h_u1(S: pd.DataFrame):
    """Date-cluster bootstrap of ROI(model unders) - ROI(all eligible unders). Called only once minimums hold."""
    S = S.copy()
    S["ret_u"] = (1 - S.over) * S.under_dec - 1
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
    for c in ("eligible", "settle", "model_under_pick", "reason", "date"):
        if c not in D:
            D[c] = None
    D["eligible"] = D.eligible.fillna(False).astype(bool)
    D["model_under_pick"] = D.model_under_pick.fillna(False).astype(bool)
    out = Path(data) / "m3" / "forward"
    out.mkdir(parents=True, exist_ok=True)
    D.to_csv(out / "props.csv", index=False)
    S = D[D.eligible & (D.settle == "settled")]
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
    (out / "status.json").write_text(json.dumps(status, indent=1, default=str))
    return status


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    a = ap.parse_args()
    print(json.dumps(run(a.data), indent=1, default=str))
