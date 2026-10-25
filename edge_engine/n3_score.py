"""N3 tranche 1 market scoring (gateA rules in config/prereg/n3-spec.yaml; futility look only).

    python -m edge_engine.n3_score --cache /home/claude/n3cache --data /tmp/edata

Contracts: Pinnacle player_receptions at the decision snapshot (kickoff - 90), exact player and line. Market = Pinnacle
multiplicative no-vig; integer lines conditional on no push. Model = frozen N3 (n3_frozen.json), 4,000 draws, seed fixed.
Blend: logit p = logit m + w (logit model - logit m); w = MAP with N(0, 0.5^2) on the summed log loss, walk-forward over
Weeks 10-14 (Week 10 burn-in; week t uses weeks < t). Futility: point estimate of (market LL - blend LL) on Weeks 11-14
<= 0 -> tranche 2 not bought. Paper DK picks (blend EV >= 4% AND model EV >= 4%, DK line == Pinnacle line) are a
diagnostic only: no close was pulled, so CLV is not available in this tranche.
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from . import n3

SEED = 20261025
DRAWS = 4000
STALE_MIN = 30


def norm(s: str) -> str:
    s = re.sub(r"\b(Jr|Sr|II|III|IV|V)\.?$", "", s.strip())
    return re.sub(r"[^a-z]", "", s.lower())


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def quotes(data: str):
    o = json.loads((Path(data) / "n3" / "tranche1_quotes.json").read_text())
    rows, log = [], defaultdict(int)
    for w in o["weeks"]:
        for g in w["games"]:
            if g["status"] != "ok":
                log["game_not_ok"] += 1
                continue
            p = json.load(gzip.open(Path(data) / g["ref"]))
            dec = ts(g["decision_at"])
            books = {}
            for b in p["data"].get("bookmakers", []):
                for m in b.get("markets", []):
                    if m["key"] != "player_receptions":
                        continue
                    if ts(m["last_update"]) < dec - timedelta(minutes=STALE_MIN):
                        log[f"stale_{b['key']}"] += 1
                        continue
                    lines = defaultdict(dict)
                    for oc in m["outcomes"]:
                        lines[(norm(oc["description"]), oc["point"])][oc["name"]] = oc["price"]
                    books[b["key"]] = lines
            if "pinnacle" not in books:
                log["no_pinnacle"] += 1
                continue
            for (pl, L), pr in books["pinnacle"].items():
                if "Over" not in pr or "Under" not in pr:
                    log["one_sided"] += 1
                    continue
                dk = books.get("draftkings", {}).get((pl, L), {})
                rows.append({"week": w["week"], "nflverse_game": g["game_id"], "key": pl, "line": L,
                             "pin_over": pr["Over"], "pin_under": pr["Under"],
                             "dk_over": dk.get("Over"), "dk_under": dk.get("Under")})
    return pd.DataFrame(rows), dict(log)


def model_rows(cache: str):
    fz = json.loads(n3.FROZEN.read_text())
    K, n0, pri = fz["K"], fz["n0"], fz["priors"]
    tg, pg, _, _ = n3.load(cache, [2024, 2025])
    tm, pl = n3.Features(tg, pg, n0, pri).run([2025])
    n3.volumes(tm, pl, K)
    qbo = set(map(tuple, tm[tm.qb_out][["game_id", "team"]].values))
    emp = {k: np.asarray(v, float) for k, v in K["rush_pool"].items()}
    rng = np.random.default_rng(SEED)
    out, seen = {}, set()
    for r in pl:
        if 10 <= r["week"] <= 14:
            seen.add((r["game_id"], norm(r["name"])))
        if not (10 <= r["week"] <= 14) or not n3.eligible(r, "receptions", K):
            continue
        x = n3.simulate(r, K, rng, DRAWS, (r["game_id"], r["team"]) in qbo, emp)["receptions"]
        out[(r["game_id"], norm(r["name"]))] = {"draws": x, "y": float(r["y_rec"]), "absence": bool(r["absent"]),
                                                "player": r["name"]}
    return out, seen


def lg(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def ll(p, y):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fit_w(m, q, y):
    f = lambda w: ll(sig(lg(m) + w * (lg(q) - lg(m))), y).sum() + w ** 2 / (2 * 0.5 ** 2)  # noqa: E731
    r = minimize_scalar(f, bounds=(-3, 3), method="bounded")
    return float(r.x)


def score(cache: str, data: str, out: str):
    Q, qlog = quotes(data)
    M, seen = model_rows(cache)
    rows, miss = [], {"ineligible_n3_universe": 0, "name_or_inactive": 0}
    for r in Q.itertuples():
        mr = M.get((r.nflverse_game, r.key))
        if mr is None:
            miss["ineligible_n3_universe" if (r.nflverse_game, r.key) in seen else "name_or_inactive"] += 1
            continue
        x, L, y = mr["draws"], r.line, mr["y"]
        push = float(np.mean(x == L))
        p_mod = float(np.mean(x > L)) / max(1 - push, 1e-9)
        io, iu = 1 / r.pin_over, 1 / r.pin_under
        p_mkt = io / (io + iu)
        if y == L:
            continue                                    # push: contract excluded
        rows.append({"week": r.week, "game": r.nflverse_game, "player": mr["player"], "line": L, "y": y,
                     "over": int(y > L), "p_mkt": p_mkt, "p_mod": p_mod, "absence": mr["absence"],
                     "dk_over": r.dk_over, "dk_under": r.dk_under})
    D = pd.DataFrame(rows)
    D["w"], D["p_blend"] = np.nan, np.nan
    for wk in sorted(D.week.unique())[1:]:
        tr = D[D.week < wk]
        w = fit_w(tr.p_mkt.values, tr.p_mod.values, tr.over.values)
        i = D.week == wk
        D.loc[i, "w"] = w
        D.loc[i, "p_blend"] = sig(lg(D.loc[i, "p_mkt"]) + w * (lg(D.loc[i, "p_mod"]) - lg(D.loc[i, "p_mkt"])))
    S = D[D.week > D.week.min()].copy()
    S["d"] = ll(S.p_mkt.values, S.over.values) - ll(S.p_blend.values, S.over.values)
    S["d_raw"] = ll(S.p_mkt.values, S.over.values) - ll(S.p_mod.values, S.over.values)
    mean, boots = n3.cluster_lb(S.d.values, S.game.values)
    # paper DK picks (diagnostic): same line as Pinnacle, blend EV >= 4% AND model EV >= 4%
    picks = []
    for r in S.itertuples():
        for side, d, pb, pm, win in (("over", r.dk_over, r.p_blend, r.p_mod, r.over),
                                     ("under", r.dk_under, 1 - r.p_blend, 1 - r.p_mod, 1 - r.over)):
            if d and not np.isnan(d) and pb * d - 1 >= 0.04 and pm * d - 1 >= 0.04:
                picks.append({"side": side, "ev_blend": pb * d - 1, "ret": win * d - 1})
    P = pd.DataFrame(picks)
    bins = pd.cut(S.p_mod, [0, .3, .4, .5, .6, .7, 1])
    res = {
        "spec": "config/prereg/n3-spec.yaml gateA (tranche 1 futility look)", "frozen_sha": __import__("hashlib").sha256(n3.FROZEN.read_bytes()).hexdigest(),
        "seed": SEED, "draws": DRAWS, "quote_log": qlog, "contracts_unmatched_to_model": miss,
        "contracts_all_weeks": int(len(D)), "scored_weeks_11_14": int(len(S)), "games": int(S.game.nunique()),
        "w_path": {int(k): float(v) for k, v in D.groupby("week").w.first().dropna().items()},
        "ll_market": float(ll(S.p_mkt.values, S.over.values).mean()),
        "ll_blend": float(ll(S.p_blend.values, S.over.values).mean()),
        "ll_model_raw": float(ll(S.p_mod.values, S.over.values).mean()),
        "gain_blend_mean": mean, "gain_blend_lb95_one_sided": float(np.quantile(boots, 0.05)),
        "gain_raw_mean": float(S.d_raw.mean()),
        "absence_subset": {"n": int(S.absence.sum()), "gain_blend": float(S[S.absence].d.mean()) if S.absence.any() else None},
        "futility": "pass (continue)" if mean > 0 else "stop: tranche 2 not bought",
        "calibration_model": S.groupby(bins, observed=True).agg(n=("over", "size"), p=("p_mod", "mean"), hit=("over", "mean")).round(3).reset_index().astype(str).to_dict("records"),
        "mean_abs_model_minus_market": float((S.p_mod - S.p_mkt).abs().mean()),
        "dk_paper_picks_diagnostic": {"n": int(len(P)), "roi": float(P.ret.mean()) if len(P) else None,
                                      "by_side": P.groupby("side").ret.agg(["count", "mean"]).round(3).to_dict() if len(P) else {},
                                      "note": "no close pulled; CLV unavailable; not a gate"},
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    D.to_csv(Path(out).with_suffix(".rows.csv"), index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/claude/n3cache")
    ap.add_argument("--data", default="/tmp/edata")
    ap.add_argument("--out", default="/tmp/edata/n3/tranche1_score.json")
    a = ap.parse_args()
    print(json.dumps(score(a.cache, a.data, a.out), indent=1, default=float))
