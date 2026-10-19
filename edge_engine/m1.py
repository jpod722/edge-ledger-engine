"""M1 Gate 0: does the NBA points model forecast the exact points total better than two simple baselines?

Free and historical (ESPN box scores). Pre-registered in config m1 (frozen 2026-10-19). Run once.

Information timing: every feature for a player-game uses only that player's earlier appearances
(strictly earlier dates). No injury, lineup or odds information is used historically.

Universe (scored rows): regular-season player-games where the player appeared (minutes > 0; props on
players who don't play are void), had >= 5 earlier appearances that season, and M1's projected median
minutes >= 12 (the NBA rule threshold).

Fitting: global constants (negative-binomial size, role minute spreads, rate prior) are fitted on
2023-24 only and frozen; 2024-25 and 2025-26 are scored.

Models
- B1 rolling10_negbin: mean = average points over the last 10 appearances; NB size fitted on 2023-24.
- B2 shrunk_rate_x_recent_minutes: (season points + k*role rate) / (season minutes + k) x mean minutes of
  the last 5 appearances; NB.
- M1: role state (starter / rotation / fringe) from the last 5 appearances; minutes distribution =
  normal around an exponentially weighted minutes mean, spread mixing the player's recent variance
  with the role's 2023-24 spread, truncated to [1, 48]; points-per-minute shrunk from recent to season
  to role rate, with its own uncertainty; points | minutes, rate ~ NB. The joint distribution is
  integrated exactly on a minutes x rate grid (no simulation noise) instead of 10,000 draws.

Gate 0 passes if M1's mean log score beats BOTH baselines with the 95% cluster-bootstrap (by game date)
lower bound of the paired difference > 0.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FIT_SEASON, EVAL_SEASONS = "2023-24", ["2024-25", "2025-26"]
MIN_APPS, MIN_PROJ_MIN = 5, 12.0
HL_MIN, HL_RATE, K_RATE_MIN = 3.0, 8.0, 150.0
M_GRID = np.arange(1, 49)
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(7)
GH_W = GH_W / GH_W.sum()


def load(data: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted((Path(data) / "m1" / "boxscores").glob("20??-??.csv.gz"))]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[(df.dnp == 0) & (df.minutes > 0) & df.pts.notna()].copy()
    df["gdate"] = df["date"].dt.tz_convert("America/New_York").dt.date
    return df.sort_values(["player_id", "date"]).reset_index(drop=True)


def ew(values: np.ndarray, hl: float) -> float:
    w = 0.5 ** (np.arange(len(values))[::-1] / hl)
    return float((w * values).sum() / w.sum())


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Per appearance, quantities from the same player's strictly earlier appearances this season."""
    out = []
    for (pid, season), g in df.groupby(["player_id", "season"], sort=False):
        g = g.sort_values("date")
        mins, pts, st, dates = g.minutes.values, g.pts.values, g.starter.values, g.date.values
        for i in range(len(g)):
            prior = dates < dates[i]          # same-day duplicates excluded too
            n = int(prior.sum())
            if n < MIN_APPS:
                continue
            m, p, s = mins[prior], pts[prior], st[prior]
            last5 = slice(max(0, n - 5), n)
            role = "starter" if s[last5].sum() >= 3 else ("rotation" if m[last5].mean() >= 18 else "fringe")
            out.append({"idx": g.index[i], "n_prior": n, "role": role,
                        "ew_min": ew(m[-10:], HL_MIN), "var_min": float(np.var(m[-10:], ddof=1)) if n > 1 else 25.0,
                        "mean_min5": float(m[last5].mean()), "mean_pts10": float(p[-10:].mean()),
                        "season_pts": float(p.sum()), "season_min": float(m.sum()),
                        "ew_rate": ew(p[-15:] / m[-15:], HL_RATE), "ew_min_w": float(m[-15:].sum())})
    f = pd.DataFrame(out).set_index("idx")
    return df.join(f, how="inner")


def nb_logpmf(k, mean, size):
    mean = np.maximum(mean, 1e-6)
    return stats.nbinom.logpmf(k, size, size / (size + mean))


def fit_size(y, mean):
    """Method of moments NB size: var = mean + mean^2 / size."""
    resid2 = ((y - mean) ** 2 - mean).sum()
    return float(max((mean ** 2).sum() / max(resid2, 1e-9), 1.0))


def fit_constants(f: pd.DataFrame) -> dict:
    t = f[f.season == FIT_SEASON]
    role_rate = (t.groupby("role").pts.sum() / t.groupby("role").minutes.sum()).to_dict()
    role_sd = t.assign(err=t.minutes - t.ew_min).groupby("role").err.std().to_dict()
    b2_mean = b2(t, role_rate)
    m1_rate, m1_rate_sd = rate_est(t, role_rate)
    resid = t.pts / t.minutes - m1_rate
    return {"role_rate": role_rate, "role_sd": role_sd,
            "size_b1": fit_size(t.pts.values, t.mean_pts10.values),
            "size_b2": fit_size(t.pts.values, b2_mean),
            "size_m1": fit_size(t.pts.values, m1_rate * t.minutes.values),   # conditional on minutes
            "rate_sd_scale": float(np.sqrt(max(np.var(resid) - np.mean(m1_rate / t.minutes.values), 1e-6)) /
                                   np.mean(m1_rate_sd))}


def b2(f, role_rate):
    rr = f.role.map(role_rate).values
    rate = (f.season_pts.values + K_RATE_MIN * rr) / (f.season_min.values + K_RATE_MIN)
    return rate * f.mean_min5.values


def rate_est(f, role_rate):
    rr = f.role.map(role_rate).values
    season_rate = (f.season_pts.values + K_RATE_MIN * rr) / (f.season_min.values + K_RATE_MIN)
    w = f.ew_min_w.values / (f.ew_min_w.values + K_RATE_MIN)
    rate = w * f.ew_rate.values + (1 - w) * season_rate
    sd = 1 / np.sqrt(f.season_min.values + K_RATE_MIN)          # shrinks as minutes accumulate
    return rate, sd


def m1_logscore(f: pd.DataFrame, c: dict) -> tuple[np.ndarray, np.ndarray]:
    rate, rsd = rate_est(f, c["role_rate"])
    rsd = rsd * c["rate_sd_scale"]
    role_sd = f.role.map(c["role_sd"]).values
    w = np.minimum(f.n_prior.values / 20, 0.5)
    sd = np.sqrt(w * f.var_min.values + (1 - w) * role_sd ** 2)
    mu = f.ew_min.values
    # minutes weights on the 1..48 grid (truncated normal)
    z = (M_GRID[None, :] - mu[:, None]) / sd[:, None]
    wm = stats.norm.pdf(z)
    wm = wm / wm.sum(1, keepdims=True)
    med = np.array([M_GRID[np.searchsorted(np.cumsum(r), 0.5)] for r in wm])
    k = f.pts.values
    logp = np.full(len(f), -np.inf)
    for x, wq in zip(GH_X, GH_W):
        r = np.maximum(rate + x * rsd, 0.01)
        lp = nb_logpmf(k[:, None], r[:, None] * M_GRID[None, :], c["size_m1"])
        logp = np.logaddexp(logp, np.log(wq) + np.log(np.maximum((wm * np.exp(lp)).sum(1), 1e-300)))
    return logp, med


def boot(diff: np.ndarray, clusters: np.ndarray, n=2000, seed=23) -> dict:
    by = pd.Series(diff).groupby(clusters).apply(np.array).values
    rng = random.Random(seed)
    means = sorted(float(np.concatenate([by[rng.randrange(len(by))] for _ in by]).mean()) for _ in range(n))
    return {"mean": round(float(diff.mean()), 5), "lo": round(means[int(0.025 * n)], 5),
            "hi": round(means[int(0.975 * n) - 1], 5), "clusters": len(by)}


def run(a, cfg) -> dict:
    df = load(a.data)
    f = features(df)
    c = fit_constants(f)
    e = f[f.season.isin(EVAL_SEASONS)].copy()
    lm1, med = m1_logscore(e, c)
    e["proj_med_min"] = med
    keep = e.proj_med_min.values >= MIN_PROJ_MIN
    e, lm1 = e[keep], lm1[keep]
    lb1 = nb_logpmf(e.pts.values, e.mean_pts10.values, c["size_b1"])
    lb2 = nb_logpmf(e.pts.values, b2(e, c["role_rate"]), c["size_b2"])
    floor = np.log(1e-6)
    lm1, lb1, lb2 = (np.maximum(x, floor) for x in (lm1, lb1, lb2))
    cl = e.gdate.astype(str).values
    vs_b1, vs_b2 = boot(lm1 - lb1, cl), boot(lm1 - lb2, cl)
    by_role = {r: {"n": int((e.role == r).sum()),
                   "m1_minus_b1": round(float((lm1 - lb1)[e.role.values == r].mean()), 5),
                   "m1_minus_b2": round(float((lm1 - lb2)[e.role.values == r].mean()), 5)} for r in ("starter", "rotation", "fringe")}
    out = {"track": "M1 Gate 0 (historical log score, exact points)", "eval_seasons": EVAL_SEASONS,
           "n_player_games": int(len(e)), "game_dates": int(len(set(cl))),
           "mean_log_score": {"m1": round(float(lm1.mean()), 5), "b1_rolling10_negbin": round(float(lb1.mean()), 5),
                              "b2_shrunk_rate_x_recent_minutes": round(float(lb2.mean()), 5)},
           "m1_minus_b1": vs_b1, "m1_minus_b2": vs_b2,
           "pass": bool(vs_b1["lo"] > 0 and vs_b2["lo"] > 0),
           "by_role_diagnostic": by_role,
           "constants_fit_on_2023_24": {k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()})
                                        for k, v in c.items()},
           "deviation_from_spec": "joint distribution integrated on a minutes x rate grid instead of 10,000 draws (same model, no simulation noise)"}
    p = Path(a.data) / "m1"
    p.mkdir(parents=True, exist_ok=True)
    (p / "gate0_summary.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))
    return out
