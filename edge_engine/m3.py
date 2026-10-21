"""M3 Baseline B (spec: config/prereg/m3-spec.yaml). Fits on 2023-24 + 2024-25 only and freezes.

    python -m edge_engine.m3 fit-baseline --data /tmp/edata --cache /home/claude/m3cache
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from .injury_pdf import norm_name

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "m3_baseline_frozen.json"
TRAIN = ["2023-24", "2024-25"]
H_GRID = [3, 5, 8, 12]
N0_GRID = [1, 2, 4, 8]
FIX = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def load(data: str, cache: str, seasons):
    rep = pd.read_csv(Path(cache) / "report_rows.csv", usecols=["report", "date", "abbr", "player", "status"])
    sel = pd.read_csv(Path(cache) / "report_by_game.csv", usecols=["game_id", "name"])
    out = rep[rep.status == "Out"].copy()
    out["key"] = out.player.astype(str).map(norm_name)
    out_keys = set(zip(out.report, out.date, out.abbr, out.key))
    frames = []
    for s in seasons:
        b = pd.read_csv(Path(data) / "m1" / "boxscores" / f"{s}.csv.gz")
        b = b[b.player.notna()].copy()
        b["abbr"] = b.team.replace(FIX)
        b = b[b.abbr.str.len().between(2, 3) & ~b.abbr.isin(["CAN", "CHK", "KEN", "SHQ"])]
        b["tip"] = pd.to_datetime(b.date, utc=True)
        b["gdate"] = b.tip.dt.tz_convert("America/New_York").dt.strftime("%m/%d/%Y")
        b = b.merge(sel, on="game_id", how="left")
        b["listed_out"] = [(n, d, a, norm_name(str(p))) in out_keys for n, d, a, p in zip(b.name, b.gdate, b.abbr, b.player)]
        b["played"] = (b.dnp != True) & (b.minutes.fillna(0) > 0)  # noqa: E712
        frames.append(b)
    return pd.concat(frames, ignore_index=True)


def features(df: pd.DataFrame, h: float, n0m: float, n0r: float, prior: dict):
    """EW minutes and ratio-of-sums EW rate from prior played games this season, per player-game."""
    rows = []
    for (season, pid), g in df[df.played].sort_values("tip").groupby(["season", "player_id"]):
        mins, pts, st = g.minutes.values.astype(float), g.pts.values.astype(float), g.starter.values.astype(bool)
        for i in range(len(g)):
            if i < 5:
                continue
            age = i - np.arange(i)
            w = 0.5 ** (age / h)
            role = "starter" if st[max(0, i - 5):i].sum() >= 3 else "bench"
            m_hat = (w @ mins[:i] + n0m * prior["min"][role]) / (w.sum() + n0m)
            r_hat = (w @ pts[:i] + n0r * prior["rate"][role] * prior["mbar"]) / (w @ mins[:i] + n0r * prior["mbar"])
            rows.append((g.index[i], role, m_hat, r_hat))
    f = pd.DataFrame(rows, columns=["idx", "role", "m_hat", "r_hat"]).set_index("idx")
    return df.loc[f.index].join(f)


def negbin_nll(k, y, mu):
    mu = np.maximum(mu, 1e-6)
    return -np.sum(gammaln(y + k) - gammaln(k) - gammaln(y + 1) + k * np.log(k / (k + mu)) + y * np.log(mu / (k + mu)))


def fit_baseline(data: str, cache: str):
    df = load(data, cache, TRAIN)
    pl = df[df.played]
    # priors from the training seasons, by role (starter flag of the game itself is NOT a forecast input; only used here
    # to compute the role means that serve as shrinkage targets)
    prior = {"min": pl.groupby(pl.starter.map({True: "starter", False: "bench"})).minutes.mean().to_dict(),
             "rate": (pl.groupby(pl.starter.map({True: "starter", False: "bench"})).pts.sum()
                      / pl.groupby(pl.starter.map({True: "starter", False: "bench"})).minutes.sum()).to_dict(),
             "mbar": float(pl.minutes.mean())}
    log, best = {}, None
    for h in H_GRID:
        for n0 in N0_GRID:
            f = features(df, h, n0, 2, prior)
            f = f[~f.listed_out]
            e = float(((f.minutes - f.m_hat) ** 2).mean())
            log[f"minutes_h{h}_n0{n0}"] = e
            if best is None or e < best[0]:
                best = (e, h, n0)
    h, n0m = best[1], best[2]
    best_r = None
    for n0r in N0_GRID:
        f = features(df, h, n0m, n0r, prior)
        f = f[~f.listed_out]
        e = float(((f.pts - f.r_hat * f.minutes) ** 2).mean())
        log[f"rate_n0{n0r}"] = e
        if best_r is None or e < best_r[0]:
            best_r = (e, n0r, f)
    n0r, f = best_r[1], best_r[2]
    sigma = (f.minutes - f.m_hat).groupby(f.role).std().to_dict()
    k = float(minimize_scalar(lambda k: negbin_nll(k, f.pts.values, f.r_hat.values * f.minutes.values),
                              bounds=(0.5, 200), method="bounded").x)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]  # noqa: E731
    frozen = {
        "model": "M3 Baseline B", "spec": "config/prereg/m3-spec.yaml",
        "code_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "train_seasons": TRAIN, "data_cutoff": str(df.tip.max()),
        "inputs_sha256_16": {s: sha(Path(data) / "m1" / "boxscores" / f"{s}.csv.gz") for s in TRAIN}
        | {"report_rows.csv": sha(Path(cache) / "report_rows.csv"), "report_by_game.csv": sha(Path(cache) / "report_by_game.csv")},
        "sample": {"player_games_fit": int(len(f)), "players": int(f.player_id.nunique()),
                   "excluded_listed_out_played": int(features(df, h, n0m, n0r, prior).listed_out.sum())},
        "constants": {"h": h, "n0_minutes": n0m, "n0_rate": n0r, "sigma": sigma, "k": k},
        "priors": prior, "fit_log_mse": log,
        "fit_diagnostics_train_only": {"minutes_rmse": float(np.sqrt(((f.minutes - f.m_hat) ** 2).mean())),
                                       "points_given_minutes_rmse": float(np.sqrt(((f.pts - f.r_hat * f.minutes) ** 2).mean()))},
        "held_out_scored": False,
    }
    FROZEN.write_text(json.dumps(frozen, indent=1, default=float))
    return frozen


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit-baseline"])
    ap.add_argument("--data", default="/tmp/edata")
    ap.add_argument("--cache", default="/home/claude/m3cache")
    a = ap.parse_args()
    fz = fit_baseline(a.data, a.cache)
    print(json.dumps({k: fz[k] for k in ("constants", "sample", "fit_diagnostics_train_only", "data_cutoff")}, indent=1, default=float))
