"""Research track M2: does an independent NFL team-strength model add information beyond the market line?

Pre-registered 2026-10-19 (config m2:), approved by the reviewer, zero API credits. Frozen before any
evaluation was run; run once.

Data: nflverse play-by-play and schedules (GitHub releases). The schedule's spread_line/total_line are
"the" recorded lines; nflverse does not document their timing, so they are labelled near-close.

Model (all hyperparameters fixed in config, never tuned):
- Per team-game offensive and defensive EPA/play on run/pass plays with 0.10 <= wp <= 0.90
  (nflverse non-Vegas wp, so no market input), plus plays per game.
- Rating before a game = exponentially weighted mean over that team's earlier games this season
  (half-life in games), with last season's final rating shrunk toward the league mean as a prior
  worth `prior_games` pseudo-games. Only games strictly before the game's date are used.
- Stage 1 (walk-forward, fit on earlier seasons only): margin ~ diff of net ratings; total ~ offensive +
  defensive ratings + pace. Gives model margin m and model total t.
- Stage 2 (walk-forward): residual vs line ~ intercept + gamma * (model - line), fitted on stage-1
  predictions that were themselves out-of-sample. Season S is scored only with fits on earlier seasons.
Baseline = line + walk-forward intercept. Blend = baseline + gamma * disagreement.
Gate: pooled out-of-sample MSE reduction (baseline - blend) > 0 with the 95% cluster-bootstrap
(by season-week) lower bound > 0, Holm across spreads and totals at one-sided alpha 0.025.
"""
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path

import numpy as np
import pandas as pd

NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"


def load(cache: str, seasons: list[int]):
    cache_p = Path(cache)
    cache_p.mkdir(parents=True, exist_ok=True)
    games_p = cache_p / "games.csv"
    if not games_p.exists():
        import requests
        games_p.write_bytes(requests.get(f"{NFLVERSE}/schedules/games.csv", timeout=120).content)
    games = pd.read_csv(games_p)
    pbps = []
    for y in seasons:
        p = cache_p / f"pbp_{y}.parquet"
        if not p.exists():
            import requests
            p.write_bytes(requests.get(f"{NFLVERSE}/pbp/play_by_play_{y}.parquet", timeout=300).content)
        pbps.append(pd.read_parquet(p, columns=["game_id", "season", "season_type", "week", "game_date", "posteam",
                                               "defteam", "play_type", "epa", "wp", "two_point_attempt"]))
    return games, pd.concat(pbps, ignore_index=True)


def team_games(pbp: pd.DataFrame, wp_lo: float, wp_hi: float) -> pd.DataFrame:
    p = pbp[pbp["play_type"].isin(["run", "pass"]) & (pbp["two_point_attempt"] != 1) & pbp["epa"].notna()]
    p = p[(p["wp"] >= wp_lo) & (p["wp"] <= wp_hi)]
    off = p.groupby(["game_id", "season", "game_date", "posteam"]).agg(off=("epa", "mean"), plays=("epa", "size")).reset_index()
    de = p.groupby(["game_id", "defteam"]).agg(deff=("epa", "mean")).reset_index()
    tg = off.merge(de, left_on=["game_id", "posteam"], right_on=["game_id", "defteam"]).rename(columns={"posteam": "team"})
    return tg[["game_id", "season", "game_date", "team", "off", "deff", "plays"]].sort_values(["game_date", "game_id"])


def ratings_before(tg: pd.DataFrame, half_life: float, prior_games: float, carry: float) -> dict:
    """(game_id, team) -> (off, deff, plays) using only that team's games on earlier dates."""
    lam = 0.5 ** (1 / half_life)
    out, final = {}, {}
    league = {}
    for season, sg in tg.groupby("season", sort=True):
        prev = final.get(season - 1, {})
        lm = league.get(season - 1)
        hist: dict = {}
        for date, dg in sg.groupby("game_date", sort=True):
            for _, r in dg.iterrows():
                h = hist.get(r.team, [])
                vals = []
                for k, col in enumerate(("off", "deff", "plays")):
                    if lm is not None and r.team in prev:
                        pv = lm[k] + carry * (prev[r.team][k] - lm[k])
                        w0 = prior_games
                    else:
                        pv, w0 = (lm[k] if lm is not None else 0.0), (prior_games if lm is not None else 0.0)
                    num, den = pv * w0, w0
                    for j, x in enumerate(reversed(h)):
                        wt = lam ** j
                        num += wt * x[k]
                        den += wt
                    vals.append(num / den if den > 0 else 0.0)
                out[(r.game_id, r.team)] = tuple(vals)
            for _, r in dg.iterrows():  # update only after every game on this date is rated
                hist.setdefault(r.team, []).append((r.off, r.deff, r.plays))
        final[season] = {t: tuple(np.mean([x[k] for x in v[-8:]]) for k in range(3)) for t, v in hist.items()}
        league[season] = tuple(sg[c].mean() for c in ("off", "deff", "plays"))
    return out


def features(games: pd.DataFrame, rat: dict) -> pd.DataFrame:
    g = games[(games["game_type"] == "REG") & games["result"].notna() & games["spread_line"].notna()
              & games["total_line"].notna()].copy()
    rows = []
    for _, r in g.iterrows():
        h, a = rat.get((r.game_id, r.home_team)), rat.get((r.game_id, r.away_team))
        if h is None or a is None:
            continue
        rows.append({"game_id": r.game_id, "season": r.season, "week": r.week, "result": r.result, "total": r.total,
                     "spread_line": r.spread_line, "total_line": r.total_line,
                     "diff": (h[0] - h[1]) - (a[0] - a[1]), "neutral": 1 if r.location == "Neutral" else 0,
                     "sum_off": h[0] + a[0], "sum_def": h[1] + a[1], "pace": h[2] + a[2]})
    return pd.DataFrame(rows)


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def walk_forward(f: pd.DataFrame, first_fit: int, eval_seasons: list[int]) -> pd.DataFrame:
    """Two nested walk-forwards. Stage 1: model margin/total for season s from fits on [first_fit, s-1].
    Stage 2: for evaluation season S, residual-vs-line regression fitted on those out-of-sample stage-1
    predictions for seasons [first_fit+1, S-1]. Only season S is scored with each fit."""
    X1 = lambda d: np.column_stack([np.ones(len(d)), d["diff"], d["neutral"]])
    X2 = lambda d: np.column_stack([np.ones(len(d)), d["sum_off"], d["sum_def"], d["pace"]])
    oos = []
    for s in sorted(f.season.unique()):
        if s <= first_fit:
            continue
        tr, te = f[(f.season >= first_fit) & (f.season < s)], f[f.season == s].copy()
        te["m"] = X1(te) @ ols(X1(tr), tr["result"].values)
        te["t"] = X2(te) @ ols(X2(tr), tr["total"].values)
        oos.append(te)
    o = pd.concat(oos, ignore_index=True)
    out = []
    for S in eval_seasons:
        tr, te = o[o.season < S], o[o.season == S].copy()
        for name, y, line, mod in (("spread", "result", "spread_line", "m"), ("total", "total", "total_line", "t")):
            res_tr, dis_tr = tr[y] - tr[line], tr[mod] - tr[line]
            g0 = res_tr.mean()
            g = ols(np.column_stack([np.ones(len(tr)), dis_tr]), res_tr.values)
            te[f"{name}_base"] = te[line] + g0
            te[f"{name}_blend"] = te[line] + g[0] + g[1] * (te[mod] - te[line])
            te[f"{name}_gamma"] = g[1]
            te[f"{name}_dis"] = te[mod] - te[line]
        out.append(te)
    return pd.concat(out, ignore_index=True)


def boot(df: pd.DataFrame, y: str, base: str, blend: str, n=2000, seed=19) -> dict:
    df = df.assign(cl=df.season.astype(str) + "-" + df.week.astype(str),
                   gain=(df[y] - df[base]) ** 2 - (df[y] - df[blend]) ** 2)
    by = [g["gain"].values for _, g in df.groupby("cl")]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = np.concatenate([by[rng.randrange(len(by))] for _ in by])
        means.append(float(s.mean()))
    means.sort()
    return {"n_games": len(df), "clusters": len(by), "mse_base": round(float(((df[y] - df[base]) ** 2).mean()), 4),
            "mse_blend": round(float(((df[y] - df[blend]) ** 2).mean()), 4),
            "gain_mean": round(float(df["gain"].mean()), 4), "lo": round(means[int(0.025 * n)], 4),
            "hi": round(means[int(0.975 * n) - 1], 4), "p_le_0": round(sum(m <= 0 for m in means) / n, 4)}


def run(a, cfg) -> dict:
    m = cfg["m2"]
    seasons = list(range(m["data_from_season"], m["eval_seasons"][-1] + 1))
    games, pbp = load(a.cache, seasons)
    tg = team_games(pbp[pbp["season_type"] == "REG"], m["wp_lo"], m["wp_hi"])
    rat = ratings_before(tg, m["half_life_games"], m["prior_games"], m["carryover"])
    f = features(games[games["season"].isin(seasons)], rat)
    ev = walk_forward(f, m["first_fit_season"], m["eval_seasons"])
    res = {"spread": boot(ev, "result", "spread_base", "spread_blend"),
           "total": boot(ev, "total", "total_base", "total_blend")}
    # Holm across the two markets at one-sided alpha
    order = sorted(res, key=lambda k: res[k]["p_le_0"])
    passed, alpha = {}, m["holm_alpha_one_sided"]
    for i, k in enumerate(order):
        ok = res[k]["p_le_0"] <= alpha / (len(order) - i) and all(passed.values())
        passed[k] = bool(ok and res[k]["lo"] > 0)
    diag = {}
    for name, y, line in (("spread", "result", "spread_line"), ("total", "total", "total_line")):
        diag[name] = {"gamma_by_season": {int(s): round(float(g[f"{name}_gamma"].iloc[0]), 4) for s, g in ev.groupby("season")},
                      "gain_by_season": {int(s): round(float((((g[y] - g[f"{name}_base"]) ** 2) - ((g[y] - g[f"{name}_blend"]) ** 2)).mean()), 4)
                                         for s, g in ev.groupby("season")}}
        big = ev[ev[f"{name}_dis"].abs() >= m["diag_disagreement_pts"]]
        side = np.sign(big[f"{name}_dis"])
        cover = np.sign(big[y] - big[line])
        dec = big[cover != 0]
        diag[name]["hit_rate_when_disagree"] = {"n": int(len(dec)), "hit": round(float((np.sign(dec[f"{name}_dis"]) == np.sign(dec[y] - dec[line])).mean()), 4) if len(dec) else None,
                                                "breakeven_at_-110": 0.5238}
    out = {"track": "M2 NFL model information check", "line_source": "nflverse schedules spread_line/total_line (timing undocumented; near-close)",
           "eval_seasons": m["eval_seasons"], "results": res, "pass": passed, "diagnostics_not_gates": diag}
    p = Path(a.data) / "m2"
    p.mkdir(parents=True, exist_ok=True)
    (p / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out
