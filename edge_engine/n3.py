"""N3 — NFL player-props model (spec: config/prereg/n3-spec.yaml). Gate 0 needs no Odds API credits.

    python -m edge_engine.n3 fit   --cache DIR       # fit constants on 2018-2022, write config/n3_frozen.json
    python -m edge_engine.n3 gate0 --cache DIR --out data/n3/gate0.json   # run once on 2023-2025
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parent.parent
FROZEN = ROOT / "config" / "n3_frozen.json"
NFLVERSE = "https://github.com/nflverse/nflverse-data/releases/download"
H = 3.0
M_PAIR = 4.0
POS_MAP = {"FB": "RB", "HB": "RB"}
SKILL = ("WR", "TE", "RB", "QB")
TRAIN = [2018, 2019, 2020, 2021, 2022]
TEST = [2023, 2024, 2025]


# ---------------------------------------------------------------- data
def fetch(cache: str, seasons):
    import requests
    c = Path(cache)
    c.mkdir(parents=True, exist_ok=True)
    files = {"players.csv": "players/players.csv", "games.csv": "schedules/games.csv"}
    for y in seasons:
        files[f"pbp_{y}.parquet"] = f"pbp/play_by_play_{y}.parquet"
        files[f"snaps_{y}.csv"] = f"snap_counts/snap_counts_{y}.csv"
        files[f"stats_{y}.csv"] = f"stats_player/stats_player_week_{y}.csv"
    for name, rel in files.items():
        if not (c / name).exists():
            r = requests.get(f"{NFLVERSE}/{rel}", timeout=300)
            r.raise_for_status()
            (c / name).write_bytes(r.content)


def load_season(cache: str, y: int, games: pd.DataFrame, pfr2gsis: dict, pos_of: dict):
    c = Path(cache)
    cols = ["game_id", "season_type", "week", "posteam", "home_team", "away_team", "qtr", "down", "qb_dropback",
            "qb_kneel", "qb_spike", "qb_scramble", "rush_attempt", "two_point_attempt", "score_differential",
            "passer_player_id", "rusher_player_id", "receiver_player_id", "air_yards", "complete_pass",
            "receiving_yards", "rushing_yards", "play_type"]
    p = pd.read_parquet(c / f"pbp_{y}.parquet", columns=cols)
    p = p[(p.season_type == "REG") & p.posteam.notna() & (p.two_point_attempt != 1)]
    p = p[p.play_type.isin(["pass", "run", "qb_kneel", "qb_spike"]) | (p.qb_dropback == 1)]
    db = (p.qb_dropback == 1) & (p.qb_spike != 1)
    dr = (p.rush_attempt == 1) & (p.qb_scramble != 1) & (p.qb_kneel != 1) & (p.qb_dropback != 1)
    neutral = (p.qtr <= 3) & (p.down <= 3) & (p.score_differential.abs() <= 7)
    p = p.assign(db=db.astype(int), dr=dr.astype(int), ndb=(db & neutral).astype(int),
                 nplay=((db | dr) & neutral).astype(int))
    tg = p.groupby(["game_id", "posteam"]).agg(week=("week", "first"), home=("home_team", "first"),
                                                 dropbacks=("db", "sum"), drushes=("dr", "sum"),
                                                 ndb=("ndb", "sum"), nplay=("nplay", "sum")).reset_index()
    tg = tg.rename(columns={"posteam": "team"})
    tg["plays"] = tg.dropbacks + tg.drushes
    g = games.set_index("game_id")
    tg["season"] = y
    tg["is_home"] = tg.team == tg.home
    sp = g.loc[tg.game_id, "spread_line"].values
    tot = g.loc[tg.game_id, "total_line"].values
    tg["spread"] = np.where(tg.is_home, sp, -sp)          # positive = this team favored
    tg["total"] = tot
    tg["itt"] = (tg.total + tg.spread) / 2
    tg["opp"] = np.where(tg.is_home, g.loc[tg.game_id, "away_team"].values, g.loc[tg.game_id, "home_team"].values)
    # dropbacks attributed to a QB (passer, or rusher on scrambles)
    who = p.passer_player_id.where(p.passer_player_id.notna(), p.rusher_player_id)
    qbdb = p[db].assign(pid=who[db]).groupby(["game_id", "pid"]).size().rename("qb_dropbacks")
    dcar = p[dr].groupby(["game_id", "rusher_player_id"]).size().rename("dcarries")
    scr = p[(p.qb_scramble == 1)].groupby(["game_id", "rusher_player_id"]).size().rename("scrambles")
    tgt = p[p.receiver_player_id.notna() & (p.play_type == "pass")]
    tgt_air = tgt.groupby(["game_id", "receiver_player_id"]).air_yards.sum().rename("air")
    # per-catch yards and per-carry yards (for distribution fitting)
    catches = tgt[tgt.complete_pass == 1][["game_id", "receiver_player_id", "receiving_yards"]]
    carries = p[(p.rush_attempt == 1) & (p.qb_kneel != 1)][["game_id", "rusher_player_id", "rushing_yards"]]
    # stats (outcomes)
    st = pd.read_csv(c / f"stats_{y}.csv", low_memory=False)
    st = st[st.season_type == "REG"][["player_id", "player_display_name", "position", "game_id", "team", "targets",
                                      "receptions", "receiving_yards", "carries", "rushing_yards"]]
    sn = pd.read_csv(c / f"snaps_{y}.csv")
    sn = sn[sn.game_type == "REG"].copy()
    sn["player_id"] = sn.pfr_player_id.map(pfr2gsis)
    sn = sn[sn.player_id.notna()][["game_id", "player_id", "team", "position", "offense_snaps", "offense_pct", "player"]]
    pg = sn.merge(st, on=["game_id", "player_id"], how="outer", suffixes=("_sn", ""))
    pg["team"] = pg.team.fillna(pg.team_sn)
    pg["name"] = pg.player_display_name.fillna(pg.player)
    pos = pg.position.fillna(pg.position_sn).map(lambda x: POS_MAP.get(x, x))
    pg["pos"] = pos.fillna(pg.player_id.map(pos_of))
    for k in ["targets", "receptions", "receiving_yards", "carries", "rushing_yards", "offense_snaps", "offense_pct"]:
        pg[k] = pg[k].fillna(0.0)
    pg = pg.merge(qbdb, left_on=["game_id", "player_id"], right_index=True, how="left")
    pg = pg.merge(dcar, left_on=["game_id", "player_id"], right_index=True, how="left")
    pg = pg.merge(scr, left_on=["game_id", "player_id"], right_index=True, how="left")
    pg = pg.merge(tgt_air, left_on=["game_id", "player_id"], right_index=True, how="left")
    for k in ["qb_dropbacks", "dcarries", "scrambles", "air"]:
        pg[k] = pg[k].fillna(0.0)
    pg["active"] = (pg.offense_snaps > 0) | (pg.targets + pg.carries + pg.qb_dropbacks > 0)
    pg = pg[pg.pos.isin(SKILL) & pg.team.notna()]
    pg = pg.merge(tg[["game_id", "team", "week"]], on=["game_id", "team"], how="inner")
    pg["season"] = y
    pg = pg[["season", "week", "game_id", "team", "player_id", "name", "pos", "active", "offense_pct", "targets",
             "receptions", "receiving_yards", "air", "dcarries", "scrambles", "carries", "rushing_yards", "qb_dropbacks"]]
    pos_by_id = pg.drop_duplicates("player_id").set_index("player_id").pos
    catches = catches.assign(pos=catches.receiver_player_id.map(pos_by_id))
    carries = carries.assign(pos=carries.rusher_player_id.map(pos_by_id))
    return tg, pg, catches, carries


def load(cache: str, seasons):
    fetch(cache, seasons)
    c = Path(cache)
    games = pd.read_csv(c / "games.csv")
    pl = pd.read_csv(c / "players.csv", usecols=["gsis_id", "pfr_id", "position"])
    pfr2gsis = dict(zip(pl.pfr_id, pl.gsis_id))
    pos_of = {k: POS_MAP.get(v, v) for k, v in zip(pl.gsis_id, pl.position)}
    T, P, C, R = [], [], [], []
    for y in seasons:
        tg, pg, ca, cr = load_season(cache, y, games, pfr2gsis, pos_of)
        T.append(tg); P.append(pg); C.append(ca.assign(season=y)); R.append(cr.assign(season=y))
    return pd.concat(T, ignore_index=True), pd.concat(P, ignore_index=True), pd.concat(C, ignore_index=True), pd.concat(R, ignore_index=True)


# ---------------------------------------------------------------- features
def ew_ratio(nums, dens, ages, n0, prior, dbar):
    """(sum w num + n0*prior*dbar) / (sum w den + n0*dbar): n0 counts pseudo-games of average size dbar."""
    if len(ages):
        w = 0.5 ** (np.asarray(ages, float) / H)
        a, b = float(np.dot(w, nums)), float(np.dot(w, dens))
    else:
        a = b = 0.0
    return (a + n0 * prior * dbar) / (b + n0 * dbar)


def ew_mean(vals, ages, n0, prior):
    return ew_ratio(vals, np.ones(len(vals)), ages, n0, prior, 1.0)


class Features:
    """One chronological pass. For every team-game: team features; for every active skill player: base volume
    rates and efficiency; absences with pairwise history; QB-out flag. Uses prior games only."""

    def __init__(self, tg: pd.DataFrame, pg: pd.DataFrame, n0: dict, pri: dict):
        self.tg, self.pg, self.n0, self.pri = tg, pg, n0, pri

    def run_team_only(self, seasons):
        n0, pri, rows = self.n0, self.pri, []
        for s in seasons:
            for team, t in self.tg[self.tg.season == s].groupby("team"):
                t = t.sort_values("week").reset_index(drop=True)
                for idx, row in t.iterrows():
                    prior = t.iloc[:idx]
                    ages = idx - np.arange(idx)
                    rows.append({"game_id": row.game_id, "team": team, "idx": idx, "itt": row.itt, "spread": row.spread,
                                 "plays": row.plays, "dropbacks": row.dropbacks,
                                 "plays_ew": ew_mean(prior.plays.values, ages, n0["team"], pri["plays"]),
                                 "npr_ew": ew_ratio(prior.ndb.values, prior.nplay.values, ages, n0["team"], pri["npr"], pri["nplay_bar"])})
        return pd.DataFrame(rows), None

    def run(self, seasons):
        n0, pri = self.n0, self.pri
        out_team, out_play, out_abs = [], [], []
        pg = self.pg
        for s in seasons:
            tgs = self.tg[self.tg.season == s].sort_values("week")
            pgs = pg[pg.season == s]
            last = pg[pg.season == s - 1]
            for team, tgt in tgs.groupby("team"):
                tgt = tgt.sort_values("week").reset_index(drop=True)
                tpg = pgs[pgs.team == team]
                tlast = last[last.team == team]
                by_game = {gid: d for gid, d in tpg.groupby("game_id")}
                for idx, row in tgt.iterrows():
                    prior = tgt.iloc[:idx]
                    ages = idx - np.arange(idx)
                    plays = ew_mean(prior.plays.values, ages, n0["team"], pri["plays"])
                    npr = ew_ratio(prior.ndb.values, prior.nplay.values, ages, n0["team"], pri["npr"], pri["nplay_bar"])
                    tf = {"game_id": row.game_id, "season": s, "week": row.week, "team": team, "idx": idx,
                          "plays_ew": plays, "npr_ew": npr, "itt": row.itt, "spread": row.spread, "total": row.total,
                          "plays": row.plays, "dropbacks": row.dropbacks, "drushes": row.drushes}
                    cur = by_game.get(row.game_id)
                    if cur is None:
                        continue
                    prior_ids = list(prior.game_id)
                    hist = tpg[tpg.game_id.isin(prior_ids)]
                    gidx = {g: i for i, g in enumerate(prior_ids)}
                    team_db = dict(zip(prior.game_id, prior.dropbacks))
                    team_dr = dict(zip(prior.game_id, prior.drushes))
                    team_tg = hist.groupby("game_id").targets.sum().to_dict()
                    team_ca = hist.groupby("game_id").dcarries.sum().to_dict()
                    # primary QB and 15%-rule absentees from the prior 4 games
                    last4 = prior_ids[-4:]
                    h4 = hist[hist.game_id.isin(last4)]
                    qb_out, primary = False, None
                    if len(last4):
                        qd = h4.groupby("player_id").qb_dropbacks.sum()
                        tdb = sum(team_db[g] for g in last4)
                        if len(qd) and tdb > 0 and qd.max() >= 0.7 * tdb:
                            primary = qd.idxmax()
                    active_ids = set(cur[cur.active].player_id)
                    if primary is not None and primary not in active_ids:
                        qb_out = True
                    snap4 = (h4.groupby("player_id").offense_pct.sum() / max(len(last4), 1)) if len(last4) else pd.Series(dtype=float)
                    absent = [a for a, v in snap4.items() if v >= 0.15 and a not in active_ids]
                    tf.update({"qb_out": qb_out, "n_absent": len(absent)})
                    out_team.append(tf)
                    # per-player features (active this game, plus absentees "as if active")
                    feats = {}
                    for pid in list(active_ids) + absent:
                        ph = hist[(hist.player_id == pid) & hist.active]
                        rowp = cur[cur.player_id == pid]
                        pos = (rowp.pos.iloc[0] if len(rowp) else (ph.pos.iloc[0] if len(ph) else None))
                        if pos is None:
                            continue
                        ag = np.array([idx - gidx[g] for g in ph.game_id], float)
                        den_t = np.array([team_db[g] for g in ph.game_id]) * ph.offense_pct.values
                        snap = ew_mean(ph.offense_pct.values, ag, n0["share"], pri["snap"][pos])
                        tprr = ew_ratio(ph.targets.values, den_t, ag, n0["share"], pri["tprr"][pos], pri["den_t_bar"])
                        cshare = ew_ratio(ph.dcarries.values, np.array([team_dr[g] for g in ph.game_id], float), ag,
                                          n0["share"], pri["cshare"][pos], pri["drush_bar"])
                        srate = ew_ratio(ph.scrambles.values, np.array([team_db[g] for g in ph.game_id], float), ag,
                                         n0["share"], pri["srate"][pos], pri["db_bar"])
                        adot = ew_ratio(ph.air.values, ph.targets.values, ag, n0["eff"], pri["adot"][pos], pri["tgt_bar"])
                        cprior = 1 / (1 + math.exp(-(pri["alpha"][pos] + pri["beta"] * adot)))
                        crate = ew_ratio(ph.receptions.values, ph.targets.values, ag, n0["eff"], cprior, pri["tgt_bar"])
                        yprior = pri["g0"] + pri["g1"] * adot
                        ypc = ew_ratio(ph.receiving_yards.values, ph.receptions.values, ag, n0["eff"], yprior, pri["rec_bar"])
                        rypc = ew_ratio(ph.rushing_yards.values, ph.carries.values, ag, n0["eff"], pri["rypc"][pos], pri["car_bar"])
                        n_prior_games = len(ph)
                        # target / carry shares per prior game, for pairwise history
                        feats[pid] = {"pos": pos, "snap": snap, "tprr": tprr, "cshare": cshare, "srate": srate,
                                      "adot": adot, "crate": crate, "ypc": ypc, "rypc": rypc, "n_prior": n_prior_games,
                                      "ag": ag, "gids": list(ph.game_id)}
                    # pairwise history for each absentee (this and last season)
                    abs_rows = []
                    for a in absent:
                        if a not in feats:
                            continue
                        span = pd.concat([tlast, hist])
                        a_games = set(span[(span.player_id == a) & span.active].game_id)
                        if not a_games:
                            continue
                        allg = span.game_id.unique()
                        without = [g for g in allg if g not in a_games]
                        shares = {}
                        for kind, col, tot in (("t", "targets", None), ("c", "dcarries", None)):
                            tot_by = span.groupby("game_id")[col].sum()
                            sh = span[span.active].assign(share=lambda d: d[col] / d.game_id.map(tot_by).replace(0, np.nan))
                            w_ = sh[sh.game_id.isin(without)].groupby("player_id").share.agg(["mean", "count"])
                            wi_ = sh[sh.game_id.isin(a_games)].groupby("player_id").share.mean()
                            shares[kind] = (w_, wi_)
                        abs_rows.append((a, shares))
                    for pid in active_ids:
                        if pid not in feats:
                            continue
                        f = feats[pid]
                        r = cur[cur.player_id == pid].iloc[0]
                        rec = {"game_id": row.game_id, "season": s, "week": row.week, "team": team, "player_id": pid,
                               "name": r["name"], "pos": f["pos"], **{k: f[k] for k in ("snap", "tprr", "cshare", "srate",
                               "adot", "crate", "ypc", "rypc", "n_prior")},
                               "y_rec": r.receptions, "y_recyds": r.receiving_yards, "y_rushyds": r.rushing_yards,
                               "y_targets": r.targets, "y_dcarries": r.dcarries, "y_scr": r.scrambles,
                               "is_primary_qb": pid == primary}
                        # season-to-date history for B1
                        ph = hist[(hist.player_id == pid) & hist.active]
                        rec["hist_rec"] = list(ph.receptions)
                        rec["hist_recyds"] = list(ph.receiving_yards)
                        rec["hist_rushyds"] = list(ph.rushing_yards)
                        # freshness: share of EW weight from games where each absentee was active
                        comp = []
                        for a, shares in abs_rows:
                            fa = feats[a]
                            a_act = set(fa["gids"])
                            w = 0.5 ** (f["ag"] / H)
                            fresh = float(w[[g in a_act for g in f["gids"]]].sum() / w.sum()) if len(w) and w.sum() > 0 else 1.0
                            pt = shares["t"][0]
                            pc = shares["c"][0]
                            dt = (pt["mean"].get(pid, np.nan) - shares["t"][1].get(pid, np.nan)) if pid in pt.index else np.nan
                            dc = (pc["mean"].get(pid, np.nan) - shares["c"][1].get(pid, np.nan)) if pid in pc.index else np.nan
                            nt = float(pt["count"].get(pid, 0)) if pid in pt.index else 0.0
                            comp.append({"a": a, "pos_a": fa["pos"], "a_snap": fa["snap"], "a_tprr": fa["tprr"],
                                         "a_cshare": fa["cshare"], "fresh": fresh, "dt": dt, "dc": dc, "n_pair": nt})
                        rec["absent"] = comp
                        out_play.append(rec)
        return pd.DataFrame(out_team), out_play


# ---------------------------------------------------------------- volume model with absence / QB out
def volumes(team: pd.DataFrame, plays: list, K: dict):
    """Mean targets and carries per player-game, given frozen constants K."""
    tm = team.set_index(["game_id", "team"])
    P = tm.plays_ew + K["c_T"] * (tm.itt - 22.5)
    pi = (K["a"] + K["b"] * tm.npr_ew + K["k"] * tm.spread).clip(0.3, 0.8)
    D = (P * pi) * np.where(tm.qb_out, K["h_D"], 1.0)
    R = P * (1 - pi)
    D, R = D.to_dict(), R.to_dict()
    by_team = defaultdict(list)
    for r in plays:
        by_team[(r["game_id"], r["team"])].append(r)
    for key, rows in by_team.items():
        d, rr = D[key], R[key]
        for r in rows:
            r["lam"] = d * r["snap"] * r["tprr"]
            r["car"] = rr * r["cshare"] + (d * r["srate"] if r["pos"] == "QB" else 0.0)
            r["D"], r["Rr"] = d, rr
        # absences
        absents = {}
        for r in rows:
            for c in r["absent"]:
                absents[c["a"]] = c
        for a, c in absents.items():
            for kind, lamkey, theta, phi, dkey in (("t", "lam", K["theta_t"], K["phi_t"], "dt"),
                                                   ("c", "car", K["theta_c"], K["phi_c"], "dc")):
                va = (d * c["a_snap"] * c["a_tprr"]) if kind == "t" else (rr * c["a_cshare"])
                V = (1 - phi) * va
                cand = [r for r in rows if r["pos"] in (("WR", "TE", "RB") if kind == "t" else ("RB", "QB", "WR"))]
                if not cand or V <= 0:
                    continue
                vac = np.array([r[lamkey] * (theta if r["pos"] == c["pos_a"] else 1.0) for r in cand])
                vac = vac / vac.sum() if vac.sum() > 0 else np.full(len(cand), 1 / len(cand))
                comps = [next(x for x in r["absent"] if x["a"] == a) for r in cand]
                dpos = np.array([max(x[dkey], 0.0) if not np.isnan(x[dkey]) else 0.0 for x in comps])
                npair = np.array([x["n_pair"] for x in comps])
                pair = dpos / dpos.sum() if dpos.sum() > 0 else np.zeros(len(cand))
                npair = np.where(dpos.sum() > 0, npair, 0.0)
                om = (npair * pair + M_PAIR * vac) / (npair + M_PAIR)
                om = om / om.sum()
                for r, o, x in zip(cand, om, comps):
                    r[lamkey] += V * o * x["fresh"]
    return plays


# ---------------------------------------------------------------- simulation and scoring
def simulate(r: dict, K: dict, rng, n: int, qb_out: bool, emp: dict):
    pos = r["pos"]
    out = {}
    hC = K["h_C"] if qb_out else 1.0
    hY = K["h_Y"] if qb_out else 1.0
    if pos in ("WR", "TE", "RB"):
        lam = max(r["lam"], 1e-6)
        size = K["r_t"][pos]
        tg = rng.negative_binomial(size, size / (size + lam), n)
        cp = min(max(r["crate"] * hC, 0.01), 0.99)
        rec = rng.binomial(tg, cp)
        mu = max(r["ypc"] * hY + 5.0, 1.0)
        kap = K["kappa"][pos]
        # sum of `rec` gamma(kap, mu/kap) draws = gamma(rec*kap, mu/kap)
        yds = np.where(rec > 0, rng.gamma(np.maximum(rec, 1) * kap, mu / kap), 0.0) - 5.0 * rec
        out["receptions"], out["receiving_yards"] = rec, yds
    if pos in ("RB", "QB"):
        c = max(r["car"], 1e-6)
        size = K["r_c"][pos]
        car = rng.negative_binomial(size, size / (size + c), n)
        pool = emp[pos]
        shift = r["rypc"] - K["rypc_mean"][pos]
        tot = np.zeros(n)
        mx = int(car.max()) if len(car) else 0
        if mx > 0:
            draws = rng.choice(pool, size=(n, mx)) + shift
            mask = np.arange(mx)[None, :] < car[:, None]
            tot = (draws * mask).sum(1)
        out["rushing_yards"] = tot
    return out


def crps(samples: np.ndarray, y: float, rng) -> float:
    s = np.asarray(samples, float)
    a = np.mean(np.abs(s - y))
    b = np.mean(np.abs(s - rng.permutation(s)))
    return float(a - 0.5 * b)


def b1_samples(hist, kind, cv, rng, n):
    h = np.asarray(hist, float)
    m = h.mean()
    if kind == "receptions":
        return rng.poisson(max(m, 1e-6), n)
    k = len(h)
    s2 = h.var(ddof=1) if k > 1 else 0.0
    sd = math.sqrt((k * s2 + 4 * (cv * max(m, 1.0)) ** 2) / (k + 4))
    return rng.normal(m, max(sd, 1.0), n)


def eligible(r, stat, K):
    if r["n_prior"] < 3:
        return False
    if stat in ("receptions", "receiving_yards"):
        return r["pos"] in ("WR", "TE", "RB") and r["lam"] >= 3.0
    if stat == "rushing_yards_RB":
        return r["pos"] == "RB" and r["car"] >= 6.0
    if stat == "rushing_yards_QB":
        return r["pos"] == "QB" and r["is_primary_qb"] and r["car"] >= 2.0
    return False


def cluster_lb(vals, clusters, n=2000, seed=31, alpha=0.05):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    u, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals)
    cnt = np.bincount(inv)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(u), size=(n, len(u)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)
    return float(vals.mean()), means


def holm(pvals: dict, alpha=0.05):
    order = sorted(pvals, key=pvals.get)
    m, res, stop = len(order), {}, False
    for i, k in enumerate(order):
        thr = alpha / (m - i)
        ok = (not stop) and pvals[k] <= thr
        stop = stop or not ok
        res[k] = {"p": pvals[k], "threshold": thr, "reject": ok}
    return res


# ---------------------------------------------------------------- fitting on 2018-2022
def priors(tg, pg, catches, carries):
    a = pg[pg.active]
    rc = a[a.pos.isin(["WR", "TE", "RB"])]
    tg_i = tg.set_index(["game_id", "team"])
    db = a.set_index(["game_id", "team"]).index.map(tg_i.dropbacks)
    dr = a.set_index(["game_id", "team"]).index.map(tg_i.drushes)
    a = a.assign(db=np.asarray(db, float), dr=np.asarray(dr, float))
    pri = {"plays": float(tg.plays.mean()), "npr": float(tg.ndb.sum() / tg.nplay.sum()),
           "nplay_bar": float(tg.nplay.mean()), "db_bar": float(tg.dropbacks.mean()), "drush_bar": float(tg.drushes.mean())}
    pri["snap"] = a.groupby("pos").offense_pct.mean().to_dict()
    pri["tprr"] = (a.groupby("pos").targets.sum() / (a.db * a.offense_pct).groupby(a.pos).sum()).to_dict()
    pri["den_t_bar"] = float((a.db * a.offense_pct)[a.pos.isin(["WR", "TE", "RB"])].mean())
    pri["cshare"] = (a.groupby("pos").dcarries.sum() / a.groupby("pos").dr.sum()).to_dict()
    pri["srate"] = (a.groupby("pos").scrambles.sum() / a.groupby("pos").db.sum()).to_dict()
    pri["adot"] = (a.groupby("pos").air.sum() / a.groupby("pos").targets.sum().replace(0, np.nan)).fillna(0).to_dict()
    pri["tgt_bar"] = float(rc.targets.mean())
    pri["rec_bar"] = float(rc.receptions.mean())
    pri["car_bar"] = float(a[a.pos.isin(["RB", "QB"])].carries.mean())
    pri["rypc"] = (a.groupby("pos").rushing_yards.sum() / a.groupby("pos").carries.sum().replace(0, np.nan)).fillna(0).to_dict()
    # catch prob ~ logistic(alpha_pos + beta * aDOT), player-game aggregated, receivers with targets
    d = rc[rc.targets > 0]
    adot = (d.air / d.targets).clip(-5, 40).values
    posi = d.pos.map({"WR": 0, "TE": 1, "RB": 2}).values
    y, n = d.receptions.values, d.targets.values

    def nll(th):
        z = th[posi] + th[3] * adot
        pr = 1 / (1 + np.exp(-z))
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(pr) + (n - y) * np.log(1 - pr))
    th = minimize(nll, np.array([0.5, 0.8, 1.5, -0.05]), method="L-BFGS-B").x
    pri["alpha"] = {"WR": float(th[0]), "TE": float(th[1]), "RB": float(th[2]), "QB": float(th[0])}
    pri["beta"] = float(th[3])
    d2 = d[d.receptions > 0]
    x = (d2.air / d2.targets).clip(-5, 40).values
    yy = (d2.receiving_yards / d2.receptions).values
    w = d2.receptions.values
    X = np.c_[np.ones(len(x)), x]
    g = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * yy))
    pri["g0"], pri["g1"] = float(g[0]), float(g[1])
    return pri


def fit(cache: str):
    tg, pg, catches, carries = load(cache, [2017] + TRAIN)
    tgT, pgT = tg[tg.season.isin(TRAIN)], pg[pg.season.isin(TRAIN)]
    pri = priors(tgT, pgT, catches[catches.season.isin(TRAIN)], carries[carries.season.isin(TRAIN)])
    log = {}
    # 1) team n0 (plays) and team constants
    best = None
    for n0t in (1, 2, 4, 8):
        team, _ = Features(tg, pg, {"team": n0t, "share": 2, "eff": 2}, pri).run_team_only(TRAIN)
        e = float(((team.plays - team.plays_ew) ** 2)[team.idx >= 3].mean())
        log[f"team_n0_{n0t}"] = e
        if best is None or e < best[0]:
            best = (e, n0t, team)
    n0 = {"team": best[1]}
    team = best[2][best[2].idx >= 3]
    X = (team.itt - 22.5).values
    c_T = float(np.dot(X, team.plays - team.plays_ew) / np.dot(X, X))
    pr = team.dropbacks / team.plays
    A = np.c_[np.ones(len(team)), team.npr_ew, team.spread]
    a_, b_, k_ = np.linalg.lstsq(A, pr.values, rcond=None)[0]
    K = {"c_T": c_T, "a": float(a_), "b": float(b_), "k": float(k_), "h_D": 1.0, "h_C": 1.0, "h_Y": 1.0,
         "phi_t": 0.0, "phi_c": 0.0, "theta_t": 1.0, "theta_c": 1.0}
    # 2) share n0 (targets SSE, no-absence games), 3) efficiency n0 (yards given volume)
    import pickle
    ck = Path(cache) / "n3_fit_runs"
    ck.mkdir(exist_ok=True)

    def run(n0s, n0e):
        f = ck / f"run_{n0['team']}_{n0s}_{n0e}.pkl"
        if f.exists():
            return pickle.loads(f.read_bytes())
        tm, pl = Features(tg, pg, {"team": n0["team"], "share": n0s, "eff": n0e}, pri).run(TRAIN)
        f.write_bytes(pickle.dumps((tm, pl)))
        return tm, pl
    cache_runs = {}
    best = None
    for n0s in (1, 2, 4, 8):
        tm, pl = run(n0s, 2)
        cache_runs[(n0s, 2)] = (tm, pl)
        volumes(tm, pl, K)
        rows = [r for r in pl if r["n_prior"] >= 3 and not r["absent"] and r["pos"] in ("WR", "TE", "RB")]
        e = float(np.mean([(r["y_targets"] - r["lam"]) ** 2 for r in rows]))
        rows_c = [r for r in pl if r["n_prior"] >= 3 and not r["absent"] and r["pos"] in ("RB", "QB")]
        e += float(np.mean([(r["y_dcarries"] + r["y_scr"] - r["car"]) ** 2 for r in rows_c]))
        log[f"share_n0_{n0s}"] = e
        if best is None or e < best[0]:
            best = (e, n0s)
    n0["share"] = best[1]
    best = None
    for n0e in (1, 2, 4, 8):
        tm, pl = cache_runs.get((n0["share"], n0e)) or run(n0["share"], n0e)
        cache_runs[(n0["share"], n0e)] = (tm, pl)
        rows = [r for r in pl if r["n_prior"] >= 3 and r["pos"] in ("WR", "TE", "RB") and r["y_targets"] > 0]
        e = float(np.mean([(r["y_recyds"] - r["y_targets"] * r["crate"] * r["ypc"]) ** 2 for r in rows]))
        rows_c = [r for r in pl if r["n_prior"] >= 3 and r["pos"] in ("RB", "QB")]
        e += float(np.mean([(r["y_rushyds"] - (r["y_dcarries"] + r["y_scr"]) * r["rypc"]) ** 2 for r in rows_c]))
        log[f"eff_n0_{n0e}"] = e
        if best is None or e < best[0]:
            best = (e, n0e)
    n0["eff"] = best[1]
    tm, pl = cache_runs[(n0["share"], n0["eff"])]
    # 4) absence: phi, theta (targets and carries), on games with absences, QB-out games excluded
    qbo = set(map(tuple, tm[tm.qb_out][["game_id", "team"]].values))
    for kind, ycol, keyp, keyt in (("t", "y_targets", "phi_t", "theta_t"), ("c", None, "phi_c", "theta_c")):
        best = None
        for phi in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            for theta in (1.0, 1.5, 2.0, 3.0, 4.0):
                K2 = {**K, keyp: phi, keyt: theta}
                plc = [dict(r) for r in pl]
                volumes(tm, plc, K2)
                if kind == "t":
                    rows = [r for r in plc if r["absent"] and r["n_prior"] >= 3 and r["pos"] in ("WR", "TE", "RB")
                            and (r["game_id"], r["team"]) not in qbo]
                    e = float(np.mean([(r["y_targets"] - r["lam"]) ** 2 for r in rows]))
                else:
                    rows = [r for r in plc if r["absent"] and r["n_prior"] >= 3 and r["pos"] in ("RB", "QB")
                            and (r["game_id"], r["team"]) not in qbo]
                    e = float(np.mean([(r["y_dcarries"] + r["y_scr"] - r["car"]) ** 2 for r in rows]))
                if best is None or e < best[0]:
                    best = (e, phi, theta)
        K[keyp], K[keyt] = best[1], best[2]
        log[f"absence_{kind}"] = {"sse": best[0], "phi": best[1], "theta": best[2]}
    # 5) QB-out haircuts
    plc = [dict(r) for r in pl]
    volumes(tm, plc, K)
    tq = tm[tm.qb_out & (tm.idx >= 3)]
    P = tq.plays_ew + K["c_T"] * (tq.itt - 22.5)
    pi = (K["a"] + K["b"] * tq.npr_ew + K["k"] * tq.spread).clip(0.3, 0.8)
    K["h_D"] = float(tq.dropbacks.sum() / (P * pi).sum())
    rq = [r for r in plc if (r["game_id"], r["team"]) in qbo and r["pos"] in ("WR", "TE", "RB") and r["n_prior"] >= 3]
    K["h_C"] = float(sum(r["y_rec"] for r in rq) / sum(r["y_targets"] * r["crate"] for r in rq))
    K["h_Y"] = float(sum(r["y_recyds"] for r in rq) / sum(r["y_rec"] * r["ypc"] for r in rq if r["y_rec"] > 0))
    log["qb_out_games"] = int(len(tq))
    # 6) dispersion (NegBin size) by position, gamma shape by position
    plc = [dict(r) for r in pl]
    volumes(tm, plc, K)
    K["r_t"], K["r_c"] = {}, {}
    for pos in ("WR", "TE", "RB"):
        rows = [r for r in plc if r["pos"] == pos and r["n_prior"] >= 3 and r["lam"] >= 1]
        lam = np.array([r["lam"] for r in rows]); y = np.array([r["y_targets"] for r in rows])
        ex = ((y - lam) ** 2 - lam).sum()
        K["r_t"][pos] = float(np.clip((lam ** 2).sum() / ex, 1.0, 200.0)) if ex > 0 else 200.0
    for pos in ("RB", "QB"):
        rows = [r for r in plc if r["pos"] == pos and r["n_prior"] >= 3 and r["car"] >= 1]
        lam = np.array([r["car"] for r in rows]); y = np.array([r["y_dcarries"] + r["y_scr"] for r in rows])
        ex = ((y - lam) ** 2 - lam).sum()
        K["r_c"][pos] = float(np.clip((lam ** 2).sum() / ex, 1.0, 200.0)) if ex > 0 else 200.0
    ca = catches[catches.season.isin(TRAIN)].copy()
    ca["ps"] = ca.receiver_player_id.astype(str) + "_" + ca.season.astype(str)
    mu = ca.groupby("ps").receiving_yards.transform("mean")
    ca = ca[ca.groupby("ps").receiving_yards.transform("size") >= 10]
    z = (ca.receiving_yards + 5) / (mu.loc[ca.index] + 5)
    K["kappa"] = {pos: float(1 / z[ca.pos == pos].var()) for pos in ("WR", "TE", "RB")}
    cr = carries[carries.season.isin(TRAIN) & carries.pos.isin(["RB", "QB"])]
    K["rypc_mean"] = cr.groupby("pos").rushing_yards.mean().to_dict()
    K["rush_pool"] = {pos: np.round(cr[cr.pos == pos].rushing_yards.values, 0).astype(int).tolist() for pos in ("RB", "QB")}
    # 7) B1 coefficient of variation by stat and position
    a = pgT[pgT.active]
    K["cv"] = {}
    for stat, col in (("receiving_yards", "receiving_yards"), ("rushing_yards", "rushing_yards")):
        g = a.groupby(["player_id", "season"])[col].agg(["mean", "std", "size"])
        g = g[(g["size"] >= 6) & (g["mean"] >= 10)]
        pos = a.drop_duplicates("player_id").set_index("player_id").pos
        g = g.assign(pos=g.index.get_level_values(0).map(pos))
        K["cv"][stat] = (g["std"] / g["mean"]).groupby(g.pos).median().to_dict()
    frozen = {"spec": "config/prereg/n3-spec.yaml", "fit_seasons": TRAIN, "n0": n0, "priors": pri, "K": K, "fit_log": log}
    FROZEN.write_text(json.dumps(frozen, indent=1, default=float))
    return frozen


# ---------------------------------------------------------------- Gate 0 (run once, 2023-2025)
TESTS = {"receptions": ("receptions", "y_rec"), "receiving_yards": ("receiving_yards", "y_recyds"),
         "rushing_yards_RB": ("rushing_yards", "y_rushyds"), "rushing_yards_QB": ("rushing_yards", "y_rushyds")}


def gate0(cache: str, out: str, n_draws: int = 4000, seed: int = 20261019):
    fz = json.loads(FROZEN.read_text())
    K, n0, pri = fz["K"], fz["n0"], fz["priors"]
    tg, pg, _, _ = load(cache, [2022] + TEST)
    tm, pl = Features(tg, pg, n0, pri).run(TEST)
    volumes(tm, pl, K)
    qbo = set(map(tuple, tm[tm.qb_out][["game_id", "team"]].values))
    emp = {k: np.asarray(v, float) for k, v in K["rush_pool"].items()}
    rng = np.random.default_rng(seed)
    by_tg = defaultdict(list)
    for r in pl:
        by_tg[(r["game_id"], r["team"])].append(r)
    rank = {}
    for key, rows in by_tg.items():
        for kind, f in (("rec", lambda r: r["snap"] * r["tprr"] if r["pos"] in ("WR", "TE", "RB") else -1),
                        ("rush", lambda r: r["cshare"] if r["pos"] in ("RB", "QB") else -1)):
            order = sorted(rows, key=f, reverse=True)
            for i, r in enumerate(order):
                rank[(r["game_id"], r["player_id"], kind)] = i + 1
    res = {k: [] for k in TESTS}
    for r in pl:
        qb_out = (r["game_id"], r["team"]) in qbo
        stats = [t for t in TESTS if eligible(r, t, K)]
        if not stats:
            continue
        sims = simulate(r, K, rng, n_draws, qb_out, emp)
        for t in stats:
            key, ycol = TESTS[t]
            y = float(r[ycol])
            x = sims[key]
            if key == "receptions":
                b = b1_samples(r["hist_rec"], "receptions", None, rng, n_draws)
            else:
                hist = r["hist_recyds"] if key == "receiving_yards" else r["hist_rushyds"]
                b = b1_samples(hist, key, K["cv"][key].get(r["pos"], 0.8), rng, n_draws)
            c3, c1 = crps(x, y, rng), crps(b, y, rng)
            pit = float(np.mean(x < y) + 0.5 * np.mean(x == y))
            kind = "rush" if key == "rushing_yards" else "rec"
            res[t].append({"game_id": r["game_id"], "season": r["season"], "week": r["week"], "player": r["name"],
                           "gain": c1 - c3, "crps_n3": c3, "crps_b1": c1, "absence": bool(r["absent"]), "qb_out": qb_out,
                           "rank": rank.get((r["game_id"], r["player_id"], kind)), "pit": pit,
                           "in80": bool(np.quantile(x, 0.1) <= y <= np.quantile(x, 0.9))})
    verdicts, pvals = {}, {}
    for t, rows in res.items():
        df = pd.DataFrame(rows)
        if df.empty:
            verdicts[t] = {"n": 0}
            continue
        mean, boots = cluster_lb(df.gain.values, df.game_id.values)
        p = float(np.mean(boots <= 0))
        pvals[t] = p
        ab = df[df.absence]
        head = ab[ab["rank"] == 1].gain.mean() if len(ab) else float("nan")
        sec = ab[ab["rank"].isin([2, 3])].gain.mean() if len(ab) else float("nan")
        verdicts[t] = {"n": int(len(df)), "games": int(df.game_id.nunique()), "gain_mean": mean,
                       "lb95_one_sided": float(np.quantile(boots, 0.05)), "p_one_sided": p,
                       "crps_n3": float(df.crps_n3.mean()), "crps_b1": float(df.crps_b1.mean()),
                       "absence_n": int(len(ab)), "absence_gain": float(ab.gain.mean()) if len(ab) else None,
                       "no_absence_gain": float(df[~df.absence].gain.mean()),
                       "qb_out_n": int(df.qb_out.sum()), "qb_out_gain": float(df[df.qb_out].gain.mean()) if df.qb_out.any() else None,
                       "headline_gain": None if pd.isna(head) else float(head),
                       "secondary_gain": None if pd.isna(sec) else float(sec),
                       "pit_mean": float(df.pit.mean()), "coverage80": float(df.in80.mean()),
                       "by_season": df.groupby("season").gain.mean().round(4).to_dict()}
    hm = holm(pvals)
    for t, v in verdicts.items():
        if t in hm:
            v["holm"] = hm[t]
            v["pass"] = bool(hm[t]["reject"] and (v["absence_gain"] or 0) > 0)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    report = {"spec": fz["spec"], "frozen_sha": __import__("hashlib").sha256(FROZEN.read_bytes()).hexdigest(),
              "seasons": TEST, "draws": n_draws, "seed": seed, "verdicts": verdicts,
              "program": "stop" if not any(v.get("pass") for v in verdicts.values()) else "continue for passing stats"}
    out_p.write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "gate0"])
    ap.add_argument("--cache", default="/tmp/n3cache")
    ap.add_argument("--out", default="data/n3/gate0.json")
    a = ap.parse_args()
    if a.cmd == "fit":
        f = fit(a.cache)
        print(json.dumps({k: v for k, v in f["K"].items() if k != "rush_pool"}, indent=1, default=float))
        print(json.dumps(f["n0"]), json.dumps(f["fit_log"], default=float))
    else:
        r = gate0(a.cache, a.out)
        print(json.dumps(r["verdicts"], indent=1, default=float))
