import numpy as np
import pandas as pd

from edge_engine import m1


def synth(n_players=30, n_games=40, seed=3):
    rng = np.random.default_rng(seed)
    rows = []
    for season, start in (("2023-24", "2023-10-24"), ("2024-25", "2024-10-22")):
        dates = pd.date_range(start, periods=n_games, freq="2D", tz="UTC")
        for p in range(n_players):
            base_min, rate = rng.uniform(14, 36), rng.uniform(0.3, 0.8)
            for d in dates:
                mins = float(np.clip(rng.normal(base_min, 4), 1, 48))
                rows.append({"season": season, "game_id": f"{d:%Y%m%d}{p//2}", "date": d, "player_id": p,
                             "starter": int(base_min > 28), "dnp": 0, "minutes": round(mins),
                             "pts": float(rng.poisson(rate * mins))})
    df = pd.DataFrame(rows)
    df["gdate"] = df["date"].dt.date
    return df.sort_values(["player_id", "date"]).reset_index(drop=True)


def test_features_use_only_earlier_games():
    df = synth()
    f = m1.features(df)
    r = f.iloc[0]
    g = df[(df.player_id == r.player_id) & (df.season == r.season)].sort_values("date")
    prior = g[g.date < r.date]
    assert r.n_prior == len(prior) >= m1.MIN_APPS
    assert abs(r.mean_pts10 - prior.pts.values[-10:].mean()) < 1e-9


def test_m1_pmf_is_a_distribution_and_beats_nothing_by_construction():
    df = synth()
    f = m1.features(df)
    c = m1.fit_constants(f)
    e = f[f.season == "2024-25"].head(20)
    tot = np.zeros(len(e))
    for k in range(0, 120):
        lp, _ = m1.m1_logscore(e.assign(pts=k), c)
        tot += np.exp(lp)
    assert np.allclose(tot, 1.0, atol=0.02)
