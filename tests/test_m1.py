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


def test_pav_weighted_and_calibration_map():
    from edge_engine.m1_live import apply_cal, pav_weighted
    f = pav_weighted(np.array([0.0, 1, 0, 1, 1, 0, 1]), np.ones(7))
    assert np.all(np.diff(f) >= 0) and abs(f.mean() - 4 / 7) < 1e-9
    knots = {"x": [0.0, 0.4, 0.6, 1.0], "y": [0.1, 0.45, 0.65, 0.9]}
    assert apply_cal(np.array([0.4]), knots)[0] == 0.45


def test_live_forecast_uses_only_prior_games_and_consensus_excludes_venue():
    from edge_engine import m1_live
    df = synth()
    f = m1.features(df)
    c = m1.fit_constants(f)
    frozen = {"constants": c, "m1_1_isotonic": {"x": [0.0, 1.0], "y": [0.0, 1.0]}}
    box = df[df.season == "2024-25"].copy()
    box["player"] = "Player " + box.player_id.map(lambda i: chr(97 + i // 26) + chr(97 + i % 26) + "x")
    box["team"] = np.where(box.player_id % 2 == 0, "BOS", "NY")
    box["dnp"] = 0
    tip = (box.date.max() + pd.Timedelta(days=2)).isoformat()
    def bk(key, o, u, line=14.5):
        return {"key": key, "markets": [{"key": "player_points", "last_update": tip, "outcomes": [
            {"name": "Over", "description": "Player acx", "price": o, "point": line},
            {"name": "Under", "description": "Player acx", "price": u, "point": line}]}]}
    ev = {"id": "e1", "commence_time": tip, "home_team": "Boston Celtics", "away_team": "New York Knicks",
          "bookmakers": [bk("draftkings", 2.2, 1.7), bk("fanduel", 1.9, 1.9), bk("betmgm", 1.9, 1.9),
                         bk("pinnacle", 1.95, 1.87), bk("fanatics", 1.9, 1.9)]}
    cfg = {"m1": {"market": "player_points", "venues": ["draftkings", "fanduel"], "min_projected_minutes": 0,
                  "consensus_books": ["pinnacle", "betmgm", "fanatics", "draftkings", "fanduel"], "consensus_min_books": 3}}
    recs = m1_live.forecasts(box, [(ev, tip)], cfg, frozen)
    dk = next(r for r in recs if r["venue"] == "draftkings")
    assert dk["eligible"] and "draftkings" not in dk["market_books"] and len(dk["market_books"]) == 4
    assert 0 < dk["m1_p_over"] < 1 and dk["m1_1_p_over"] == dk["m1_p_over"]
