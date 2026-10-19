import numpy as np
import pandas as pd

from edge_engine import k1, m1
from tests.test_m1 import synth


def test_orders_floor_and_never_cross():
    o = k1.orders(0.555, yes_bid=0.50, yes_ask=0.53, offset=0.03)
    assert o["yes_bid"] == 0.52 and o["yes_placed"]
    assert o["no_bid"] == 0.41 and o["no_placed"]          # no_ask = 1 - 0.50 = 0.50
    o = k1.orders(0.60, yes_bid=0.50, yes_ask=0.55, offset=0.03)
    assert o["yes_bid"] == 0.57 and not o["yes_placed"]    # would cross the ask: not placed


def test_event_teams():
    assert k1.event_teams("KXNBAPTS-26OCT20OKCSAS") == {"OKC", "SA"}
    assert k1.event_teams("KXNBAPTS-26OCT20PHINYK") == {"PHI", "NY"}


def test_quotes_price_rungs_from_frozen_model():
    df = synth()
    c = m1.fit_constants(m1.features(df))
    frozen = {"constants": c, "m1_1_isotonic": {"x": [0.0, 1.0], "y": [0.05, 0.95]}}
    box = df[df.season == "2024-25"].copy()
    box["player"] = "Player " + box.player_id.map(lambda i: chr(97 + i // 26) + chr(97 + i % 26) + "x")
    box["team"] = np.where(box.player_id % 2 == 0, "OKC", "SA")
    tip = (box.date.max() + pd.Timedelta(days=2)).isoformat()
    games = [{"home_team": "San Antonio Spurs", "away_team": "Oklahoma City Thunder", "commence_time": tip}]
    mk = lambda s: {"ticker": f"T{s}", "event_ticker": "KXNBAPTS-26OCT20OKCSAS", "yes_sub_title": f"Player acx: {s + 0.5:g}+",
                    "floor_strike": s, "strike_type": "greater", "yes_bid_dollars": "0.30", "yes_ask_dollars": "0.40"}
    cfg = {"k1": {"offset": 0.03}, "m1": {"min_projected_minutes": 0}}
    recs = k1.quotes(box, [mk(9.5), mk(14.5), mk(19.5)], tip, games, cfg, frozen)
    fairs = [r["fair_m1"] for r in recs]
    assert all(r["eligible"] for r in recs) and fairs[0] > fairs[1] > fairs[2]   # ladder is monotone
    assert all("orders_naive" in r for r in recs)
