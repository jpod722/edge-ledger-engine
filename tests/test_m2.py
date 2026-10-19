import pandas as pd

from edge_engine.m2 import ratings_before


def test_ratings_use_only_earlier_dates():
    tg = pd.DataFrame([
        {"game_id": "g1", "season": 2020, "game_date": "2020-09-13", "team": "A", "off": 0.5, "deff": 0.0, "plays": 60},
        {"game_id": "g1", "season": 2020, "game_date": "2020-09-13", "team": "B", "off": 0.0, "deff": 0.5, "plays": 60},
        {"game_id": "g2", "season": 2020, "game_date": "2020-09-20", "team": "A", "off": 9.9, "deff": 9.9, "plays": 99},
    ])
    r = ratings_before(tg, 4, 3, 0.5)
    assert r[("g1", "A")] == (0.0, 0.0, 0.0)          # no history, no prior season: nothing from its own game
    assert r[("g2", "A")][0] == 0.5                    # only g1 informs g2; g2's own 9.9 never leaks in
