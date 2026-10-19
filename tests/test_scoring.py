import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from edge_engine import scoring
from edge_engine.cli import load_cfg

TIP = datetime(2026, 11, 2, 0, 0, tzinfo=timezone.utc)      # Nov 1, 8 PM ET
DEC = TIP - timedelta(hours=3)


def z(t):
    return t.isoformat().replace("+00:00", "Z")


def put_raw(root: Path, rows):
    man = root / "raw" / "manifest.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    with open(man, "a") as f:
        for i, (path, obj, at) in enumerate(rows):
            rel = f"raw/x{i}_{abs(hash(path)) % 10**8}.json.gz"
            with gzip.open(root / rel, "wb") as g:
                g.write(json.dumps(obj).encode())
            f.write(json.dumps({"retrieved_at": z(at), "path": path, "status": 200, "kind": "kalshi", "file": rel,
                                "credits_last": 0, "credits_remaining": "9000"}) + "\n")


def k1_quote(ticker="T1", yes_bid=0.40, no_bid=0.50):
    o = {"yes_bid": yes_bid, "yes_placed": True, "no_bid": no_bid, "no_placed": True}
    return {"ticker": ticker, "event_ticker": "KXNBAPTS-26NOV01BOSNYK", "decision_at": z(DEC), "tip": z(TIP), "eligible": True,
            "orders_m1_1": o, "orders_m1": o, "orders_naive": {**o, "yes_bid": 0.30}}


def setup(tmp_path, injury_at=None):
    (tmp_path / "k1" / "quotes").mkdir(parents=True)
    (tmp_path / "k1" / "quotes" / "2026-11-01.jsonl").write_text(json.dumps(k1_quote()) + "\n")
    trades = {"trades": [
        {"created_time": z(DEC + timedelta(minutes=30)), "yes_price_dollars": "0.38", "no_price_dollars": "0.62", "count_fp": "4"},
        {"created_time": z(DEC + timedelta(minutes=40)), "yes_price_dollars": "0.40", "no_price_dollars": "0.60", "count_fp": "9"},
        {"created_time": z(DEC + timedelta(minutes=50)), "yes_price_dollars": "0.52", "no_price_dollars": "0.48", "count_fp": "20"},
        {"created_time": z(TIP + timedelta(minutes=5)), "yes_price_dollars": "0.10", "no_price_dollars": "0.90", "count_fp": "50"}]}
    book = {"markets": [{"ticker": "T1", "yes_bid_dollars": "0.44", "yes_ask_dollars": "0.48"}]}
    put_raw(tmp_path, [("/markets/trades?ticker=T1&limit=1000", trades, TIP + timedelta(hours=3)),
                       ("/markets?series_ticker=KXNBAPTS&status=open&limit=200", book, TIP - timedelta(minutes=4))])
    (tmp_path / "monitor").mkdir()
    ev = [{"at": z(TIP), "kind": "games", "n": 1}, {"at": z(TIP), "kind": "done"}]
    if injury_at:
        ev.append({"at": z(injury_at), "kind": "injury_change", "team": "Boston Celtics", "first_seen": z(injury_at)})
    (tmp_path / "monitor" / "2026-11-01.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")


def test_k1_through_only_fills_and_clv(tmp_path):
    setup(tmp_path)
    cfg = load_cfg()
    raw = scoring.Raw(str(tmp_path))
    logs = scoring.jsonl((tmp_path / "monitor").glob("*.jsonl"))
    s = scoring.k1_score(str(tmp_path), cfg, raw, logs)
    assert s["fills"] == 2 and s["results"] == "sealed"
    # internal check of the fills
    cfg["k1"]["gate"]["min_fills"], cfg["k1"]["gate"]["min_game_dates"] = 1, 1
    s = scoring.k1_score(str(tmp_path), cfg, raw, logs)
    r = s["results"]
    assert s["complete"]
    # YES 0.40: only the 0.38 print is through (4 contracts); the at-price 0.40 print does not count; post-tip ignored
    # NO 0.50: the 0.48 NO print is through (20 -> capped at 10). Close mid 0.46: YES clv +6c, NO clv (0.54 - 0.50) = +4c
    assert abs(r["m1_1_clv_cents"]["mean"] - 5.0) < 1e-6


def test_injury_cancel_stops_fills(tmp_path):
    setup(tmp_path, injury_at=DEC + timedelta(minutes=10))
    cfg = load_cfg()
    s = scoring.k1_score(str(tmp_path), cfg, scoring.Raw(str(tmp_path)), scoring.jsonl((tmp_path / "monitor").glob("*.jsonl")))
    assert s["fills"] == 0


def test_scoreboard_runs_and_seals(tmp_path):
    setup(tmp_path)
    sb = scoring.run(str(tmp_path), load_cfg(), now=TIP + timedelta(hours=6))
    assert sb["tracks"]["gate_a"]["results"] == "sealed" and sb["tracks"]["k1"]["state"] == "collecting"
    assert sb["health"]["status"] == "red"            # no injury-archive snapshots in this fixture
    assert (tmp_path / "scoreboard.json").exists()


def test_gate_a_passes_when_model_is_informative(tmp_path):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(5)
    cfg = load_cfg()
    cfg["m1"]["gateA"]["min_forecasts"], cfg["m1"]["gateA"]["min_game_dates"] = 300, 30
    recs, box = [], []
    for d in range(60):
        day = datetime(2026, 11, 1, 23, tzinfo=timezone.utc) + timedelta(days=d)
        for i in range(20):
            p_true = rng.uniform(0.25, 0.75)
            p_mkt = float(np.clip(p_true + rng.normal(0, 0.15), 0.05, 0.95))
            pts = 21.0 if rng.random() < p_true else 19.0
            pid = f"{d}-{i}"
            recs.append({"event_id": f"e{d}", "player": pid, "line": 20.5, "venue": "draftkings", "eligible": True,
                         "commence_time": z(day), "player_id": pid, "m1_p_over": p_true, "m1_1_p_over": p_true,
                         "market_p_over": p_mkt})
            box.append({"player_id": pid, "date": z(day), "dnp": 0, "minutes": 30, "pts": pts})
    (tmp_path / "m1" / "forecasts").mkdir(parents=True)
    (tmp_path / "m1" / "forecasts" / "all.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    (tmp_path / "m1" / "boxscores").mkdir(parents=True)
    pd.DataFrame(box).to_csv(tmp_path / "m1" / "boxscores" / "2026-27.csv.gz", index=False)
    a = scoring.gate_a(str(tmp_path), cfg, scoring.Raw(str(tmp_path)), [])
    assert a["complete"] and a["results"]["M1.1"]["pass"]
    assert a["results"]["M1.1"]["blend_logloss"] < a["results"]["M1.1"]["market_logloss"]
