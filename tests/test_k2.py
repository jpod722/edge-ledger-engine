from datetime import datetime, timedelta, timezone

import pandas as pd

from edge_engine import k2, k2_pull

T0 = datetime(2026, 11, 14, 0, 0, tzinfo=timezone.utc)


def iso(m):
    return (T0 + timedelta(minutes=m)).isoformat().replace("+00:00", "Z")


def sec(m):
    return (T0 + timedelta(minutes=m)).timestamp()


def test_parse_event_and_codes():
    assert k2_pull.parse_event("KXNBAPTS-26NOV14DALCLE") == ("2026-11-14", "DAL", "CLE")
    assert k2_pull.parse_event("KXNBAPTS-25DEC01NOPGSW") == ("2025-12-01", "NOP", "GSW")
    assert k2_pull.parse_event("bad") is None
    assert {k2_pull.espn_code(c) for c in ("NOP", "NYK", "SAS", "GSW", "UTA", "WAS")} == {"NO", "NY", "SA", "GS", "UTAH", "WSH"}


def test_fee_is_cent_ceiling_per_order():
    assert k2.fee(0.50, 20) == 0.35                     # 0.07*20*0.25 = 0.35 exactly
    assert k2.fee(0.45, 20) == 0.35                     # 0.3465 -> 0.35
    assert k2.fee(0.10, 20) == 0.13                     # 0.126 -> 0.13
    assert k2.fee(0.50, 20, 0.0175) == 0.09             # maker sensitivity 0.0875 -> 0.09
    assert k2.fee(0.50, 1) == 0.02                      # 0.0175 -> 0.02


def test_compact_candles_both_schemas():
    recent = {"end_period_ts": 60, "volume_fp": "3.0", "yes_bid": {"close_dollars": "0.41"}, "yes_ask": {"close_dollars": "0.43"}}
    archived = {"end_period_ts": 120, "volume": "0.00", "yes_bid": {"close": "0.40"}, "yes_ask": {"close": None}}
    assert k2_pull.compact_candles([recent, archived]) == [[60, 0.41, 0.43, 3.0], [120, 0.40, None, 0.0]]


def test_book_carries_last_change_forward():
    b = k2.Book([[sec(1), 0.40, 0.42, 0], [sec(5), 0.30, 0.33, 0]])
    assert b.at(sec(0.5)) is None
    assert b.at(sec(4)) == (0.40, 0.42)                 # no candle 2-4: unchanged
    assert b.at(sec(5)) == (0.30, 0.33)
    assert k2.no_quote((0.40, 0.42)) == (0.6, 0.58)
    assert k2.no_quote((0.0, 1.0)) == (None, None)


def test_points_so_far():
    p = k2.Points([[iso(1), "7", 2, 1], [iso(3), "7", 3, 1], [iso(2), "8", 1, 1]])
    assert p.at("7", sec(0)) == 0 and p.at("7", sec(2)) == 2 and p.at("7", sec(3)) == 5 and p.final("7") == 5
    assert p.at("9", sec(3)) == 0


def test_select_closest_to_half_ties_lower_and_band():
    c = [{"floor_strike": 14.5, "no_ask": 0.45}, {"floor_strike": 19.5, "no_ask": 0.55}, {"floor_strike": 24.5, "no_ask": 0.95}]
    assert k2.select(c)["floor_strike"] == 14.5
    assert k2.select([{"floor_strike": 9.5, "no_ask": 0.05}, {"floor_strike": 29.5, "no_ask": 0.95}]) is None
    assert k2.select([{"floor_strike": 9.5, "no_ask": 0.10}])["floor_strike"] == 9.5     # band edges inclusive


def test_tradable_and_maker_fill():
    trades = [[sec(10) + 30, 0.42, 5, "no"], [sec(20), 0.45, 5, "yes"]]           # NO prices 0.58, 0.55
    assert k2.tradable(trades, sec(10), 0.58)
    assert not k2.tradable(trades, sec(10), 0.57)                                 # trade at 0.58 is worse for the buyer
    assert not k2.tradable(trades, sec(12), 0.60)                                 # outside +/- 60 s
    assert k2.maker_fill(trades, sec(10), sec(30), 0.56) == sec(20)               # 0.55 < 0.56: strictly through
    assert k2.maker_fill(trades, sec(10), sec(30), 0.55) is None                  # at the bid is not a fill
    assert k2.maker_fill(trades, sec(10), sec(15), 0.56) is None                  # cancelled before the print


def test_bootstrap_resamples_dates_and_is_seeded():
    df = pd.DataFrame({"date": ["2026-01-01"] * 3 + ["2026-01-02"] * 3, "pnl": [1, 1, 1, -1, -1, -1], "cost": [10] * 6})
    a, b = k2.boot(df), k2.boot(df)
    assert a == b and a[0] == 0.0 and a[1] < 0 < a[2]
    s = k2.summarize(df)
    assert s["bets"] == 6 and s["share_profitable_nights"] == 0.5 and set(s["by_month"]) == {"2026-01"}


def test_background_positions():
    # one YES-buyer trade at 0.40 for 150 contracts (capped at 100) and one NO-buyer trade at yes 0.70; market settles no
    trades = [[sec(5), 0.40, 150, "yes"], [sec(6), 0.70, 10, "no"], [sec(-5), 0.50, 10, "yes"], [sec(7), 0.95, 10, "no"]]
    r = k2.background_rows(trades, "no", sec(0), sec(60), 0.0)
    assert r["yes_taker"][0] == 1 and abs(r["yes_taker"][1] - (40 + k2.fee(0.40, 100))) < 1e-9 and r["yes_taker"][2] < 0
    assert abs(r["maker_vs_yes_buyers"][2] - (100 - 60)) < 1e-9                    # bought 100 NO at 0.60, no fee
    assert r["no_taker"][0] == 1 and r["no_taker"][2] > 0                          # pre-tip and 0.95 trades excluded


def test_replay_game_halftime_rules():
    g = {"date": "2026-11-13", "game_id": "1", "event_ticker": "KXNBAPTS-26NOV13DALCLE", "away": "DAL", "home": "CLE",
         "actual_tip": iso(0),
         "period_end": {"1": iso(30), "2": iso(70), "3": iso(100), "4": iso(130)}, "game_end": iso(130),
         "roster": {"7": ["Star Player", "DAL"]}, "scoring": [[iso(10), "7", 2, 1], [iso(50), "7", 10, 2]]}
    mk = lambda s, res: {"ticker": f"T{s}", "player": "Star Player", "floor_strike": s, "result": res,
                         "open_time": iso(-600), "close_time": iso(140)}
    markets = [mk(9.5, "yes"), mk(19.5, "no"), mk(24.5, "no")]
    tape = {"T9.5": ([[sec(0), 0.9, 0.92, 0]], []),                    # already decided at halftime (12 > 9.5)
            "T19.5": ([[sec(60), 0.52, 0.54, 0], [sec(95), 0.60, 0.62, 0]],
                      [[sec(71), 0.52, 3, "no"], [sec(80), 0.40, 3, "yes"], [sec(90), 0.56, 3, "yes"]]),
            "T24.5": ([[sec(60), 0.20, 0.22, 0]], [])}
    bets, st = k2.replay_game(g, markets, lambda t: tape[t])
    half = [b for b in bets if b["checkpoint"] == "half"]
    tk = [b for b in half if b["rule"] == "taker"][0]
    assert tk["floor_strike"] == 19.5 and tk["price"] == 0.48 and tk["pts_so_far"] == 12 and tk["tradable"]
    assert abs(tk["pnl"] - (20 - (20 * 0.48 + k2.fee(0.48, 20)))) < 1e-9
    mkr = [b for b in half if b["rule"] == "maker"][0]
    # NO bid 1 - 0.54 = 0.46; the NO 0.60 print is not through, the NO 0.44 print at minute 90 is
    assert mkr["price"] == 0.46 and mkr["filled"] and mkr["fill_delay_s"] == sec(90) - sec(71)
    assert abs(mkr["mid5"] - (0.40 + 0.38) / 2) < 1e-9                # NO mid 5 min after the fill (candle at 95)
    q3 = [b for b in bets if b["checkpoint"] == "q3" and b["rule"] == "taker"][0]
    assert q3["price"] == 0.40 and q3["floor_strike"] == 19.5
    assert st["decided_by_pbp"] >= 1 and st["settle_agree"] == st["settle_checked"]


def _plays(stamps):
    return [{"wallclock": s, "period": {"number": i // 2 + 1}} for i, s in enumerate(stamps)]


def test_fix_wallclock_dst_segments():
    # Mar 8 2026 pattern: Q1-Q3 stamped +1 h, Q4 right (jump back 59 min)
    p = _plays(["2026-03-08T23:11:51Z", "2026-03-09T00:59:22Z", "2026-03-09T00:00:25Z", "2026-03-09T00:30:26Z"])
    assert k2_pull.fix_wallclock(p, "2026-03-08T22:00Z") == [-1, 0]
    assert [x["wallclock"] for x in p] == ["2026-03-08T22:11:51Z", "2026-03-08T23:59:22Z", "2026-03-09T00:00:25Z",
                                           "2026-03-09T00:30:26Z"]
    # Nov 1 2026 pattern: -1 h until mid-Q3, then right (jump forward 62 min)
    p = _plays(["2026-11-01T21:10:49Z", "2026-11-01T22:58:12Z", "2026-11-02T00:00:28Z", "2026-11-02T00:27:13Z"])
    assert k2_pull.fix_wallclock(p, "2026-11-01T22:00Z") == [1, 0]
    assert p[0]["wallclock"] == "2026-11-01T22:10:49Z" and p[3]["wallclock"] == "2026-11-02T00:27:13Z"
    # a normal game is untouched
    p = _plays(["2026-11-02T00:11:00Z", "2026-11-02T00:50:00Z", "2026-11-02T01:30:00Z"])
    assert k2_pull.fix_wallclock(p, "2026-11-02T00:00Z") == [0]
    assert p[0]["wallclock"] == "2026-11-02T00:11:00Z"
