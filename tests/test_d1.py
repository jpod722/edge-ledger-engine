import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from edge_engine import d1
from edge_engine.cli import load_cfg
from edge_engine.client import CreditGuard, OddsAPI, RawStore
from edge_engine.normalize import Quote

HOME, AWAY = "Alpha Hawks", "Beta Bears"
KICK = "2025-10-26T17:00:00Z"          # week 8 Sunday, 1 PM ET
EID = "evt0000d1test"


def tt(over, under, lu, market="team_totals", pt=24.5):
    return {"key": market, "last_update": lu, "outcomes": [
        {"name": "Over", "description": HOME, "price": over, "point": pt},
        {"name": "Under", "description": HOME, "price": under, "point": pt}]}


def ev(books, ts):
    return {"timestamp": ts, "data": {"id": EID, "commence_time": KICK, "home_team": HOME, "away_team": AWAY,
                                      "bookmakers": books}}


class Resp:
    def __init__(self, obj, remaining):
        self.content = json.dumps(obj).encode()
        self.text = self.content.decode()
        self.status_code = 200
        self.headers = {"x-requests-last": str(obj.pop("_cost", 0) if isinstance(obj, dict) else 0),
                        "x-requests-remaining": str(remaining)}
        self.content = json.dumps(obj).encode()

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        pass


class World:
    """Fake historical API: decision, +10, +30 and close snapshots for one team-total contract."""

    def __init__(self, close_ts="2025-10-26T16:50:37Z", remaining=20000):
        self.remaining, self.close_ts, self.calls = remaining, close_ts, []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.calls.append((url, dict(params)))
        date = params.get("date", "")
        if "scoreboard" in url:
            obj = {"events": [{"id": "1", "competitions": [{"status": {"type": {"completed": True}}, "competitors": [
                {"homeAway": "home", "team": {"displayName": HOME}, "score": "27", "linescores": [{"value": 7}, {"value": 10}]},
                {"homeAway": "away", "team": {"displayName": AWAY}, "score": "20", "linescores": [{"value": 3}, {"value": 7}]}]}]}]}
            return Resp(obj, self.remaining)
        if url.endswith("/odds") and "/events/" not in url:      # bulk main snapshot
            obj = {"timestamp": "2025-10-26T13:45:37Z", "_cost": 30, "data": [
                {"id": EID, "commence_time": KICK, "home_team": HOME, "away_team": AWAY, "bookmakers": []}]}
        elif date == "2025-10-26T13:48:00Z":                     # decision
            obj = ev([{"key": "pinnacle", "markets": [tt(1.95, 1.95, "2025-10-26T13:45:00Z")]},
                      {"key": "draftkings", "markets": [tt(2.15, 1.75, "2025-10-26T13:45:10Z", "alternate_team_totals")]},
                      {"key": "fanduel", "markets": [tt(2.40, 1.60, "2025-10-26T13:49:00Z", "alternate_team_totals")]}],
                     "2025-10-26T13:45:37Z")
            obj["_cost"] = 60
        elif date == "2025-10-26T13:55:37Z":                     # +10: still offered, slightly worse
            obj = ev([{"key": "draftkings", "markets": [tt(2.10, 1.78, "2025-10-26T13:55:00Z", "alternate_team_totals")]}],
                     "2025-10-26T13:55:37Z")
            obj["_cost"] = 20
        elif date == "2025-10-26T14:15:37Z":                     # +30: gone
            obj = ev([], "2025-10-26T14:15:37Z")
            obj["_cost"] = 20
        elif date == "2025-10-26T16:55:00Z":                     # close
            obj = ev([{"key": "pinnacle", "markets": [tt(1.80, 2.05, "2025-10-26T16:50:00Z")]}], self.close_ts)
            obj["_cost"] = 20
        else:
            obj = {"timestamp": date, "data": [], "_cost": 0}
        self.remaining -= obj.get("_cost", 0)
        return Resp(obj, self.remaining)


@pytest.fixture
def cfg():
    return load_cfg()


def run(tmp_path, cfg, world, weeks=(8,)):
    import edge_engine.client as client
    orig = client.requests.Session
    client.requests.Session = lambda: world
    d1.api_key = lambda: "test"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "2026-10-19T06:00:00", "credits_last": 0,
                                                        "credits_remaining": str(world.remaining)}) + "\n")
    try:
        d1.run_weeks(SimpleNamespace(data=str(tmp_path), weeks=list(weeks)), cfg)
    finally:
        client.requests.Session = orig
    return [json.loads(p.read_text()) for p in (tmp_path / "d1" / "picks").glob("*.json")]


def test_no_lookahead_drops_quotes_stamped_after_scan():
    scan = datetime(2025, 10, 26, 13, 48, tzinfo=timezone.utc)
    q = lambda lu: Quote("NFL", "e", KICK, HOME, AWAY, "draftkings", "team_totals", "Over", HOME, 24.5, 2.0, lu)
    keep, dropped = d1.no_lookahead([q("2025-10-26T13:47:59Z"), q("2025-10-26T13:48:01Z")], scan)
    assert len(keep) == 1 and dropped == 1


def test_close_gap_bounds():
    kick = datetime(2025, 10, 26, 17, 0, tzinfo=timezone.utc)
    at = lambda h, m, s=0: datetime(2025, 10, 26, h, m, s, tzinfo=timezone.utc)
    assert d1.close_gap_ok(kick, at(16, 50, 37), 10) == (True, 9.38)
    assert d1.close_gap_ok(kick, at(16, 45, 0), 10)[0] is False       # too early to count as a close
    assert d1.close_gap_ok(kick, at(17, 0, 0), 10)[0] is False        # at or after kickoff
    assert d1.close_gap_ok(kick, None, 10) == (False, None)


def test_vanished_quote_is_unavailable_not_a_loss():
    o = d1.opportunity({"fair_prob": 0.5, "push_prob": 0}, None, 0.03)
    assert o == {"available": False, "price_dec": None, "ev_at_orig_fair": None, "eligible": False}


def test_holm_step_down():
    r = d1.holm({"a": 0.001, "b": 0.02, "c": 0.2}, 0.05)
    assert r == {"a": True, "b": True, "c": False}
    assert d1.holm({"a": 0.03, "b": 0.04}, 0.05) == {"a": False, "b": False}


def test_account_floor_guard(tmp_path):
    api = OddsAPI("k", RawStore(tmp_path), 3000, session=World(), min_remaining=9000, remaining=9050)
    with pytest.raises(CreditGuard):
        api.historical_event_odds("americanfootball_nfl", EID, "2025-10-26T13:48:00Z",
                                  "team_totals,alternate_team_totals,spreads_h1,totals_h1,alternate_spreads,alternate_totals", "pinnacle")


def test_d1_end_to_end(tmp_path, cfg):
    picks = run(tmp_path, cfg, World())
    assert len(picks) == 1
    p = picks[0]
    # FanDuel's better price was stamped after the scan time, so it never entered the decision
    assert p["venue"] == "draftkings" and p["family"] == "team_totals" and p["group"] == "alt"
    assert p["tier"] == "research" and p["sources"][0]["book"] == "pinnacle" and len(p["sources"]) == 1
    assert p["threshold"] == cfg["single_source_min_ev"]
    assert p["exec"]["10"]["available"] and p["exec"]["10"]["actual_delay_min"] == 10.0
    assert p["exec"]["30"] == {**p["exec"]["30"], "available": False, "eligible": False, "clv_pinnacle": None}
    assert p["close_gap_ok"] and p["close_pinnacle_p"] is not None
    assert p["clv_pinnacle"]["prob"] > 0 and p["result"] == "win"
    meta = json.loads((tmp_path / "d1" / "weeks" / "08.json").read_text())
    assert meta["credits_by_kind"] == {"main": 60, "decision": 60, "delay_10": 20, "delay_30": 20, "close": 20}
    # Stage 1 incomplete -> results sealed
    s = d1.report(SimpleNamespace(data=str(tmp_path), final=False), cfg)
    assert isinstance(s["results"], str) and "sealed" in s["results"]
    assert s["candidates_by_family"] == {"team_totals": 1}
    s = d1.report(SimpleNamespace(data=str(tmp_path), final=True), cfg)
    tt_ = s["results"]["team_totals"]
    assert tt_["available_10"] == 1.0 and tt_["available_30"] == 0.0 and tt_["opportunity_30"] == 0.0
    assert s["verdicts"]["team_totals"] == "inconclusive"     # n far below 30


def test_close_outside_gap_is_missing_benchmark(tmp_path, cfg):
    picks = run(tmp_path, cfg, World(close_ts="2025-10-26T16:40:37Z"))
    p = picks[0]
    assert p["close_gap_ok"] is False and p["close_pinnacle_p"] is None
    assert p["close_pinnacle_reason"].startswith("close_gap_")
    assert p["clv_pinnacle"]["prob"] is None


def test_unknown_balance_fails_closed(tmp_path, cfg):
    api = OddsAPI("k", RawStore(tmp_path), 3000, session=World(), min_remaining=9000, remaining=None)
    with pytest.raises(CreditGuard):
        api.historical_event_odds("americanfootball_nfl", EID, "2025-10-26T13:48:00Z", "team_totals", "pinnacle")


def test_floor_stops_run_before_spending(tmp_path, cfg):
    run(tmp_path, cfg, World(remaining=9010))
    assert not (tmp_path / "d1" / "weeks" / "08.json").exists()
