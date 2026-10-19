import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from edge_engine import s1
from edge_engine.cli import load_cfg
from edge_engine.client import RawStore
from edge_engine.normalize import quotes_from

HOME, AWAY = "Alpha Hawks", "Beta Bears"
KICK = "2025-09-28T17:00:00Z"          # week 4 Sunday, 1 PM ET
EID = "evt0000s1test"
T0 = datetime(2025, 9, 28, 14, 58, 37, tzinfo=timezone.utc)
SNAPS = [T0 + timedelta(minutes=5 * i) for i in range(24)]      # 10:58:37 .. 12:53:37 ET


def z(t):
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def spread(book, home_dec, away_dec, lu, pt=-3.0):
    return {"key": book, "markets": [{"key": "spreads", "last_update": lu, "outcomes": [
        {"name": HOME, "price": home_dec, "point": pt}, {"name": AWAY, "price": away_dec, "point": -pt}]}]}


def snapshot(i):
    t = SNAPS[i]
    if i < 2:
        pin = (1.95, 1.95, z(SNAPS[0] - timedelta(minutes=1)))
    elif i < 23:
        pin = (1.85, 2.05, z(SNAPS[2] - timedelta(seconds=20)))           # Pinnacle moves toward Home at k=2
    else:
        pin = (1.80, 2.10, z(t - timedelta(seconds=30)))                   # close
    dk = (2.00, 1.80, z(t - timedelta(seconds=50))) if i < 4 else (1.85, 1.95, z(SNAPS[4] - timedelta(seconds=10)))
    books = [spread("pinnacle", *pin), spread("lowvig", *pin), spread("draftkings", *dk)]
    return {"timestamp": z(t), "previous_timestamp": z(t - timedelta(minutes=5)),
            "next_timestamp": z(t + timedelta(minutes=5)),
            "data": [{"id": EID, "commence_time": KICK, "home_team": HOME, "away_team": AWAY, "bookmakers": books}]}


class Resp:
    def __init__(self, obj, cost, remaining):
        self.content = json.dumps(obj).encode()
        self.text, self.status_code = self.content.decode(), 200
        self.headers = {"x-requests-last": str(cost), "x-requests-remaining": str(remaining)}

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        pass


class World:
    def __init__(self, remaining=20000):
        self.remaining, self.calls = remaining, []

    def get(self, url, params=None, timeout=None):
        params = params or {}
        self.calls.append(params.get("date"))
        if "scoreboard" in url:
            return Resp({"events": [{"id": "1", "competitions": [{"status": {"type": {"completed": True}}, "competitors": [
                {"homeAway": "home", "team": {"displayName": HOME}, "score": "24", "linescores": []},
                {"homeAway": "away", "team": {"displayName": AWAY}, "score": "17", "linescores": []}]}]}]}, 0, self.remaining)
        want = datetime.fromisoformat(params["date"].replace("Z", "+00:00"))
        i = max(j for j, t in enumerate(SNAPS) if t <= want) if want >= SNAPS[0] else 0
        if want > SNAPS[-1] + timedelta(minutes=5):
            i = len(SNAPS) - 1
        self.remaining -= 20
        return Resp(snapshot(i), 20, self.remaining)


@pytest.fixture
def cfg():
    c = load_cfg()
    c["s1"]["weeks"] = [4]
    return c


def run(tmp_path, cfg, world):
    import edge_engine.client as client
    orig = client.requests.Session
    client.requests.Session = lambda: world
    s1.api_key = lambda: "test"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "2026-10-19T06:00:00", "credits_last": 0,
                                                        "credits_remaining": str(world.remaining)}) + "\n")
    try:
        s1.run(SimpleNamespace(data=str(tmp_path)), cfg)
    finally:
        client.requests.Session = orig
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "s1" / "picks").glob("*.json"))]


def test_trigger_requires_pinnacle_move_and_unchanged_older_venue_quote():
    prev, cur = quotes_from([snapshot(1)["data"][0]], "NFL"), quotes_from([snapshot(2)["data"][0]], "NFL")
    t = s1.triggers(prev, cur, ["draftkings"], 0.015)
    assert len(t) == 1 and t[0]["key"][3] == HOME and t[0]["move"] >= 0.015
    assert s1.triggers(prev, cur, ["draftkings"], 0.10) == []               # move too small
    assert s1.triggers(prev, prev, ["draftkings"], 0.015) == []             # Pinnacle did not move
    moved = quotes_from([snapshot(4)["data"][0]], "NFL")                    # venue repriced
    assert s1.triggers(quotes_from([snapshot(3)["data"][0]], "NFL"), moved, ["draftkings"], 0.0) == []


def test_s1_end_to_end(tmp_path, cfg):
    world = World()
    picks = run(tmp_path, cfg, world)
    meta = json.loads((tmp_path / "s1" / "sundays" / "04.json").read_text())
    assert meta["snapshots"] == 24 and meta["gaps_s"] == [300] and meta["credits"] == 480
    trig = [p for p in picks if p["track"] == "triggered"]
    assert len(trig) == 1
    p = trig[0]
    assert p["k"] == 2 and p["venue"] == "draftkings" and p["side"] == HOME and p["tier"] == "research"
    assert p["exec"]["5"]["available"] and p["exec"]["5"]["eligible"]        # still 2.00 at k=3
    assert p["exec"]["10"]["available"] and not p["exec"]["10"]["eligible"]  # repriced to 1.85 at k=4
    assert p["close_gap_ok"] and p["close_gap_min"] == pytest.approx(6.38, abs=0.01)
    assert p["clv_pinnacle"]["prob"] > 0 and p["result"] == "win"
    s = s1.report(SimpleNamespace(data=str(tmp_path)), cfg)
    assert s["raw_triggers"] >= 1 and s["triggered_research"] == 1
    assert s["results"]["triggered"]["opportunity_5"] == 1.0 and s["results"]["triggered"]["opportunity_10"] == 0.0
    assert s["verdict"] == "stop"                                            # fewer than 10 triggered


def test_s1_floor_stops_before_spending(tmp_path, cfg):
    world = World(remaining=9010)
    run(tmp_path, cfg, world)
    assert not (tmp_path / "s1" / "sundays" / "04.json").exists()
    assert [d for d in world.calls if d] == []
