import json
from datetime import datetime, timedelta, timezone

from edge_engine import monitor
from edge_engine.cli import load_cfg
from edge_engine.client import RawStore

T0 = datetime(2026, 10, 20, 22, 0, tzinfo=timezone.utc)          # 6:00 PM ET
TIP = T0 + timedelta(minutes=60)


def inj(status):
    return {"injuries": [{"id": "2", "displayName": "Boston Celtics", "injuries": [
        {"id": "9", "status": status, "date": "2026-10-20T21:00Z", "athlete": {"displayName": "Star Player"}}]}]}


class Clock:
    def __init__(self):
        self.t = T0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += timedelta(seconds=s)


class R:
    def __init__(self, obj, code=200, cost=0):
        self.content = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.status_code, self.text = code, self.content.decode("utf-8", "ignore")
        self.headers = {"x-requests-last": str(cost), "x-requests-remaining": "10000"}

    def json(self):
        return json.loads(self.content)


class S:
    def __init__(self, clock):
        self.clock, self.polls = clock, []

    def get(self, url, params=None, timeout=None, headers=None):
        if url.endswith("/events"):
            return R([{"id": "g1", "home_team": "Boston Celtics", "away_team": "New York Knicks",
                       "commence_time": TIP.isoformat().replace("+00:00", "Z")}])
        if url == monitor.ESPN_INJ:     # star ruled out 20 minutes in
            return R(inj("Out" if self.clock() >= T0 + timedelta(minutes=20) else "Questionable"))
        if url in monitor.INDEX_PAGES:
            return R(b"<html></html>")
        if "/events/g1/odds" in url:
            self.polls.append(self.clock())
            return R({"id": "g1", "bookmakers": []}, cost=1)
        return R(b"", 404)


def test_status_change_triggers_reaction_polls_and_close(tmp_path):
    cfg = load_cfg()
    clock = Clock()
    s = S(clock)
    monitor.api_key = lambda: "k"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep).run()
    log = [json.loads(l) for l in (tmp_path / "monitor").glob("*.jsonl").__next__().read_text().splitlines()]
    kinds = [x["kind"] for x in log]
    ch = [x for x in log if x["kind"] == "injury_change"]
    assert len(ch) == 1 and ch[0]["status"] == "Out" and ch[0]["prev_status"] == "Questionable"
    reasons = [x["reason"] for x in log if x["kind"] == "poll"]
    assert any("inj+0:Star Player:Out" in r for r in reasons)
    assert any("inj+10:" in r for r in reasons)
    close = [x for x in log if x["kind"] == "poll" and "close" in x["reason"]]
    assert len(close) == 1 and 0 < close[0]["minutes_to_tip"] <= 6
    assert all(p < TIP for p in s.polls)                                   # never polls after tip
    assert kinds[-1] == "done"


def test_diff_injuries():
    old = monitor.injury_state(inj("Questionable"))
    new = monitor.injury_state(inj("Out"))
    assert monitor.diff_injuries(old, new)[0]["status"] == "Out"
    assert monitor.diff_injuries(new, {})[0]["status"] == "removed"
    assert monitor.diff_injuries(new, new) == []
