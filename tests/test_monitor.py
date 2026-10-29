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
        if "kalshi.com" in url:
            if "/markets/trades" in url:
                return R({"trades": [{"ticker": "KXNBAPTS-X-1", "yes_price_dollars": "0.40", "count_fp": "5"}]})
            return R({"markets": [{"ticker": "KXNBAPTS-X-1"}], "cursor": ""})
        if "/events/g1/odds" in url:
            self.polls.append(self.clock())
            return R({"id": "g1", "bookmakers": []}, cost=1)
        return R(b"", 404)


def test_status_change_triggers_reaction_polls_and_close(tmp_path):
    cfg = load_cfg()
    cfg["monitor"]["lean"] = False
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
    assert "kalshi_snapshot" in kinds and "kalshi_trades" in kinds


def test_diff_injuries():
    old = monitor.injury_state(inj("Questionable"))
    new = monitor.injury_state(inj("Out"))
    assert monitor.diff_injuries(old, new)[0]["status"] == "Out"
    assert monitor.diff_injuries(new, {})[0]["status"] == "removed"
    assert monitor.diff_injuries(new, new) == []


def test_budget_reserves_closes(tmp_path):
    cfg = load_cfg()
    cfg["monitor"]["lean"] = False
    cfg["monitor"]["nightly_cap_credits"] = 3          # tiny cap: baseline polls must stop before the close
    clock = Clock()
    s = S(clock)
    monitor.api_key = lambda: "k"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep).run()
    log = [json.loads(l) for l in next((tmp_path / "monitor").glob("*.jsonl")).read_text().splitlines()]
    polls = [x for x in log if x["kind"] == "poll"]
    assert any("close" in x["reason"] for x in polls)                  # the close still happened
    assert any(x["kind"] == "poll_skipped_budget" for x in log)
    assert len(polls) <= 3


def test_stop_at_caps_runtime(tmp_path):
    """A one-off early run stops at its --stop-at-et time even when its game has not tipped."""
    cfg = load_cfg()
    clock = Clock()
    s = S(clock)
    monitor.api_key = lambda: "k"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    stop = T0 + timedelta(minutes=30)
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep, stop_at=stop).run()
    log = [json.loads(l) for l in next((tmp_path / "monitor").glob("*.jsonl")).read_text().splitlines()]
    assert log[-1]["kind"] == "done"
    assert all(p <= stop for p in s.polls)
    assert not any(x["kind"] == "poll" and "close" in x["reason"] for x in log)
    assert "kalshi_trades" in [x["kind"] for x in log]                # still collects trades on the way out


def test_too_early_run_exits_without_polling(tmp_path):
    """A run fired 8 hours before the first tip polls nothing and spends no credits."""
    cfg = load_cfg()
    clock = Clock()
    clock.t = TIP - timedelta(hours=8)
    s = S(clock)
    monitor.api_key = lambda: "k"
    RawStore(tmp_path).manifest.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep).run()
    log = [json.loads(l) for l in next((tmp_path / "monitor").glob("*.jsonl")).read_text().splitlines()]
    assert log[-1]["kind"] == "done" and log[-1]["reason"] == "too_early"
    assert s.polls == []


def run_lean(tmp_path, start):
    cfg = load_cfg()
    assert cfg["monitor"]["lean"] is True
    clock = Clock()
    clock.t = start
    s = S(clock)
    monitor.api_key = lambda: "k"
    m = RawStore(tmp_path).manifest
    if not m.exists():
        m.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep).run()
    log = [json.loads(l) for l in next((tmp_path / "monitor").glob("*.jsonl")).read_text().splitlines()]
    return s, log


def test_lean_one_decision_one_close_and_no_reaction(tmp_path):
    """Lean mode (reviewer Oct 24): tip 7 PM ET -> decision 17:30 ET; one poll at ~17:25 and one close; injury
    changes are logged but trigger no polls."""
    start = T0 - timedelta(hours=1, minutes=30)                         # 4:30 PM ET
    s, log = run_lean(tmp_path, start)
    polls = [x for x in log if x["kind"] == "poll"]
    assert [p["reason"] for p in polls] == ["decision", "close"]
    dec = s.polls[0].astimezone(monitor.ET)
    assert (dec.hour, dec.minute) == (17, 25)
    assert any(x["kind"] == "injury_change" for x in log)               # still archived, no reaction poll


def test_lean_second_run_same_day_does_not_repeat(tmp_path):
    s1, _ = run_lean(tmp_path, T0 - timedelta(hours=1, minutes=30))
    s2, log = run_lean(tmp_path, T0 - timedelta(minutes=30))           # a late duplicate run before tip
    assert len(s1.polls) == 2 and s2.polls == []
    assert not any(x["kind"] == "late_decision" for x in log[-20:] if x["at"] >= (T0 - timedelta(minutes=30)).isoformat())


def run_lean_relay(tmp_path, start, relays):
    cfg = load_cfg()
    clock = Clock()
    clock.t = start
    s = S(clock)
    monitor.api_key = lambda: "k"
    m = RawStore(tmp_path).manifest
    if not m.exists():
        m.write_text(json.dumps({"retrieved_at": "x", "credits_remaining": "10000"}) + "\n")
    monitor.Monitor(str(tmp_path), cfg, session=s, clock=clock, sleep=clock.sleep,
                    relay=lambda: relays.append(clock()) or True).run()
    log = [json.loads(l) for l in next((tmp_path / "monitor").glob("*.jsonl")).read_text().splitlines()]
    return s, clock, log


def test_relay_chain_from_an_early_morning_run_takes_the_decision_on_time(tmp_path):
    """Oct 28: the only run before the decision fired at 08:42 UTC and exited too_early; the next started after
    17:30 ET. Now the early run waits and hands over, and the chain takes the snapshot at 17:25 ET."""
    relays, t, polls = [], T0 - timedelta(hours=13, minutes=20), []      # 08:40 UTC
    for _ in range(5):
        n = len(relays)
        s, clock, log = run_lean_relay(tmp_path, t, relays)
        polls += s.polls
        if len(relays) == n:
            break
        if log[-1]["reason"] == "relayed":
            assert s.polls == []                                        # waiting runs spend nothing
            assert clock() <= t + timedelta(minutes=330)                # hands over inside its own job limit
        t = clock() + timedelta(minutes=3)                              # successor start-up delay
    dec = polls[0].astimezone(monitor.ET)
    assert (dec.hour, dec.minute) == (17, 25)
    assert [x["reason"] for x in log if x["kind"] == "poll"] == ["decision", "close"]   # whole night's log
    assert 1 <= len(relays) <= 4


def test_relay_hands_over_pending_closes_at_the_job_limit(tmp_path):
    relays = []
    start = TIP - timedelta(minutes=350)                                # decision reachable, tip after the hard stop
    _, _, log = run_lean_relay(tmp_path, start, relays)
    assert [x["reason"] for x in log if x["kind"] == "poll"] == ["decision"]
    assert [x["why"] for x in log if x["kind"] == "relay"] == ["pending_closes"] and len(relays) == 1


def test_relays_are_capped_per_night(tmp_path):
    relays = []
    for i in range(8):
        run_lean_relay(tmp_path, T0 - timedelta(hours=13, minutes=20), relays)
    assert len(relays) == load_cfg()["monitor"].get("max_relays_per_day", 6)
