import gzip
import json
from types import SimpleNamespace

from edge_engine import cli
from edge_engine.client import OddsAPI, RawStore
from tests.test_engine import market_payload


class FakeResp:
    def __init__(self, obj):
        self.content = json.dumps(obj).encode()
        self.status_code = 200
        self.headers = {"x-requests-last": "3", "x-requests-remaining": "19000", "x-requests-used": "1000"}
        self.text = self.content.decode()

    def json(self):
        return json.loads(self.content)


class FakeSession:
    def __init__(self, obj):
        self.obj = obj
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        return FakeResp(self.obj)


def test_scan_end_to_end_stores_raw_and_ledger(tmp_path, monkeypatch):
    payload = market_payload()
    fake = FakeSession(payload)
    monkeypatch.setenv("ODDS_API_KEY", "test")
    orig = OddsAPI.__init__

    def init(self, key, store, max_credits, session=None):
        orig(self, key, store, max_credits, session=fake)
    monkeypatch.setattr(OddsAPI, "__init__", init)
    monkeypatch.setattr(cli, "utcnow", lambda: __import__("tests.test_engine", fromlist=["NOW"]).NOW)
    cli.main(["--data", str(tmp_path), "scan", "--sport", "NFL", "--scan", "t1"])
    # raw payload stored byte-for-byte with a manifest line
    man = [json.loads(x) for x in (tmp_path / "raw" / "manifest.jsonl").read_text().splitlines()]
    assert man and man[0]["credits_last"] == 3 and "apiKey" not in json.dumps(man[0]["params"])
    raw = gzip.open(tmp_path / man[0]["file"]).read()
    assert json.loads(raw) == payload
    picks = list((tmp_path / "ledger" / "picks").glob("*.json"))
    assert picks
    p = json.loads(picks[0].read_text())
    assert p["paper"] is True and p["raw_refs"] and p["engine_version"]
    sync = list((tmp_path / "sync").glob("*.json"))
    assert sync and json.loads(sync[0].read_text())["files"]
