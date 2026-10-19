"""Direct HTTP clients. Every response is written to the immutable raw store before use."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://api.the-odds-api.com/v4"


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RawStore:
    """Append-only store: data/raw/<date>/<time>_<label>_<sha8>.json.gz plus a manifest line per response."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.manifest = self.root / "raw" / "manifest.jsonl"
        self.manifest.parent.mkdir(parents=True, exist_ok=True)

    def save(self, label: str, body: bytes, meta: dict) -> str:
        sha = hashlib.sha256(body).hexdigest()
        ts = meta["retrieved_at"]
        rel = Path("raw") / ts[:10] / f"{ts[11:19].replace(':', '')}_{label}_{sha[:8]}.json.gz"
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with gzip.open(path, "wb") as f:
                f.write(body)
        with open(self.manifest, "a") as f:
            f.write(json.dumps({**meta, "file": str(rel), "sha256": sha, "bytes": len(body)}) + "\n")
        return str(rel)

    @staticmethod
    def cycle_start(now: datetime, reset_day: int) -> datetime:
        """Start of the current billing cycle (credits reset on `reset_day` of each month, UTC)."""
        y, m = now.year, now.month
        if now.day < reset_day:
            y, m = (y - 1, 12) if m == 1 else (y, m - 1)
        return datetime(y, m, reset_day, tzinfo=timezone.utc)

    def credits_this_month(self, now: datetime, reset_day: int = 1) -> int:
        """Credits billed since the current billing cycle began."""
        if not self.manifest.exists():
            return 0
        start = self.cycle_start(now, reset_day).isoformat()
        total = 0
        for line in self.manifest.read_text().splitlines():
            m = json.loads(line)
            if m.get("retrieved_at", "") >= start:
                total += int(m.get("credits_last") or 0)
        return total


class CreditGuard(Exception):
    pass


class OddsAPI:
    def __init__(self, key: str, store: RawStore, max_credits: int, session=None):
        self.key, self.store, self.max_credits = key, store, max_credits
        self.used = 0
        self.s = session or requests.Session()
        self.refs: list[str] = []

    def _get(self, path: str, params: dict, label: str, est_cost: int):
        if self.used + est_cost > self.max_credits:
            raise CreditGuard(f"run credit cap {self.max_credits} reached ({self.used} used)")
        url = f"{BASE}{path}"
        for attempt in range(2):
            r = self.s.get(url, params={**params, "apiKey": self.key}, timeout=30)
            if r.status_code < 500:
                break
            time.sleep(2)
        retrieved = utcnow().isoformat()
        cost = int(r.headers.get("x-requests-last", est_cost) or 0)
        meta = {"retrieved_at": retrieved, "path": path, "params": params, "status": r.status_code,
                "credits_last": cost, "credits_remaining": r.headers.get("x-requests-remaining"),
                "credits_used_total": r.headers.get("x-requests-used")}
        ref = self.store.save(label, r.content, meta)
        self.refs.append(ref)
        self.used += cost
        if r.status_code != 200:
            raise RuntimeError(f"Odds API {r.status_code} on {path}: {r.text[:200]}")  # fail closed
        return r.json(), retrieved, ref

    def odds(self, sport_key: str, markets: str, bookmakers: str):
        n = len(markets.split(","))
        return self._get(f"/sports/{sport_key}/odds/",
                         {"markets": markets, "bookmakers": bookmakers, "oddsFormat": "decimal", "dateFormat": "iso"},
                         f"{sport_key}_odds", n)

    def event_odds(self, sport_key: str, event_id: str, markets: str, bookmakers: str):
        n = len(markets.split(","))
        return self._get(f"/sports/{sport_key}/events/{event_id}/odds",
                         {"markets": markets, "bookmakers": bookmakers, "oddsFormat": "decimal", "dateFormat": "iso"},
                         f"{sport_key}_event_{event_id[:8]}", n)

    def historical_odds(self, sport_key: str, date_iso: str, markets: str, bookmakers: str):
        n = len(markets.split(",")) * 10
        return self._get(f"/historical/sports/{sport_key}/odds",
                         {"date": date_iso, "markets": markets, "bookmakers": bookmakers,
                          "oddsFormat": "decimal", "dateFormat": "iso"},
                         f"{sport_key}_hist", n)


def espn_get(store: RawStore, url: str, label: str, session=None) -> dict:
    s = session or requests.Session()
    r = s.get(url, timeout=30)
    meta = {"retrieved_at": utcnow().isoformat(), "path": url, "params": {}, "status": r.status_code,
            "credits_last": 0}
    store.save(label, r.content, meta)
    r.raise_for_status()
    return r.json()


def api_key() -> str:
    k = os.environ.get("ODDS_API_KEY")
    if not k:
        raise SystemExit("ODDS_API_KEY is not set")
    return k
