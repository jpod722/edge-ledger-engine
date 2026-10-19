"""Timestamped NBA availability archive (no Odds API credits).

The reviewer's rule: a historical test may only use injury/availability information that was published
before the decision time. Nothing public archives that, so we archive it ourselves from now on. Each run
stores, with its retrieval time and SHA-256 in the raw manifest:
- ESPN's NBA injuries JSON (every run);
- every official NBA injury-report PDF linked from the league's index page that we have not stored yet.
Failures are logged and never fail the job: a missed snapshot is a gap, not a reason to stop archiving.

Run: python -m edge_engine.archive --data data
"""
from __future__ import annotations

import argparse
import json
import re
import sys

import requests

from .client import RawStore, utcnow

ESPN_INJ = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
INDEX_PAGES = ["https://official.nba.com/nba-injury-report-2026-27-season/",
               "https://official.nba.com/nba-injury-report-2025-26-season/"]
PDF_RE = re.compile(r"https://ak-static\.cms\.nba\.com/referee/injury/Injury-Report_[0-9_\-APM]+\.pdf")
UA = {"User-Agent": "Mozilla/5.0 (edge-ledger archive)"}


def stored_paths(store: RawStore) -> set[str]:
    if not store.manifest.exists():
        return set()
    return {json.loads(l).get("path") for l in store.manifest.read_text().splitlines()}


def fetch(store: RawStore, s, url: str, label: str) -> tuple[int | None, bytes]:
    try:
        r = s.get(url, timeout=30, headers=UA)
        store.save(label, r.content, {"retrieved_at": utcnow().isoformat(), "path": url, "params": {},
                                      "status": r.status_code, "credits_last": 0, "kind": "availability"})
        return r.status_code, r.content
    except requests.RequestException as e:
        print(f"fetch failed {url}: {e}", file=sys.stderr)
        return None, b""


def run(data: str, session=None) -> dict:
    store, s = RawStore(data), session or requests.Session()
    have = stored_paths(store)
    out = {"at": utcnow().isoformat(), "espn": None, "index": None, "new_pdfs": 0}
    out["espn"], _ = fetch(store, s, ESPN_INJ, "nba_injuries_espn")
    links: list[str] = []
    for page in INDEX_PAGES:
        code, body = fetch(store, s, page, "nba_injury_index")
        if code == 200:
            out["index"] = page
            links = sorted(set(PDF_RE.findall(body.decode("utf-8", "ignore"))))
            break
    for url in links:
        if url in have:
            continue
        code, _ = fetch(store, s, url, "nba_injury_report_pdf")
        out["new_pdfs"] += int(code == 200)
    print(json.dumps(out))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    run(ap.parse_args().data)
