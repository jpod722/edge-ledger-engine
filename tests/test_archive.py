import json

from edge_engine import archive
from edge_engine.client import RawStore

PDF = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_2026-10-19_05PM.pdf"


class R:
    def __init__(self, code, body):
        self.status_code, self.content = code, body


class S:
    def __init__(self):
        self.calls = []

    def get(self, url, timeout=None, headers=None):
        self.calls.append(url)
        if url == archive.ESPN_INJ:
            return R(200, b'{"injuries": []}')
        if url == archive.INDEX_PAGES[0]:
            return R(200, f'<a href="{PDF}">5 PM</a>'.encode())
        return R(200, b"%PDF-1.4")


def test_archive_stores_once_with_timestamps(tmp_path):
    s = S()
    out = archive.run(str(tmp_path), s)
    assert out["espn"] == 200 and out["new_pdfs"] == 1
    out = archive.run(str(tmp_path), s)
    assert out["new_pdfs"] == 0 and s.calls.count(PDF) == 1          # a PDF is never fetched twice
    rows = [json.loads(l) for l in RawStore(tmp_path).manifest.read_text().splitlines()]
    assert all(r["retrieved_at"] and r["sha256"] and r["kind"] == "availability" for r in rows)
