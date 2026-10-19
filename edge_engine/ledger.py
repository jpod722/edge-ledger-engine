"""File-based ledger on the data branch. The agent copies these files verbatim into the dashboard database."""
from __future__ import annotations

import gzip
import json
from pathlib import Path


class Ledger:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.picks = self.root / "ledger" / "picks"
        self.bets = self.root / "ledger" / "bets"
        self.runs = self.root / "runs"
        self.sync = self.root / "sync"
        for d in (self.picks, self.bets, self.runs, self.sync):
            d.mkdir(parents=True, exist_ok=True)
        self.changed: list[tuple[str, str, str]] = []   # (collection, doc_id, relative file)

    def _write(self, path: Path, obj: dict, collection: str, doc_id: str):
        path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
        self.changed.append((collection, doc_id, str(path.relative_to(self.root))))

    def get_pick(self, pid: str) -> dict | None:
        p = self.picks / f"{pid}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def all_picks(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.picks.glob("*.json"))]

    def all_bets(self) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted(self.bets.glob("*.json"))]

    def upsert_pick(self, cand: dict, seen_at: str):
        """First flag wins: selection fields are never overwritten by later scans."""
        cur = self.get_pick(cand["id"])
        if cur is None:
            cand = {**cand, "first_seen": seen_at, "last_seen": seen_at, "times_seen": 1}
            self._write(self.picks / f"{cand['id']}.json", cand, "picks", cand["id"])
            return
        cur["last_seen"], cur["times_seen"] = seen_at, cur.get("times_seen", 1) + 1
        cur["last_seen_ev_quoted"] = cand.get("ev_quoted")
        self._write(self.picks / f"{cur['id']}.json", cur, "picks", cur["id"])

    def update_pick(self, pid: str, fields: dict):
        cur = self.get_pick(pid)
        if cur is None:
            return
        cur.update(fields)
        self._write(self.picks / f"{pid}.json", cur, "picks", pid)

    def write_run(self, run_id: str, summary: dict, candidates: list[dict], report_md: str):
        with gzip.open(self.runs / f"{run_id}.candidates.jsonl.gz", "wt") as f:
            for c in candidates:
                f.write(json.dumps(c, default=str) + "\n")
        (self.runs / f"{run_id}.md").write_text(report_md)
        self._write(self.runs / f"{run_id}.json", {**summary, "report_md": report_md}, "runs", run_id)
        manifest = {"run_id": run_id, "files": [{"collection": c, "doc_id": d, "file": f} for c, d, f in self.changed]}
        (self.sync / f"{run_id}.json").write_text(json.dumps(manifest, indent=1))
