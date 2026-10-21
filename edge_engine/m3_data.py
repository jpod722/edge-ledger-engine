"""M3 historical data (free): ESPN play-by-play per game and the official injury report selected by the
pre-registered rule (latest report with file-name time <= decision - 15 min, decision = min(17:30 ET, tip - 90)).

    python -m edge_engine.m3_data pbp     --data DATA --cache CACHE
    python -m edge_engine.m3_data reports --data DATA --cache CACHE
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
SEASONS = ["2023-24", "2024-25", "2025-26"]
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={}"
REPORT = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{}.pdf"
KEEP = ("sequenceNumber", "period", "clock", "wallclock", "type", "text", "team", "participants", "scoringPlay",
        "scoreValue", "shootingPlay", "coordinate", "homeScore", "awayScore", "pointsAttempted")


def games(data: str, season: str) -> pd.DataFrame:
    b = pd.read_csv(Path(data) / "m1" / "boxscores" / f"{season}.csv.gz", usecols=["game_id", "date"])
    g = b.drop_duplicates("game_id").copy()
    g["tip"] = pd.to_datetime(g.date, utc=True)
    return g


def decision_time(tip_utc: pd.Timestamp) -> datetime:
    tip = tip_utc.tz_convert(ET).to_pydatetime()
    return min(tip.replace(hour=17, minute=30, second=0, microsecond=0), tip - timedelta(minutes=90))


def fetch_pbp(gid: str, s: requests.Session):
    for attempt in range(3):
        try:
            r = s.get(SUMMARY.format(gid), timeout=30)
            if r.status_code == 200:
                d = r.json()
                plays = [{k: p.get(k) for k in KEEP} for p in d.get("plays", [])]
                off = [o.get("displayName") for o in d.get("gameInfo", {}).get("officials", [])]
                return {"game_id": gid, "plays": plays, "officials": off}
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (attempt + 1))
    return {"game_id": gid, "error": True}


def pbp(data: str, cache: str):
    c = Path(cache) / "pbp"
    c.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    for season in SEASONS:
        out = c / f"{season}.jsonl.gz"
        done = set()
        if out.exists():
            with gzip.open(out, "rt") as f:
                done = {json.loads(l)["game_id"] for l in f}
        todo = [str(g) for g in games(data, season).game_id if str(g) not in done]
        with gzip.open(out, "at") as f, cf.ThreadPoolExecutor(6) as ex:
            for i, rec in enumerate(ex.map(lambda g: fetch_pbp(g, s), todo)):
                f.write(json.dumps(rec) + "\n")
                if i % 200 == 0:
                    print(season, i, len(todo), flush=True)
        print(season, "done", flush=True)


def candidates(cutoff: datetime, hours: int = 10):
    """File-name timestamps at 15-minute steps, newest first, in both naming styles the league has used."""
    t = cutoff.replace(minute=(cutoff.minute // 15) * 15, second=0, microsecond=0)
    for _ in range(hours * 4):
        h12 = t.strftime("%I")
        ap = t.strftime("%p")
        d = t.strftime("%Y-%m-%d")
        yield t, f"{d}_{h12}_{t.strftime('%M')}{ap}"
        if t.minute == 0:
            yield t, f"{d}_{h12}{ap}"
        t -= timedelta(minutes=15)


def header_time(path: Path):
    """The report's own time, printed at the top of page 1."""
    import re
    import pdfplumber
    try:
        with pdfplumber.open(path) as p:
            t = p.pages[0].extract_text() or ""
    except Exception:  # noqa: BLE001
        return None
    m = re.search(r"Injury Report:\s*(\d{2}/\d{2}/\d{2})\s*(\d{1,2}:\d{2})\s*([AP]M)", t)
    if not m:
        return None
    return datetime.strptime(f"{m.group(1)} {m.group(2)} {m.group(3)}", "%m/%d/%y %I:%M %p").replace(tzinfo=ET)


def reports(data: str, cache: str):
    c = Path(cache) / "reports"
    c.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    sel_p = Path(cache) / "report_selection_v2.jsonl"
    seen = {}
    if sel_p.exists():
        for l in open(sel_p):
            x = json.loads(l)
            seen[x["cutoff"]] = x
    rows = []
    for season in SEASONS:
        g = games(data, season)
        for gid, tip in zip(g.game_id, g.tip):
            dec = decision_time(tip)
            cutoff = dec - timedelta(minutes=15)
            rows.append((season, str(gid), dec, cutoff))
    by_cut = sorted({r[3] for r in rows})
    found = dict((k, v) for k, v in seen.items())

    def probe(cut):
        key = cut.isoformat()
        if key in found:
            return found[key]
        for t, name in candidates(cut):
            if t.date() != cut.date():
                break
            u = REPORT.format(name)
            try:
                r = s.get(u, timeout=30)
            except requests.RequestException:
                continue
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                (c / f"{name}.pdf").write_bytes(r.content)
                printed = header_time(c / f"{name}.pdf")
                if printed is None or printed > cut:
                    continue                      # printed later than the cutoff (or unreadable): keep looking earlier
                return {"cutoff": key, "name": name, "name_time": t.isoformat(), "report_time": printed.isoformat(), "url": u}
        return {"cutoff": key, "name": None, "report_time": None, "url": None}

    with open(sel_p, "a") as f, cf.ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(probe, [x for x in by_cut if x.isoformat() not in found])):
            f.write(json.dumps(res) + "\n")
            found[res["cutoff"]] = res
            if i % 50 == 0:
                print("reports", i, len(by_cut), flush=True)
    m = pd.DataFrame([{"season": se, "game_id": gid, "decision": d.isoformat(), "cutoff": cu.isoformat(),
                       **{k: found[cu.isoformat()].get(k) for k in ("name", "name_time", "report_time", "url")}} for se, gid, d, cu in rows])
    m.to_csv(Path(cache) / "report_by_game.csv", index=False)
    print("games", len(m), "with report", m.name.notna().sum(), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pbp", "reports"])
    ap.add_argument("--data", default="/tmp/edata")
    ap.add_argument("--cache", default="/home/claude/m3cache")
    a = ap.parse_args()
    {"pbp": pbp, "reports": reports}[a.cmd](a.data, a.cache)
