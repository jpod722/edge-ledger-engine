"""Ingest NBA regular-season box scores from ESPN into compact player-game tables (free, no credits).

Output: data/m1/boxscores/<season>.csv.gz (one row per player per game, DNPs included) and
data/m1/ingest_manifest.json (retrieval times, counts, failures).

Run: python -m edge_engine.nba_ingest --data data --season 2023-24 [--probe]
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from .client import utcnow

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SEASONS = {"2023-24": (date(2023, 10, 24), date(2024, 4, 14)),
           "2024-25": (date(2024, 10, 22), date(2025, 4, 13)),
           "2025-26": (date(2025, 10, 21), date(2026, 4, 12))}
FIELDS = ["season", "game_id", "date", "team", "opponent", "home", "team_score", "opp_score", "periods",
          "player_id", "player", "starter", "dnp", "dnp_reason", "minutes", "pts", "fgm", "fga", "tpm", "tpa",
          "ftm", "fta", "oreb", "dreb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus"]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _pair(x):
    if not x or "-" not in str(x):
        return None, None
    a, b = str(x).split("-", 1)
    return _num(a), _num(b)


def parse_summary(js: dict, season: str) -> list[dict]:
    comp = (js.get("header", {}).get("competitions") or [{}])[0]
    gid = js.get("header", {}).get("id") or comp.get("id")
    gdate = comp.get("date")
    teams = {}
    for c in comp.get("competitors", []):
        teams[str(c.get("team", {}).get("id"))] = {"abbr": c.get("team", {}).get("abbreviation"),
                                                    "home": c.get("homeAway") == "home", "score": _num(c.get("score"))}
    periods = len((comp.get("competitors") or [{}])[0].get("linescores", []) or [])
    rows = []
    for tb in js.get("boxscore", {}).get("players", []):
        tid = str(tb.get("team", {}).get("id"))
        me = teams.get(tid, {})
        opp = next((v for k, v in teams.items() if k != tid), {})
        for st in tb.get("statistics", [])[:1]:
            keys = st.get("keys") or st.get("names") or []
            for a in st.get("athletes", []):
                vals = dict(zip(keys, a.get("stats", [])))
                fgm, fga = _pair(vals.get("fieldGoalsMade-fieldGoalsAttempted"))
                tpm, tpa = _pair(vals.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted"))
                ftm, fta = _pair(vals.get("freeThrowsMade-freeThrowsAttempted"))
                ath = a.get("athlete", {})
                rows.append({
                    "season": season, "game_id": gid, "date": gdate, "team": me.get("abbr"), "opponent": opp.get("abbr"),
                    "home": int(bool(me.get("home"))), "team_score": me.get("score"), "opp_score": opp.get("score"),
                    "periods": periods, "player_id": ath.get("id"), "player": ath.get("displayName"),
                    "starter": int(bool(a.get("starter"))), "dnp": int(bool(a.get("didNotPlay")) or not a.get("stats")),
                    "dnp_reason": a.get("reason") or "", "minutes": _num(vals.get("minutes")), "pts": _num(vals.get("points")),
                    "fgm": fgm, "fga": fga, "tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta,
                    "oreb": _num(vals.get("offensiveRebounds")), "dreb": _num(vals.get("defensiveRebounds")),
                    "reb": _num(vals.get("rebounds")), "ast": _num(vals.get("assists")), "stl": _num(vals.get("steals")),
                    "blk": _num(vals.get("blocks")), "tov": _num(vals.get("turnovers")), "pf": _num(vals.get("fouls")),
                    "plus_minus": _num(vals.get("plusMinus"))})
    return rows


def get(s, url, tries=3):
    for i in range(tries):
        try:
            r = s.get(url, timeout=30)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(2 * (i + 1))
    return None


def run(data: str, season: str, probe: bool = False, session=None, pause: float = 0.15):
    s = session or requests.Session()
    out = Path(data) / "m1"
    (out / "boxscores").mkdir(parents=True, exist_ok=True)
    d0, d1 = SEASONS[season]
    if probe:
        d1 = d0
    rows, games, fails = [], 0, []
    d = d0
    while d <= d1:
        sb = get(s, f"{BASE}/scoreboard?dates={d:%Y%m%d}")
        for ev in (sb or {}).get("events", []):
            if (ev.get("season") or {}).get("type") not in (2, None):
                continue
            js = get(s, f"{BASE}/summary?event={ev['id']}")
            if probe and js:
                (out / "probe_summary.json").write_text(json.dumps(js)[:400000])
            if not js:
                fails.append(ev["id"])
                continue
            rows += parse_summary(js, season)
            games += 1
            time.sleep(pause)
            if probe:
                break
        if sb is None:
            fails.append(f"scoreboard {d}")
        d += timedelta(days=1)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FIELDS)
    w.writeheader()
    w.writerows(rows)
    with gzip.open(out / "boxscores" / f"{season}{'-probe' if probe else ''}.csv.gz", "wt") as f:
        f.write(buf.getvalue())
    man_p = out / "ingest_manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    man[season + ("-probe" if probe else "")] = {"retrieved_at": utcnow().isoformat(), "games": games, "rows": len(rows),
                                                 "failures": fails[:200], "n_failures": len(fails)}
    man_p.write_text(json.dumps(man, indent=1))
    print(json.dumps({"season": season, "games": games, "rows": len(rows), "failures": len(fails)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--season", required=True, choices=list(SEASONS))
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    run(a.data, a.season, a.probe)
