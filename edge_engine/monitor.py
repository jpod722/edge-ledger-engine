"""Nightly NBA points-prop monitor: the prospective dataset the reviewer asked for (Oct 19, 2026).

Collects, with timestamps, for every NBA game tonight:
- availability: ESPN injuries every 60 s (diffed into status-change events) and every new official
  NBA injury-report PDF (archived; parsed offline). Each event keeps its first-seen time.
- prices: player_points from 10 books (one Odds API region, 1 credit per game per poll)
    * baseline every `baseline_min` minutes per game until tip;
    * after an ESPN status change for a team: that game at +0, +2, +5 and +10 minutes;
    * close: one poll inside the last `close_window_min` minutes before tip.
Nothing here prices, selects or bets. Everything goes to the raw store plus a monitor log, so the
frozen M1 model, book-reaction analysis and segment calibration can be scored later on exactly what
was known at each moment.

Run: python -m edge_engine.monitor --data data
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests
import yaml

from .client import CreditGuard, OddsAPI, RawStore, api_key, last_remaining_from_manifest, utcnow
from .archive import ESPN_INJ, INDEX_PAGES, PDF_RE, UA, stored_paths

ET = ZoneInfo("America/New_York")
SK = "basketball_nba"


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def injury_state(payload: dict) -> dict:
    """athlete id -> (team, player, status, date) from ESPN's injuries JSON."""
    out = {}
    for team in payload.get("injuries", []):
        for i in team.get("injuries", []):
            a = i.get("athlete", {})
            key = str(i.get("id") or a.get("displayName"))
            out[key] = (team.get("displayName", ""), a.get("displayName", ""), i.get("status", ""), i.get("date", ""))
    return out


def diff_injuries(old: dict, new: dict) -> list[dict]:
    ev = []
    for k, v in new.items():
        o = old.get(k)
        if o is None or o[2] != v[2]:
            ev.append({"team": v[0], "player": v[1], "status": v[2], "prev_status": o[2] if o else None, "espn_date": v[3]})
    for k, o in old.items():
        if k not in new:
            ev.append({"team": o[0], "player": o[1], "status": "removed", "prev_status": o[2], "espn_date": o[3]})
    return ev


class Monitor:
    def __init__(self, data: str, cfg: dict, session=None, clock=None, sleep=None):
        self.m = cfg["monitor"]
        self.store = RawStore(data)
        self.log_p = Path(data) / "monitor" / f"{utcnow().astimezone(ET):%Y-%m-%d}.jsonl"
        self.log_p.parent.mkdir(parents=True, exist_ok=True)
        self.s = session or requests.Session()
        self.api = OddsAPI(api_key(), self.store, self.m["nightly_cap_credits"], session=self.s,
                           min_remaining=self.m["min_remaining_credits"], remaining=last_remaining_from_manifest(self.store))
        self.now = clock or utcnow
        self.sleep = sleep or time.sleep
        self.books = cfg["odds_api"]["event_bookmakers"]
        self.cfg, self.data = cfg, data
        self.first_payload: dict = {}
        self.forecasts_done = False

    def log(self, kind: str, **kw):
        with open(self.log_p, "a") as f:
            f.write(json.dumps({"at": self.now().isoformat(), "kind": kind, **kw}) + "\n")

    def games_tonight(self) -> list[dict]:
        # the events list is free (0 credits)
        evs, _, ref = self.api._get(f"/sports/{SK}/events", {"dateFormat": "iso"}, f"{SK}_events", 0)
        today = self.now().astimezone(ET).date()
        out = [e for e in evs if parse_ts(e["commence_time"]).astimezone(ET).date() == today
               and parse_ts(e["commence_time"]) > self.now()]
        self.log("games", n=len(out), events=[{"id": e["id"], "home": e["home_team"], "away": e["away_team"],
                                                "commence_time": e["commence_time"]} for e in out], ref=ref)
        return out

    def poll(self, e: dict, reason: str):
        try:
            payload, retrieved, ref = self.api.event_odds(SK, e["id"], self.m["market"], self.books)
            if e["id"] not in self.first_payload:
                self.first_payload[e["id"]] = (payload, retrieved)
            self.log("poll", event_id=e["id"], reason=reason, retrieved_at=retrieved, ref=ref,
                     minutes_to_tip=round((parse_ts(e["commence_time"]) - self.now()).total_seconds() / 60, 2))
            return True
        except CreditGuard as g:
            self.log("credit_guard", event_id=e["id"], reason=reason, detail=str(g))
            return False
        except RuntimeError as err:
            self.log("poll_error", event_id=e["id"], reason=reason, detail=str(err)[:200])
            return True

    def prepare_box(self):
        """Current-season box scores through yesterday (ET), free from ESPN. Failure never stops the monitor."""
        try:
            from .nba_ingest import incremental
            yday = (self.now().astimezone(ET) - timedelta(days=1)).date()
            n = incremental(self.data, self.m["season"], yday, session=self.s)
            self.log("box_ingest", season=self.m["season"], games_added=n, through=str(yday))
        except Exception as err:  # noqa: BLE001
            self.log("box_ingest_error", detail=str(err)[:300])

    def forecast(self, games):
        """Frozen M1 / M1.1 forecasts for every contract in each game's first snapshot (the decision time)."""
        self.forecasts_done = True
        try:
            import pandas as pd
            from . import m1_live
            frozen = json.loads(m1_live.FROZEN.read_text())
            path = Path(self.data) / "m1" / "boxscores" / f"{self.m['season']}.csv.gz"
            box = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["dnp", "minutes", "pts", "date", "player_id", "player", "team"])
            recs = m1_live.forecasts(box, [self.first_payload[e["id"]] for e in games if e["id"] in self.first_payload],
                                     self.cfg, frozen)
            day = f"{self.now().astimezone(ET):%Y-%m-%d}"
            out = m1_live.write_and_push(self.data, day, recs)
            self.log("forecasts", n=len(recs), eligible=sum(r["eligible"] for r in recs), file=out,
                     frozen_at=frozen["frozen_at"])
        except Exception as err:  # noqa: BLE001
            self.log("forecast_error", detail=str(err)[:300])

    def fetch_free(self, url: str, label: str):
        try:
            r = self.s.get(url, timeout=30, headers=UA)
            ref = self.store.save(label, r.content, {"retrieved_at": self.now().isoformat(), "path": url, "params": {},
                                                     "status": r.status_code, "credits_last": 0, "kind": "availability"})
            return r, ref
        except requests.RequestException as err:
            self.log("fetch_error", url=url, detail=str(err)[:200])
            return None, None

    def run(self):
        start = self.now()
        hard_stop = start + timedelta(minutes=self.m["max_runtime_min"])
        games = self.games_tonight()
        if games:
            self.prepare_box()
        if not games:
            self.log("done", reason="no games")
            return
        by_team = {}
        for e in games:
            by_team.setdefault(e["home_team"], []).append(e)
            by_team.setdefault(e["away_team"], []).append(e)
        last_tip = max(parse_ts(e["commence_time"]) for e in games)
        next_base = {e["id"]: start for e in games}
        closed, queue, last_poll = set(), [], {}
        inj_state, next_inj, next_pdf = None, start, start
        have_pdfs = stored_paths(self.store)
        stop_credits = False
        while self.now() < min(hard_stop, last_tip + timedelta(minutes=self.m["after_last_tip_min"])):
            now = self.now()
            if now >= next_inj:
                r, ref = self.fetch_free(ESPN_INJ, "nba_injuries_espn")
                if r is not None and r.status_code == 200:
                    st = injury_state(r.json())
                    if inj_state is not None:
                        for ch in diff_injuries(inj_state, st):
                            self.log("injury_change", first_seen=now.isoformat(), ref=ref, **ch)
                            for e in by_team.get(ch["team"], []):
                                if parse_ts(e["commence_time"]) > now:
                                    for dm in self.m["reaction_polls_min"]:
                                        queue.append((now + timedelta(minutes=dm), e, f"inj+{dm}:{ch['player']}:{ch['status']}"))
                    inj_state = st
                next_inj = now + timedelta(seconds=self.m["injury_poll_s"])
            if now >= next_pdf:
                r, _ = self.fetch_free(INDEX_PAGES[0], "nba_injury_index")
                if r is not None and r.status_code == 200:
                    for url in sorted(set(PDF_RE.findall(r.content.decode("utf-8", "ignore")))):
                        if url not in have_pdfs:
                            pr, pref = self.fetch_free(url, "nba_injury_report_pdf")
                            if pr is not None and pr.status_code == 200:
                                have_pdfs.add(url)
                                self.log("official_report", first_seen=now.isoformat(), url=url, ref=pref)
                next_pdf = now + timedelta(seconds=self.m["official_poll_s"])
            due = []
            for e in games:
                tip = parse_ts(e["commence_time"])
                if tip <= now:
                    continue
                if tip - now <= timedelta(minutes=self.m["close_window_min"]) and e["id"] not in closed:
                    due.append((e, "close"))
                    closed.add(e["id"])
                elif now >= next_base[e["id"]]:
                    due.append((e, "baseline"))
                    next_base[e["id"]] = now + timedelta(minutes=self.m["baseline_min"])
            keep = []
            for t, e, why in queue:
                if t <= now:
                    if parse_ts(e["commence_time"]) > now:
                        due.append((e, why))
                else:
                    keep.append((t, e, why))
            queue = keep
            grouped: dict = {}
            for e, why in due:
                grouped.setdefault(e["id"], (e, []))[1].append(why)
            for eid, (e, whys) in grouped.items():
                if stop_credits:
                    break
                why = "|".join(whys)
                # reaction polls within 50 s of the last poll of this game are already answered by it
                if "close" not in whys and last_poll.get(eid) and now - last_poll[eid] < timedelta(seconds=50):
                    self.log("poll_merged", event_id=eid, reason=why)
                    continue
                if not self.poll(e, why):
                    stop_credits = True
                last_poll[eid] = now
            if not self.forecasts_done and all(e["id"] in self.first_payload for e in games
                                                   if parse_ts(e["commence_time"]) > now):
                self.forecast(games)
            self.sleep(self.m["loop_s"])
        self.log("done", reason="finished", credits=self.api.used, remaining=self.api.remaining)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--config")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config or Path(__file__).resolve().parent.parent / "config" / "rules.yaml"))
    Monitor(a.data, cfg).run()
