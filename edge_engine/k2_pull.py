"""K2 data pull (config/prereg/k2-replay.yaml). Paper only: Kalshi public, unauthenticated GET endpoints and ESPN.

    python -m edge_engine.k2_pull --cache ~/k2-cache markets|espn|tape|all

Everything is cached gzipped OUTSIDE the repo (never committed); a cached request is never repeated. No order endpoint
is ever called and no API key is read. Zero Odds API credits.

Stages (each resumable):
  markets  every KXNBAPTS market from /historical/markets and /markets?status=settled -> markets.csv.gz
  espn     ESPN scoreboard for every Kalshi event date, then the summary (play-by-play) of every matched game, reduced to
           schedule + game-state fields only (games/{id}.json.gz). Box-score STATS are never read (2025-26 holdout).
  tape     per market: minute candlesticks (bid/ask) and trades for the game window -> candles/, trades/
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

API = "https://api.elections.kalshi.com/trade-api/v2"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
SERIES = "KXNBAPTS"
ET = ZoneInfo("America/New_York")
FIRST_DATE, LAST_DATE = "2025-11-18", "2026-11-15"          # the background look's window
# Kalshi -> ESPN team codes (same map as edge_engine/k1.py; copied so K2 never imports a frozen study module)
KALSHI_TO_ESPN = {"SAS": "SA", "NYK": "NY", "GSW": "GS", "NOP": "NO", "UTA": "UTAH", "WAS": "WSH", "PHO": "PHX",
                  "BRK": "BKN", "CHO": "CHA"}
MON = {m: i + 1 for i, m in enumerate("JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split())}
CANDLE_LOOKBACK_H, CANDLE_AFTER_H, TRADE_LEAD_MIN = 48, 5, 30


def norm_name(s: str) -> str:
    """Same normalisation as m1_live.norm_name: ascii, lower, no punctuation or generational suffix."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[^a-z ]", "", s.replace("-", " "))
    return " ".join(w for w in s.split() if w not in {"jr", "sr", "ii", "iii", "iv"})


def parse_event(event_ticker: str):
    """'KXNBAPTS-26NOV14DALCLE' -> ('2026-11-14', 'DAL', 'CLE') (ET date, away, home); None if malformed."""
    m = re.fullmatch(r"[A-Z]+-(\d\d)([A-Z]{3})(\d\d)([A-Z]{3})([A-Z]{3})", event_ticker or "")
    if not m or m.group(2) not in MON:
        return None
    return f"20{m.group(1)}-{MON[m.group(2)]:02d}-{m.group(3)}", m.group(4), m.group(5)


def espn_code(k: str) -> str:
    return KALSHI_TO_ESPN.get(k, k)


def ts(s) -> datetime:
    return datetime.fromisoformat(str(s).replace("Z", "+00:00"))


# ---------------------------------------------------------------- http + cache
class Http:
    """Thread-safe GET with a global request-rate cap and retry on 429 / 5xx / network errors."""
    def __init__(self, rps: float = 12.0):
        self.gap, self.lock, self.next, self.local, self.n = 1.0 / rps, threading.Lock(), 0.0, threading.local(), 0

    def _s(self):
        if not hasattr(self.local, "s"):
            self.local.s = requests.Session()
            self.local.s.headers["User-Agent"] = "edge-ledger-research/1.0"
        return self.local.s

    def get(self, url: str, params: dict | None = None):
        for attempt in range(12):
            with self.lock:
                now = time.monotonic()
                wait = self.next - now
                self.next = max(now, self.next) + self.gap
                self.n += 1
            if wait > 0:
                time.sleep(wait)
            try:
                r = self._s().get(url, params=params, timeout=40)
            except requests.RequestException:
                time.sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                with self.lock:                       # back everyone off
                    self.next = max(self.next, time.monotonic() + min(60, 2 ** attempt))
                continue
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"gave up: {url} {params}")


def load(p: Path):
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def save(p: Path, obj) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp{os.getpid()}{threading.get_ident()}")
    with gzip.open(tmp, "wt") as fh:
        json.dump(obj, fh, separators=(",", ":"))
    tmp.replace(p)


# ---------------------------------------------------------------- markets
MKT_COLS = ["ticker", "event_ticker", "player", "floor_strike", "strike_type", "result", "created_time", "open_time",
            "close_time", "volume", "src"]


def _pages(http: Http, cache: Path, name: str, url: str, params: dict):
    """Cursor-paginate a market listing; every page cached under markets/{name}/, resumable."""
    d = cache / "markets" / name
    d.mkdir(parents=True, exist_ok=True)
    i, cursor = 0, None
    while True:
        p = d / f"{i:05d}.json.gz"
        if p.exists():
            page = load(p)
        else:
            q = dict(params, limit=1000)
            if cursor:
                q["cursor"] = cursor
            page = http.get(url, q)
            save(p, page)
        yield page.get("markets", [])
        cursor = page.get("cursor")
        if not cursor:
            (d / "DONE").write_text(str(i + 1))
            return
        i += 1


def pull_markets(http: Http, cache: Path) -> pd.DataFrame:
    rows = []
    for name, url, params in [("historical", f"{API}/historical/markets", {"series_ticker": SERIES}),
                              ("settled", f"{API}/markets", {"series_ticker": SERIES, "status": "settled"})]:
        for mk in _pages(http, cache, name, url, params):
            for m in mk:
                sub = m.get("yes_sub_title") or ""
                vol = m.get("volume_fp", m.get("volume"))
                rows.append({"ticker": m["ticker"], "event_ticker": m.get("event_ticker"),
                             "player": sub.rsplit(":", 1)[0].strip() if ":" in sub else None,
                             "floor_strike": m.get("floor_strike"), "strike_type": m.get("strike_type"),
                             "result": m.get("result"), "created_time": m.get("created_time"), "open_time": m.get("open_time"),
                             "close_time": m.get("close_time"), "volume": float(vol) if vol not in (None, "") else 0.0,
                             "src": name})
        print(f"markets/{name}: {sum(r['src'] == name for r in rows)}", flush=True)
    M = pd.DataFrame(rows, columns=MKT_COLS)
    # a market settled after the archive cutoff appears only in /markets; if both list it, the live copy wins
    M = M.sort_values("src", key=lambda s: s.map({"settled": 0, "historical": 1})).drop_duplicates("ticker")
    ev = M.event_ticker.map(parse_event)
    M["date"] = ev.map(lambda x: x[0] if x else None)
    M["away"] = ev.map(lambda x: x[1] if x else None)
    M["home"] = ev.map(lambda x: x[2] if x else None)
    M.to_csv(cache / "markets.csv.gz", index=False)
    return M


def markets(cache: Path) -> pd.DataFrame:
    return pd.read_csv(cache / "markets.csv.gz")


# ---------------------------------------------------------------- ESPN
HOUR = 3600


def fix_wallclock(plays: list[dict], sched_tip: str | None) -> list[int]:
    """ESPN stamps some plays one hour off on daylight-saving change days (found 2026-11-16: Mar 8 2026 games had
    Q1-Q3 stamped +1 h and Q4 right; Nov 1 2026 games -1 h, one of them until mid-Q3). Split the plays at jumps of
    50-70 min (either sign), give the first segment the whole-hour offset that puts the tip within 45 min of the
    schedule, and each later segment the offset that cancels its jump. Mutates wallclock in place; returns the
    per-segment offsets in hours (all zero = unchanged). The DST jumps seen were 59-62 min; 50-70 keeps an ordinary
    long halftime (one game had a 45-min gap) from being treated as one."""
    wc = [i for i, p in enumerate(plays) if p.get("wallclock")]
    if not wc:
        return []
    t = [ts(plays[i]["wallclock"]).timestamp() for i in wc]
    cuts = [j for j in range(1, len(t)) if 50 * 60 <= abs(t[j] - t[j - 1]) <= 70 * 60]
    delay = (t[0] - ts(sched_tip).timestamp()) if sched_tip else 0
    off = [-1 if delay > 45 * 60 else 1 if delay < -30 * 60 else 0]
    for j in cuts:
        off.append(off[-1] - round((t[j] - t[j - 1]) / HOUR))
    if not any(off):
        return off
    bounds = [0] + cuts + [len(t)]
    for k in range(len(off)):
        for j in range(bounds[k], bounds[k + 1]):
            p = plays[wc[j]]
            p["wallclock"] = (datetime.fromtimestamp(t[j] + off[k] * HOUR, timezone.utc)
                              .isoformat().replace("+00:00", "Z"))
    return off


def reduce_summary(s: dict, game_id: str, sched_tip: str | None = None) -> dict:
    """Keep only schedule + game-state fields. From the box score we read athlete id, name and team ONLY (identity, to map
    Kalshi names); the stats arrays are never touched (2025-26 box scores are a sealed holdout)."""
    plays = s.get("plays") or []
    offsets = fix_wallclock(plays, sched_tip)
    roster = {}
    for tm in (s.get("boxscore") or {}).get("players") or []:
        abbr = (tm.get("team") or {}).get("abbreviation")
        for grp in tm.get("statistics") or []:
            for a in grp.get("athletes") or []:
                ath = a.get("athlete") or {}
                if ath.get("id"):
                    roster[str(ath["id"])] = [ath.get("displayName"), abbr]
    pe, scoring, first, end = {}, [], None, None
    for p in plays:
        wc = p.get("wallclock")
        if not wc:
            continue
        per = int((p.get("period") or {}).get("number") or 0)
        if first is None:
            first = wc
        typ = (p.get("type") or {}).get("text", "")
        if typ == "End Period":
            pe[str(per)] = wc
        if typ == "End Game":
            end = wc
        if p.get("scoringPlay") and p.get("participants"):
            aid = str(((p["participants"][0] or {}).get("athlete") or {}).get("id") or "")
            if aid:
                scoring.append([wc, aid, int(p.get("scoreValue") or 0), per])
    last = max((p["wallclock"] for p in plays if p.get("wallclock")), default=None)
    return {"game_id": game_id, "n_plays": len(plays), "n_wallclock": sum(1 for p in plays if p.get("wallclock")),
            "wallclock_available": s.get("wallclockAvailable"), "wallclock_offsets_h": offsets,
            "actual_tip": first, "period_end": pe,
            "game_end": end or last, "roster": roster, "scoring": scoring}


def pull_espn(http: Http, cache: Path, M: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(d for d in M.date.dropna().unique() if FIRST_DATE <= d <= LAST_DATE)
    games = []
    for d in dates:
        p = cache / "espn" / "scoreboard" / f"{d}.json.gz"
        if not p.exists():
            save(p, http.get(f"{ESPN}/scoreboard", {"dates": d.replace("-", "")}))
        for e in load(p).get("events", []):
            c = e["competitions"][0]
            side = {t["homeAway"]: t["team"]["abbreviation"] for t in c["competitors"]}
            games.append({"date": d, "game_id": str(e["id"]), "sched_tip": e["date"], "home_espn": side.get("home"),
                          "away_espn": side.get("away"), "status": e["status"]["type"]["name"],
                          "season_type": (e.get("season") or {}).get("type")})
    G = pd.DataFrame(games).drop_duplicates("game_id")
    G.to_csv(cache / "espn_games.csv.gz", index=False)
    # match Kalshi events to ESPN games by ET date + both teams
    E = M[["event_ticker", "date", "away", "home"]].drop_duplicates("event_ticker").dropna()
    E = E[(E.date >= FIRST_DATE) & (E.date <= LAST_DATE)]
    key = {(g.date, g.away_espn, g.home_espn): g.game_id for g in G.itertuples()}
    E["game_id"] = [key.get((r.date, espn_code(r.away), espn_code(r.home))) for r in E.itertuples()]
    E.to_csv(cache / "event_games.csv.gz", index=False)
    todo = sorted(set(E.game_id.dropna()))
    sched = dict(zip(G.game_id, G.sched_tip))

    def one(gid):
        p = cache / "espn" / "games" / f"{gid}.json.gz"
        if p.exists() and "wallclock_offsets_h" in load(p):
            return
        raw = cache / "espn" / "summary" / f"{gid}.json.gz"   # raw summary kept (cache only) so a fix never re-fetches
        if raw.exists():
            s = load(raw)
        else:
            s = http.get(f"{ESPN}/summary", {"event": gid}) or {}
            save(raw, {"plays": s.get("plays"), "wallclockAvailable": s.get("wallclockAvailable"),
                       "boxscore": {"players": [{"team": t.get("team"),
                                                 "statistics": [{"athletes": [{"athlete": a.get("athlete")}
                                                                              for a in g.get("athletes") or []]}
                                                                for g in t.get("statistics") or []]}
                                                for t in (s.get("boxscore") or {}).get("players") or []]}})
        save(p, reduce_summary(load(raw), gid, sched.get(gid)))

    with ThreadPoolExecutor(6) as ex:
        for i, _ in enumerate(as_completed([ex.submit(one, g) for g in todo])):
            if i % 200 == 0:
                print(f"espn summaries {i}/{len(todo)}", flush=True)
    print(f"events {len(E)}, matched {E.game_id.notna().sum()}, games {len(todo)}", flush=True)
    return E


# ---------------------------------------------------------------- tape
def compact_candles(c: list[dict]) -> list[list]:
    """[end_ts, yes_bid_close, yes_ask_close, volume]; recent data uses *_dollars / volume_fp, archived data close / volume."""
    out = []
    for x in c:
        yb, ya = x.get("yes_bid") or {}, x.get("yes_ask") or {}
        b = yb.get("close_dollars", yb.get("close"))
        a = ya.get("close_dollars", ya.get("close"))
        v = x.get("volume_fp", x.get("volume"))
        out.append([int(x["end_period_ts"]), None if b is None else float(b), None if a is None else float(a),
                    float(v or 0)])
    return out


def compact_trades(t: list[dict]) -> list[list]:
    """[created_ts (float s), yes_price, count, taker_side]"""
    return [[ts(x["created_time"]).timestamp(), float(x["yes_price_dollars"]), float(x["count_fp"]), x["taker_side"]]
            for x in t]


def tape_one(http: Http, cache: Path, m: dict, sched_tip: datetime) -> str:
    t, hist = m["ticker"], m["src"] == "historical"
    close = ts(m["close_time"])
    opened = ts(m["open_time"])
    cp = cache / "candles" / f"{t}.json.gz"
    if not cp.exists():
        start = max(opened, sched_tip - timedelta(hours=CANDLE_LOOKBACK_H))
        end = min(close + timedelta(minutes=1), sched_tip + timedelta(hours=CANDLE_AFTER_H))
        if end <= start:
            save(cp, [])
        else:
            url = f"{API}/historical/markets/{t}/candlesticks" if hist else f"{API}/series/{SERIES}/markets/{t}/candlesticks"
            r = http.get(url, {"start_ts": int(start.timestamp()) // 60 * 60, "end_ts": int(end.timestamp()) // 60 * 60 + 60,
                               "period_interval": 1})
            save(cp, compact_candles((r or {}).get("candlesticks", [])))
    tp = cache / "trades" / f"{t}.json.gz"
    if not tp.exists():
        out = []
        if m["volume"] > 0:
            url = f"{API}/historical/trades" if hist else f"{API}/markets/trades"
            q = {"ticker": t, "limit": 1000, "min_ts": int((sched_tip - timedelta(minutes=TRADE_LEAD_MIN)).timestamp()),
                 "max_ts": int(close.timestamp()) + 60}
            while True:
                r = http.get(url, q) or {}
                out += compact_trades(r.get("trades", []))
                if not r.get("cursor"):
                    break
                q["cursor"] = r["cursor"]
        save(tp, out)
    return t


def pull_tape(http: Http, cache: Path, M: pd.DataFrame, workers: int = 10) -> None:
    E = pd.read_csv(cache / "event_games.csv.gz", dtype={"game_id": str})
    G = pd.read_csv(cache / "espn_games.csv.gz", dtype={"game_id": str})
    tip = dict(zip(G.game_id, G.sched_tip))
    eg = dict(zip(E.event_ticker, E.game_id))
    W = M[M.result.isin(["yes", "no"]) & M.event_ticker.map(lambda e: isinstance(eg.get(e), str))].copy()
    todo = [r for r in W.to_dict("records")
            if not ((cache / "candles" / f"{r['ticker']}.json.gz").exists() and (cache / "trades" / f"{r['ticker']}.json.gz").exists())]
    print(f"tape: {len(W)} markets, {len(todo)} to pull", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        fut = [ex.submit(tape_one, http, cache, r, ts(tip[eg[r["event_ticker"]]])) for r in todo]
        for i, f in enumerate(as_completed(fut)):
            f.result()
            if i % 500 == 0:
                print(f"tape {i}/{len(todo)} requests={http.n} {time.time() - t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["markets", "espn", "tape", "all"])
    ap.add_argument("--cache", default=os.path.expanduser("~/k2-cache"))
    ap.add_argument("--rps", type=float, default=12.0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    cache, http = Path(a.cache), Http(a.rps)
    cache.mkdir(parents=True, exist_ok=True)
    M = pull_markets(http, cache) if a.stage in ("markets", "all") or not (cache / "markets.csv.gz").exists() else markets(cache)
    if a.stage in ("espn", "all"):
        pull_espn(http, cache, M)
    if a.stage in ("tape", "all"):
        pull_tape(http, cache, M, a.workers)
    print(f"done; requests this run: {http.n}", flush=True)


if __name__ == "__main__":
    main()
