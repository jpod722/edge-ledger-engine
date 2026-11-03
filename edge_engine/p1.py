"""Parallel track, Step 1 — thesis test (config/prereg/p1-thesis.yaml).

Do DraftKings NBA points lines under-react (or over-react) to fresh teammate absences?

    python -m edge_engine.p1 sample  --data DATA            # writes config/prereg/p1-sample.json (no credits)
    python -m edge_engine.cli --data DATA p1-pull           # buys the lines (GitHub; hard stop 3,900 credits)
    python -m edge_engine.p1 analyze --data DATA --cache M3CACHE

Code, seed and game list are committed before the first purchase. The analysis runs once; a stop is reported like a
continue.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "config" / "prereg" / "p1-sample.json"
ET = ZoneInfo("America/New_York")
SK = "basketball_nba"
SEED = 20261103
N_GAMES = 380
HARD_STOP = 3900
FAIL_CLOSED_AFTER = 40
COVERAGE_MIN = 0.90
STALE_MIN = 30
K_STD = 7.2
THRESH = 0.094
BOOT, BOOT_SEED = 10000, 20261103
ALLSTAR = {"CAN", "CHK", "KEN", "SHQ"}
FULL = {"ATL": "Atlanta Hawks", "BKN": "Brooklyn Nets", "BOS": "Boston Celtics", "CHA": "Charlotte Hornets",
        "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers", "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets",
        "DET": "Detroit Pistons", "GS": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies", "MIA": "Miami Heat",
        "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves", "NO": "New Orleans Pelicans", "NY": "New York Knicks",
        "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SA": "San Antonio Spurs", "SAC": "Sacramento Kings", "TOR": "Toronto Raptors",
        "UTAH": "Utah Jazz", "WSH": "Washington Wizards"}
REPORT_ABBR = {"GS": "GSW", "NY": "NYK", "NO": "NOP", "SA": "SAS", "UTAH": "UTA", "WSH": "WAS"}


def ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def decision(tip):
    t = tip.astimezone(ET)
    return min(t.replace(hour=17, minute=30, second=0, microsecond=0), t - timedelta(minutes=90))


# ------------------------------------------------------------------ sample (free)
def sample(data: str):
    import pandas as pd
    b = pd.read_csv(Path(data) / "m1" / "boxscores" / "2024-25.csv.gz", usecols=["game_id", "date", "team", "home"])
    b = b[~b.team.isin(ALLSTAR)]
    g = {}
    for gid, d in b.groupby("game_id"):
        if len(d.team.unique()) != 2 or not d.team.isin(FULL).all():
            continue
        home = d[d.home == True].team  # noqa: E712
        away = d[d.home == False].team  # noqa: E712
        if len(home) == 0 or len(away) == 0:
            continue
        tip = pd.Timestamp(d.date.iloc[0]).tz_convert("UTC")
        g[str(gid)] = {"game_id": str(gid), "home": home.iloc[0], "away": away.iloc[0], "tip": tip.strftime("%Y-%m-%dT%H:%M:%SZ"),
                       "month": tip.tz_convert(ET).strftime("%Y-%m")}
    by_m = defaultdict(list)
    for x in g.values():
        by_m[x["month"]].append(x["game_id"])
    tot = sum(len(v) for v in by_m.values())
    quota = {m: N_GAMES * len(v) / tot for m, v in by_m.items()}
    alloc = {m: int(q) for m, q in quota.items()}
    for m in sorted(quota, key=lambda m: -(quota[m] - alloc[m]))[: N_GAMES - sum(alloc.values())]:
        alloc[m] += 1
    rng = random.Random(SEED)
    pick = []
    for m in sorted(by_m):
        pick += rng.sample(sorted(by_m[m]), alloc[m])
    games = sorted((g[i] for i in pick), key=lambda x: x["tip"])
    for x in games:
        x["decision_at"] = decision(ts(x["tip"])).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
    # one historical events lookup per ISO week, at noon ET of the week's first sampled date
    weeks = defaultdict(list)
    for x in games:
        wk = ts(x["tip"]).astimezone(ET).strftime("%G-W%V")
        x["week"] = wk
        weeks[wk].append(x)
    lookups = {wk: (min(ts(x["tip"]) for x in v).astimezone(ET).replace(hour=12, minute=0, second=0, microsecond=0)
                    .astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")) for wk, v in weeks.items()}
    out = {"seed": SEED, "n_games": len(games), "allocation": alloc, "population": {m: len(v) for m, v in by_m.items()},
           "week_lookups": lookups, "credits_planned": len(lookups) + 10 * len(games), "hard_stop": HARD_STOP,
           "games": games}
    SAMPLE.write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ pull (credits)
def pull(api, data: str, books: str):
    """Buys one player_points snapshot per sampled game at the decision time. No retries. Fail closed after the
    first 40 games if fewer than 90% have a fresh two-sided DraftKings player_points market."""
    from .client import CreditGuard
    plan = json.loads(SAMPLE.read_text())
    out_p = Path(data) / "p1" / "quotes.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out = {"sample": str(SAMPLE.relative_to(ROOT)), "games": [], "lookups": {}, "stopped": None}

    def save():
        out["credits_used"], out["remaining_after"] = api.used, api.remaining
        out_p.write_text(json.dumps(out, indent=1))

    idx = {}

    def lookup(at, key):
        try:
            evs, _, ref = api._get(f"/historical/sports/{SK}/events", {"date": at, "dateFormat": "iso"}, f"{SK}_hist_events", 1)
        except (CreditGuard, RuntimeError) as err:
            out["lookups"][key] = {"at": at, "error": str(err)[:200]}
            return False
        out["lookups"][key] = {"at": at, "ref": ref, "n": len(evs.get("data", []))}
        for e in evs.get("data", []):
            idx.setdefault((e["home_team"], e["away_team"]), []).append(e)
        return True

    done_weeks = set()
    for i, g in enumerate(plan["games"]):
        row = {k: g[k] for k in ("game_id", "home", "away", "tip", "decision_at")}
        if g["week"] not in done_weeks:
            done_weeks.add(g["week"])
            lookup(plan["week_lookups"][g["week"]], g["week"])

        def find():
            for e in idx.get((FULL[g["home"]], FULL[g["away"]]), []):
                if abs((ts(e["commence_time"]) - ts(g["tip"])).total_seconds()) <= 3 * 3600:
                    return e
            return None
        e = find()
        if e is None:
            # per-date fallback, only if it cannot push the remaining planned purchases past the hard stop
            left = sum(1 for x in plan["games"][i:]) * 10
            if api.used + 1 + left <= HARD_STOP:
                at = ts(g["tip"]).astimezone(ET).replace(hour=12, minute=0, second=0, microsecond=0).astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%dT%H:%M:%SZ")
                lookup(at, f"date:{at[:10]}:{g['game_id']}")
                e = find()
        if e is None:
            row["status"] = "unmatched"
            out["games"].append(row)
            continue
        row["event_id"] = e["id"]
        try:
            payload, _, ref = api.historical_event_odds(SK, e["id"], g["decision_at"], "player_points", books, kind="p1")
        except CreditGuard as err:
            row["status"] = "credit_guard"
            out["games"].append(row)
            out["stopped"] = f"hard stop: {str(err)[:200]}"
            save()
            return out
        except RuntimeError as err:
            row.update(status="error", error=str(err)[:200])
            out["games"].append(row)
            continue
        row.update(status="ok", ref=ref, snapshot=payload.get("timestamp"))
        dk = None
        for b in payload.get("data", {}).get("bookmakers", []):
            if b["key"] == "draftkings":
                for m in b.get("markets", []):
                    if m["key"] == "player_points":
                        dk = m
        dec = ts(g["decision_at"])
        row["dk_ok"] = bool(dk and dk.get("outcomes") and ts(dk["last_update"]) >= dec - timedelta(minutes=STALE_MIN))
        out["games"].append(row)
        if len(out["games"]) == FAIL_CLOSED_AFTER:
            cov = sum(x.get("dk_ok", False) for x in out["games"]) / FAIL_CLOSED_AFTER
            out["coverage_first_40"] = cov
            if cov < COVERAGE_MIN:
                out["stopped"] = f"fail closed: DraftKings coverage {cov:.2f} < {COVERAGE_MIN} after {FAIL_CLOSED_AFTER} games"
                save()
                return out
        if i % 20 == 0:
            save()
    save()
    return out


# ------------------------------------------------------------------ analysis (free, once)
def _norm(s):
    from .injury_pdf import norm_name
    return norm_name(str(s))


def features(data: str, cache: str):
    """Per player-game for sampled games: Baseline B minutes projection (frozen h, n0, role priors), 5+ prior games,
    official Out list (report printed <= decision - 15), fresh/stale vacated minutes of absentees projected 15+."""
    import numpy as np
    import pandas as pd
    fz = json.loads((ROOT / "config" / "m3_baseline_frozen.json").read_text())
    C, P = fz["constants"], fz["priors"]
    b = pd.read_csv(Path(data) / "m1" / "boxscores" / "2024-25.csv.gz")
    b = b[b.player.notna() & ~b.team.isin(ALLSTAR)].copy()
    b["tip"] = pd.to_datetime(b.date, utc=True)
    b["key"] = b.player.map(_norm)
    b["played"] = (b.dnp.fillna(0).astype(int) == 0) & (b.minutes.fillna(0) > 0)
    b["starter"] = b.starter.fillna(0).astype(int).astype(bool)
    b["gid"] = b.game_id.astype(str)
    team_games = {t: list(d.sort_values("tip").drop_duplicates("gid").gid) for t, d in b.groupby("team")}
    rep = pd.read_csv(Path(cache) / "report_rows.csv")
    sel = pd.read_csv(Path(cache) / "report_by_game.csv")
    sel = sel[sel.season == "2024-25"].assign(gid=lambda d: d.game_id.astype(str)).set_index("gid")
    outs = defaultdict(set)
    for r in rep[rep.status == "Out"].itertuples():
        outs[(r.report, r.date, r.abbr)].add(r.key)
    plan = json.loads(SAMPLE.read_text())
    want = {g["game_id"]: g for g in plan["games"]}
    by_player = {k: d.sort_values("tip") for k, d in b.groupby("key")}
    played_by_game = {gid: set(d[d.played].key) for gid, d in b.groupby("gid")}

    def proj(key, before):
        h = by_player.get(key)
        if h is None:
            return None, 0
        h = h[(h.tip < before) & h.played]
        n = len(h)
        if n == 0:
            return None, 0
        w = 0.5 ** ((n - np.arange(n)) / C["h"])
        role = "starter" if h.starter.values[-5:].sum() >= 3 else "bench"
        return (w @ h.minutes.values + C["n0_minutes"] * P["min"][role]) / (w.sum() + C["n0_minutes"]), n

    rows = []
    for gid, g in want.items():
        tip = pd.Timestamp(g["tip"])
        dec = decision(ts(g["tip"]))
        gd = b[b.gid == gid]
        report = sel.loc[gid, "name"] if gid in sel.index else None
        gdate = tip.tz_convert(ET).strftime("%m/%d/%Y")
        for team in (g["home"], g["away"]):
            out_keys = outs.get((report, gdate, REPORT_ABBR.get(team, team)), set()) if isinstance(report, str) else set()
            tg = team_games[team]
            pos = tg.index(gid)
            prev = tg[max(0, pos - 5):pos]
            fresh = stale = 0.0
            for a in out_keys:
                m, n = proj(a, tip)
                if m is None or m < 15:
                    continue
                last = [j for j, pg in enumerate(prev) if a in played_by_game.get(pg, set())]
                if not last:
                    continue
                if last[-1] == len(prev) - 1:
                    fresh += m
                else:
                    stale += m
            for r in gd[gd.team == team].itertuples():
                m, n = proj(r.key, tip)
                rows.append({"game_id": gid, "team": team, "key": r.key, "player": r.player, "played": bool(r.played),
                             "pts": float(r.pts) if r.played else None, "m_hat": m, "n_prior": n,
                             "listed_out": r.key in out_keys, "report": report if isinstance(report, str) else None,
                             "fresh_vac": fresh, "stale_vac": stale, "decision": dec.isoformat()})
    return pd.DataFrame(rows)


def dk_lines(data: str):
    import pandas as pd
    q = json.loads((Path(data) / "p1" / "quotes.json").read_text())
    rows = []
    for g in q["games"]:
        if g.get("status") != "ok":
            continue
        p = json.load(gzip.open(Path(data) / g["ref"]))
        dec = ts(g["decision_at"])
        for b in p.get("data", {}).get("bookmakers", []):
            if b["key"] != "draftkings":
                continue
            for m in b.get("markets", []):
                if m["key"] != "player_points" or ts(m["last_update"]) < dec - timedelta(minutes=STALE_MIN):
                    continue
                two = defaultdict(dict)
                for o in m["outcomes"]:
                    two[(o["description"], o["point"])][o["name"]] = o["price"]
                per_player = defaultdict(list)
                for (pl, L), pr in two.items():
                    if "Over" in pr and "Under" in pr:
                        per_player[pl].append((L, pr))
                for pl, lst in per_player.items():
                    if len(lst) != 1:          # DK main line only: a player with several lines is ambiguous -> dropped
                        continue
                    L, pr = lst[0]
                    io, iu = 1 / pr["Over"], 1 / pr["Under"]
                    rows.append({"game_id": g["game_id"], "key": _norm(pl), "line": float(L), "over_dec": pr["Over"],
                                 "under_dec": pr["Under"], "p_dk_over": io / (io + iu)})
    return pd.DataFrame(rows)


def analyze(data: str, cache: str, out: str):
    import numpy as np
    import pandas as pd
    F = features(data, cache)
    D = dk_lines(data)
    X = D.merge(F, on=["game_id", "key"], how="left")
    log = {"dk_props": int(len(D)), "unmatched_to_box": int(X.m_hat.isna().sum() + X.n_prior.isna().sum() * 0)}
    X = X[X.n_prior.fillna(0) >= 5]
    X = X[X.m_hat >= 12]
    X = X[~X.listed_out.fillna(False).astype(bool)]
    log["eligible"] = int(len(X))
    X = X[X.played.fillna(False).astype(bool)]
    log["after_void_dnp"] = int(len(X))
    X = X[~((X.line % 1 == 0) & (X.pts == X.line))]
    log["after_void_push"] = int(len(X))
    L = X.line.values
    X["margin"] = (X.pts - X.line) / np.sqrt(L + L ** 2 / K_STD)
    X["over"] = (X.pts > X.line).astype(int)
    X["fresh"] = (X.fresh_vac >= 15).astype(int)
    X["stale"] = (X.stale_vac >= 15).astype(int)
    p = X.p_dk_over.clip(1e-3, 1 - 1e-3)
    X["lp"] = np.log(p / (1 - p))

    def ols(d, y):
        A = np.column_stack([np.ones(len(d)), d.fresh.values, d.stale.values, d.lp.values])
        return np.linalg.lstsq(A, d[y].values, rcond=None)[0]

    beta = ols(X, "margin")
    games = np.array(sorted(X.game_id.unique()))
    grp = {gid: d for gid, d in X.groupby("game_id")}
    rng = np.random.default_rng(BOOT_SEED)
    bs, bs_w = [], []
    for _ in range(BOOT):
        pick = rng.choice(games, len(games), replace=True)
        d = pd.concat([grp[gid] for gid in pick])
        bs.append(ols(d, "margin")[1])
        bs_w.append(ols(d, "over")[1])
    lo, hi = np.quantile(bs, [0.025, 0.975])
    excl = lo > 0 or hi < 0
    big = abs(beta[1]) >= THRESH
    res = {"spec": "config/prereg/p1-thesis.yaml", "sample": "config/prereg/p1-sample.json", "log": log,
           "n_props": int(len(X)), "n_games": int(X.game_id.nunique()), "n_fresh": int(X.fresh.sum()), "n_stale": int(X.stale.sum()),
           "coef_margin": {"intercept": beta[0], "fresh": beta[1], "stale": beta[2], "logit_dk": beta[3]},
           "fresh_ci95": [float(lo), float(hi)], "threshold": THRESH,
           "decision": ("continue (" + ("over" if beta[1] > 0 else "under") + " side)") if (excl and big) else "stop",
           "secondary_winloss": {"fresh_coef": float(ols(X, "over")[1]), "ci95": [float(x) for x in np.quantile(bs_w, [0.025, 0.975])]},
           "over_rate": {"fresh": float(X[X.fresh == 1].over.mean()) if X.fresh.any() else None,
                         "no_fresh": float(X[X.fresh == 0].over.mean())},
           "dk_mean_p_over": {"fresh": float(X[X.fresh == 1].p_dk_over.mean()) if X.fresh.any() else None,
                              "no_fresh": float(X[X.fresh == 0].p_dk_over.mean())}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1, default=float))
    X.to_csv(Path(out).with_suffix(".rows.csv"), index=False)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "analyze"])
    ap.add_argument("--data", default="/tmp/edata")
    ap.add_argument("--cache", default="/home/claude/m3cache")
    ap.add_argument("--out", default="/tmp/edata/p1/result.json")
    a = ap.parse_args()
    if a.cmd == "sample":
        s = sample(a.data)
        print(json.dumps({k: v for k, v in s.items() if k != "games"}, indent=1))
    else:
        print(json.dumps(analyze(a.data, a.cache, a.out), indent=1, default=float))
