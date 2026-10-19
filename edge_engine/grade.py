"""Settle candidates from ESPN results. Pushes are explicit; unknown outcomes stay open, then void."""
from __future__ import annotations

ESPN = {
    "NFL": "https://site.api.espn.com/apis/site/v2/sports/football/nfl",
    "NBA": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba",
}

PROP_KEYS = {
    "player_points": ("points",),
    "player_rebounds": ("rebounds",),
    "player_assists": ("assists",),
    "player_threes": ("threePointFieldGoalsMade-threePointFieldGoalsAttempted",),
    "player_pass_yds": ("passingYards",),
    "player_rush_yds": ("rushingYards",),
    "player_reception_yds": ("receivingYards",),
    "player_receptions": ("receptions",),
}


def nickname(team: str) -> str:
    return team.strip().split()[-1].lower()


def parse_scoreboard(sb: dict) -> list[dict]:
    games = []
    for ev in sb.get("events", []):
        comp = ev["competitions"][0]
        done = comp.get("status", {}).get("type", {}).get("completed", False)
        teams = {}
        for t in comp["competitors"]:
            ls = [float(x.get("value", 0)) for x in t.get("linescores", [])]
            teams[t["homeAway"]] = {"name": t["team"]["displayName"], "score": float(t.get("score", 0) or 0),
                                    "h1": sum(ls[:2]) if len(ls) >= 2 else None}
        games.append({"espn_id": ev["id"], "date": ev.get("date"), "completed": done, **teams})
    return games


def find_game(games: list[dict], game: str) -> dict | None:
    away, home = [s.strip() for s in game.split(" @ ")]
    for g in games:
        if nickname(g["home"]["name"]) == nickname(home) and nickname(g["away"]["name"]) == nickname(away):
            return g
    return None


def _team(g: dict, name: str) -> tuple[dict, dict] | None:
    for match in (lambda x: x.lower() == name.lower(), lambda x: nickname(x) == nickname(name)):
        for me, opp in (("home", "away"), ("away", "home")):
            if match(g[me]["name"]):
                return g[me], g[opp]
    return None


def outcome(c: dict, g: dict, player_stat: float | None = None) -> str | None:
    """'win' / 'loss' / 'push', or None if it can't be settled from this data."""
    fam, side, pt = c["family"], c["side"], c.get("point")
    half = fam.endswith("_h1")
    key = "h1" if half else "score"
    if fam in ("h2h", "h2h_h1", "spreads", "spreads_h1"):
        t = _team(g, side)
        if not t or t[0][key] is None:
            return None
        diff = t[0][key] - t[1][key] + ((pt or 0) if fam.startswith("spreads") else 0)
    elif fam in ("totals", "totals_h1"):
        if g["home"][key] is None:
            return None
        tot = g["home"][key] + g["away"][key]
        diff = (tot - pt) if side == "Over" else (pt - tot)
    elif fam == "team_totals":
        t = _team(g, c.get("description", ""))
        if not t:
            return None
        diff = (t[0]["score"] - pt) if side == "Over" else (pt - t[0]["score"])
    elif fam.startswith("player_"):
        if player_stat is None:
            return None
        diff = (player_stat - pt) if side == "Over" else (pt - player_stat)
    else:
        return None
    return "win" if diff > 0 else ("push" if diff == 0 else "loss")


def units(result: str, dec: float) -> float:
    return {"win": round(dec - 1, 6), "loss": -1.0, "push": 0.0, "void": 0.0}[result]


def player_stat(summary: dict, player: str, market: str) -> float | None:
    keys = PROP_KEYS.get(market.removesuffix("_alternate"))
    if not keys:
        return None
    want = player.lower().replace(".", "").strip()
    for team in summary.get("boxscore", {}).get("players", []):
        for cat in team.get("statistics", []):
            names = cat.get("keys", [])
            for k in keys:
                if k not in names:
                    continue
                i = names.index(k)
                for a in cat.get("athletes", []):
                    nm = a.get("athlete", {}).get("displayName", "").lower().replace(".", "").strip()
                    if nm == want:
                        val = a.get("stats", [])[i]
                        return float(str(val).split("-")[0])
    return None
