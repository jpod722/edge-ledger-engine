"""Turn raw Odds API payloads into flat quotes and two-way pairs.

Contracts are identified by (event, family, description, side, point) where
alternate markets share a family with their main market, so an alternate
-2.5 and a main -2.5 are the same contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

FAMILY = {
    "h2h": "h2h",
    "spreads": "spreads",
    "alternate_spreads": "spreads",
    "totals": "totals",
    "alternate_totals": "totals",
    "team_totals": "team_totals",
    "alternate_team_totals": "team_totals",
    "h2h_h1": "h2h_h1",
    "spreads_h1": "spreads_h1",
    "alternate_spreads_h1": "spreads_h1",
    "totals_h1": "totals_h1",
    "alternate_totals_h1": "totals_h1",
}


def family(market: str) -> str:
    if market.startswith("player_"):
        return market.removesuffix("_alternate")
    return FAMILY.get(market, market)


def group(market: str) -> str:
    fam = family(market)
    if fam.startswith("player_"):
        return "prop"
    if market.startswith("alternate_"):
        return "alt"
    if fam in ("h2h", "spreads", "totals"):
        return "main"
    return "deriv"


def parse_ts(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Quote:
    sport: str
    event_id: str
    commence_time: str
    home: str
    away: str
    book: str
    market: str
    side: str            # team name, "Over" / "Under"
    description: str     # player or team for props / team totals, else ""
    point: float | None
    dec: float
    last_update: str | None

    @property
    def family(self) -> str:
        return family(self.market)

    @property
    def group(self) -> str:
        return group(self.market)

    @property
    def game(self) -> str:
        return f"{self.away} @ {self.home}"


def events_of(payload) -> list[dict]:
    """Accept bulk (list), per-event (dict) or historical ({'data': ...}) payloads."""
    if isinstance(payload, dict) and "data" in payload:
        payload = payload["data"]
    if isinstance(payload, dict):
        return [payload]
    return list(payload or [])


def quotes_from(payload, sport: str) -> list[Quote]:
    out: list[Quote] = []
    for ev in events_of(payload):
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                lu = mk.get("last_update") or bk.get("last_update")
                for o in mk.get("outcomes", []):
                    price = o.get("price")
                    if price is None or price <= 1:
                        continue  # fail closed: decimal format required
                    out.append(Quote(
                        sport=sport, event_id=ev["id"], commence_time=ev["commence_time"],
                        home=ev["home_team"], away=ev["away_team"], book=bk["key"],
                        market=mk["key"], side=o["name"], description=o.get("description", "") or "",
                        point=o.get("point"), dec=float(price), last_update=lu,
                    ))
    return out


@dataclass
class Pair:
    """Both sides of one two-way line at one book."""
    event_id: str
    book: str
    family: str
    description: str
    a: Quote
    b: Quote
    extras: dict = field(default_factory=dict)


def _pair_key(q: Quote):
    fam = q.family
    if fam in ("spreads", "spreads_h1"):
        return (q.event_id, q.book, fam, q.description, abs(q.point) if q.point is not None else None)
    if fam in ("h2h", "h2h_h1"):
        return (q.event_id, q.book, fam, q.description, None)
    return (q.event_id, q.book, fam, q.description, q.point)


def pairs_from(quotes: list[Quote]) -> list[Pair]:
    buckets: dict = {}
    for q in quotes:
        buckets.setdefault(_pair_key(q), []).append(q)
    out = []
    for key, qs in buckets.items():
        fam = key[2]
        if fam in ("spreads", "spreads_h1"):
            # a spread pair is side X at +p with the other side at -p;
            # main and alternate quotes of the same line are the same contract
            uniq = {}
            for q in qs:
                uniq.setdefault((q.side, q.point), q)
            qs = list(uniq.values())
            for q in qs:
                for r in qs:
                    if q.side < r.side and q.point is not None and r.point is not None \
                            and abs(q.point + r.point) < 1e-9 and q.side != r.side:
                        out.append(Pair(key[0], key[1], fam, key[3], q, r))
            continue
        sides = {}
        for q in qs:
            sides.setdefault(q.side, q)  # duplicates (main vs alt) are the same contract
        if len(sides) == 2:
            a, b = sorted(sides.values(), key=lambda q: q.side)
            out.append(Pair(key[0], key[1], fam, key[3], a, b))
    return out


def contract_id(q: Quote) -> tuple:
    return (q.event_id, q.family, q.description, q.side, q.point)
