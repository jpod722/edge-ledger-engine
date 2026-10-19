"""Candidate generation: price every venue quote, apply eligibility and risk checks,
and size paper/real stakes from an uncertainty-haircut probability."""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime

from . import ENGINE_VERSION
from .fair import consensus, index_pairs, quote_age_s, staleness_cutoff_s
from .ladder import build_ladder
from .normalize import Quote, pairs_from, parse_ts
from .odds import ev_no_push_prob, kalshi_effective_decimal, kelly_fraction


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def contract_key(q: Quote) -> str:
    pt = "ml" if q.point is None else f"{q.point:+g}"
    d = slug(q.description) if q.description else ""
    return "|".join([q.sport, q.event_id, q.family, d, slug(q.side), pt])


def doc_id(q: Quote) -> str:
    day = q.commence_time[:10].replace("-", "")
    pt = "ml" if q.point is None else f"{q.point:g}".replace("-", "m").replace(".", "p")
    who = slug(q.description or q.side)[:40]
    return f"{q.sport.lower()}-{day}-{q.event_id[:8]}-{q.family}-{who}-{slug(q.side)[:12]}-{pt}"


@dataclass
class Candidate:
    id: str
    contract: str
    sport: str
    event_id: str
    game: str
    commence_time: str
    family: str
    group: str
    side: str
    description: str
    point: float | None
    venue: str
    has_account: bool
    price_dec: float
    effective_dec: float
    venue_quote_age_s: float | None
    fair_prob: float | None
    fair_method: str
    devig_method: str
    fair_by_method: dict
    method_sensitive: bool
    sources: list
    dispersion: float
    push_prob: float
    se: float
    p_adj: float | None
    ev_quoted: float | None
    ev_adj_net: float | None
    tier: str                      # rejected / research / bet
    paper: bool
    stake_usd: float
    reasons: list
    sharp_detail: list = field(default_factory=list)   # per-book reasons a sharp source was not used
    best_account_venue: str | None = None
    best_account_ev: float | None = None
    decision_at: str = ""
    api_retrieved_at: str = ""
    raw_refs: list = field(default_factory=list)
    scan: str = ""
    engine_version: str = ENGINE_VERSION
    config_version: str = ""


def _push_prob(q: Quote, pairs_by_book: dict, keys: list, method: str) -> float:
    if q.point is None or not float(q.point).is_integer():
        return 0.0
    pin = [p for p in pairs_by_book.get("pinnacle", []) if p.family == q.family
           and p.event_id == q.event_id and p.description == q.description]
    lad = build_ladder(pin, q.side, keys, method) if pin else None
    return lad.push(q.point) if lad and lad.quality_ok else 0.0


def price_quotes(quotes: list[Quote], cfg: dict, decision_at: datetime, *, scan: str = "",
                 news_window: bool = False, api_retrieved_at: str = "", raw_refs: list | None = None) -> list[Candidate]:
    pairs = pairs_from(quotes)
    pbb = index_pairs(pairs)
    method = cfg["devig_method"]
    out: list[Candidate] = []
    for q in quotes:
        if q.book not in cfg["bettable_books"]:
            continue
        start = parse_ts(q.commence_time)
        mins = (start - decision_at).total_seconds() / 60
        if mins <= 0:
            continue
        keys = cfg["key_numbers"].get(q.sport, {}).get(q.family, [])
        cutoff = staleness_cutoff_s(cfg, q.group, mins, news_window)
        reasons: list[str] = []
        vage = quote_age_s(q, decision_at)
        if vage is not None and vage > cutoff:
            reasons.append(f"venue_stale_{int(vage)}s")
        if q.group == "prop":
            fair = consensus(q, cfg["props"]["fair_books"], cfg["props"]["weights"], pbb, keys, method,
                             decision_at, cutoff, exclude={q.book}, allow_ladder=False)
            if fair.n < cfg["props"]["min_books"]:
                reasons.append(f"prop_sources_{fair.n}_lt_{cfg['props']['min_books']}")
        else:
            fair = consensus(q, cfg["sharp_books"], cfg["weights"], pbb, keys, method,
                             decision_at, cutoff, exclude={q.book}, allow_ladder=True)
        reasons += [r for r in fair.reasons if r.startswith("no_price_")]
        fair_method = "+".join(sorted({s.how for s in fair.sources})) or "none"
        dec = q.dec
        if q.book == "kalshi":
            dec = kalshi_effective_decimal(min(max(1 / q.dec, 0.01), 0.99), 100, cfg["kalshi"]["taker_rate"])
        push = _push_prob(q, pbb, keys, method)
        c = Candidate(
            id=doc_id(q), contract=contract_key(q), sport=q.sport, event_id=q.event_id, game=q.game,
            commence_time=q.commence_time, family=q.family, group=q.group, side=q.side,
            description=q.description, point=q.point, venue=q.book, has_account=q.book in cfg["accounts"],
            price_dec=q.dec, effective_dec=round(dec, 6), venue_quote_age_s=vage, fair_prob=fair.p,
            fair_method=fair_method, devig_method=method, fair_by_method={}, method_sensitive=False,
            sources=[{"book": s.book, "p": round(s.p, 6), "age_s": s.age_s, "how": s.how} for s in fair.sources],
            dispersion=round(fair.dispersion, 6), push_prob=round(push, 6), se=0.0, p_adj=None,
            ev_quoted=None, ev_adj_net=None, tier="rejected", paper=cfg["mode"] == "paper", stake_usd=0.0,
            reasons=reasons, sharp_detail=[r for r in fair.reasons if not r.startswith("no_price_")], decision_at=decision_at.isoformat(), api_retrieved_at=api_retrieved_at,
            raw_refs=list(raw_refs or []), scan=scan, config_version=cfg["version"],
        )
        if fair.p is not None:
            if fair.n < 2 and q.group != "prop":
                c.reasons.append("single_source")
            if fair.dispersion > cfg["split_threshold"]:
                c.reasons.append(f"split_{fair.dispersion:.3f}")
            unc = cfg["uncertainty"]
            se = math.sqrt((fair.dispersion / 2) ** 2 + fair.method_spread ** 2 + unc["base_se"][q.group] ** 2)
            p_adj = max(0.0, fair.p - unc["z"] * se)
            c.se, c.p_adj = round(se, 6), round(p_adj, 6)
            c.ev_quoted = round(ev_no_push_prob(fair.p, dec, push), 6)
            c.ev_adj_net = round(ev_no_push_prob(p_adj, dec, push) - cfg["execution_drag"][q.group], 6)
            # does the devig method decide whether this qualifies?
            by = {}
            for s in fair.sources:
                for m, v in s.by_method.items():
                    by.setdefault(m, []).append(v)
            c.fair_by_method = {m: round(sum(v) / len(v), 6) for m, v in by.items()}
            evs = [ev_no_push_prob(v, dec, push) for v in c.fair_by_method.values()]
            thr = cfg["research_min_ev"]
            c.method_sensitive = len(evs) > 1 and (min(evs) < thr <= max(evs))
            hard = [r for r in c.reasons if not r.startswith("single_source")]
            single_ok = fair.n >= 2 or c.ev_quoted >= cfg["single_source_min_ev"]
            if not hard and single_ok and c.ev_quoted >= thr:
                c.tier = "research"
                if (q.group in cfg["bet_groups"] and fair_method != "none"
                        and c.ev_adj_net >= cfg["bet_min_ev"][q.group]):
                    c.tier = "bet"
        out.append(c)
    return best_per_contract(out, cfg)


def best_per_contract(cands: list[Candidate], cfg: dict) -> list[Candidate]:
    """Keep every rejected/research record for study; mark the best venue per contract."""
    by: dict[str, list[Candidate]] = {}
    for c in cands:
        by.setdefault(c.contract, []).append(c)
    out = []
    rank = {"bet": 2, "research": 1, "rejected": 0}
    for cs in by.values():
        cs.sort(key=lambda c: (rank[c.tier], c.ev_adj_net if c.ev_adj_net is not None else -9), reverse=True)
        best = cs[0]
        acct = [c for c in cs if c.has_account and c.ev_quoted is not None]
        if acct:
            a = max(acct, key=lambda c: c.ev_adj_net)
            best.best_account_venue, best.best_account_ev = a.venue, a.ev_adj_net
        out.append(best)
    return out


def size_stakes(cands: list[Candidate], cfg: dict, existing_exposure: dict | None = None) -> list[Candidate]:
    """Quarter Kelly on the haircut probability, then per-bet, per-game, per-team and per-player caps."""
    st = cfg["staking"]
    bank = st["bankroll"]
    expo = dict(existing_exposure or {})
    for c in sorted([c for c in cands if c.tier == "bet"], key=lambda c: c.ev_adj_net, reverse=True):
        # Kelly on P(win | no push); a push returns the stake, so scale by the chance of action
        k = kelly_fraction(c.p_adj, c.effective_dec) * (1 - c.push_prob)
        cap = st["max_prop_pct"] if c.group == "prop" else st["max_bet_pct"]
        stake = min(bank * st["kelly_fraction"] * k, bank * cap)
        limits = [("game:" + c.event_id, st["max_game_pct"])]
        team = c.description if c.family == "team_totals" else (c.side if c.family in ("h2h", "spreads", "h2h_h1", "spreads_h1") else "")
        if team:
            limits.append(("team:" + team, st["max_team_pct"]))
        if c.group == "prop" and c.description:
            limits.append(("player:" + c.description, st["max_player_pct"]))
        for key, pct in limits:
            stake = min(stake, max(0.0, bank * pct - expo.get(key, 0.0)))
        stake = math.floor(stake)
        if stake < st["min_stake_usd"]:
            c.tier, c.stake_usd = "research", 0.0
            c.reasons.append("stake_below_min_or_exposure_cap")
            continue
        c.stake_usd = float(stake)
        for key, _ in limits:
            expo[key] = expo.get(key, 0.0) + stake
    return cands


def to_dict(c: Candidate) -> dict:
    return asdict(c)
