"""Fair probabilities for one exact contract from sharp sources.

Selection uses a weighted sharp consensus. Evaluation (CLV) uses Pinnacle
alone, with an independent consensus that excludes Pinnacle as a secondary
benchmark, so the scorecard never inherits the selection blend's assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .ladder import build_ladder
from .normalize import Pair, Quote, parse_ts
from .odds import DEVIG


def quote_age_s(q: Quote, decision_at: datetime) -> float | None:
    lu = parse_ts(q.last_update)
    return None if lu is None else (decision_at - lu).total_seconds()


def staleness_cutoff_s(cfg: dict, group: str, minutes_to_start: float, news_window: bool) -> float:
    s = cfg["staleness"]
    if news_window:
        return s["news_window_s"]
    if minutes_to_start <= s["near_start_minutes"]:
        return s["near_start_s"]
    return s["prop_s"] if group == "prop" else s["main_s"]


def same_contract(pair: Pair, q: Quote) -> tuple[Quote, Quote] | None:
    """Return (this side, other side) from a pair that prices exactly q's contract."""
    for x, y in ((pair.a, pair.b), (pair.b, pair.a)):
        if x.side == q.side and x.description == q.description and x.point == q.point \
                and pair.event_id == q.event_id and pair.family == q.family:
            return x, y
    return None


@dataclass
class SourceFair:
    book: str
    p: float
    age_s: float | None
    how: str                          # "direct" or "ladder"
    by_method: dict = field(default_factory=dict)


@dataclass
class Fair:
    p: float | None
    sources: list[SourceFair]
    dispersion: float
    method_spread: float
    reasons: list[str]

    @property
    def n(self) -> int:
        return len(self.sources)


def source_fair(book: str, q: Quote, pairs_by_book: dict, keys: list[float], method: str,
                decision_at: datetime, cutoff_s: float, allow_ladder: bool) -> tuple[SourceFair | None, str | None]:
    pairs = pairs_by_book.get(book, [])
    for pr in pairs:
        hit = same_contract(pr, q)
        if hit:
            x, y = hit
            age = quote_age_s(x, decision_at)
            if age is not None and age > cutoff_s:
                return None, f"{book}_stale_{int(age)}s"
            by = {m: f(x.dec, y.dec)[0] for m, f in DEVIG.items() if _safe(f, x.dec, y.dec)}
            return SourceFair(book, by[method], age, "direct", by), None
    if allow_ladder:
        fam_pairs = [p for p in pairs if p.family == q.family and p.description == q.description
                     and p.event_id == q.event_id]
        if fam_pairs:
            lad = build_ladder(fam_pairs, q.side, keys, method)
            if lad and lad.quality_ok:
                ages = [quote_age_s(p.a, decision_at) for p in fam_pairs]
                age = max((a for a in ages if a is not None), default=None)
                if age is not None and age > cutoff_s:
                    return None, f"{book}_ladder_stale_{int(age)}s"
                p = lad.cover(q.point)
                if p is not None:
                    return SourceFair(book, p, age, "ladder", {method: p}), None
                return None, f"{book}_ladder_no_safe_interp"
            if lad:
                return None, f"{book}_ladder_low_quality"
    return None, f"{book}_no_exact_quote"


def _safe(f, a, b) -> bool:
    try:
        f(a, b)
        return True
    except (ValueError, RuntimeError):
        return False


def consensus(q: Quote, books: list[str], weights: dict, pairs_by_book: dict, keys: list[float],
              method: str, decision_at: datetime, cutoff_s: float, exclude: set[str],
              allow_ladder: bool = True) -> Fair:
    srcs, reasons = [], []
    for b in books:
        if b in exclude:
            continue
        sf, why = source_fair(b, q, pairs_by_book, keys, method, decision_at, cutoff_s, allow_ladder)
        if sf:
            srcs.append(sf)
        elif why:
            reasons.append(why)
    if not srcs:
        return Fair(None, [], 0.0, 0.0, reasons + ["no_fresh_sources"])
    wsum = sum(weights.get(s.book, 0) for s in srcs)
    if wsum <= 0:
        return Fair(None, srcs, 0.0, 0.0, reasons + ["zero_weight"])
    p = sum(weights.get(s.book, 0) * s.p for s in srcs) / wsum
    disp = max(s.p for s in srcs) - min(s.p for s in srcs)
    # how much the answer depends on the devig method
    alts = []
    for m in DEVIG:
        vals = [(weights.get(s.book, 0), s.by_method.get(m)) for s in srcs]
        if all(v is not None for _, v in vals):
            alts.append(sum(w * v for w, v in vals) / wsum)
    mspread = (max(alts) - min(alts)) if len(alts) > 1 else 0.0
    return Fair(p, srcs, disp, mspread, reasons)


def index_pairs(pairs: list[Pair]) -> dict:
    out: dict = {}
    for p in pairs:
        out.setdefault(p.book, []).append(p)
    return out
