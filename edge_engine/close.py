"""Closing-line value.

Primary benchmark: Pinnacle's close for the exact contract, devigged.
Secondary: an independent consensus that excludes Pinnacle (Kalshi + LowVig).
If a benchmark has no exact-contract close, the metric is left empty with a
reason; one benchmark is never silently swapped for another.
"""
from __future__ import annotations

import math
from datetime import datetime

from .fair import consensus, index_pairs, source_fair
from .normalize import Quote, pairs_from
from .odds import ev_no_push_prob

NO_CUTOFF = 10 ** 9


def quote_from_candidate(c: dict) -> Quote:
    return Quote(sport=c["sport"], event_id=c["event_id"], commence_time=c["commence_time"], home="", away="",
                 book=c["venue"], market=c["family"], side=c["side"], description=c.get("description", ""),
                 point=c.get("point"), dec=c["price_dec"], last_update=None)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def metrics(p_close: float | None, dec: float, push: float) -> dict:
    if p_close is None:
        return {"prob": None, "ev": None, "logodds": None}
    be = 1 / dec
    return {"prob": round(p_close - be, 6), "ev": round(ev_no_push_prob(p_close, dec, push), 6),
            "logodds": round(_logit(p_close) - _logit(be), 6)}


def close_candidate(c: dict, close_quotes: list[Quote], cfg: dict, close_requested_at: datetime,
                    api_retrieved_at: str, raw_refs: list) -> dict:
    q = quote_from_candidate(c)
    pbb = index_pairs(pairs_from(close_quotes))
    keys = cfg["key_numbers"].get(q.sport, {}).get(q.family, [])
    method = cfg["devig_method"]
    dec, push = c["effective_dec"], c.get("push_prob", 0.0)

    pin, why = source_fair("pinnacle", q, pbb, keys, method, close_requested_at, NO_CUTOFF, allow_ladder=True)
    indep_books = [b for b in cfg["sharp_books"] if b != "pinnacle"]
    ind = consensus(q, indep_books, cfg["weights"], pbb, keys, method, close_requested_at, NO_CUTOFF,
                    exclude={c["venue"]}, allow_ladder=False)
    out = {
        "close_requested_at": close_requested_at.isoformat(),
        "close_api_retrieved_at": api_retrieved_at,
        "close_raw_refs": raw_refs,
        "close_pinnacle_p": round(pin.p, 6) if pin else None,
        "close_pinnacle_how": pin.how if pin else None,
        "close_pinnacle_age_s": pin.age_s if pin else None,
        "close_pinnacle_reason": None if pin else why,
        "close_indep_p": round(ind.p, 6) if ind.p is not None else None,
        "close_indep_books": [s.book for s in ind.sources],
        "clv_pinnacle": metrics(pin.p if pin else None, dec, push),
        "clv_indep": metrics(ind.p, dec, push),
    }
    return out
