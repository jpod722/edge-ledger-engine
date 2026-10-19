"""Odds conversions, vig removal and fees. Pure functions, fully tested."""
from __future__ import annotations

import math

from scipy.optimize import brentq


def american_to_decimal(american: float) -> float:
    if american == 0 or -100 < american < 100:
        raise ValueError(f"invalid American odds: {american}")
    return 1 + (american / 100 if american > 0 else 100 / -american)


def decimal_to_american(dec: float) -> float:
    if dec <= 1:
        raise ValueError(f"invalid decimal odds: {dec}")
    return (dec - 1) * 100 if dec >= 2 else -100 / (dec - 1)


def implied(dec: float) -> float:
    if dec <= 1:
        raise ValueError(f"invalid decimal odds: {dec}")
    return 1 / dec


def overround(dec_a: float, dec_b: float) -> float:
    return implied(dec_a) + implied(dec_b) - 1


def devig_multiplicative(dec_a: float, dec_b: float) -> tuple[float, float]:
    a, b = implied(dec_a), implied(dec_b)
    s = a + b
    return a / s, b / s


def devig_additive(dec_a: float, dec_b: float) -> tuple[float, float]:
    a, b = implied(dec_a), implied(dec_b)
    m = (a + b - 1) / 2
    pa, pb = a - m, b - m
    if pa <= 0 or pb <= 0:
        raise ValueError("additive devig produced a non-positive probability")
    return pa, pb


def devig_power(dec_a: float, dec_b: float) -> tuple[float, float]:
    a, b = implied(dec_a), implied(dec_b)
    if abs(a + b - 1) < 1e-12:
        return a, b
    k = brentq(lambda k: a ** k + b ** k - 1, 0.2, 5.0)
    return a ** k, b ** k


DEVIG = {
    "power": devig_power,
    "multiplicative": devig_multiplicative,
    "additive": devig_additive,
}


def devig(dec_a: float, dec_b: float, method: str = "power") -> tuple[float, float]:
    return DEVIG[method](dec_a, dec_b)


def ev(p_win: float, dec: float, p_push: float = 0.0) -> float:
    """Expected profit per unit staked. p_win is the unconditional win probability."""
    p_loss = 1 - p_win - p_push
    return p_win * (dec - 1) - p_loss


def ev_no_push_prob(p_cond: float, dec: float, p_push: float = 0.0) -> float:
    """EV when p_cond is P(win | no push), as a two-way devig gives at an integer line."""
    return (1 - p_push) * (p_cond * dec - 1)


def kalshi_taker_fee(price: float, contracts: int = 1, rate: float = 0.07) -> float:
    """Kalshi taker fee in dollars for one order, rounded up to the cent."""
    if not 0 < price < 1:
        raise ValueError("Kalshi price must be between 0 and 1")
    raw = rate * contracts * price * (1 - price)
    return math.ceil(round(raw * 100, 9)) / 100


def kalshi_effective_decimal(price: float, contracts: int = 100, rate: float = 0.07) -> float:
    """Decimal odds of a Kalshi YES/NO contract after the taker fee (per-order rounding)."""
    cost = price * contracts + kalshi_taker_fee(price, contracts, rate)
    return contracts / cost


def kelly_fraction(p_win: float, dec: float) -> float:
    b = dec - 1
    f = (p_win * b - (1 - p_win)) / b
    return max(0.0, f)
