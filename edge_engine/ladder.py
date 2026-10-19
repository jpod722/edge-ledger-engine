"""Alt-ladder distributions.

A sharp book's alternate lines, each devigged as a two-way pair, give the
probability that one side covers at each point. Those probabilities are made
monotonic with pool-adjacent-violators, kept discrete at whole numbers (push
mass), and only interpolated between rungs when no key number lies between.
Devigging a line and estimating the margin distribution are separate steps.
"""
from __future__ import annotations

from dataclasses import dataclass

from .normalize import Pair
from .odds import devig


def pav(values: list[float], increasing: bool = True) -> list[float]:
    """Pool-adjacent-violators isotonic fit with equal weights."""
    v = list(values) if increasing else [-x for x in values]
    blocks = [[x, 1] for x in v]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            tot = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            n = blocks[i][1] + blocks[i + 1][1]
            blocks[i] = [tot / n, n]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = []
    for mean, n in blocks:
        out += [mean] * n
    return out if increasing else [-x for x in out]


@dataclass
class Ladder:
    """P(cover) for one side (a team, or 'Over') as a function of its point."""
    family: str
    side: str
    points: list[float]
    probs: list[float]       # monotone fitted
    raw: list[float]         # devigged, before the fit
    keys: list[float]
    max_adjustment: float

    @property
    def quality_ok(self) -> bool:
        return len(self.points) >= 3 and self.max_adjustment <= 0.02

    def _increasing(self) -> bool:
        return self.side != "Over"

    def _crosses_key(self, lo: float, hi: float) -> bool:
        """True if a key number sits in the settlement mass between two points."""
        spread_like = self.family in ("spreads", "spreads_h1")
        for k in self.keys:
            if spread_like:
                # covering at p means margin > -p; the mass between p1 < p2 is margins in (-p2, -p1]
                if any(-hi < s <= -lo for s in (k, -k)):
                    return True
            elif lo < k < hi or k == lo or k == hi:
                # totals, team totals, props: the mass between two lines is the totals between them
                return True
        return False

    def cover(self, point: float) -> float | None:
        """P(this side covers | no push) at point, or None if it can't be priced safely."""
        if point in self.points:
            return self.probs[self.points.index(point)]
        if float(point).is_integer():
            below, above = self.cover(point - 0.5), self.cover(point + 0.5)
            if below is None or above is None:
                return None
            push = abs(above - below)
            win = below if self._increasing() else above
            return win / (1 - push) if push < 1 else None
        lo = max((p for p in self.points if p < point), default=None)
        hi = min((p for p in self.points if p > point), default=None)
        if lo is None or hi is None or self._crosses_key(lo, hi):
            return None
        plo, phi = self.probs[self.points.index(lo)], self.probs[self.points.index(hi)]
        return plo + (phi - plo) * (point - lo) / (hi - lo)

    def push(self, point: float) -> float:
        """Probability of a push at a whole-number point (0 for half points or if unknown)."""
        if not float(point).is_integer():
            return 0.0
        a, b = self.probs_at(point - 0.5), self.probs_at(point + 0.5)
        return abs(b - a) if a is not None and b is not None else 0.0

    def probs_at(self, point: float) -> float | None:
        return self.probs[self.points.index(point)] if point in self.points else None


def build_ladder(pairs: list[Pair], side: str, keys: list[float], method: str = "power") -> Ladder | None:
    """Build one side's ladder from a single book's pairs of one family and event."""
    pts: dict[float, float] = {}
    for pr in pairs:
        for q, other in ((pr.a, pr.b), (pr.b, pr.a)):
            if q.side == side and q.point is not None:
                p, _ = devig(q.dec, other.dec, method)
                pts.setdefault(float(q.point), p)
    if not pts:
        return None
    points = sorted(pts)
    raw = [pts[p] for p in points]
    fitted = pav(raw, increasing=(side != "Over"))
    adj = max(abs(a - b) for a, b in zip(raw, fitted))
    return Ladder(pairs[0].family, side, points, fitted, raw, keys, adj)
