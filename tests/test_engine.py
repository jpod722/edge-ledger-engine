import copy
from datetime import datetime, timedelta, timezone

import pytest

from edge_engine.cli import in_window, load_cfg
from edge_engine.close import close_candidate
from edge_engine.evaluate import price_quotes, size_stakes, to_dict
from edge_engine.grade import outcome, units
from edge_engine.ladder import build_ladder, pav
from edge_engine.normalize import pairs_from, quotes_from
from edge_engine.odds import (american_to_decimal, decimal_to_american, devig_additive, devig_multiplicative,
                              devig_power, ev, ev_no_push_prob, kalshi_effective_decimal, kalshi_taker_fee,
                              kelly_fraction)

NOW = datetime(2026, 10, 25, 14, 0, tzinfo=timezone.utc)
KICK = (NOW + timedelta(hours=3)).isoformat().replace("+00:00", "Z")


def ts(seconds_ago):
    return (NOW - timedelta(seconds=seconds_ago)).isoformat().replace("+00:00", "Z")


def book(key, markets, age=30):
    return {"key": key, "title": key, "last_update": ts(age),
            "markets": [{"key": k, "last_update": ts(age), "outcomes": o} for k, o in markets.items()]}


def spread(home_pt, home_dec, away_dec):
    return [{"name": "Home", "price": home_dec, "point": home_pt}, {"name": "Away", "price": away_dec, "point": -home_pt}]


def event(books):
    return {"id": "evt12345abcdef", "sport_key": "americanfootball_nfl", "commence_time": KICK,
            "home_team": "Home", "away_team": "Away", "bookmakers": books}


@pytest.fixture
def cfg():
    return load_cfg()


# ---------- odds ----------

def test_american_decimal_roundtrip():
    for a in (-250, -110, 100, 150, 370):
        assert decimal_to_american(american_to_decimal(a)) == pytest.approx(a)
    assert american_to_decimal(-110) == pytest.approx(1.909090909)
    with pytest.raises(ValueError):
        american_to_decimal(50)


def test_devig_methods_sum_to_one_and_order():
    a, b = american_to_decimal(333), american_to_decimal(-415)
    for f in (devig_power, devig_multiplicative, devig_additive):
        pa, pb = f(a, b)
        assert pa + pb == pytest.approx(1)
    # power shades the longshot below multiplicative (the Giants +370 lesson)
    assert devig_power(a, b)[0] < devig_multiplicative(a, b)[0]
    assert devig_power(a, b)[0] == pytest.approx(0.2070, abs=5e-4)


def test_ev_and_push():
    assert ev(0.55, 2.0) == pytest.approx(0.10)
    assert ev(0.50, 2.0, p_push=0.1) == pytest.approx(0.50 - 0.40)
    assert ev_no_push_prob(0.55, 2.0, 0.1) == pytest.approx(0.9 * 0.10)


def test_kalshi_fee_rounds_up_per_order():
    assert kalshi_taker_fee(0.50, 1) == 0.02
    assert kalshi_taker_fee(0.50, 100) == 1.75
    assert kalshi_effective_decimal(0.50, 100) == pytest.approx(100 / 51.75)


def test_kelly():
    assert kelly_fraction(0.55, 2.0) == pytest.approx(0.10)
    assert kelly_fraction(0.40, 2.0) == 0.0


# ---------- ladders ----------

def test_pav_monotone():
    assert pav([0.4, 0.35, 0.5, 0.6]) == pytest.approx([0.375, 0.375, 0.5, 0.6])
    assert pav([0.6, 0.65, 0.5], increasing=False) == pytest.approx([0.625, 0.625, 0.5])


def ladder_payload():
    # Home ladder: cover probability rises with the points Home receives
    rungs = {-7.5: (2.9, 1.42), -6.5: (2.6, 1.52), -3.5: (2.12, 1.76), -2.5: (1.80, 2.06), -1.5: (1.72, 2.16), 1.5: (1.55, 2.48)}
    outs = []
    for pt, (h, a) in rungs.items():
        outs += spread(pt, h, a)
    return [event([book("pinnacle", {"alternate_spreads": outs})])]


def test_ladder_no_interpolation_across_key_number(cfg):
    q = quotes_from(ladder_payload(), "NFL")
    pairs = [p for p in pairs_from(q) if p.book == "pinnacle"]
    lad = build_ladder(pairs, "Home", cfg["key_numbers"]["NFL"]["spreads"])
    assert lad.quality_ok
    assert lad.cover(-2.5) == pytest.approx(lad.probs_at(-2.5))
    assert lad.cover(-5.5) is None          # between -6.5 and -3.5 the 4 and 6 key numbers sit in the gap
    assert lad.cover(-2.0) is not None and lad.probs_at(-2.5) < lad.cover(-2.0) < lad.probs_at(-1.5) + 0.2
    assert lad.push(-2.0) == pytest.approx(lad.probs_at(-1.5) - lad.probs_at(-2.5))


# ---------- candidates ----------

def market_payload(dk_home_dec=2.10, pin_age=30, kal=True, lowvig=True, dk_age=30):
    books = [book("pinnacle", {"spreads": spread(-2.5, 1.95, 1.95)}, pin_age),
             book("draftkings", {"spreads": spread(-2.5, dk_home_dec, 1.74)}, dk_age)]
    if kal:
        books.append(book("kalshi", {"spreads": spread(-2.5, 1.96, 1.96)}))
    if lowvig:
        books.append(book("lowvig", {"spreads": spread(-2.5, 1.96, 1.94)}))
    return [event(books)]


def home_dk(cands):
    return next(c for c in cands if c.side == "Home" and c.venue == "draftkings")


def test_clear_edge_is_bet_tier_in_paper_mode(cfg):
    c = home_dk(price_quotes(quotes_from(market_payload(), "NFL"), cfg, NOW))
    assert c.tier == "bet" and c.paper
    assert c.fair_prob == pytest.approx(0.5, abs=0.01)
    assert c.ev_quoted == pytest.approx(c.fair_prob * 2.10 - 1, abs=1e-6)
    assert c.ev_adj_net < c.ev_quoted        # haircut and execution drag applied
    assert {s["book"] for s in c.sources} == {"pinnacle", "kalshi", "lowvig"}


def test_stale_sharp_quote_is_dropped(cfg):
    c = home_dk(price_quotes(quotes_from(market_payload(pin_age=600), "NFL"), cfg, NOW))
    assert "pinnacle" not in {s["book"] for s in c.sources}


def test_stale_venue_quote_rejected(cfg):
    c = home_dk(price_quotes(quotes_from(market_payload(dk_age=600), "NFL"), cfg, NOW))
    assert c.tier == "rejected" and any(r.startswith("venue_stale") for r in c.reasons)


def test_single_source_needs_higher_ev(cfg):
    c = home_dk(price_quotes(quotes_from(market_payload(dk_home_dec=2.03, kal=False, lowvig=False), "NFL"), cfg, NOW))
    assert "single_source" in c.reasons and c.tier == "rejected"


def test_split_market_rejected(cfg):
    p = market_payload()
    p[0]["bookmakers"][2] = book("kalshi", {"spreads": spread(-2.5, 1.70, 2.30)})
    c = home_dk(price_quotes(quotes_from(p, "NFL"), cfg, NOW))
    assert any(r.startswith("split_") for r in c.reasons) and c.tier == "rejected"


def test_kalshi_excluded_from_its_own_fair(cfg):
    p = market_payload(dk_home_dec=1.80)
    p[0]["bookmakers"][2] = book("kalshi", {"spreads": spread(-2.5, 2.25, 1.80)})
    cands = price_quotes(quotes_from(p, "NFL"), cfg, NOW)
    k = next(c for c in cands if c.contract.endswith("home|-2.5"))
    if k.venue == "kalshi":
        assert "kalshi" not in {s["book"] for s in k.sources}
        assert k.effective_dec < 2.25


def test_stakes_respect_caps(cfg):
    cands = price_quotes(quotes_from(market_payload(dk_home_dec=2.6), "NFL"), cfg, NOW)
    size_stakes(cands, cfg)
    c = home_dk(cands)
    assert 0 < c.stake_usd <= cfg["staking"]["bankroll"] * cfg["staking"]["max_bet_pct"]
    cands2 = price_quotes(quotes_from(market_payload(dk_home_dec=2.6), "NFL"), cfg, NOW)
    size_stakes(cands2, cfg, {"game:evt12345abcdef": 99.0})
    assert home_dk(cands2).stake_usd == 0 and home_dk(cands2).tier == "research"


# ---------- close / CLV ----------

def test_clv_uses_exact_contract_and_never_substitutes(cfg):
    c = to_dict(home_dk(price_quotes(quotes_from(market_payload(), "NFL"), cfg, NOW)))
    moved = [event([book("pinnacle", {"spreads": spread(-3.5, 1.95, 1.95)}),
                    book("lowvig", {"spreads": spread(-3.5, 1.95, 1.95)})])]
    out = close_candidate(c, quotes_from(moved, "NFL"), cfg, NOW, NOW.isoformat(), [])
    assert out["close_pinnacle_p"] is None and out["clv_pinnacle"]["prob"] is None
    assert out["close_pinnacle_reason"]
    same = [event([book("pinnacle", {"spreads": spread(-2.5, 1.80, 2.10)})])]
    out2 = close_candidate(c, quotes_from(same, "NFL"), cfg, NOW, NOW.isoformat(), [])
    assert out2["clv_pinnacle"]["prob"] > 0 and out2["clv_indep"]["prob"] is None


# ---------- grading ----------

def game(h, a, h1h=None, h1a=None):
    return {"completed": True, "home": {"name": "Home Team", "score": h, "h1": h1h},
            "away": {"name": "Away Team", "score": a, "h1": h1a}}


def test_grading_with_pushes():
    g = game(24, 21, 10, 10)
    assert outcome({"family": "spreads", "side": "Home Team", "point": -3.0}, g) == "push"
    assert outcome({"family": "spreads", "side": "Home Team", "point": -2.5}, g) == "win"
    assert outcome({"family": "spreads", "side": "Away Team", "point": 2.5}, g) == "loss"
    assert outcome({"family": "totals", "side": "Over", "point": 45.0}, g) == "push"
    assert outcome({"family": "totals", "side": "Under", "point": 45.5}, g) == "win"
    assert outcome({"family": "spreads_h1", "side": "Home Team", "point": -0.5}, g) == "loss"
    assert outcome({"family": "h2h", "side": "Away Team", "point": None}, g) == "loss"
    assert units("win", 1.91) == pytest.approx(0.91) and units("push", 1.91) == 0


def test_windows():
    assert in_window(KICK, "within:200", NOW) and not in_window(KICK, "within:60", NOW)
    assert not in_window(ts(10), "all", NOW)
