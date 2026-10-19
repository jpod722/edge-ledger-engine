from edge_engine.probe import coverage


def test_coverage_counts_two_sided_rungs_and_age():
    payload = {"timestamp": "2025-11-09T14:45:00Z", "data": {"bookmakers": [
        {"key": "pinnacle", "last_update": "2025-11-09T14:44:00Z", "markets": [
            {"key": "alternate_spreads", "last_update": "2025-11-09T14:44:30Z", "outcomes": [
                {"name": "A", "point": -3.5, "price": 2.1}, {"name": "B", "point": 3.5, "price": 1.8},
                {"name": "A", "point": -6.5, "price": 2.9}]},
            {"key": "team_totals", "outcomes": [
                {"name": "Over", "description": "A", "point": 24.5, "price": 1.9},
                {"name": "Under", "description": "A", "point": 24.5, "price": 1.9}]}]}]}}
    c = coverage(payload)
    assert c["alternate_spreads"]["pinnacle"] == {"outcomes": 3, "points": 2, "two_sided_rungs": 1, "age_s": 30}
    assert c["team_totals"]["pinnacle"]["two_sided_rungs"] == 1
    assert c["team_totals"]["pinnacle"]["age_s"] == 60


def test_billing_cycle_start():
    from datetime import datetime, timezone
    from edge_engine.client import RawStore
    t = lambda *a: datetime(*a, tzinfo=timezone.utc)
    assert RawStore.cycle_start(t(2026, 11, 5), 18) == t(2026, 10, 18)
    assert RawStore.cycle_start(t(2026, 11, 18), 18) == t(2026, 11, 18)
    assert RawStore.cycle_start(t(2027, 1, 2), 18) == t(2026, 12, 18)
