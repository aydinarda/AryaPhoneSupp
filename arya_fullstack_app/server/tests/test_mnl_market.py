from __future__ import annotations

import math

import pytest

from app.customer_segment import CustomerSegment
from app.mnl_market import BuyerProfile, run_mnl_market


def test_single_buyer_realized_utility_uses_shift_then_scale() -> None:
    profiles = [
        BuyerProfile(
            team_name="Solo",
            price_per_user=10.0,
            avg_env=1.0,
            avg_social=2.0,
        )
    ]
    segments = [
        CustomerSegment(
            segment_id="S1",
            density=1.0,
            w_env=2.0,
            w_social=1.0,
            w_cost=0.5,
        )
    ]

    result = run_mnl_market(profiles, segments, delta=1.0, quality_sensitivity=1.0)
    buyer = result.buyer_results["Solo"]

    raw_utility = 2.0 * (5.0 - 1.0) + 1.0 * (5.0 - 2.0) - 0.5 * 10.0
    expected = (raw_utility + 50.0) * 1.2

    assert buyer.total_demand == pytest.approx(1.0)
    assert buyer.realized_utility == pytest.approx(expected)


def test_scale_changes_mnl_shares_after_positive_shift() -> None:
    profiles = [
        BuyerProfile(team_name="A", price_per_user=10.0, avg_env=1.0, avg_social=1.0),
        BuyerProfile(team_name="B", price_per_user=10.0, avg_env=2.0, avg_social=1.0),
    ]
    segments = [
        CustomerSegment(
            segment_id="S1",
            density=1.0,
            w_env=1.0,
            w_social=0.0,
            w_cost=0.0,
        )
    ]

    result = run_mnl_market(profiles, segments, delta=1.0, quality_sensitivity=1.0)
    shares = result.segment_allocations[0].shares

    raw_a = 5.0 - 1.0
    raw_b = 5.0 - 2.0
    expected_share_a = math.exp((raw_a + 50.0) * 1.2)
    expected_share_b = math.exp((raw_b + 50.0) * 1.2)
    expected = expected_share_a / (expected_share_a + expected_share_b)

    assert shares["A"] == pytest.approx(expected)
    assert shares["B"] == pytest.approx(1.0 - expected)
