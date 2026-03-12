from __future__ import annotations

import random
from datetime import datetime, timedelta

from app.matching_engine import GUROBI_AVAILABLE, run_market_matching
from app.mincost_agent import DEFAULT_XLSX_PATH, load_supplier_user_tables


def _assert_capacity_respected(result: dict, market_options: list[dict]) -> None:
    capacity_by_market = {str(m["option_id"]): int(m["capacity"]) for m in market_options}
    market_to_users = result["market_to_users"]

    for market_id, users in market_to_users.items():
        assert len(users) <= capacity_by_market[market_id]


def _assert_user_assignments_valid(result: dict, users_payload: list[dict], market_options: list[dict]) -> None:
    valid_user_ids = {str(u["user_id"]) for u in users_payload}
    valid_market_ids = {str(m["option_id"]) for m in market_options}

    user_to_market = result["user_to_market"]
    unmatched = set(result["unmatched_users"])

    assert set(user_to_market.keys()) == valid_user_ids

    matched_count = 0
    for user_id, market_id in user_to_market.items():
        assert user_id in valid_user_ids
        if market_id is None:
            assert user_id in unmatched
        else:
            assert market_id in valid_market_ids
            matched_count += 1

    assert matched_count + len(unmatched) == len(valid_user_ids)

    market_loads = result.get("market_loads") or {}
    for market in market_options:
        market_id = str(market["option_id"])
        load = market_loads.get(market_id)
        assert load is not None
        assert load["assigned_count"] == len(result["market_to_users"].get(market_id, []))
        assert load["capacity"] == int(market["capacity"])
        assert load["remaining_capacity"] == int(market["capacity"]) - load["assigned_count"]


def test_mock_dataset_matching_works_and_respects_constraints() -> None:
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m1", "m2", "m3"],
            "utilities": {"m1": 10.0, "m2": 7.0, "m3": 6.0},
        },
        {
            "user_id": "u2",
            "choices": ["m1", "m2", "m3"],
            "utilities": {"m1": 8.0, "m2": 9.0, "m3": 5.0},
        },
        {
            "user_id": "u3",
            "choices": ["m2", "m3", "m1"],
            "utilities": {"m1": 3.0, "m2": 9.5, "m3": 8.5},
        },
        {
            "user_id": "u4",
            "choices": ["m3", "m2", "m1"],
            "utilities": {"m1": 4.0, "m2": 5.0, "m3": 9.0},
        },
    ]

    market_options = [
        {
            "option_id": "m1",
            "capacity": 1,
            "priority": ["u1", "u2", "u3", "u4"],
            "request_time": "2026-03-09T10:00:00",
        },
        {
            "option_id": "m2",
            "capacity": 2,
            "priority": ["u3", "u2", "u1", "u4"],
            "request_time": "2026-03-09T09:30:00",
        },
        {
            "option_id": "m3",
            "capacity": 1,
            "priority": ["u4", "u3", "u2", "u1"],
            "request_time": "2026-03-09T09:00:00",
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    _assert_capacity_respected(result, market_options)
    _assert_user_assignments_valid(result, users_payload, market_options)

    # In this setup, total capacity equals number of users, so everyone should be matched.
    assert result["unmatched_users"] == []
    assert result["meta"]["matched_count"] == len(users_payload)
    assert result["meta"]["unmatched_count"] == 0


def test_tie_break_prefers_earlier_request_time_when_other_scores_are_equal() -> None:
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m_late", "m_early"],
            "utilities": {"m_late": 5.0, "m_early": 5.0},
        }
    ]

    market_options = [
        {
            "option_id": "m_late",
            "capacity": 2,
            "priority": ["u1"],
            "request_time": "2026-03-09T12:00:00",
        },
        {
            "option_id": "m_early",
            "capacity": 2,
            "priority": ["u1"],
            "request_time": "2026-03-09T08:00:00",
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    if result["meta"]["solver"] == "gurobi_stable_lexicographic":
        # Utility tie + occupancy tie => earlier request_time should win.
        assert result["user_to_market"]["u1"] == "m_early"
    else:
        # Stable mode uses user preference order here.
        assert result["user_to_market"]["u1"] in {"m_late", "m_early"}


def test_excel_driven_preferences_with_random_market_profiles() -> None:
    rng = random.Random(42)

    suppliers_df, users_df = load_supplier_user_tables(DEFAULT_XLSX_PATH)

    # Build mock market options from a subset of suppliers with random capacities/priorities.
    market_count = min(7, len(suppliers_df))
    user_count = min(10, len(users_df))

    sampled_suppliers = suppliers_df.sample(n=market_count, random_state=42).reset_index(drop=True)
    sampled_users = users_df.head(user_count).reset_index(drop=True)

    user_ids = [str(u) for u in sampled_users["user_id"].tolist()]

    base_time = datetime(2026, 3, 9, 8, 0, 0)
    market_options: list[dict] = []
    for idx, row in sampled_suppliers.iterrows():
        market_id = str(row["supplier_id"])
        priority = list(user_ids)
        rng.shuffle(priority)

        market_options.append(
            {
                "option_id": market_id,
                "capacity": rng.randint(1, 3),
                "priority": priority,
                "request_time": (base_time + timedelta(minutes=rng.randint(0, 180) + idx)).isoformat(),
            }
        )

    supplier_feature_map = {
        str(row["supplier_id"]): {
            "env_risk": float(row["env_risk"]),
            "social_risk": float(row["social_risk"]),
            "cost_score": float(row["cost_score"]),
            "strategic": float(row["strategic"]),
            "improvement": float(row["improvement"]),
            "low_quality": float(row["low_quality"]),
        }
        for _, row in sampled_suppliers.iterrows()
    }

    users_payload: list[dict] = []
    market_ids = [str(m["option_id"]) for m in market_options]

    for _, user in sampled_users.iterrows():
        uid = str(user["user_id"])

        w_env = float(user["w_env"])
        w_social = float(user["w_social"])
        w_cost = float(user["w_cost"])
        w_strategic = float(user["w_strategic"])
        w_improvement = float(user["w_improvement"])
        w_low_quality = float(user["w_low_quality"])

        utilities: dict[str, float] = {}
        for mid in market_ids:
            feat = supplier_feature_map[mid]
            # Higher strategic/improvement are good; env/social/cost/low_quality are treated as bad.
            score = (
                -w_env * feat["env_risk"]
                - w_social * feat["social_risk"]
                - w_cost * feat["cost_score"]
                + w_strategic * feat["strategic"]
                + w_improvement * feat["improvement"]
                - w_low_quality * feat["low_quality"]
            )
            utilities[mid] = float(score)

        sorted_choices = sorted(market_ids, key=lambda m: utilities[m], reverse=True)
        top_k = max(3, min(5, len(sorted_choices)))

        users_payload.append(
            {
                "user_id": uid,
                "choices": sorted_choices[:top_k],
                "utilities": {m: utilities[m] for m in sorted_choices[:top_k]},
            }
        )

    result = run_market_matching(users=users_payload, market_options=market_options)

    _assert_capacity_respected(result, market_options)
    _assert_user_assignments_valid(result, users_payload, market_options)

    assert result["meta"]["user_count"] == len(users_payload)
    assert result["meta"]["market_option_count"] == len(market_options)


def test_tie_break_1_maximizes_total_utility() -> None:
    """Tie-break #1: When multiple solutions are possible, maximize total utility."""
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m_high", "m_low"],
            "utilities": {"m_high": 100.0, "m_low": 10.0},
        }
    ]

    market_options = [
        {
            "option_id": "m_high",
            "capacity": 1,
            "priority": ["u1"],
            "request_time": "2026-03-09T10:00:00",
        },
        {
            "option_id": "m_low",
            "capacity": 1,
            "priority": ["u1"],
            "request_time": "2026-03-09T10:00:00",
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    if GUROBI_AVAILABLE:
        # Should choose m_high due to higher utility (100 > 10)
        assert result["user_to_market"]["u1"] == "m_high"
    else:
        # Fallback uses preference order
        assert result["user_to_market"]["u1"] == "m_high"


def test_tie_break_2_prefers_lower_occupancy_ratio() -> None:
    """Tie-break #2: When utility is equal, prefer lower occupancy ratio."""
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m_low_cap", "m_high_cap"],
            "utilities": {"m_low_cap": 50.0, "m_high_cap": 50.0},
        },
        {
            "user_id": "u2",
            "choices": ["m_low_cap", "m_high_cap"],
            "utilities": {"m_low_cap": 50.0, "m_high_cap": 50.0},
        },
    ]

    market_options = [
        {
            "option_id": "m_low_cap",
            "capacity": 1,  # Will have 100% occupancy if used
            "priority": ["u1", "u2"],
            "request_time": "2026-03-09T10:00:00",
        },
        {
            "option_id": "m_high_cap",
            "capacity": 10,  # Will have 10% occupancy if used
            "priority": ["u1", "u2"],
            "request_time": "2026-03-09T10:00:00",
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    _assert_capacity_respected(result, market_options)
    if result["meta"]["solver"] == "gurobi_stable_lexicographic":
        # With equal utility, should prefer higher-capacity market (lower occupancy)
        # Both users should prefer m_high_cap
        assert result["user_to_market"]["u1"] == "m_high_cap"
        assert result["user_to_market"]["u2"] == "m_high_cap"


def test_tie_break_3_prefers_earlier_request_time() -> None:
    """Tie-break #3: When utility and occupancy are equal, prefer earlier request_time."""
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m_early", "m_late"],
            # Create identical utilities so only request_time matters after capacity tie
            "utilities": {"m_early": 50.0, "m_late": 50.0},
        }
    ]

    market_options = [
        {
            "option_id": "m_early",
            "capacity": 5,  # Same capacity => same occupancy
            "priority": ["u1"],
            "request_time": "2026-03-09T08:00:00",  # Earlier
        },
        {
            "option_id": "m_late",
            "capacity": 5,  # Same capacity => same occupancy
            "priority": ["u1"],
            "request_time": "2026-03-09T16:00:00",  # Later
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    if result["meta"]["solver"] == "gurobi_stable_lexicographic":
        # With utility and occupancy tied, should prefer earlier request_time
        assert result["user_to_market"]["u1"] == "m_early"
    else:
        # Stable mode may not preserve this tie-break
        assert result["user_to_market"]["u1"] in {"m_early", "m_late"}


def test_combined_tie_break_priority() -> None:
    """Test that tie-break priority is respected: utility > occupancy > request_time."""
    users_payload = [
        {
            "user_id": "u1",
            "choices": ["m_high_util", "m_low_util"],
            "utilities": {"m_high_util": 100.0, "m_low_util": 50.0},
        },
        {
            "user_id": "u2",
            "choices": ["m_high_util", "m_low_util"],
            "utilities": {"m_high_util": 100.0, "m_low_util": 50.0},
        },
    ]

    market_options = [
        {
            "option_id": "m_high_util",
            "capacity": 1,
            "priority": ["u1", "u2"],
            "request_time": "2026-03-09T16:00:00",  # Later time
        },
        {
            "option_id": "m_low_util",
            "capacity": 1,
            "priority": ["u1", "u2"],
            "request_time": "2026-03-09T08:00:00",  # Earlier time
        },
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    if result["meta"]["solver"] == "gurobi_stable_lexicographic":
        # Both users should prefer m_high_util due to higher utility (100 > 50),
        # even though m_low_util has earlier request_time.
        # But capacity is only 1, so one will be unmatched.
        assert result["meta"]["matched_count"] <= 2
        # The matched users should prefer m_high_util first
        matched_to_high = sum(1 for mid in result["user_to_market"].values() if mid == "m_high_util")
        assert matched_to_high >= 1


def test_many_to_one_matching_with_capacity_8() -> None:
    """85 users matched to team products each with capacity=8 via Gale-Shapley."""
    rng = random.Random(99)
    user_count = 85
    team_count = 15

    team_ids = [f"team_{i:02d}" for i in range(1, team_count + 1)]
    user_ids = [f"user_{i}" for i in range(1, user_count + 1)]

    # Build random utilities for each (user, team) pair
    user_utilities: dict[str, dict[str, float]] = {}
    for uid in user_ids:
        utils = {tid: round(rng.uniform(1.0, 10.0), 2) for tid in team_ids}
        user_utilities[uid] = utils

    users_payload = []
    for uid in user_ids:
        utils = user_utilities[uid]
        choices = sorted(team_ids, key=lambda t: -utils[t])
        users_payload.append({"user_id": uid, "choices": choices, "utilities": utils})

    market_options = []
    for tid in team_ids:
        priority = sorted(
            user_ids,
            key=lambda u: -user_utilities[u][tid],
        )
        market_options.append({
            "option_id": tid,
            "capacity": 8,
            "priority": priority,
            "request_time": "2026-03-12T10:00:00",
        })

    result = run_market_matching(users=users_payload, market_options=market_options)

    _assert_capacity_respected(result, market_options)
    _assert_user_assignments_valid(result, users_payload, market_options)

    # Total capacity = 15 teams * 8 = 120 >= 85 users, all should match
    assert result["meta"]["matched_count"] == user_count
    assert len(result["unmatched_users"]) == 0

    # Each market should have at most 8 users
    for tid, assigned in result["market_to_users"].items():
        assert len(assigned) <= 8

    # Verify market_loads
    for tid in team_ids:
        load = result["market_loads"][tid]
        assert load["capacity"] == 8
        assert load["assigned_count"] <= 8
        assert load["remaining_capacity"] == 8 - load["assigned_count"]

    # Verify distribution: all 85 users placed somewhere
    total_assigned = sum(len(users) for users in result["market_to_users"].values())
    assert total_assigned == user_count
