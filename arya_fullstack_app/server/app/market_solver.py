from __future__ import annotations

import hashlib
import math
import os
from typing import Any

from .matching_models import MarketNode, MatchingResult, UserNode, parse_request_time, safe_float


def from_market_payload(
    users: list[dict[str, Any]],
    market_options: list[dict[str, Any]],
) -> tuple[
    list[UserNode],
    list[MarketNode],
    dict[tuple[str, str], float],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """
    Convert DB/API payloads into typed matching entities.

    Expected shape:
    - users: [{"user_id": "u1", "choices": ["m1", "m2"], "utilities": {"m1": 5.0}}]
    - market_options: [{"option_id": "m1", "capacity": 2, "priority": ["u2", "u1"], "request_time": "..."}]
    """
    user_nodes: list[UserNode] = []
    market_nodes: list[MarketNode] = []
    utility_map: dict[tuple[str, str], float] = {}
    user_sensitivity_map: dict[str, dict[str, float]] = {}
    market_feature_map: dict[str, dict[str, float]] = {}

    market_ids: list[str] = []
    for option in market_options:
        market_id = str(option.get("option_id", "")).strip()
        if not market_id:
            continue

        capacity = int(option.get("capacity", 0))
        priority_raw = option.get("priority") or []
        priority = [str(x).strip() for x in priority_raw if str(x).strip()]

        request_time = option.get("request_time", option.get("request_ts", option.get("created_at")))
        request_time_score = parse_request_time(request_time)

        market_nodes.append(
            MarketNode(
                market_id=market_id,
                capacity=capacity,
                preference_list=priority,
                request_time_score=request_time_score,
            )
        )
        market_feature_map[market_id] = {
            "price": safe_float(
                option.get("price", option.get("unit_price", option.get("sale_price_per_user", option.get("avg_cost")))),
                0.0,
            ),
            "sustainability": safe_float(
                option.get(
                    "sustainability",
                    option.get("sustainability_score", option.get("general_sustainability", option.get("esg_score"))),
                ),
                0.0,
            ),
        }
        market_ids.append(market_id)

    market_ids_set = set(market_ids)

    for user in users:
        user_id = str(user.get("user_id", "")).strip()
        if not user_id:
            continue

        choices_raw = user.get("choices") or []
        choices = [str(x).strip() for x in choices_raw if str(x).strip() and str(x).strip() in market_ids_set]

        # If explicit choices are missing, user can be considered for all market options.
        if not choices:
            choices = list(market_ids)

        user_nodes.append(UserNode(user_id=user_id, preference_list=choices))

        raw_price_sens = user.get("price_sensitivity", user.get("w_cost"))
        sustainability_default = (
            safe_float(user.get("w_env"), 0.0)
            + safe_float(user.get("w_social"), 0.0)
            + safe_float(user.get("w_low_quality"), 0.0)
        ) / 3.0
        raw_sustainability_sens = user.get("sustainability_sensitivity", sustainability_default)

        user_sensitivity_map[user_id] = {
            "price": safe_float(raw_price_sens, 1.0),
            "sustainability": safe_float(raw_sustainability_sens, 1.0),
        }

        # Utility source priority:
        # 1) user['utilities'][market_id]
        # 2) inverse rank from choices (top choice gets higher utility)
        explicit_util = user.get("utilities") or {}
        if isinstance(explicit_util, dict):
            for market_id in choices:
                if market_id in explicit_util:
                    utility_map[(user_id, market_id)] = safe_float(explicit_util[market_id], 0.0)

        ranked_default = {market_id: float(len(choices) - idx) for idx, market_id in enumerate(choices)}
        for market_id, score in ranked_default.items():
            utility_map.setdefault((user_id, market_id), score)

    return user_nodes, market_nodes, utility_map, user_sensitivity_map, market_feature_map


def run_market_demand_allocation(
    users: list[UserNode],
    market_options: list[MarketNode],
    utility_map: dict[tuple[str, str], float],
    user_sensitivity_map: dict[str, dict[str, float]],
    market_feature_map: dict[str, dict[str, float]],
) -> MatchingResult:
    market_by_id = {m.market_id: m for m in market_options}
    market_to_users: dict[str, list[str]] = {m.market_id: [] for m in market_options}
    user_to_market: dict[str, str | None] = {u.user_id: None for u in users}
    remaining_capacity: dict[str, int] = {m.market_id: max(0, int(m.capacity)) for m in market_options}

    temperature = safe_float(os.getenv("MARKET_SOFTMAX_TEMPERATURE", "1.0"), 1.0)
    if temperature <= 0.0:
        temperature = 1.0

    def _softmax_probabilities(logits: list[float]) -> list[float]:
        if not logits:
            return []
        max_logit = max(logits)
        exps = [math.exp((x - max_logit) / temperature) for x in logits]
        total = sum(exps)
        if total <= 0.0:
            return [1.0 / float(len(logits)) for _ in logits]
        return [v / total for v in exps]

    def _deterministic_uniform_0_1(key: str) -> float:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
        integer = int(digest, 16)
        return integer / float(16**16)

    # Softmax-like per-user demand over market options.
    # Note: utilities and sustainability scores are pre-inverted by sessions.py:
    #   utilities = (higher user preference = higher score)
    #   sustainability = (5-env + 5-social + ... ) / 5 = (higher = better)
    #   price = actual value (lower = better, so negate)
    # demand_score = utility + sustainability_sensitivity*sustainability - price_sensitivity*price
    per_user_market_probs: dict[str, list[tuple[str, float]]] = {}
    user_priority: list[tuple[float, str]] = []
    for user in users:
        sens = user_sensitivity_map.get(user.user_id, {})
        price_sens = safe_float(sens.get("price"), 1.0)
        sustainability_sens = safe_float(sens.get("sustainability"), 1.0)

        valid_markets: list[str] = []
        logits: list[float] = []

        for market_id in user.preference_list:
            market = market_by_id.get(market_id)
            if market is None:
                continue

            feat = market_feature_map.get(market_id, {})
            market_price = safe_float(feat.get("price"), 0.0)
            market_sustainability = safe_float(feat.get("sustainability"), 0.0)
            base_utility = safe_float(utility_map.get((user.user_id, market_id)), 0.0)

            # Utilities and sustainability already "higher is better" (pre-inverted in sessions.py).
            # Only price needs negation (lower is better = subtract).
            demand_score = base_utility + (sustainability_sens * market_sustainability) - (price_sens * market_price)
            logits.append(demand_score)
            valid_markets.append(market_id)

        probs = _softmax_probabilities(logits)
        per_user_market_probs[user.user_id] = list(zip(valid_markets, probs))
        user_priority.append((max(probs) if probs else 0.0, user.user_id))

    # Assign high-confidence users first while preserving deterministic behavior.
    user_priority.sort(key=lambda x: (-x[0], x[1]))

    for _confidence, user_id in user_priority:
        market_prob_pairs = per_user_market_probs.get(user_id, [])
        available = [(market_id, p) for market_id, p in market_prob_pairs if remaining_capacity.get(market_id, 0) > 0]
        if not available:
            continue

        total_available_prob = sum(p for _, p in available)
        if total_available_prob <= 0.0:
            continue

        normalized = [(market_id, p / total_available_prob) for market_id, p in available]
        draw = _deterministic_uniform_0_1(user_id)
        cumulative = 0.0
        selected_market: str | None = None
        for market_id, prob in normalized:
            cumulative += prob
            if draw <= cumulative:
                selected_market = market_id
                break

        if selected_market is None:
            selected_market = normalized[-1][0]

        if remaining_capacity.get(selected_market, 0) <= 0:
            continue

        market_to_users[selected_market].append(user_id)
        user_to_market[user_id] = selected_market
        remaining_capacity[selected_market] = max(0, remaining_capacity[selected_market] - 1)

    unmatched = [uid for uid, market_id in user_to_market.items() if market_id is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
    )


def run_market_demand_allocation_proportional(
    users: list[UserNode],
    market_options: list[MarketNode],
    utility_map: dict[tuple[str, str], float],
    user_sensitivity_map: dict[str, dict[str, float]],
    market_feature_map: dict[str, dict[str, float]],
) -> MatchingResult:
    """
    Proportional allocation: each user's demand is split across markets according to softmax probabilities.
    In proportional mode, capacities are not enforced.
    """
    market_by_id = {m.market_id: m for m in market_options}
    temperature = safe_float(os.getenv("MARKET_SOFTMAX_TEMPERATURE", "1.0"), 1.0)
    if temperature <= 0.0:
        temperature = 1.0

    def _softmax_probabilities(logits: list[float]) -> list[float]:
        if not logits:
            return []
        max_logit = max(logits)
        exps = [math.exp((x - max_logit) / temperature) for x in logits]
        total = sum(exps)
        if total <= 0.0:
            return [1.0 / float(len(logits)) for _ in logits]
        return [v / total for v in exps]

    # Step 1: Calculate probabilities for each user-market pair.
    # Note: utilities and sustainability scores are pre-inverted by sessions.py:
    #   utilities = (higher user preference = higher score)
    #   sustainability = (5-env + 5-social + ... ) / 5 = (higher = better)
    #   price = actual value (lower = better, so negate)
    # demand_score = utility + sustainability_sensitivity*sustainability - price_sensitivity*price
    per_user_market_probs: dict[str, list[tuple[str, float]]] = {}
    for user in users:
        sens = user_sensitivity_map.get(user.user_id, {})
        price_sens = safe_float(sens.get("price"), 1.0)
        sustainability_sens = safe_float(sens.get("sustainability"), 1.0)

        valid_markets: list[str] = []
        logits: list[float] = []

        for market_id in user.preference_list:
            market = market_by_id.get(market_id)
            if market is None:
                continue

            feat = market_feature_map.get(market_id, {})
            market_price = safe_float(feat.get("price"), 0.0)
            market_sustainability = safe_float(feat.get("sustainability"), 0.0)
            base_utility = safe_float(utility_map.get((user.user_id, market_id)), 0.0)

            # Utilities and sustainability already "higher is better" (pre-inverted in sessions.py).
            # Only price needs negation (lower is better = subtract).
            demand_score = base_utility + (sustainability_sens * market_sustainability) - (price_sens * market_price)
            logits.append(demand_score)
            valid_markets.append(market_id)

        probs = _softmax_probabilities(logits)
        per_user_market_probs[user.user_id] = list(zip(valid_markets, probs))

    # Step 2: Use raw probabilities as final fractional allocations.
    fractional_allocations: dict[str, dict[str, float]] = {}
    for user in users:
        fractional_allocations[user.user_id] = {}
        market_probs = per_user_market_probs.get(user.user_id, [])
        for market_id, prob in market_probs:
            if prob > 0.0:
                fractional_allocations[user.user_id][market_id] = prob

    # Step 3: Create backward-compatible output (primary market = largest allocation)
    market_to_users: dict[str, list[str]] = {m.market_id: [] for m in market_options}
    user_to_market: dict[str, str | None] = {}
    for user in users:
        allocations = fractional_allocations.get(user.user_id, {})
        if allocations:
            primary_market = max(allocations.items(), key=lambda x: x[1])[0]
            user_to_market[user.user_id] = primary_market
            market_to_users[primary_market].append(user.user_id)
        else:
            user_to_market[user.user_id] = None

    unmatched = [uid for uid, market_id in user_to_market.items() if market_id is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
        fractional_allocations=fractional_allocations,
    )
