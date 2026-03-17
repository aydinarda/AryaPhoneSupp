from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False


@dataclass(slots=True)
class UserNode:
    user_id: str
    preference_list: list[str]


@dataclass(slots=True)
class MarketNode:
    market_id: str
    capacity: int
    preference_list: list[str]
    request_time_score: float = 0.0
    _rank: dict[str, int] = field(default_factory=dict, repr=False)

    def build_rank(self) -> None:
        # Smaller index means higher priority for the market option.
        self._rank = {user_id: idx for idx, user_id in enumerate(self.preference_list)}

    def rank_of(self, user_id: str) -> int:
        # Unknown users are treated as lowest priority.
        return self._rank.get(user_id, len(self.preference_list) + 10**6)


@dataclass(slots=True)
class MatchingResult:
    market_to_users: dict[str, list[str]]
    user_to_market: dict[str, str | None]
    unmatched_users: list[str]
    fractional_allocations: dict[str, dict[str, float]] = field(default_factory=dict)


DEPRECATED_MATCHING_SOLVERS = {"stable", "gurobi"}
ACTIVE_MATCHING_SOLVER = "market_demand"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_request_time(value: Any) -> float:
    if value is None:
        return float(10**12)

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return float(10**12)

    # Supports ISO strings like 2026-03-09T15:02:01
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return float(10**12)


def stable_many_to_one_matching(users: list[UserNode], market_options: list[MarketNode]) -> MatchingResult:
    """
    Deprecated.

    User-proposing Gale-Shapley fallback for many-to-one market matching.

    This path is used when Gurobi is unavailable.
    """
    warnings.warn(
        "stable_many_to_one_matching is deprecated and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    market_map = {m.market_id: m for m in market_options}
    for market in market_options:
        market.build_rank()
        if market.capacity < 0:
            raise ValueError(f"Market capacity cannot be negative: {market.market_id}")

    user_to_market: dict[str, str | None] = {u.user_id: None for u in users}
    proposal_index: dict[str, int] = {u.user_id: 0 for u in users}
    free_users: list[str] = [u.user_id for u in users]
    pref_map: dict[str, list[str]] = {u.user_id: list(u.preference_list) for u in users}

    market_to_users: dict[str, list[str]] = {m.market_id: [] for m in market_options}

    while free_users:
        user_id = free_users.pop(0)
        prefs = pref_map[user_id]
        idx = proposal_index[user_id]

        if idx >= len(prefs):
            continue

        market_id = prefs[idx]
        proposal_index[user_id] += 1
        market = market_map.get(market_id)

        if market is None:
            free_users.append(user_id)
            continue

        accepted = market_to_users[market_id]
        accepted.append(user_id)
        accepted.sort(key=market.rank_of)

        if len(accepted) <= market.capacity:
            user_to_market[user_id] = market_id
            continue

        removed = accepted.pop()
        user_to_market[removed] = None

        if removed != user_id:
            user_to_market[user_id] = market_id

        if proposal_index[removed] < len(pref_map[removed]):
            free_users.append(removed)

    unmatched = [uid for uid, mid in user_to_market.items() if mid is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
    )


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
        request_time_score = _parse_request_time(request_time)

        market_nodes.append(
            MarketNode(
                market_id=market_id,
                capacity=capacity,
                preference_list=priority,
                request_time_score=request_time_score,
            )
        )
        market_feature_map[market_id] = {
            "price": _safe_float(
                option.get("price", option.get("unit_price", option.get("sale_price_per_user", option.get("avg_cost")))),
                0.0,
            ),
            "sustainability": _safe_float(
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
            _safe_float(user.get("w_env"), 0.0)
            + _safe_float(user.get("w_social"), 0.0)
            + _safe_float(user.get("w_low_quality"), 0.0)
        ) / 3.0
        raw_sustainability_sens = user.get("sustainability_sensitivity", sustainability_default)

        user_sensitivity_map[user_id] = {
            "price": _safe_float(raw_price_sens, 1.0),
            "sustainability": _safe_float(raw_sustainability_sens, 1.0),
        }

        # Utility source priority:
        # 1) user['utilities'][market_id]
        # 2) inverse rank from choices (top choice gets higher utility)
        explicit_util = user.get("utilities") or {}
        if isinstance(explicit_util, dict):
            for mid in choices:
                if mid in explicit_util:
                    utility_map[(user_id, mid)] = _safe_float(explicit_util[mid], 0.0)

        ranked_default = {mid: float(len(choices) - idx) for idx, mid in enumerate(choices)}
        for mid, score in ranked_default.items():
            utility_map.setdefault((user_id, mid), score)

    return user_nodes, market_nodes, utility_map, user_sensitivity_map, market_feature_map


def _run_market_demand_allocation(
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

    temperature = _safe_float(os.getenv("MARKET_SOFTMAX_TEMPERATURE", "1.0"), 1.0)
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
        price_sens = _safe_float(sens.get("price"), 1.0)
        sustainability_sens = _safe_float(sens.get("sustainability"), 1.0)

        valid_markets: list[str] = []
        logits: list[float] = []

        for market_id in user.preference_list:
            market = market_by_id.get(market_id)
            if market is None:
                continue

            feat = market_feature_map.get(market_id, {})
            market_price = _safe_float(feat.get("price"), 0.0)
            market_sustainability = _safe_float(feat.get("sustainability"), 0.0)
            base_utility = _safe_float(utility_map.get((user.user_id, market_id)), 0.0)

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
        available = [(mid, p) for mid, p in market_prob_pairs if remaining_capacity.get(mid, 0) > 0]
        if not available:
            continue

        total_available_prob = sum(p for _, p in available)
        if total_available_prob <= 0.0:
            continue

        normalized = [(mid, p / total_available_prob) for mid, p in available]
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

    unmatched = [uid for uid, mid in user_to_market.items() if mid is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
    )


def _solve_with_gurobi_tie_breaks(
    users: list[UserNode],
    market_options: list[MarketNode],
    utility_map: dict[tuple[str, str], float],
) -> MatchingResult:
    warnings.warn(
        "_solve_with_gurobi_tie_breaks is deprecated and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    user_ids = [u.user_id for u in users]
    market_ids = [m.market_id for m in market_options]
    market_by_id = {m.market_id: m for m in market_options}
    user_by_id = {u.user_id: u for u in users}

    # Build strict market-side rankings by completing missing users at the end.
    # This avoids ties for users omitted from explicit priority lists.
    market_rank: dict[str, dict[str, int]] = {}
    for market in market_options:
        explicit = [uid for uid in market.preference_list if uid in user_by_id]
        seen = set(explicit)
        missing = sorted(uid for uid in user_ids if uid not in seen)
        ordered = explicit + missing
        market_rank[market.market_id] = {uid: idx for idx, uid in enumerate(ordered)}

    # Build user-side ranking over listed choices (strict by list order).
    user_rank: dict[str, dict[str, int]] = {}
    for u in users:
        user_rank[u.user_id] = {mid: idx for idx, mid in enumerate(u.preference_list)}

    model = gp.Model("market_user_matching")
    model.Params.OutputFlag = 0
    model.ModelSense = GRB.MAXIMIZE

    x: dict[tuple[str, str], Any] = {}
    for u in users:
        for mid in u.preference_list:
            if mid not in market_by_id:
                continue
            x[(u.user_id, mid)] = model.addVar(vtype=GRB.BINARY, name=f"x_{u.user_id}_{mid}")

    # Each user can match to at most one market option.
    for uid in user_ids:
        user_edges = [x[(u, m)] for (u, m) in x if u == uid]
        if user_edges:
            model.addConstr(gp.quicksum(user_edges) <= 1, name=f"user_cap_{uid}")

    # Market capacities.
    for mid in market_ids:
        cap = max(0, int(market_by_id[mid].capacity))
        market_edges = [x[(u, m)] for (u, m) in x if m == mid]
        if market_edges:
            model.addConstr(gp.quicksum(market_edges) <= cap, name=f"market_cap_{mid}")

    # Stability constraints (blocking-pair elimination) for Hospital-Residents.
    # For each acceptable pair (u, m):
    # If u is not assigned to m or a better market, then m must be full with users
    # that m strictly prefers over u.
    for (uid, mid), _var in x.items():
        cap = max(0, int(market_by_id[mid].capacity))
        if cap == 0:
            continue

        u_rank_m = user_rank[uid][mid]
        better_or_equal_markets = [
            m2 for m2 in user_by_id[uid].preference_list if user_rank[uid].get(m2, 10**6) <= u_rank_m and (uid, m2) in x
        ]

        m_rank_u = market_rank[mid][uid]
        better_users_for_market = [
            u2 for u2 in user_ids if (u2, mid) in x and market_rank[mid].get(u2, 10**6) < m_rank_u
        ]

        lhs = gp.quicksum(x[(u2, mid)] for u2 in better_users_for_market)
        rhs = cap * (1 - gp.quicksum(x[(uid, m2)] for m2 in better_or_equal_markets))
        model.addConstr(lhs >= rhs, name=f"stable_{uid}_{mid}")

    utility_expr = gp.quicksum(utility_map.get((u, m), 0.0) * var for (u, m), var in x.items())

    # Tie-break #2: lower occupancy ratio.
    occupancy_expr = gp.quicksum((1.0 / max(1, int(market_by_id[m].capacity))) * var for (u, m), var in x.items())

    # Tie-break #3: earlier request time.
    request_time_expr = gp.quicksum(market_by_id[m].request_time_score * var for (u, m), var in x.items())

    # Lexicographic objectives:
    # 1) maximize utility
    # 2) minimize occupancy ratio (implemented as maximize negative occupancy)
    # 3) minimize request time (implemented as maximize negative timestamp)
    model.setObjectiveN(utility_expr, index=0, priority=3, weight=1.0, name="max_total_utility")
    model.setObjectiveN(-occupancy_expr, index=1, priority=2, weight=1.0, name="min_occupancy_ratio")
    model.setObjectiveN(-request_time_expr, index=2, priority=1, weight=1.0, name="earliest_request_time")

    model.optimize()

    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi failed to find an optimal matching (status={model.Status})")

    market_to_users: dict[str, list[str]] = {mid: [] for mid in market_ids}
    user_to_market: dict[str, str | None] = {uid: None for uid in user_ids}

    for (uid, mid), var in x.items():
        if var.X > 0.5:
            market_to_users[mid].append(uid)
            user_to_market[uid] = mid

    unmatched = [uid for uid, mid in user_to_market.items() if mid is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
    )


def _run_market_demand_allocation_proportional(
    users: list[UserNode],
    market_options: list[MarketNode],
    utility_map: dict[tuple[str, str], float],
    user_sensitivity_map: dict[str, dict[str, float]],
    market_feature_map: dict[str, dict[str, float]],
) -> MatchingResult:
    """
    Proportional allocation: each user's demand is split across markets according to softmax probabilities.
    Demands are capacity-constrained via scaling factors.
    """
    market_by_id = {m.market_id: m for m in market_options}
    temperature = _safe_float(os.getenv("MARKET_SOFTMAX_TEMPERATURE", "1.0"), 1.0)
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
        price_sens = _safe_float(sens.get("price"), 1.0)
        sustainability_sens = _safe_float(sens.get("sustainability"), 1.0)

        valid_markets: list[str] = []
        logits: list[float] = []

        for market_id in user.preference_list:
            market = market_by_id.get(market_id)
            if market is None:
                continue

            feat = market_feature_map.get(market_id, {})
            market_price = _safe_float(feat.get("price"), 0.0)
            market_sustainability = _safe_float(feat.get("sustainability"), 0.0)
            base_utility = _safe_float(utility_map.get((user.user_id, market_id)), 0.0)

            # Utilities and sustainability already "higher is better" (pre-inverted in sessions.py).
            # Only price needs negation (lower is better = subtract).
            demand_score = base_utility + (sustainability_sens * market_sustainability) - (price_sens * market_price)
            logits.append(demand_score)
            valid_markets.append(market_id)

        probs = _softmax_probabilities(logits)
        per_user_market_probs[user.user_id] = list(zip(valid_markets, probs))

    # Step 2: Aggregate demand per market and compute scaling factors
    total_demand_per_market: dict[str, float] = {m.market_id: 0.0 for m in market_options}
    for user_id, market_probs in per_user_market_probs.items():
        for market_id, prob in market_probs:
            total_demand_per_market[market_id] += prob

    # Calculate scaling: if total_demand > capacity, scale down
    market_scaling: dict[str, float] = {}
    for market_id, total_demand in total_demand_per_market.items():
        capacity = max(0, int(market_by_id[market_id].capacity))
        if total_demand > 0.0:
            market_scaling[market_id] = min(1.0, capacity / total_demand)
        else:
            market_scaling[market_id] = 1.0

    # Step 3: Apply scaling to get final fractional allocations
    fractional_allocations: dict[str, dict[str, float]] = {}
    for user in users:
        fractional_allocations[user.user_id] = {}
        market_probs = per_user_market_probs.get(user.user_id, [])
        for market_id, prob in market_probs:
            scaled_prob = prob * market_scaling.get(market_id, 1.0)
            if scaled_prob > 0.0:
                fractional_allocations[user.user_id][market_id] = scaled_prob

    # Step 4: Create backward-compatible output (primary market = largest allocation)
    market_to_users: dict[str, list[str]] = {m.market_id: [] for m in market_options}
    user_to_market: dict[str, str | None] = {}
    for user in users:
        allocs = fractional_allocations.get(user.user_id, {})
        if allocs:
            primary_market = max(allocs.items(), key=lambda x: x[1])[0]
            user_to_market[user.user_id] = primary_market
            market_to_users[primary_market].append(user.user_id)
        else:
            user_to_market[user.user_id] = None

    unmatched = [uid for uid, mid in user_to_market.items() if mid is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
        fractional_allocations=fractional_allocations,
    )


def run_market_matching(users: list[dict[str, Any]], market_options: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Orchestration layer for future DB integration.

    Active mode is market demand allocation.

    Deprecated modes (stable/gurobi) are currently ignored and routed to market demand.

    Demand score:
    base_utility + sustainability_sensitivity * sustainability - price_sensitivity * price
    """
    user_nodes, market_nodes, utility_map, user_sensitivity_map, market_feature_map = from_market_payload(
        users=users,
        market_options=market_options,
    )

    requested_solver_mode = os.getenv("MATCHING_SOLVER", ACTIVE_MATCHING_SOLVER).strip().lower()
    deprecated_mode_requested = requested_solver_mode in DEPRECATED_MATCHING_SOLVERS
    if deprecated_mode_requested:
        warnings.warn(
            f"MATCHING_SOLVER={requested_solver_mode} is deprecated; using {ACTIVE_MATCHING_SOLVER}.",
            DeprecationWarning,
            stacklevel=2,
        )

    allocation_mode = os.getenv("ALLOCATION_MODE", "deterministic").strip().lower()
    if allocation_mode == "proportional":
        result = _run_market_demand_allocation_proportional(
            users=user_nodes,
            market_options=market_nodes,
            utility_map=utility_map,
            user_sensitivity_map=user_sensitivity_map,
            market_feature_map=market_feature_map,
        )
    else:
        result = _run_market_demand_allocation(
            users=user_nodes,
            market_options=market_nodes,
            utility_map=utility_map,
            user_sensitivity_map=user_sensitivity_map,
            market_feature_map=market_feature_map,
        )
    solver_name = "market_demand_v1"

    market_loads = {
        market.market_id: {
            "assigned_count": len(result.market_to_users.get(market.market_id, [])),
            "capacity": int(market.capacity),
            "remaining_capacity": max(0, int(market.capacity) - len(result.market_to_users.get(market.market_id, []))),
        }
        for market in market_nodes
    }

    output = {
        "market_to_users": result.market_to_users,
        "user_to_market": result.user_to_market,
        "unmatched_users": result.unmatched_users,
        "market_loads": market_loads,
        "meta": {
            "user_count": len(user_nodes),
            "market_option_count": len(market_nodes),
            "matched_count": len(user_nodes) - len(result.unmatched_users),
            "unmatched_count": len(result.unmatched_users),
            "total_capacity": sum(max(0, int(market.capacity)) for market in market_nodes),
            "solver": solver_name,
            "requested_solver": requested_solver_mode,
            "deprecated_solver_requested": deprecated_mode_requested,
            "allocation_mode": allocation_mode,
        },
    }

    if result.fractional_allocations:
        output["fractional_allocations"] = result.fractional_allocations

    return output
