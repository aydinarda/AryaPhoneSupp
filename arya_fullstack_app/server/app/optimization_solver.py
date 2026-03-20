from __future__ import annotations

import warnings
from typing import Any

from .matching_models import MarketNode, MatchingResult, UserNode

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False


def solve_with_gurobi_tie_breaks(
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
    for user in users:
        user_rank[user.user_id] = {mid: idx for idx, mid in enumerate(user.preference_list)}

    model = gp.Model("market_user_matching")
    model.Params.OutputFlag = 0
    model.ModelSense = GRB.MAXIMIZE

    x: dict[tuple[str, str], Any] = {}
    for user in users:
        for market_id in user.preference_list:
            if market_id not in market_by_id:
                continue
            x[(user.user_id, market_id)] = model.addVar(vtype=GRB.BINARY, name=f"x_{user.user_id}_{market_id}")

    # Each user can match to at most one market option.
    for user_id in user_ids:
        user_edges = [x[(u, m)] for (u, m) in x if u == user_id]
        if user_edges:
            model.addConstr(gp.quicksum(user_edges) <= 1, name=f"user_cap_{user_id}")

    # Market capacities.
    for market_id in market_ids:
        cap = max(0, int(market_by_id[market_id].capacity))
        market_edges = [x[(u, m)] for (u, m) in x if m == market_id]
        if market_edges:
            model.addConstr(gp.quicksum(market_edges) <= cap, name=f"market_cap_{market_id}")

    # Stability constraints (blocking-pair elimination) for Hospital-Residents.
    # For each acceptable pair (u, m):
    # If u is not assigned to m or a better market, then m must be full with users
    # that m strictly prefers over u.
    for (user_id, market_id), _var in x.items():
        cap = max(0, int(market_by_id[market_id].capacity))
        if cap == 0:
            continue

        user_rank_market = user_rank[user_id][market_id]
        better_or_equal_markets = [
            m2
            for m2 in user_by_id[user_id].preference_list
            if user_rank[user_id].get(m2, 10**6) <= user_rank_market and (user_id, m2) in x
        ]

        market_rank_user = market_rank[market_id][user_id]
        better_users_for_market = [
            u2 for u2 in user_ids if (u2, market_id) in x and market_rank[market_id].get(u2, 10**6) < market_rank_user
        ]

        lhs = gp.quicksum(x[(u2, market_id)] for u2 in better_users_for_market)
        rhs = cap * (1 - gp.quicksum(x[(user_id, m2)] for m2 in better_or_equal_markets))
        model.addConstr(lhs >= rhs, name=f"stable_{user_id}_{market_id}")

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

    market_to_users: dict[str, list[str]] = {market_id: [] for market_id in market_ids}
    user_to_market: dict[str, str | None] = {user_id: None for user_id in user_ids}

    for (user_id, market_id), var in x.items():
        if var.X > 0.5:
            market_to_users[market_id].append(user_id)
            user_to_market[user_id] = market_id

    unmatched = [user_id for user_id, market_id in user_to_market.items() if market_id is None]
    return MatchingResult(
        market_to_users=market_to_users,
        user_to_market=user_to_market,
        unmatched_users=unmatched,
    )
