from __future__ import annotations

import os
import warnings
from typing import Any

from .market_solver import (
    from_market_payload,
    run_market_demand_allocation,
    run_market_demand_allocation_proportional,
)
from .matching_models import MarketNode, MatchingResult, UserNode
from .matching_solver import stable_many_to_one_matching
from .optimization_solver import GUROBI_AVAILABLE, solve_with_gurobi_tie_breaks

DEPRECATED_MATCHING_SOLVERS = {"stable", "gurobi"}
ACTIVE_MATCHING_SOLVER = "market_demand"

# Backward-compatible aliases for existing imports/tests.
_run_market_demand_allocation = run_market_demand_allocation
_run_market_demand_allocation_proportional = run_market_demand_allocation_proportional
_solve_with_gurobi_tie_breaks = solve_with_gurobi_tie_breaks


def run_market_matching(users: list[dict[str, Any]], market_options: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Thin orchestration layer.

    Active mode is market demand allocation.
    Deprecated modes (stable/gurobi) are ignored and routed to market demand.
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

    allocation_mode = os.getenv("ALLOCATION_MODE", "proportional").strip().lower()
    if allocation_mode == "proportional":
        result = run_market_demand_allocation_proportional(
            users=user_nodes,
            market_options=market_nodes,
            utility_map=utility_map,
            user_sensitivity_map=user_sensitivity_map,
            market_feature_map=market_feature_map,
        )
    else:
        result = run_market_demand_allocation(
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
