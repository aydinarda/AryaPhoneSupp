from __future__ import annotations

import os
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
    User-proposing Gale-Shapley fallback for many-to-one market matching.

    This path is used when Gurobi is unavailable.
    """
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
) -> tuple[list[UserNode], list[MarketNode], dict[tuple[str, str], float]]:
    """
    Convert DB/API payloads into typed matching entities.

    Expected shape:
    - users: [{"user_id": "u1", "choices": ["m1", "m2"], "utilities": {"m1": 5.0}}]
    - market_options: [{"option_id": "m1", "capacity": 2, "priority": ["u2", "u1"], "request_time": "..."}]
    """
    user_nodes: list[UserNode] = []
    market_nodes: list[MarketNode] = []
    utility_map: dict[tuple[str, str], float] = {}

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

    return user_nodes, market_nodes, utility_map


def _solve_with_gurobi_tie_breaks(
    users: list[UserNode],
    market_options: list[MarketNode],
    utility_map: dict[tuple[str, str], float],
) -> MatchingResult:
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


def run_market_matching(users: list[dict[str, Any]], market_options: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Orchestration layer for future DB integration.

    Default solver mode is stable many-to-one Gale-Shapley.
    Set MATCHING_SOLVER=gurobi to opt into utility-driven lexicographic optimization.

    Tie-break policy (inside the stable feasible set when using Gurobi):
    1) Maximize total utility with Gurobi.
    2) Among equal utility solutions, prefer lower occupancy ratio.
    3) If still tied, prefer earlier request time.
    """
    user_nodes, market_nodes, utility_map = from_market_payload(users=users, market_options=market_options)

    solver_mode = os.getenv("MATCHING_SOLVER", "stable").strip().lower()

    if solver_mode not in {"stable", "gurobi"}:
        solver_mode = "stable"

    if solver_mode == "gurobi" and GUROBI_AVAILABLE:
        result = _solve_with_gurobi_tie_breaks(
            users=user_nodes,
            market_options=market_nodes,
            utility_map=utility_map,
        )
        solver_name = "gurobi_stable_lexicographic"
    else:
        result = stable_many_to_one_matching(users=user_nodes, market_options=market_nodes)
        solver_name = "stable_gale_shapley"

    market_loads = {
        market.market_id: {
            "assigned_count": len(result.market_to_users.get(market.market_id, [])),
            "capacity": int(market.capacity),
            "remaining_capacity": max(0, int(market.capacity) - len(result.market_to_users.get(market.market_id, []))),
        }
        for market in market_nodes
    }

    return {
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
        },
    }
