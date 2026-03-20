from __future__ import annotations

import warnings

from .matching_models import MarketNode, MatchingResult, UserNode


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
