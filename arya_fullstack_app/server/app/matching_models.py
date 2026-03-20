from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_request_time(value: Any) -> float:
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
