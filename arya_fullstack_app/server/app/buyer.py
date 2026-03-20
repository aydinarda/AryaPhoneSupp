from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Buyer:
    team_name: str
    session_token: str
    buyer_utility: float = 0.0
    market_utility: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Buyer":
        buyer_utility = 0.0
        if payload.get("buyer_utility") is not None:
            try:
                buyer_utility = float(payload.get("buyer_utility"))
            except Exception:
                buyer_utility = 0.0

        market_utility = 0.0
        if payload.get("market_utility") is not None:
            try:
                market_utility = float(payload.get("market_utility"))
            except Exception:
                market_utility = 0.0

        return cls(
            team_name=str(payload.get("team_name", payload.get("team", payload.get("buyer_name", "")))).strip(),
            session_token=str(payload.get("session_token", payload.get("sessionToken", ""))).strip(),
            buyer_utility=buyer_utility,
            market_utility=market_utility,
        )
