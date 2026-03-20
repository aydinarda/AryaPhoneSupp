from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Customer:
    customer_id: str
    env_risk: float = 0.0
    social_risk: float = 0.0
    price: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Customer":
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        return cls(
            customer_id=str(payload.get("customer_id", payload.get("user_id", payload.get("id", "")))).strip(),
            env_risk=_to_float(payload.get("env_risk"), 0.0),
            social_risk=_to_float(payload.get("social_risk"), 0.0),
            price=_to_float(payload.get("price", payload.get("cost_score", payload.get("w_cost"))), 0.0),
        )
