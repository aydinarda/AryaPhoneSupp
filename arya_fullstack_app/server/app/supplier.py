from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Supplier:
    supplier_id: str
    env_risk: float = 0.0
    social_risk: float = 0.0
    cost_score: float = 0.0
    strategic: float = 0.0
    improvement: float = 0.0
    low_quality: float = 0.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Supplier":
        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return default

        return cls(
            supplier_id=str(payload.get("supplier_id", payload.get("id", ""))).strip(),
            env_risk=_to_float(payload.get("env_risk"), 0.0),
            social_risk=_to_float(payload.get("social_risk"), 0.0),
            cost_score=_to_float(payload.get("cost_score"), 0.0),
            strategic=_to_float(payload.get("strategic"), 0.0),
            improvement=_to_float(payload.get("improvement"), 0.0),
            low_quality=_to_float(payload.get("low_quality"), 0.0),
        )
