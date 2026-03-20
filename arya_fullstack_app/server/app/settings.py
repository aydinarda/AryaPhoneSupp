from __future__ import annotations

from dataclasses import dataclass

from .optimization_controller import Policy


@dataclass(frozen=True)
class GameSettings:
    served_users: int = 8          # used for profit calc (profit_total = served_users * profit_per_user)
    default_market_capacity: int = 8  # max users matched per team product
    env_cap: float = 2.75
    social_cap: float = 3.0
    cost_scale: float = 10.0
    price_per_user: float = 100.0


GAME_SETTINGS = GameSettings()

FIXED_POLICY = Policy(
    env_mult=1.0,
    social_mult=1.0,
    cost_mult=0.0,
    strategic_mult=1.0,
    improvement_mult=1.0,
    low_quality_mult=1.0,
    child_labor_penalty=0.0,
    banned_chem_penalty=0.0,
)
