from __future__ import annotations

from dataclasses import dataclass

from .mincost_agent import Policy


@dataclass(frozen=True)
class GameSettings:
    served_users: int = 8
    env_cap: float = 2.75
    social_cap: float = 3.0
    cost_scale: float = 10.0
    price_per_user: float = 100.0


GAME_SETTINGS = GameSettings()

FIXED_POLICY = Policy(
    env_mult=1.0,
    social_mult=1.0,
    cost_mult=1.0,
    strategic_mult=1.0,
    improvement_mult=1.0,
    low_quality_mult=1.0,
    child_labor_penalty=0.0,
    banned_chem_penalty=0.0,
)
