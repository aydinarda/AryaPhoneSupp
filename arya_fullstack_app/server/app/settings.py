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
    # delta: price sensitivity in MNL utility  U = quality - delta * w_cost * price
    # delta=0.1 means a $10 price increase (at w_cost=1) shifts utility by -1.0 (≈ one quality unit)
    price_sensitivity_delta: float = 0.1
    # Audit parameters (admin-configurable per session, default = audit disabled)
    audit_probability: float = 0.0   # P(supplier selected for audit)
    catch_probability: float = 1.0   # P(violation found | audited AND has violation)


GAME_SETTINGS = GameSettings()

FIXED_POLICY = Policy(
    env_mult=1.0,
    social_mult=1.0,
    cost_mult=0.0,
)
