from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buyer import Buyer, BuyerPotential
    from .customer_segment import CustomerSegment


# ---------------------------------------------------------------------------
# Per-segment utility
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SegmentUtility:
    """Utility that a single customer segment derives from a buyer's product."""

    segment_id: str
    utility: float


# ---------------------------------------------------------------------------
# Frictionless outcome calculator
# ---------------------------------------------------------------------------

def compute_segment_utility(
    segment: "CustomerSegment",
    avg_env: float,
    avg_social: float,
) -> SegmentUtility:
    """
    Utility a segment gets from a buyer's product — quality dimensions only.

    max{ Σ_i  density_i * f_i(avg_attributes) }

    f_i = w_env_i   * (5 - avg_env)
        + w_social_i * (5 - avg_social)

    Price is intentionally excluded — it would make the benchmark unbounded.
    Profit is tracked separately via earnings fields.
    """
    return SegmentUtility(
        segment_id=segment.segment_id,
        utility=(
            segment.w_env      * (5.0 - avg_env)
            + segment.w_social * (5.0 - avg_social)
        ),
    )


def compute_potential(
    buyer: "Buyer",
    segments: list["CustomerSegment"],
    total_users: int,
    cost_scale: float,
    u_outside: float = 0.0,  # outside option utility — reserved for future use
) -> "BuyerPotential":
    """
    Compute frictionless outcomes for a buyer — as if they were the sole
    market option (no competitors, no outside option pressure).

    utility  = total_users * Σ_i (d_i / Σd) * f_i(avg_attributes)
    earnings = total_users * (price - cost_scale * avg_cost)
    """
    from .buyer import BuyerPotential

    if not segments:
        return BuyerPotential(utility=0.0, earnings=0.0)

    avg_env  = buyer.avg_env  or 0.0
    avg_social = buyer.avg_social or 0.0
    avg_cost = buyer.avg_cost or 0.0
    price    = buyer.price_per_user or 0.0

    total_density = sum(s.density for s in segments)
    if total_density <= 0.0:
        return BuyerPotential(utility=0.0, earnings=0.0)

    weighted_utility = sum(
        (s.density / total_density)
        * compute_segment_utility(s, avg_env, avg_social).utility
        for s in segments
    )

    unit_margin = price - cost_scale * avg_cost

    return BuyerPotential(
        utility=weighted_utility * float(total_users),
        earnings=unit_margin * float(total_users),
    )
