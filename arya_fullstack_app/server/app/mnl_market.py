"""
mnl_market.py
-------------
Multinomial logit (MNL) demand model for the Arya Phone market game.

Full dot-product formulation
-----------------------------
The MNL logit for buyer i in segment s is the inner product of the buyer's
attribute vector and the segment's weight vector, with price scaled by delta:

    U_{i,s} = w_env_s    * (5 - avg_env_i)
             + w_social_s * (5 - avg_social_i)
             - delta * w_cost_s * price_i

delta (default 1.0) scales the price sensitivity globally.
  delta = 0  →  price has no effect on market share (quality-only competition)
  delta = 1  →  standard MNL (price weighted by each segment's w_cost)
  delta > 1  →  amplified price competition

Realized outcomes per buyer across all segments:

    share_{i,s}           = exp(U_{i,s}) / sum_j exp(U_{j,s})
    demand_{i,s}          = (d_s / sum d) * share_{i,s}
    realized_earnings_i   = sum_s  price_i * demand_{i,s}
    realized_utility_i    = sum_s  q_{i,s} * demand_{i,s}
                            (no price/delta term — comparable with frictionless benchmark)

Computation is parallelised across segments via ThreadPoolExecutor.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buyer import Buyer
    from .customer_segment import CustomerSegment


# ---------------------------------------------------------------------------
# Input / output types
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuyerProfile:
    """
    Snapshot of one buyer's product attributes for a single round.
    Decoupled from the Buyer dataclass so MNL can be called standalone.
    """
    team_name: str
    price_per_user: float
    avg_env: float
    avg_social: float

    @classmethod
    def from_buyer(cls, buyer: "Buyer") -> "BuyerProfile":
        return cls(
            team_name=buyer.team_name,
            price_per_user=buyer.price_per_user or 0.0,
            avg_env=buyer.avg_env or 0.0,
            avg_social=buyer.avg_social or 0.0,
        )


@dataclass(frozen=True, slots=True)
class SegmentAllocation:
    """MNL result for one segment."""
    segment_id: str
    density: float
    # team_name -> MNL share (sums to <= 1 across buyers)
    shares: dict[str, float]


@dataclass(slots=True)
class BuyerMarketResult:
    """Aggregated market outcome for one buyer across all segments."""
    team_name: str
    # sum of (d_s / sum_d) * share_{i,s}  — effective demand fraction
    total_demand: float = 0.0
    # sum of price_i * demand_{i,s}
    realized_earnings: float = 0.0
    # sum of q_{i,s} * demand_{i,s}  — no price/delta term
    realized_utility: float = 0.0


@dataclass
class MarketResult:
    """Full MNL market outcome for one round."""
    buyer_results: dict[str, BuyerMarketResult] = field(default_factory=dict)
    segment_allocations: list[SegmentAllocation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core per-segment computation
# ---------------------------------------------------------------------------

def _quality_score(profile: BuyerProfile, segment: "CustomerSegment") -> float:
    """
    Quality component of U_{i,s} — price-free dot product of buyer and segment vectors.
    Used both in the MNL logit and in realized_utility (benchmark-comparable).
    """
    return (
        segment.w_env      * (5.0 - profile.avg_env)
        + segment.w_social * (5.0 - profile.avg_social)
    )


def _mnl_for_segment(
    segment: "CustomerSegment",
    profiles: list[BuyerProfile],
    delta: float,
    u_outside: float | None,
) -> SegmentAllocation:
    """
    Compute MNL shares for one segment.

    U_{i,s} = q_{i,s} - delta * w_cost_s * price_i

    u_outside: utility of the outside option.
               None = no outside option (all demand is served by buyers).
               Float = adds exp(u_outside) to denominator to model "no purchase".
               Caution: set carefully — buyer logits can be large negative numbers,
               making a fixed u_outside dominate unless the logit scale is calibrated.
    """
    logits: list[float] = []

    for p in profiles:
        q = _quality_score(p, segment)
        mnl_u = q - delta * segment.w_cost * p.price_per_user
        logits.append(mnl_u)

    if not logits:
        return SegmentAllocation(segment_id=segment.segment_id, density=segment.density, shares={})

    # Numerically stable softmax
    max_logit = max(logits)
    exps = [math.exp(u - max_logit) for u in logits]
    denom = sum(exps)

    if u_outside is not None:
        denom += math.exp(u_outside - max_logit)

    shares = {p.team_name: exps[i] / denom for i, p in enumerate(profiles)}

    return SegmentAllocation(
        segment_id=segment.segment_id,
        density=segment.density,
        shares=shares,
    )


# ---------------------------------------------------------------------------
# Parallel market runner
# ---------------------------------------------------------------------------

def run_mnl_market(
    profiles: list[BuyerProfile],
    segments: list["CustomerSegment"],
    delta: float = 1.0,
    u_outside: float | None = None,
    max_workers: int | None = None,
) -> MarketResult:
    """
    Run the MNL demand model across all segments in parallel.

    Parameters
    ----------
    profiles    : one BuyerProfile per competing team
    segments    : customer segments with density weights
    delta       : price sensitivity multiplier (default 1.0)
                  0 = price ignored in MNL, 1 = standard, >1 = amplified
    u_outside   : outside option utility; None = no outside option
    max_workers : thread pool size (None = cpu_count)

    Returns
    -------
    MarketResult with per-buyer realized_earnings, realized_utility, total_demand.
    """
    if not profiles or not segments:
        return MarketResult()

    total_density = sum(s.density for s in segments)
    if total_density <= 0.0:
        return MarketResult()

    result = MarketResult(
        buyer_results={
            p.team_name: BuyerMarketResult(team_name=p.team_name)
            for p in profiles
        }
    )

    profile_map = {p.team_name: p for p in profiles}
    segment_map  = {s.segment_id: s for s in segments}

    # --- parallel per-segment MNL ---
    allocations: list[SegmentAllocation] = [None] * len(segments)  # type: ignore

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_mnl_for_segment, seg, profiles, delta, u_outside): idx
            for idx, seg in enumerate(segments)
        }
        for future in as_completed(futures):
            idx = futures[future]
            allocations[idx] = future.result()

    # --- aggregate across segments ---
    for alloc in allocations:
        result.segment_allocations.append(alloc)
        norm_density = alloc.density / total_density
        seg = segment_map[alloc.segment_id]

        for team_name, share in alloc.shares.items():
            br = result.buyer_results[team_name]
            p  = profile_map[team_name]

            demand = norm_density * share
            br.total_demand      += demand
            br.realized_earnings += p.price_per_user * demand
            br.realized_utility  += _quality_score(p, seg) * demand

    return result
