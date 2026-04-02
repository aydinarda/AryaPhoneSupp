from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buyer import Buyer
    from .customer_segment import CustomerSegment


# ---------------------------------------------------------------------------
# Benchmark snapshot — one per objective
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """
    Theoretical optimum for one objective (max utility or max profit).

    Sourced from the Gurobi MILP agents; marked unavailable when Gurobi
    is not installed.
    """
    available: bool
    feasible: bool
    utility_total: float
    earnings_total: float


# ---------------------------------------------------------------------------
# Full player view for one round
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class PlayerView:
    """
    Everything a player sees after submitting their round choices.

    frictionless_*   — what this buyer would achieve with no competitors
                       (computed from their own product profile + segment weights)
    benchmark_*      — theoretical optima from the MILP agents
                       (Gurobi-required; marked unavailable if not installed)
    """
    team_name: str
    round_no: int

    frictionless_utility: float
    frictionless_earnings: float

    benchmark_max_utility: BenchmarkResult
    benchmark_max_profit: BenchmarkResult


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_player_view(
    buyer: "Buyer",
    segments: list["CustomerSegment"],
    total_users: int,
    cost_scale: float,
) -> PlayerView:
    """
    Compute the full player view for a buyer after their round submission.

    Parameters
    ----------
    buyer       : must have round submission fields populated
    segments    : all customer segments with density weights
    total_users : market size (GameSettings.served_users)
    cost_scale  : GameSettings.cost_scale
    """
    from .potential_outcome import compute_potential
    from .service import get_both_benchmarks

    # --- Frictionless outcomes (no Gurobi needed) ---
    potential = compute_potential(buyer, segments, total_users, cost_scale)
    potential.apply(buyer)

    # --- Benchmarks (Gurobi-cached; graceful fallback if unavailable) ---
    raw = get_both_benchmarks()

    def _parse(entry: dict) -> BenchmarkResult:
        metrics = entry.get("metrics") or {}
        return BenchmarkResult(
            available=bool(entry.get("available", False)),
            feasible=bool(entry.get("feasible", False)),
            utility_total=float(metrics.get("utility_total", 0.0)),
            earnings_total=float(metrics.get("profit_total", 0.0)),
        )

    return PlayerView(
        team_name=buyer.team_name,
        round_no=buyer.round_no or 0,
        frictionless_utility=buyer.potential_utility or 0.0,
        frictionless_earnings=buyer.potential_earnings or 0.0,
        benchmark_max_utility=_parse(raw.get("max_utility", {})),
        benchmark_max_profit=_parse(raw.get("max_profit", {})),
    )
