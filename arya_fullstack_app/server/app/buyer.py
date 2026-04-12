from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Buyer
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class Buyer:
    """
    Represents a team (seller) in a game round.

    Identity fields are set at construction.
    Round submission fields are populated via update_from_round().
    Outcome fields are populated externally via BuyerPotential / BuyerRealized.
    """

    team_name: str
    session_token: str

    # --- Round submission state (populated per round) ---
    round_no: int | None = None
    selected_suppliers: list[str] = field(default_factory=list)
    price_per_user: float | None = None
    avg_env: float | None = None
    avg_social: float | None = None
    avg_cost: float | None = None
    avg_strategic: float | None = None
    avg_child_labor: float | None = None
    avg_banned_chem: float | None = None

    # --- Potential outcomes (fractionless — no competition, full capacity) ---
    potential_utility: float | None = None
    potential_earnings: float | None = None

    # --- Realized outcomes (post-matching — actual market result) ---
    realized_utility: float | None = None
    realized_earnings: float | None = None
    matched_user_count: int | None = None

    def update_from_round(
        self,
        round_no: int,
        submission: dict[str, Any],
        supplier_averages: dict[str, float] | None = None,
    ) -> None:
        """
        Apply a round submission to this buyer.

        submission       : raw row from the DB / API (selected_suppliers, price_per_user, …)
        supplier_averages: pre-computed avg_env, avg_social, avg_cost, etc.
                           If omitted, the values stay None until set externally.
        """
        self.round_no = round_no

        raw_suppliers = str(submission.get("selected_suppliers") or "")
        self.selected_suppliers = [x.strip() for x in raw_suppliers.split(",") if x.strip()]

        self.price_per_user = _safe_float(submission.get("price_per_user"))

        if supplier_averages:
            self.avg_env = _safe_float(supplier_averages.get("avg_env"))
            self.avg_social = _safe_float(supplier_averages.get("avg_social"))
            self.avg_cost = _safe_float(supplier_averages.get("avg_cost"))
            self.avg_strategic = _safe_float(supplier_averages.get("avg_strategic"))
            self.avg_child_labor = _safe_float(supplier_averages.get("avg_child_labor"))
            self.avg_banned_chem = _safe_float(supplier_averages.get("avg_banned_chem"))

        # Clear stale outcomes when the round changes.
        self.potential_utility = None
        self.potential_earnings = None
        self.realized_utility = None
        self.realized_earnings = None
        self.matched_user_count = None

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Buyer":
        """Construct a Buyer from an API/DB payload (identity only)."""
        return cls(
            team_name=str(
                payload.get("team_name", payload.get("team", payload.get("buyer_name", "")))
            ).strip(),
            session_token=str(
                payload.get("session_token", payload.get("sessionToken", ""))
            ).strip(),
        )


# ---------------------------------------------------------------------------
# BuyerPotential — fractionless outcome; set before matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuyerPotential:
    """
    Fractionless outcome: what this buyer would achieve with no competitors
    (full capacity, every customer segment served).

    Computed once from the product profile alone, independent of competition.
    Apply to a Buyer with apply().
    """

    utility: float
    earnings: float

    def apply(self, buyer: Buyer) -> None:
        buyer.potential_utility = self.utility
        buyer.potential_earnings = self.earnings


# ---------------------------------------------------------------------------
# BuyerRealized — post-matching outcome; set after matching
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class BuyerRealized:
    """
    Realized outcome: actual results after market matching.

    Apply to a Buyer with apply().
    """

    utility: float
    earnings: float
    matched_user_count: int

    def apply(self, buyer: Buyer) -> None:
        buyer.realized_utility = self.utility
        buyer.realized_earnings = self.earnings
        buyer.matched_user_count = self.matched_user_count
