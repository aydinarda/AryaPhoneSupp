from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Supplier — immutable, one instance per row in the Excel sheet.
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Supplier:
    """
    A single supplier record loaded from the Excel workbook.

    Frozen (immutable) — suppliers are fixed for the entire game session.

    Risk/quality attributes are on a 1-5 scale.  Directional conventions:
      env_risk, social_risk, low_quality  →  lower raw value = better
      strategic, improvement              →  lower raw value = better (no need to confuse)
      cost_score                          →  lower raw value = cheaper (better for profit)
      child_labor, banned_chem            →  binary flag (0 or 1); 1 = violation
    """

    supplier_id: str

    env_risk: float = 0.0
    social_risk: float = 0.0
    cost_score: float = 0.0
    strategic: float = 0.0
    improvement: float = 0.0
    low_quality: float = 0.0
    child_labor: float = 0.0   # binary flag
    banned_chem: float = 0.0   # binary flag

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Supplier":
        """Build a Supplier from a DataFrame row or API dict."""
        supplier_id = str(
            row.get("supplier_id", row.get("id", row.get("supplier", "")))
        ).strip()
        return cls(
            supplier_id=supplier_id,
            env_risk=_safe_float(row.get("env_risk")),
            social_risk=_safe_float(row.get("social_risk")),
            cost_score=_safe_float(row.get("cost_score")),
            strategic=_safe_float(row.get("strategic")),
            improvement=_safe_float(row.get("improvement")),
            low_quality=_safe_float(row.get("low_quality")),
            child_labor=_safe_float(row.get("child_labor")),
            banned_chem=_safe_float(row.get("banned_chem")),
        )


# ---------------------------------------------------------------------------
# Suppliers — the full catalogue; created once at game start, never mutated.
# ---------------------------------------------------------------------------

class Suppliers:
    """
    Immutable catalogue of all suppliers for the current game session.

    Loaded once from the Excel workbook at startup; every round reads from
    this same object.
    """

    __slots__ = ("_by_id",)

    def __init__(self, suppliers: list[Supplier]) -> None:
        self._by_id: dict[str, Supplier] = {s.supplier_id: s for s in suppliers}

    def get(self, supplier_id: str) -> Supplier | None:
        """Return a single supplier by ID, or None if not found."""
        return self._by_id.get(str(supplier_id).strip())

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, supplier_id: str) -> bool:
        return str(supplier_id).strip() in self._by_id

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @classmethod
    def load_from_dataframe(cls, df: Any) -> "Suppliers":
        """
        Build the catalogue from a pandas DataFrame (already normalized columns).

        Skips rows with a blank supplier_id.
        """
        suppliers: list[Supplier] = []
        for _, row in df.iterrows():
            s = Supplier.from_row(row.to_dict())
            if s.supplier_id:
                suppliers.append(s)
        return cls(suppliers)

    @classmethod
    def load_from_excel(cls, xlsx_path: str | Path) -> "Suppliers":
        """
        Load directly from the Excel workbook.

        In production, prefer load_from_dataframe() with the already-loaded DF
        from service.get_tables() to avoid reading the file twice.
        """
        from .optimization_controller import load_supplier_user_tables

        suppliers_df, _ = load_supplier_user_tables(Path(xlsx_path))
        return cls.load_from_dataframe(suppliers_df)
