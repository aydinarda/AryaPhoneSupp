from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


@dataclass(slots=True)
class CustomerSegment:
    """
    Represents a segment of end-customers with shared preferences.

    Segments are loaded once at session start from the Excel dataset and do not
    change during the game.  The `density` field captures the relative size of
    the segment (e.g. number of individual users, or a probability weight).
    """

    segment_id: str

    # How large / representative this segment is.
    # Treated as a weight when aggregating across segments.
    density: float

    # Preference weights — higher means the segment cares more about that dimension.
    w_env: float = 0.0
    w_social: float = 0.0
    w_cost: float = 1.0
    w_low_quality: float = 0.0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "CustomerSegment":
        """
        Build a CustomerSegment from a DataFrame row or API dict.

        Accepted column name variants:
          segment_id  : "segment_id" | "user_id" | "id"
          density     : "density" | "weight" | "size" | "count" | "n_users"
          w_env       : "w_env" | "env_weight"
          w_social    : "w_social" | "social_weight"
          w_cost      : "w_cost" | "price_sensitivity" | "cost_score" | "cost_weight"
          w_low_quality: "w_low_quality" | "quality_weight"
        """
        segment_id = str(
            row.get("segment_id", row.get("user_id", row.get("id", "")))
        ).strip()

        density_raw = row.get(
            "density",
            row.get("weight", row.get("size", row.get("count", row.get("n_users", 1.0)))),
        )
        density = _safe_float(density_raw, default=1.0)
        if density <= 0.0:
            density = 1.0

        return cls(
            segment_id=segment_id,
            density=density,
            w_env=_safe_float(row.get("w_env", row.get("env_weight")), 0.0),
            w_social=_safe_float(row.get("w_social", row.get("social_weight")), 0.0),
            w_cost=_safe_float(
                row.get("w_cost", row.get("price_sensitivity", row.get("cost_score", row.get("cost_weight")))), 1.0
            ),
            w_low_quality=_safe_float(
                row.get("w_low_quality", row.get("quality_weight")), 0.0
            ),
        )

    @classmethod
    def load_from_dataframe(cls, df: Any) -> list["CustomerSegment"]:
        """
        Load all segments from a pandas DataFrame.

        Returns segments ordered by segment_id, skipping rows with no valid id.
        """
        segments: list[CustomerSegment] = []
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            seg = cls.from_row(row_dict)
            if seg.segment_id:
                segments.append(seg)
        segments.sort(key=lambda s: s.segment_id)
        return segments
