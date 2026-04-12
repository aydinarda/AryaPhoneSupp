"""
audit.py
--------
Independent audit module for the Arya Phone game.

Government regulations prohibit child labour and banned chemicals, but some
suppliers may still use them.  At the end of each round (before MNL matching)
the admin can trigger an audit phase with two probabilistic parameters:

  audit_probability : P(a supplier is selected for audit)
                      Applied independently to every unique supplier picked by
                      at least one team.  Equal for all suppliers.

  catch_probability : P(violation is found | supplier is audited AND has a
                      violation)
                      No false positives — a clean supplier is never caught.

If a supplier is audited and its violation is discovered, every team that
selected that supplier is immediately marked as NOT FEASIBLE and excluded from
the MNL market calculation.  The audit outcome is revealed only at the end of
the round so it cannot be exploited in advance or leak into optimisation models.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditResult:
    """Full outcome of the audit phase for one round."""

    # Parameters actually used (echoed for transparency)
    audit_probability: float = 0.0
    catch_probability: float = 0.0

    # supplier_id -> True if that supplier was drawn for audit
    audited_suppliers: dict[str, bool] = field(default_factory=dict)

    # supplier_id -> True if that supplier has a violation in the data
    violation_flags: dict[str, bool] = field(default_factory=dict)

    # set of supplier IDs that were audited AND caught with a violation
    caught_suppliers: set[str] = field(default_factory=set)

    # team_name -> list of caught supplier IDs they selected
    team_violations: dict[str, list[str]] = field(default_factory=dict)

    # teams disqualified from market matching this round
    excluded_teams: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_probability": self.audit_probability,
            "catch_probability": self.catch_probability,
            "audited_suppliers": self.audited_suppliers,
            "violation_flags": self.violation_flags,
            "caught_suppliers": sorted(self.caught_suppliers),
            "team_violations": self.team_violations,
            "excluded_teams": self.excluded_teams,
        }


def run_audit(
    team_profiles: dict[str, dict[str, Any]],
    suppliers_df: Any,  # pandas DataFrame with supplier data
    audit_probability: float,
    catch_probability: float,
    rng: random.Random | None = None,
) -> AuditResult:
    """
    Run the audit phase for one round.

    Parameters
    ----------
    team_profiles    : dict of team_name -> profile dict, as built by
                       _build_team_product_profiles().  Each profile must
                       contain "picked_suppliers" (list[str]).
    suppliers_df     : DataFrame with at least columns supplier_id,
                       child_labor, banned_chem.
    audit_probability: P(supplier selected for audit).  0 = audit disabled.
    catch_probability: P(violation found | audited AND has violation).
                       Clean suppliers are never caught (no false positives).
    rng              : optional seeded Random instance for reproducibility.

    Returns
    -------
    AuditResult with caught suppliers and excluded teams populated.
    """
    if rng is None:
        rng = random.Random()

    result = AuditResult(
        audit_probability=audit_probability,
        catch_probability=catch_probability,
    )

    # Collect all unique supplier IDs in play this round
    all_picked: set[str] = set()
    for profile in team_profiles.values():
        for sid in profile.get("picked_suppliers", []):
            all_picked.add(str(sid))

    if not all_picked or audit_probability <= 0.0:
        return result

    # Build violation map from supplier DataFrame
    violation_map: dict[str, bool] = {}
    for _, row in suppliers_df.iterrows():
        sid = str(row["supplier_id"])
        has_violation = (
            float(row.get("child_labor", 0.0)) >= 0.5
            or float(row.get("banned_chem", 0.0)) >= 0.5
        )
        violation_map[sid] = has_violation

    # Audit each unique supplier independently
    for sid in all_picked:
        has_violation = violation_map.get(sid, False)
        result.violation_flags[sid] = has_violation

        audited = rng.random() < audit_probability
        result.audited_suppliers[sid] = audited

        if audited and has_violation:
            caught = rng.random() < catch_probability
            if caught:
                result.caught_suppliers.add(sid)

    # Determine which teams are excluded
    for team, profile in team_profiles.items():
        caught_picks = [
            sid for sid in profile.get("picked_suppliers", [])
            if str(sid) in result.caught_suppliers
        ]
        if caught_picks:
            result.team_violations[team] = caught_picks
            result.excluded_teams.append(team)

    result.excluded_teams.sort()
    return result
