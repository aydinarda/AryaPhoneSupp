"""Unit tests for the audit module (app/audit.py).

The audit system simulates government investigations into supplier violations.
Each round, the admin can configure:
  audit_probability  — P(a unique supplier is selected for investigation)
  catch_probability  — P(violation found | investigated AND has violation)

If a supplier is caught, every team that selected it is excluded from MNL
matching for that round (the "regulatory exclusion" mechanic).
"""
from __future__ import annotations

import random

import pandas as pd
import pytest

from app.audit import run_audit, AuditResult


# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------

def _suppliers(records: list[dict]) -> pd.DataFrame:
    base = {"env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
            "strategic": 3.0, "child_labor": 0.0, "banned_chem": 0.0, "category": "camera"}
    rows = []
    for r in records:
        row = {**base, **r}
        rows.append(row)
    return pd.DataFrame(rows)


def _team_profiles(*supplier_lists: tuple[str, list[str]]) -> dict:
    """Build a minimal team_profiles dict.  Each arg is (team_name, [supplier_ids])."""
    return {
        team: {"team": team, "picked_suppliers": picks}
        for team, picks in supplier_lists
    }


# ---------------------------------------------------------------------------
# Audit disabled (audit_probability = 0)
# ---------------------------------------------------------------------------

class TestAuditDisabled:
    def test_zero_probability_returns_empty_result(self):
        profiles = _team_profiles(("A", ["S1", "S2"]))
        df = _suppliers([
            {"supplier_id": "S1", "child_labor": 1.0},
            {"supplier_id": "S2", "banned_chem": 1.0},
        ])
        result = run_audit(profiles, df, audit_probability=0.0, catch_probability=1.0)

        assert result.caught_suppliers == set()
        assert result.excluded_teams == []
        assert result.audited_suppliers == {}
        assert result.violation_flags == {}

    def test_audit_off_no_teams_excluded_even_with_violations(self):
        profiles = _team_profiles(("TeamX", ["FLAGGED"]))
        df = _suppliers([{"supplier_id": "FLAGGED", "child_labor": 1.0, "banned_chem": 1.0}])
        result = run_audit(profiles, df, audit_probability=0.0, catch_probability=1.0)
        assert "TeamX" not in result.excluded_teams


# ---------------------------------------------------------------------------
# Audit always triggers (audit_probability = 1.0)
# ---------------------------------------------------------------------------

class TestAuditAlwaysTriggers:
    def test_flagged_supplier_always_caught_when_both_probs_are_one(self):
        profiles = _team_profiles(("T1", ["S_BAD"]))
        df = _suppliers([{"supplier_id": "S_BAD", "child_labor": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        assert "S_BAD" in result.caught_suppliers
        assert "T1" in result.excluded_teams

    def test_clean_supplier_never_caught_even_at_full_probability(self):
        """No false positives: a clean supplier cannot be caught regardless of probabilities."""
        profiles = _team_profiles(("T2", ["S_CLEAN"]))
        df = _suppliers([{"supplier_id": "S_CLEAN", "child_labor": 0.0, "banned_chem": 0.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(42))
        assert "S_CLEAN" not in result.caught_suppliers
        assert "T2" not in result.excluded_teams

    def test_banned_chem_flag_also_triggers_catch(self):
        profiles = _team_profiles(("T3", ["S_BC"]))
        df = _suppliers([{"supplier_id": "S_BC", "banned_chem": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(7))
        assert "S_BC" in result.caught_suppliers
        assert "T3" in result.excluded_teams

    def test_multiple_teams_sharing_caught_supplier_all_excluded(self):
        """If two teams both pick the same caught supplier, both are excluded."""
        profiles = _team_profiles(("Alpha", ["SHARED"]), ("Beta", ["SHARED"]))
        df = _suppliers([{"supplier_id": "SHARED", "child_labor": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        assert "Alpha" in result.excluded_teams
        assert "Beta" in result.excluded_teams

    def test_only_teams_using_caught_supplier_are_excluded(self):
        """Teams whose suppliers are clean are never caught collaterally."""
        profiles = _team_profiles(
            ("Dirty", ["S_BAD"]),
            ("Clean", ["S_CLEAN"]),
        )
        df = _suppliers([
            {"supplier_id": "S_BAD",   "child_labor": 1.0},
            {"supplier_id": "S_CLEAN", "child_labor": 0.0},
        ])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        assert "Dirty" in result.excluded_teams
        assert "Clean" not in result.excluded_teams


# ---------------------------------------------------------------------------
# catch_probability = 0 (audit runs but never finds anything)
# ---------------------------------------------------------------------------

class TestCatchProbabilityZero:
    def test_audited_but_never_caught_when_catch_prob_is_zero(self):
        profiles = _team_profiles(("TZ", ["FLAGGED"]))
        df = _suppliers([{"supplier_id": "FLAGGED", "child_labor": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=0.0,
                           rng=random.Random(99))
        # Supplier was audited
        assert result.audited_suppliers.get("FLAGGED") is True
        # But nothing was caught
        assert "FLAGGED" not in result.caught_suppliers
        assert "TZ" not in result.excluded_teams


# ---------------------------------------------------------------------------
# AuditResult.to_dict serialization
# ---------------------------------------------------------------------------

class TestAuditResultSerialization:
    def test_to_dict_contains_all_keys(self):
        profiles = _team_profiles(("TA", ["S1"]))
        df = _suppliers([{"supplier_id": "S1", "child_labor": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        d = result.to_dict()
        for key in ("audit_probability", "catch_probability", "audited_suppliers",
                    "violation_flags", "caught_suppliers", "team_violations", "excluded_teams"):
            assert key in d, f"Missing key in to_dict output: {key!r}"

    def test_caught_suppliers_is_sorted_list(self):
        profiles = _team_profiles(("TB", ["C", "A", "B"]))
        df = _suppliers([
            {"supplier_id": "A", "child_labor": 1.0},
            {"supplier_id": "B", "child_labor": 1.0},
            {"supplier_id": "C", "child_labor": 1.0},
        ])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        d = result.to_dict()
        assert d["caught_suppliers"] == sorted(d["caught_suppliers"])

    def test_excluded_teams_is_sorted_list(self):
        profiles = _team_profiles(("Zebra", ["S1"]), ("Alpha", ["S1"]))
        df = _suppliers([{"supplier_id": "S1", "child_labor": 1.0}])
        result = run_audit(profiles, df, audit_probability=1.0, catch_probability=1.0,
                           rng=random.Random(0))
        d = result.to_dict()
        assert d["excluded_teams"] == sorted(d["excluded_teams"])


# ---------------------------------------------------------------------------
# Seeded RNG — reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_gives_same_outcome(self):
        profiles = _team_profiles(("T", ["S1", "S2", "S3"]))
        df = _suppliers([
            {"supplier_id": "S1", "child_labor": 1.0},
            {"supplier_id": "S2", "child_labor": 0.0},
            {"supplier_id": "S3", "banned_chem": 1.0},
        ])
        r1 = run_audit(profiles, df, audit_probability=0.5, catch_probability=0.7,
                       rng=random.Random(123))
        r2 = run_audit(profiles, df, audit_probability=0.5, catch_probability=0.7,
                       rng=random.Random(123))
        assert r1.caught_suppliers == r2.caught_suppliers
        assert r1.excluded_teams == r2.excluded_teams

    def test_different_seeds_can_give_different_outcomes(self):
        """With intermediate probabilities, different seeds should eventually diverge."""
        profiles = _team_profiles(("T", ["S_BAD"]))
        df = _suppliers([{"supplier_id": "S_BAD", "child_labor": 1.0}])
        outcomes = set()
        for seed in range(50):
            r = run_audit(profiles, df, audit_probability=0.5, catch_probability=0.5,
                          rng=random.Random(seed))
            outcomes.add(bool(r.caught_suppliers))
        # Over 50 seeds with p=0.5*0.5=0.25, we expect both True and False
        assert len(outcomes) == 2, "Expected both caught and not-caught outcomes over 50 seeds"
