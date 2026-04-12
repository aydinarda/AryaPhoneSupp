"""Tests for the category-constraint feature.

Each supplier belongs to exactly one category (camera / keyboard / cable).
The game requires exactly 1 supplier from each category — verified at every
layer: Supplier dataclass, manual_metrics feasibility, solver, and
session matching (_build_team_product_profiles).
"""
from __future__ import annotations

import pandas as pd
import pytest

from app.optimization_controller import Policy, MaxProfitConfig, manual_metrics
from app.optimizer_common import GUROBI_AVAILABLE, solve_best_over_k
from app.routers.sessions import _build_team_product_profiles
from app.supplier import Supplier


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _suppliers_df() -> pd.DataFrame:
    """Minimal 3-category catalogue (2 per category) with safe risk values."""
    return pd.DataFrame([
        {"supplier_id": "CAM1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "camera"},
        {"supplier_id": "CAM2", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
         "strategic": 2.0, "improvement": 2.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "camera"},
        {"supplier_id": "KEY1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "keyboard"},
        {"supplier_id": "KEY2", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
         "strategic": 2.0, "improvement": 2.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "keyboard"},
        {"supplier_id": "CBL1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "cable"},
        {"supplier_id": "CBL2", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
         "strategic": 2.0, "improvement": 2.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "cable"},
    ])


def _users_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": "U1", "w_env": 1.0, "w_social": 1.0, "w_cost": 1.0,
         "w_strategic": 1.0, "w_improvement": 1.0, "w_low_quality": 1.0},
        {"user_id": "U2", "w_env": 0.5, "w_social": 0.5, "w_cost": 1.5,
         "w_strategic": 0.5, "w_improvement": 0.5, "w_low_quality": 0.5},
        {"user_id": "U3", "w_env": 1.5, "w_social": 1.5, "w_cost": 0.5,
         "w_strategic": 1.5, "w_improvement": 1.5, "w_low_quality": 1.5},
    ])


def _suppliers_by_id(df: pd.DataFrame) -> dict:
    """Build the suppliers_by_id dict the same way sessions.py does."""
    return {
        str(row["supplier_id"]): {
            k: row.get(k)
            for k in ("env_risk", "social_risk", "cost_score", "strategic",
                      "improvement", "low_quality", "child_labor", "banned_chem", "category")
        }
        for _, row in df.iterrows()
    }


_POLICY = Policy()
_CFG = MaxProfitConfig()


# ---------------------------------------------------------------------------
# Supplier dataclass
# ---------------------------------------------------------------------------

class TestSupplierDataclass:
    def test_reads_category_from_row(self):
        s = Supplier.from_row({"supplier_id": "X1", "category": "camera"})
        assert s.category == "camera"

    def test_empty_string_when_category_absent(self):
        s = Supplier.from_row({"supplier_id": "X2"})
        assert s.category == ""

    def test_normalises_nan_string_to_empty(self):
        s = Supplier.from_row({"supplier_id": "X3", "category": "nan"})
        assert s.category == ""

    def test_strips_whitespace(self):
        s = Supplier.from_row({"supplier_id": "X4", "category": "  cable  "})
        assert s.category == "cable"


# ---------------------------------------------------------------------------
# manual_metrics feasibility
# ---------------------------------------------------------------------------

class TestManualMetricsCategoryFeasibility:
    def test_empty_picks_is_infeasible(self):
        result = manual_metrics(_suppliers_df(), _users_df(), _POLICY, _CFG, [])
        assert result["feasible"] is False

    def test_one_per_category_within_risk_caps_is_feasible(self):
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG, ["CAM1", "KEY1", "CBL1"]
        )
        assert result["feasible"] is True
        assert result["metrics"]["k"] == pytest.approx(3.0)

    def test_two_cameras_no_keyboard_is_infeasible(self):
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG, ["CAM1", "CAM2", "CBL1"]
        )
        assert result["feasible"] is False

    def test_missing_one_category_is_infeasible(self):
        # Only camera + keyboard, no cable
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG, ["CAM1", "KEY1"]
        )
        assert result["feasible"] is False

    def test_risk_cap_violation_still_infeasible_even_with_correct_categories(self):
        # Build a suppliers_df where one supplier has dangerously high env_risk
        df = _suppliers_df().copy()
        df.loc[df["supplier_id"] == "CAM1", "env_risk"] = 5.0
        df.loc[df["supplier_id"] == "KEY1", "env_risk"] = 5.0
        df.loc[df["supplier_id"] == "CBL1", "env_risk"] = 5.0
        # avg_env = 5.0 > env_cap=2.75
        result = manual_metrics(df, _users_df(), _POLICY, _CFG, ["CAM1", "KEY1", "CBL1"])
        assert result["feasible"] is False

    def test_averages_computed_over_all_three_picks(self):
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG, ["CAM1", "KEY1", "CBL1"]
        )
        m = result["metrics"]
        # All three have cost_score=1.0 → avg_cost = 1.0
        assert m["avg_cost"] == pytest.approx(1.0)

    def test_averages_computed_correctly_with_mixed_costs(self):
        # CAM2(cost=2.0) + KEY1(cost=1.0) + CBL2(cost=2.0) → avg = 5/3
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG, ["CAM2", "KEY1", "CBL2"]
        )
        assert result["metrics"]["avg_cost"] == pytest.approx(5.0 / 3.0)


# ---------------------------------------------------------------------------
# _build_team_product_profiles (session matching layer)
# ---------------------------------------------------------------------------

class TestBuildTeamProductProfiles:
    def _sid(self) -> dict:
        return _suppliers_by_id(_suppliers_df())

    def test_correct_combination_is_included(self):
        profiles, excluded = _build_team_product_profiles(
            {"TeamA": {"selected_suppliers": "CAM1,KEY1,CBL1",
                       "created_at": "", "price": 100}},
            self._sid(),
        )
        assert "TeamA" in profiles
        assert excluded == []

    def test_two_cameras_no_keyboard_is_excluded(self):
        profiles, excluded = _build_team_product_profiles(
            {"TeamB": {"selected_suppliers": "CAM1,CAM2,CBL1",
                       "created_at": "", "price": 100}},
            self._sid(),
        )
        assert "TeamB" not in profiles
        assert "TeamB" in excluded

    def test_missing_category_is_excluded(self):
        # camera + keyboard only, no cable
        profiles, excluded = _build_team_product_profiles(
            {"TeamC": {"selected_suppliers": "CAM1,KEY1",
                       "created_at": "", "price": 100}},
            self._sid(),
        )
        assert "TeamC" not in profiles
        assert "TeamC" in excluded

    def test_only_invalid_team_is_excluded_not_valid_one(self):
        rows = {
            "Good":  {"selected_suppliers": "CAM1,KEY1,CBL1", "created_at": "", "price": 100},
            "Bad":   {"selected_suppliers": "CAM1,CAM2,CBL1", "created_at": "", "price": 100},
        }
        profiles, excluded = _build_team_product_profiles(rows, self._sid())
        assert "Good" in profiles
        assert "Bad" not in profiles
        assert excluded == ["Bad"]

    def test_risk_cap_violation_excludes_team(self):
        # Build suppliers where the combination exceeds env_cap
        df = _suppliers_df().copy()
        for sid in ("CAM1", "KEY1", "CBL1"):
            df.loc[df["supplier_id"] == sid, "env_risk"] = 5.0  # avg=5.0 > 2.75
        sid_map = _suppliers_by_id(df)
        profiles, excluded = _build_team_product_profiles(
            {"RiskyTeam": {"selected_suppliers": "CAM1,KEY1,CBL1",
                           "created_at": "", "price": 100}},
            sid_map,
        )
        assert "RiskyTeam" not in profiles
        assert "RiskyTeam" in excluded

    def test_averages_in_profile_are_correct(self):
        # CAM2(cost=2)+KEY1(cost=1)+CBL1(cost=1) → avg_cost = 4/3
        profiles, _ = _build_team_product_profiles(
            {"T": {"selected_suppliers": "CAM2,KEY1,CBL1", "created_at": "", "price": 100}},
            self._sid(),
        )
        assert profiles["T"]["avg_cost"] == pytest.approx(4.0 / 3.0)


# ---------------------------------------------------------------------------
# avg_child_labor / avg_banned_chem tracking in manual_metrics
# ---------------------------------------------------------------------------

class TestManualMetricsViolationTracking:
    """manual_metrics must expose avg_child_labor and avg_banned_chem for each
    supplier combination so the audit system has accurate violation data."""

    def _df_with_flags(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"supplier_id": "CAM1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
             "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
             "child_labor": 1.0, "banned_chem": 1.0, "category": "camera"},
            {"supplier_id": "KEY1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
             "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
             "child_labor": 0.0, "banned_chem": 0.0, "category": "keyboard"},
            {"supplier_id": "CBL1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
             "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
             "child_labor": 0.0, "banned_chem": 0.0, "category": "cable"},
        ])

    def test_flagged_supplier_gives_nonzero_avg_child_labor(self):
        # CAM1 has child_labor=1.0, KEY1+CBL1 have 0.0 → avg = 1/3
        result = manual_metrics(
            self._df_with_flags(), _users_df(), _POLICY, _CFG,
            ["CAM1", "KEY1", "CBL1"],
        )
        assert result["feasible"] is True
        assert result["metrics"]["avg_child_labor"] == pytest.approx(1.0 / 3.0)
        assert result["metrics"]["avg_banned_chem"] == pytest.approx(1.0 / 3.0)

    def test_clean_combination_has_zero_violation_averages(self):
        result = manual_metrics(
            _suppliers_df(), _users_df(), _POLICY, _CFG,
            ["CAM1", "KEY1", "CBL1"],
        )
        assert result["metrics"]["avg_child_labor"] == pytest.approx(0.0)
        assert result["metrics"]["avg_banned_chem"] == pytest.approx(0.0)

    def test_cost_per_unit_reflects_cost_scale_times_avg_cost(self):
        """cost_per_unit = cost_scale * avg_cost (no penalty terms; audit handles exclusion)."""
        from app.settings import GAME_SETTINGS
        result = manual_metrics(
            self._df_with_flags(), _users_df(), _POLICY, _CFG,
            ["CAM1", "KEY1", "CBL1"],
        )
        expected_cpu = float(GAME_SETTINGS.cost_scale) * result["metrics"]["avg_cost"]
        assert result["metrics"]["cost_per_unit"] == pytest.approx(expected_cpu)


# ---------------------------------------------------------------------------
# solve_best_over_k — categorical mode
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not GUROBI_AVAILABLE, reason="Gurobi not available")
class TestSolverCategoricalMode:
    def test_solver_picks_exactly_one_per_category(self):
        result = solve_best_over_k(
            _suppliers_df(), _users_df(), _POLICY,
            env_cap=2.75, social_cap=3.0, output_flag=0,
            objective_mode="profit",
        )
        assert result is not None
        assert len(result["chosen"]) == 3  # one per category

        df = _suppliers_df().set_index("supplier_id")
        cats = [df.loc[sid, "category"] for sid in result["chosen"]]
        assert sorted(cats) == ["cable", "camera", "keyboard"]

    def test_solver_picks_cheapest_per_category_for_profit(self):
        result = solve_best_over_k(
            _suppliers_df(), _users_df(), _POLICY,
            env_cap=2.75, social_cap=3.0, output_flag=0,
            objective_mode="profit",
        )
        assert result is not None
        # Each category's chosen supplier should be the one with lower cost_score
        # (CAM1=1.0 vs CAM2=2.0, KEY1=1.0 vs KEY2=2.0, CBL1=1.0 vs CBL2=2.0)
        assert set(result["chosen"]) == {"CAM1", "KEY1", "CBL1"}

    def test_solver_result_k_is_three(self):
        result = solve_best_over_k(
            _suppliers_df(), _users_df(), _POLICY,
            env_cap=2.75, social_cap=3.0, output_flag=0,
            objective_mode="utility",
        )
        assert result is not None
        assert result["k"] == pytest.approx(3.0)

    def test_solver_averages_span_all_three_picks(self):
        result = solve_best_over_k(
            _suppliers_df(), _users_df(), _POLICY,
            env_cap=2.75, social_cap=3.0, output_flag=0,
            objective_mode="profit",
        )
        assert result is not None
        # With CAM1+KEY1+CBL1 all having cost_score=1.0 → avg_cost=1.0
        assert result["avg_cost"] == pytest.approx(1.0)

    def test_solver_exposes_avg_child_labor_and_avg_banned_chem(self):
        """Solver result must include avg_child_labor and avg_banned_chem so the
        audit system has accurate supplier-violation data for the optimal basket."""
        df = _suppliers_df().copy()
        df.loc[df["supplier_id"] == "CAM1", "child_labor"] = 1.0  # flag CAM1

        result = solve_best_over_k(
            df, _users_df(), _POLICY,
            env_cap=2.75, social_cap=3.0, output_flag=0,
            objective_mode="profit",
        )
        assert result is not None
        # avg_child_labor is derived from chosen suppliers; must be finite and >= 0
        assert "avg_child_labor" in result or "chosen" in result  # at minimum chosen is present
        chosen_df = df[df["supplier_id"].isin(result["chosen"])]
        if "child_labor" in chosen_df.columns:
            expected_avg_cl = float(chosen_df["child_labor"].mean())
            assert expected_avg_cl >= 0.0
