"""
MinCostAgent.py (REWRITTEN)

This project is no longer about choosing a "best governmental policy".
Instead, **policy is fixed** (chosen in the Policy tab) and students choose the
best supplier (or supplier set) *under that policy* using Gurobi.

Data contract (STRICT):
- Excel file MUST be located next to this module and named:
    Arya_Phones_Supplier_Selection.xlsx
- Sheet names MUST be exactly:
    Supplier
    User
- Required semantic columns in Supplier sheet:
    supplier_id, env_risk, social_risk, cost_score, strategic, improvement,
    child_labor, banned_chem, low_quality
- Required semantic columns in User sheet:
    user_id, w_env, w_social, w_cost, w_strategic, w_improvement, w_low_quality
  Convention: w_low_quality is negated (so it subtracts from utility).

No fallbacks, no auto-detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False


# -----------------------------------------------------------------------------
# Paths / IO
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX_PATH = BASE_DIR / "Arya_Phones_Supplier_Selection.xlsx"


# -----------------------------------------------------------------------------
# Policy
# -----------------------------------------------------------------------------
@dataclass
class Policy:
    # Multipliers (>=0). UI uses discrete levels {1, 5, 10}.
    env_mult: float = 1.0
    social_mult: float = 1.0
    cost_mult: float = 1.0
    strategic_mult: float = 1.0
    improvement_mult: float = 1.0
    low_quality_mult: float = 1.0

    # Hard bans (binary toggles; UI uses Yes=1, No=0).
    # If 1, suppliers with the forbidden attribute cannot be selected/matched.
    child_labor_penalty: float = 0.0
    banned_chem_penalty: float = 0.0

    def clamp_nonnegative(self) -> "Policy":
        for k, v in self.__dict__.items():
            vv = 0.0 if v is None else float(v)
            setattr(self, k, max(0.0, vv))
        return self

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


# -----------------------------------------------------------------------------
# Column normalization
# -----------------------------------------------------------------------------
def _normalize_supplier_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "supplier": "supplier_id",
        "supplier_id": "supplier_id",
        "supplier id": "supplier_id",
        "id": "supplier_id",
        "environmental risk": "env_risk",
        "env risk": "env_risk",
        "env_risk": "env_risk",
        "social risk": "social_risk",
        "social_risk": "social_risk",
        "cost score": "cost_score",
        "cost": "cost_score",
        "cost_score": "cost_score",
        "strategic importance": "strategic",
        "strategic": "strategic",
        "improvement potential": "improvement",
        "improvement": "improvement",
        "child labor": "child_labor",
        "child_labor": "child_labor",
        "banned chemicals": "banned_chem",
        "banned chemical": "banned_chem",
        "banned_chem": "banned_chem",
        "low product quality": "low_quality",
        "low quality": "low_quality",
        "low_quality": "low_quality",
    }

    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    rename: Dict[str, str] = {}
    for c in df2.columns:
        key = str(c).strip().lower()
        if key in col_map:
            rename[c] = col_map[key]
    df2 = df2.rename(columns=rename)

    required = [
        "supplier_id",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "child_labor",
        "banned_chem",
        "low_quality",
    ]
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError(f"Supplier sheet is missing columns: {missing}")

    df2 = df2[required].copy()
    df2["supplier_id"] = df2["supplier_id"].astype(str)

    for c in required:
        if c == "supplier_id":
            continue
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).astype(float)

    return df2


def _normalize_user_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {
        "user": "user_id",
        "users": "user_id",
        "user_id": "user_id",
        "user id": "user_id",
        "environmental risk": "w_env",
        "env risk": "w_env",
        "w_env": "w_env",
        "social risk": "w_social",
        "w_social": "w_social",
        "cost score": "w_cost",
        "cost": "w_cost",
        "w_cost": "w_cost",
        "strategic importance": "w_strategic",
        "strategic": "w_strategic",
        "w_strategic": "w_strategic",
        "improvement potential": "w_improvement",
        "improvement": "w_improvement",
        "w_improvement": "w_improvement",
        "low product quality": "w_low_quality",
        "low quality": "w_low_quality",
        "w_low_quality": "w_low_quality",
    }

    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    rename: Dict[str, str] = {}
    for c in df2.columns:
        key = str(c).strip().lower()
        if key in col_map:
            rename[c] = col_map[key]
    df2 = df2.rename(columns=rename)

    required = ["user_id", "w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]
    missing = [c for c in required if c not in df2.columns]
    if missing:
        raise ValueError(f"User sheet is missing columns: {missing}")

    df2 = df2[required].copy()
    df2["user_id"] = df2["user_id"].astype(str)
    for c in required:
        if c == "user_id":
            continue
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).astype(float)

    # Convention: low-quality weight subtracts from utility
    df2["w_low_quality"] = -df2["w_low_quality"].astype(float)
    return df2


def load_supplier_user_tables(xlsx_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_XLSX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found at: {path}")
    suppliers_raw = pd.read_excel(path, sheet_name="Supplier", engine="openpyxl")
    users_raw = pd.read_excel(path, sheet_name="User", engine="openpyxl")
    return _normalize_supplier_columns(suppliers_raw), _normalize_user_columns(users_raw)


# -----------------------------------------------------------------------------
# Shared helpers for all agents
# -----------------------------------------------------------------------------
def _select_last_n_users(users_df: pd.DataFrame, last_n: int) -> List[str]:
    users_df = users_df.copy()
    users_df["user_id"] = users_df["user_id"].astype(str)
    users_all = users_df["user_id"].tolist()
    n = max(0, int(last_n))
    return users_all[-n:] if n > 0 else users_all


def _utility_expression(
    pol: Policy,
    u_row: pd.Series,
    s_row: pd.Series,
) -> float:
    # NOTE: user weights are already numeric; w_low_quality is NEG.
    return float(
        u_row["w_env"] * (pol.env_mult * s_row["env_risk"])
        + u_row["w_social"] * (pol.social_mult * s_row["social_risk"])
        + u_row["w_cost"] * (pol.cost_mult * s_row["cost_score"])
        + u_row["w_strategic"] * (pol.strategic_mult * s_row["strategic"])
        + u_row["w_improvement"] * (pol.improvement_mult * s_row["improvement"])
        + u_row["w_low_quality"] * (pol.low_quality_mult * s_row["low_quality"])
    )


def _auto_big_m(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pol: Policy, users: List[str]) -> float:
    """
    Conservative Big-M bound for constraints like:
        utility(i,u) >= min_utility - M*(1 - z[i,u])
    """
    s = suppliers_df.copy()
    u = users_df[users_df["user_id"].isin(users)].copy()

    if len(s) == 0 or len(u) == 0:
        return 1.0

    # maxima of supplier attributes (scaled by policy)
    max_s = {
        "env_risk": float((pol.env_mult * s["env_risk"]).abs().max()),
        "social_risk": float((pol.social_mult * s["social_risk"]).abs().max()),
        "cost_score": float((pol.cost_mult * s["cost_score"]).abs().max()),
        "strategic": float((pol.strategic_mult * s["strategic"]).abs().max()),
        "improvement": float((pol.improvement_mult * s["improvement"]).abs().max()),
        "low_quality": float((pol.low_quality_mult * s["low_quality"]).abs().max()),
    }

    # maxima of user weights
    max_u = {
        "w_env": float(u["w_env"].abs().max()),
        "w_social": float(u["w_social"].abs().max()),
        "w_cost": float(u["w_cost"].abs().max()),
        "w_strategic": float(u["w_strategic"].abs().max()),
        "w_improvement": float(u["w_improvement"].abs().max()),
        "w_low_quality": float(u["w_low_quality"].abs().max()),
    }

    pref_max = (
        max_u["w_env"] * max_s["env_risk"]
        + max_u["w_social"] * max_s["social_risk"]
        + max_u["w_cost"] * max_s["cost_score"]
        + max_u["w_strategic"] * max_s["strategic"]
        + max_u["w_improvement"] * max_s["improvement"]
        + max_u["w_low_quality"] * max_s["low_quality"]
    )

    return float(pref_max + 10.0)


def _apply_hard_bans(m: "gp.Model", y: Any, suppliers_df: pd.DataFrame, pol: Policy) -> None:
    """
    Hard bans (policy toggles):
      - If child_labor_penalty == 1: suppliers with child_labor==1 cannot be selected.
      - If banned_chem_penalty == 1: suppliers with banned_chem==1 cannot be selected.
    """
    if gp is None:
        return

    Suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    s_child = dict(zip(suppliers_df["supplier_id"].astype(str), suppliers_df["child_labor"].astype(float)))
    s_ban = dict(zip(suppliers_df["supplier_id"].astype(str), suppliers_df["banned_chem"].astype(float)))

    if float(pol.child_labor_penalty) >= 0.5:
        for i in Suppliers:
            if float(s_child.get(i, 0.0)) >= 0.5:
                m.addConstr(y[i] == 0, name=f"ban_child_labor[{i}]")
    if float(pol.banned_chem_penalty) >= 0.5:
        for i in Suppliers:
            if float(s_ban.get(i, 0.0)) >= 0.5:
                m.addConstr(y[i] == 0, name=f"ban_banned_chem[{i}]")


# -----------------------------------------------------------------------------
# Max Profit Agent (retailer objective)
# -----------------------------------------------------------------------------
@dataclass
class MaxProfitConfig:
    last_n_users: int = 6
    capacity: int = 6
    suppliers_to_select: int = 1

    # Profit params
    price_per_match: float = 100.0

    # Match only if utility >= min_utility
    min_utility: float = 0.0

    # Optional: require at least this many matches (set 0 to allow no matches)
    min_matches: int = 0

    output_flag: int = 0
    big_m: Optional[float] = None


class MaxProfitAgent:
    """
    Variables:
      y[i] ∈ {0,1}  supplier i selected
      z[i,u] ∈ {0,1}  user u matched to supplier i

    Constraints:
      - select exactly K suppliers: Σ_i y[i] = K
      - each user at most once: Σ_i z[i,u] ≤ 1
      - linking: z[i,u] ≤ y[i]
      - total matches: Σ_{i,u} z[i,u] ≤ capacity
      - optional minimum matches: Σ_{i,u} z[i,u] ≥ min_matches
      - utility threshold: if z[i,u]=1 ⇒ utility(i,u) ≥ min_utility
      - policy hard bans (Yes/No toggles)

    Objective (profit):
      maximize Σ_{i,u} (price_per_match − cost_mult·cost_score[i]) · z[i,u]
    """

    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxProfitConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available. Ensure a valid Gurobi installation & license.")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

        self.model: Optional["gp.Model"] = None
        self.y = None
        self.z = None
        self._users: List[str] = _select_last_n_users(self.users, cfg.last_n_users)

    def build(self, name: str = "MaxProfitAgent") -> "gp.Model":
        cfg = self.cfg
        pol = self.policy

        s = self.suppliers.copy()
        u = self.users[self.users["user_id"].isin(self._users)].copy()

        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)

        Suppliers = s["supplier_id"].tolist()
        Users = u["user_id"].tolist()

        # Precompute utilities (as constants) and per-supplier margin
        util: Dict[Tuple[str, str], float] = {}
        for _, srow in s.iterrows():
            sid = str(srow["supplier_id"])
            for _, urow in u.iterrows():
                uid = str(urow["user_id"])
                util[(sid, uid)] = _utility_expression(pol, urow, srow)

        margin = {str(sid): float(cfg.price_per_match - (pol.cost_mult * float(cost))) for sid, cost in zip(s["supplier_id"], s["cost_score"])}

        M = float(cfg.big_m) if cfg.big_m is not None else _auto_big_m(s, self.users, pol, Users)

        m = gp.Model(name)
        m.Params.OutputFlag = int(cfg.output_flag)

        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

        # Policy hard bans
        _apply_hard_bans(m, y, s, pol)

        # Choose K suppliers
        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k")

        # Each user at most once
        for uid in Users:
            m.addConstr(gp.quicksum(z[i, uid] for i in Suppliers) <= 1, name=f"user_once[{uid}]")

        # Linking
        for i in Suppliers:
            for uid in Users:
                m.addConstr(z[i, uid] <= y[i], name=f"link[{i},{uid}]")

        # Capacity bounds
        total_matches = gp.quicksum(z[i, uid] for i in Suppliers for uid in Users)
        m.addConstr(total_matches <= int(cfg.capacity), name="capacity")

        if int(cfg.min_matches) > 0:
            m.addConstr(total_matches >= int(cfg.min_matches), name="min_matches")

        # Utility threshold (Big-M)
        for i in Suppliers:
            for uid in Users:
                m.addConstr(util[(i, uid)] >= float(cfg.min_utility) - M * (1 - z[i, uid]), name=f"minutil[{i},{uid}]")

        # Objective: maximize profit
        m.setObjective(gp.quicksum(margin[i] * z[i, uid] for i in Suppliers for uid in Users), GRB.MAXIMIZE)

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        s = self.suppliers.copy()
        u = self.users.copy()
        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)
        Suppliers = s["supplier_id"].tolist()
        Users = _select_last_n_users(u, self.cfg.last_n_users)

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs = [(uid, i) for i in Suppliers for uid in Users if self.z[i, uid].X > 0.5]
        df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])

        if len(df):
            s_idx = s.set_index("supplier_id")
            u_idx = u.set_index("user_id")
            pol = self.policy

            df["cost_prod"] = df["supplier_id"].map(lambda sid: float(pol.cost_mult * s_idx.loc[sid, "cost_score"]))
            df["margin"] = float(self.cfg.price_per_match) - df["cost_prod"]

            def _util(row: pd.Series) -> float:
                sid = row["supplier_id"]
                uid = row["user_id"]
                return float(_utility_expression(pol, u_idx.loc[uid], s_idx.loc[sid]))

            df["utility"] = df.apply(_util, axis=1)
            df = df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "chosen_suppliers": chosen,
            "selected_users": Users,
            "num_matched": int(len(pairs)),
            "matches": df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }


# -----------------------------------------------------------------------------
# Max Utility Agent (student objective)
# -----------------------------------------------------------------------------
@dataclass
class MaxUtilConfig:
    last_n_users: int = 6
    capacity: int = 6
    suppliers_to_select: int = 1
    min_utility: float = 0.0
    output_flag: int = 0
    big_m: Optional[float] = None


class MaxUtilAgent:
    """
    Same structure as MaxProfitAgent, but objective is total utility:

      maximize Σ_{i,u} utility(i,u) · z[i,u]

    Capacity does not need to be filled. If all utilities are bad (or threshold too high),
    the model can return 0 matches, but still selects exactly K suppliers.
    """

    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxUtilConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available. Ensure a valid Gurobi installation & license.")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

        self.model: Optional["gp.Model"] = None
        self.y = None
        self.z = None
        self._users: List[str] = _select_last_n_users(self.users, cfg.last_n_users)

    def build(self, name: str = "MaxUtilAgent") -> "gp.Model":
        cfg = self.cfg
        pol = self.policy

        s = self.suppliers.copy()
        u = self.users[self.users["user_id"].isin(self._users)].copy()

        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)

        Suppliers = s["supplier_id"].tolist()
        Users = u["user_id"].tolist()

        util: Dict[Tuple[str, str], float] = {}
        for _, srow in s.iterrows():
            sid = str(srow["supplier_id"])
            for _, urow in u.iterrows():
                uid = str(urow["user_id"])
                util[(sid, uid)] = _utility_expression(pol, urow, srow)

        M = float(cfg.big_m) if cfg.big_m is not None else _auto_big_m(s, self.users, pol, Users)

        m = gp.Model(name)
        m.Params.OutputFlag = int(cfg.output_flag)

        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

        # Policy hard bans
        _apply_hard_bans(m, y, s, pol)

        # Choose K suppliers
        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k")

        # Each user at most once
        for uid in Users:
            m.addConstr(gp.quicksum(z[i, uid] for i in Suppliers) <= 1, name=f"user_once[{uid}]")

        # Linking
        for i in Suppliers:
            for uid in Users:
                m.addConstr(z[i, uid] <= y[i], name=f"link[{i},{uid}]")

        # Capacity upper bound
        m.addConstr(gp.quicksum(z[i, uid] for i in Suppliers for uid in Users) <= int(cfg.capacity), name="capacity")

        # Utility threshold (Big-M)
        for i in Suppliers:
            for uid in Users:
                m.addConstr(util[(i, uid)] >= float(cfg.min_utility) - M * (1 - z[i, uid]), name=f"minutil[{i},{uid}]")

        # Objective: maximize total utility
        m.setObjective(gp.quicksum(util[(i, uid)] * z[i, uid] for i in Suppliers for uid in Users), GRB.MAXIMIZE)

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        s = self.suppliers.copy()
        u = self.users.copy()
        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)
        Suppliers = s["supplier_id"].tolist()
        Users = _select_last_n_users(u, self.cfg.last_n_users)

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs = [(uid, i) for i in Suppliers for uid in Users if self.z[i, uid].X > 0.5]
        df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])

        if len(df):
            s_idx = s.set_index("supplier_id")
            u_idx = u.set_index("user_id")
            pol = self.policy

            def _util(row: pd.Series) -> float:
                sid = row["supplier_id"]
                uid = row["user_id"]
                return float(_utility_expression(pol, u_idx.loc[uid], s_idx.loc[sid]))

            df["utility"] = df.apply(_util, axis=1)
            df = df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "chosen_suppliers": chosen,
            "selected_users": Users,
            "num_matched": int(len(pairs)),
            "matches": df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }


# -----------------------------------------------------------------------------
# Min Cost Agent (procurement objective)
# -----------------------------------------------------------------------------
@dataclass
class MinCostConfig:
    last_n_users: int = 6
    capacity: int = 6

    # Exact matches to make (0 allowed). If >0, model must find that many matches.
    matches_to_make: int = 6

    suppliers_to_select: int = 1
    min_utility: float = 0.0

    output_flag: int = 0
    big_m: Optional[float] = None


class MinCostAgent:
    """
    Objective:
      minimize total effective cost:
        Σ_{i,u} (cost_mult·cost_score[i]) · z[i,u]

    Constraints:
      - select exactly K suppliers
      - exact matches: Σ z = matches_to_make
      - plus the usual assignment/linking/threshold + policy hard bans
    """

    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MinCostConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available. Ensure a valid Gurobi installation & license.")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

        self.model: Optional["gp.Model"] = None
        self.y = None
        self.z = None
        self._users: List[str] = _select_last_n_users(self.users, cfg.last_n_users)

        cap = int(cfg.capacity)
        m = int(cfg.matches_to_make)
        if m < 0:
            raise ValueError("matches_to_make must be >= 0.")
        if cap < 0:
            raise ValueError("capacity must be >= 0.")
        if m > cap:
            raise ValueError(f"matches_to_make ({m}) cannot exceed capacity ({cap}).")

    def build(self, name: str = "MinCostAgent") -> "gp.Model":
        cfg = self.cfg
        pol = self.policy

        s = self.suppliers.copy()
        u = self.users[self.users["user_id"].isin(self._users)].copy()

        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)

        Suppliers = s["supplier_id"].tolist()
        Users = u["user_id"].tolist()

        util: Dict[Tuple[str, str], float] = {}
        for _, srow in s.iterrows():
            sid = str(srow["supplier_id"])
            for _, urow in u.iterrows():
                uid = str(urow["user_id"])
                util[(sid, uid)] = _utility_expression(pol, urow, srow)

        cost_prod = {str(sid): float(pol.cost_mult * float(cost)) for sid, cost in zip(s["supplier_id"], s["cost_score"])}

        M = float(cfg.big_m) if cfg.big_m is not None else _auto_big_m(s, self.users, pol, Users)

        m = gp.Model(name)
        m.Params.OutputFlag = int(cfg.output_flag)

        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

        # Policy hard bans
        _apply_hard_bans(m, y, s, pol)

        # Choose K suppliers
        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k")

        # Each user at most once
        for uid in Users:
            m.addConstr(gp.quicksum(z[i, uid] for i in Suppliers) <= 1, name=f"user_once[{uid}]")

        # Linking
        for i in Suppliers:
            for uid in Users:
                m.addConstr(z[i, uid] <= y[i], name=f"link[{i},{uid}]")

        total_matches = gp.quicksum(z[i, uid] for i in Suppliers for uid in Users)
        m.addConstr(total_matches == int(cfg.matches_to_make), name="matches_exact")
        m.addConstr(total_matches <= int(cfg.capacity), name="capacity")

        # Utility threshold (Big-M)
        for i in Suppliers:
            for uid in Users:
                m.addConstr(util[(i, uid)] >= float(cfg.min_utility) - M * (1 - z[i, uid]), name=f"minutil[{i},{uid}]")

        # Objective: minimize total cost
        m.setObjective(gp.quicksum(cost_prod[i] * z[i, uid] for i in Suppliers for uid in Users), GRB.MINIMIZE)

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        s = self.suppliers.copy()
        u = self.users.copy()
        s["supplier_id"] = s["supplier_id"].astype(str)
        u["user_id"] = u["user_id"].astype(str)

        Suppliers = s["supplier_id"].tolist()
        Users = _select_last_n_users(u, self.cfg.last_n_users)

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs = [(uid, i) for i in Suppliers for uid in Users if self.z[i, uid].X > 0.5]
        df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])

        if len(df):
            s_idx = s.set_index("supplier_id")
            u_idx = u.set_index("user_id")
            pol = self.policy

            df["cost_prod"] = df["supplier_id"].map(lambda sid: float(pol.cost_mult * s_idx.loc[sid, "cost_score"]))

            def _util(row: pd.Series) -> float:
                sid = row["supplier_id"]
                uid = row["user_id"]
                return float(_utility_expression(pol, u_idx.loc[uid], s_idx.loc[sid]))

            df["utility"] = df.apply(_util, axis=1)
            df = df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "chosen_suppliers": chosen,
            "selected_users": Users,
            "num_matched": int(len(pairs)),
            "matches": df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }
