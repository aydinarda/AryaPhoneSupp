"""MinCostAgent.py

This project uses a **supplier-set averaging game**:

- Students pick any non-empty set of suppliers.
- The delivered product is the **average** of the chosen suppliers' attributes.
- Risk is **only a hard constraint**:
    avg(env_risk)   <= env_cap
    avg(social_risk)<= social_cap
- Served end-users is fixed (served_users).

Objectives:
- Max Profit:
    profit_per_user = price_per_user - cost_scale * avg(cost_score)
    profit_total    = served_users * profit_per_user

- Max Utility:
    utility_total = sum_{u in selected_users}
        [ w_env*u * env_mult*avg(env_risk) + ... + w_low_quality*u * low_quality_mult*avg(low_quality) ]

Benchmarks:
- The optimizer is allowed to choose **any number of suppliers** (no fixed K).
- Because objectives depend on the average (division by k), we solve by enumerating k=1..N
  and solving a MILP with the constraint sum(y)=k for each k.

UI requirements supported:
- Manual evaluation never calls the solver (so students never see solver infeasibility messages).
- Benchmark hides the chosen supplier IDs (UI does not display them).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re
import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX_PATH = BASE_DIR / "Arya_Phones_Supplier_Selection.xlsx"


@dataclass
class Policy:
    env_mult: float = 1.0
    social_mult: float = 1.0
    cost_mult: float = 1.0
    strategic_mult: float = 1.0
    improvement_mult: float = 1.0
    low_quality_mult: float = 1.0
    child_labor_penalty: float = 0.0  # if >=0.5 => ban suppliers with child_labor==1
    banned_chem_penalty: float = 0.0  # if >=0.5 => ban suppliers with banned_chem==1

    def clamp_nonnegative(self) -> "Policy":
        for k, v in self.__dict__.items():
            vv = 0.0 if v is None else float(v)
            setattr(self, k, max(0.0, vv))
        return self

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


def _canon_col(c: str) -> str:
    """Canonicalize a column name for fuzzy matching."""
    c = str(c).strip().lower()
    c = re.sub(r"[^0-9a-z]+", " ", c)
    c = re.sub(r"\s+", " ", c).strip()
    return c


def _fuzzy_rename(df: pd.DataFrame, patterns_to_target: Dict[str, str]) -> pd.DataFrame:
    """Rename columns using regex patterns on canonicalized names."""
    df = df.copy()
    new_cols = {}
    canon = {col: _canon_col(col) for col in df.columns}
    for col, ccol in canon.items():
        for pat, target in patterns_to_target.items():
            if re.search(pat, ccol):
                new_cols[col] = target
                break
    return df.rename(columns=new_cols)


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
        "low quality": "low_quality",
            "low product quality": "low_quality",
            "product quality": "low_quality",
        "low_quality": "low_quality",
        "child labor": "child_labor",
        "child_labor": "child_labor",
        "banned chem": "banned_chem",
            "banned chemicals": "banned_chem",
            "banned chemical": "banned_chem",
            "restricted chemicals": "banned_chem",
        "banned_chem": "banned_chem",
    }

    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    required = [
        "supplier_id",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "low_quality",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Suppliers sheet missing columns: {missing}")

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    for b in ["child_labor", "banned_chem"]:
        if b not in df.columns:
            df[b] = 0.0
        df[b] = pd.to_numeric(df[b], errors="coerce").fillna(0.0).astype(float)

    df["supplier_id"] = df["supplier_id"].astype(str)
    return df


def _normalize_user_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    col_map = {
        "users": "user_id",
        "user": "user_id",
        "user_id": "user_id",
        "user id": "user_id",
        "id": "user_id",

        "environmental risk": "w_env",
        "env risk": "w_env",
        "env_risk": "w_env",
        "w_env": "w_env",
        "w_environment": "w_env",

        "social risk": "w_social",
        "social_risk": "w_social",
        "w_social": "w_social",

        "cost score": "w_cost",
        "cost": "w_cost",
        "cost_score": "w_cost",
        "w_cost": "w_cost",

        "strategic importance": "w_strategic",
        "strategic": "w_strategic",
        "w_strategic": "w_strategic",

        "improvement potential": "w_improvement",
        "improvement": "w_improvement",
        "w_improvement": "w_improvement",

        "low product quality": "w_low_quality",
        "product quality": "w_low_quality",
        "low quality": "w_low_quality",
        "low_quality": "w_low_quality",
        "w_low_quality": "w_low_quality",
    }

    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={c: col_map.get(c, c) for c in df.columns})

    patterns_to_target = {
        r"^users?$": "user_id",
        r"^user( id)?$": "user_id",
        r"^(env|environment|environmental)( risk| score)?$": "w_env",
        r"^social( risk| score)?$": "w_social",
        r"^(cost|price)( risk| score)?$": "w_cost",
        r"^strategic( importance)?( score)?$": "w_strategic",
        r"^improvement( potential)?( score)?$": "w_improvement",
        r"^(low product quality|product quality|low quality|lowquality|quality)( risk| score)?$": "w_low_quality",
    
            r"^(banned|restricted) (chem|chemical|chemicals)$": "banned_chem",
        
            r"^child( labour| labor)?$": "child_labor",
        }
    df = _fuzzy_rename(df, patterns_to_target)

    required = ["user_id", "w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        detected = list(df.columns)
        raise ValueError(
            "Users sheet missing required columns: "
            f"{missing}. Detected columns: {detected}. "
            "Expected: Users/User ID, plus weights for Environmental Risk, Social Risk, Cost Score, "
            "Strategic Importance, Improvement Potential, Low Product Quality."
        )

    df["user_id"] = df["user_id"].astype(str)
    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df


def load_supplier_user_tables(xlsx_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load supplier and user tables from the canonical workbook.

    The workbook contains many sheets. We prefer exact sheet names:
    - Supplier
    - User
    If not found, fall back to the first sheet containing 'supplier' / 'user' (case-insensitive).
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(str(xlsx_path))

    xl = pd.ExcelFile(xlsx_path)
    sheet_names = xl.sheet_names

    def pick_exact_or_contains(preferred_exact: str, token: str) -> str:
        for s in sheet_names:
            if s.strip().lower() == preferred_exact.lower():
                return s
        for s in sheet_names:
            if token in s.strip().lower():
                return s
        return sheet_names[0]

    supplier_sheet = pick_exact_or_contains("Supplier", "supplier")
    user_sheet = pick_exact_or_contains("User", "user")

    suppliers = _normalize_supplier_columns(pd.read_excel(xlsx_path, sheet_name=supplier_sheet))
    users = _normalize_user_columns(pd.read_excel(xlsx_path, sheet_name=user_sheet))

    suppliers = suppliers.sort_values("supplier_id").reset_index(drop=True)
    users = users.reset_index(drop=True)
    return suppliers, users


def _select_last_n_users(users_df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(n)
    if n <= 0:
        return users_df.iloc[0:0].copy()
    if len(users_df) <= n:
        return users_df.copy()
    return users_df.iloc[-n:].copy()


def _avg_of_selected(suppliers_df: pd.DataFrame, picks: List[str]) -> Dict[str, float]:
    if not picks:
        return {
            "k": 0.0,
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "avg_strategic": 0.0,
            "avg_improvement": 0.0,
            "avg_low_quality": 0.0,
        }

    sel = suppliers_df[suppliers_df["supplier_id"].astype(str).isin([str(x) for x in picks])].copy()
    if sel.empty:
        return {
            "k": 0.0,
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "avg_strategic": 0.0,
            "avg_improvement": 0.0,
            "avg_low_quality": 0.0,
        }

    return {
        "k": float(len(sel)),
        "avg_env": float(sel["env_risk"].mean()),
        "avg_social": float(sel["social_risk"].mean()),
        "avg_cost": float(sel["cost_score"].mean()),
        "avg_strategic": float(sel["strategic"].mean()),
        "avg_improvement": float(sel["improvement"].mean()),
        "avg_low_quality": float(sel["low_quality"].mean()),
    }


@dataclass
class MaxProfitConfig:
    served_users: int = 10
    price_per_user: float = 100.0
    cost_scale: float = 10.0
    env_cap: float = 2.75
    social_cap: float = 3.0
    output_flag: int = 0


@dataclass
class MaxUtilConfig:
    served_users: int = 10
    price_per_user: float = 100.0
    cost_scale: float = 10.0
    env_cap: float = 2.75
    social_cap: float = 3.0
    output_flag: int = 0



def _user_utilities_from_avg(
    users_df: pd.DataFrame,
    policy: Policy,
    avg_env: float,
    avg_social: float,
    avg_cost: float,
    avg_strategic: float,
    avg_improvement: float,
    avg_low_quality: float,
) -> pd.Series:
    """Compute per-user utility given averaged supplier attributes."""
    pol = policy.clamp_nonnegative()
    u = users_df.copy()
    return (
        u["w_env"] * (pol.env_mult * float(avg_env))
        + u["w_social"] * (pol.social_mult * float(avg_social))
        + u["w_cost"] * (pol.cost_mult * float(avg_cost))
        + u["w_strategic"] * (pol.strategic_mult * float(avg_strategic))
        + u["w_improvement"] * (pol.improvement_mult * float(avg_improvement))
        + u["w_low_quality"] * (pol.low_quality_mult * float(avg_low_quality))
    ).astype(float)


def manual_metrics(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    policy: Policy,
    cfg: MaxProfitConfig | MaxUtilConfig,
    picks: List[str],
) -> Dict[str, Any]:
    """Evaluate a manual supplier pick with *customer eligibility + capacity*.

    Rules (fixed):
      - We can sell **only** to users with utility >= 2.35
      - Maximum customers served (capacity) = 8
    """
    pol = policy.clamp_nonnegative()
    a = _avg_of_selected(suppliers_df, picks)

    feasible = True
    if a["k"] <= 0:
        feasible = False
    if a["avg_env"] > float(cfg.env_cap) + 1e-12:
        feasible = False
    if a["avg_social"] > float(cfg.social_cap) + 1e-12:
        feasible = False

    # Fixed selling rules
    UTILITY_MIN = 2.35
    CAPACITY = 8

    profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * a["avg_cost"]

    # Compute utilities for *all* users and serve the best eligible ones up to capacity
    if len(users_df):
        util = _user_utilities_from_avg(
            users_df,
            pol,
            a["avg_env"],
            a["avg_social"],
            a["avg_cost"],
            a["avg_strategic"],
            a["avg_improvement"],
            a["avg_low_quality"],
        )
        eligible = users_df.loc[util >= UTILITY_MIN].copy()
        eligible["utility"] = util[util >= UTILITY_MIN].values

        served_df = eligible.sort_values("utility", ascending=False).head(CAPACITY)
        served = int(len(served_df))
        utility_total = float(served_df["utility"].sum())
        served_user_ids = served_df["user_id"].astype(str).tolist()
    else:
        served = 0
        utility_total = 0.0
        served_user_ids = []

    profit_total = float(served) * float(profit_per_user)

    return {
        "feasible": bool(feasible),
        "served_user_ids": served_user_ids,
        "metrics": {
            "k": float(a["k"]),
            "avg_env": float(a["avg_env"]),
            "avg_social": float(a["avg_social"]),
            "avg_cost": float(a["avg_cost"]),
            "profit_total": float(profit_total),
            "utility_total": float(utility_total),
            "served_users": float(served),
        },
    }


def _apply_policy_bans(model: "gp.Model", y, suppliers_df: pd.DataFrame, policy: Policy) -> None:
    pol = policy.clamp_nonnegative()
    if float(pol.child_labor_penalty) >= 0.5 and "child_labor" in suppliers_df.columns:
        for _, r in suppliers_df.iterrows():
            if float(r.get("child_labor", 0.0)) >= 0.5:
                model.addConstr(y[str(r["supplier_id"])] == 0)

    if float(pol.banned_chem_penalty) >= 0.5 and "banned_chem" in suppliers_df.columns:
        for _, r in suppliers_df.iterrows():
            if float(r.get("banned_chem", 0.0)) >= 0.5:
                model.addConstr(y[str(r["supplier_id"])] == 0)


def _solve_best_over_k(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    policy: Policy,
    served_users: int,
    env_cap: float,
    social_cap: float,
    output_flag: int,
    objective_mode: str,
) -> Optional[Dict[str, Any]]:
    """Benchmark optimizer with *customer eligibility + capacity*.

    Notes:
      - `served_users` is ignored for backward compatibility.
      - Fixed rules:
          UTILITY_MIN = 2.35
          CAPACITY    = 8
      - We enumerate k=1..N suppliers to avoid division by k (average) inside the MILP.
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    pol = policy.clamp_nonnegative()
    df = suppliers_df.copy()
    df["supplier_id"] = df["supplier_id"].astype(str)

    S = df["supplier_id"].tolist()
    N = len(S)
    if N == 0:
        return None

    UDF = users_df.copy()
    UDF["user_id"] = UDF["user_id"].astype(str)
    user_ids = UDF["user_id"].tolist()

    # Fixed selling rules
    UTILITY_MIN = 2.35
    CAPACITY = 8

    best_obj = None
    best_sol: Optional[Dict[str, Any]] = None

    for k in range(1, N + 1):
        model = gp.Model(f"bench_{objective_mode}_k{k}")
        model.Params.OutputFlag = int(output_flag)

        # Supplier decision
        y = {s: model.addVar(vtype=GRB.BINARY, name=f"y[{s}]") for s in S}
        model.addConstr(gp.quicksum(y[s] for s in S) == k, name="k_suppliers")

        # Policy bans (hard zeroing)
        _apply_policy_bans(model, y, df, pol)

        # Risk caps on averages: sum(y*attr) <= k*cap
        model.addConstr(
            gp.quicksum(y[s] * float(df.loc[df["supplier_id"] == s, "env_risk"].iloc[0]) for s in S)
            <= float(env_cap) * k,
            name="env_cap",
        )
        model.addConstr(
            gp.quicksum(y[s] * float(df.loc[df["supplier_id"] == s, "social_risk"].iloc[0]) for s in S)
            <= float(social_cap) * k,
            name="social_cap",
        )

        # Customer decision
        x = {u: model.addVar(vtype=GRB.BINARY, name=f"x[{u}]") for u in user_ids}
        model.addConstr(gp.quicksum(x[u] for u in user_ids) <= CAPACITY, name="capacity")

        # Precompute coefficient per (user, supplier) for the utility numerator (without /k)
        # num_u = sum_s coeff[u,s] * y_s
        coeff = {}
        for _, ur in UDF.iterrows():
            uid = str(ur["user_id"])
            w_env = float(ur["w_env"])
            w_soc = float(ur["w_social"])
            w_cost = float(ur["w_cost"])
            w_str = float(ur["w_strategic"])
            w_imp = float(ur["w_improvement"])
            w_lq = float(ur["w_low_quality"])
            for _, sr in df.iterrows():
                sid = str(sr["supplier_id"])
                coeff[(uid, sid)] = (
                    w_env * (pol.env_mult * float(sr["env_risk"]))
                    + w_soc * (pol.social_mult * float(sr["social_risk"]))
                    + w_cost * (pol.cost_mult * float(sr["cost_score"]))
                    + w_str * (pol.strategic_mult * float(sr["strategic"]))
                    + w_imp * (pol.improvement_mult * float(sr["improvement"]))
                    + w_lq * (pol.low_quality_mult * float(sr["low_quality"]))
                )

        # Eligibility: (1/k)*num_u >= UTILITY_MIN when served
        # => num_u >= UTILITY_MIN * k * x_u
        for uid in user_ids:
            num_u = gp.quicksum(coeff[(uid, sid)] * y[sid] for sid in S)
            model.addConstr(num_u >= float(UTILITY_MIN) * k * x[uid], name=f"utility_min[{uid}]")

        # Average cost pieces
        cost_sum = gp.quicksum(
            y[s] * float(df.loc[df["supplier_id"] == s, "cost_score"].iloc[0]) for s in S
        )

        t = model.addVar(lb=0.0, ub=float(CAPACITY), vtype=GRB.CONTINUOUS, name="served_count")
        model.addConstr(t == gp.quicksum(x[u] for u in user_ids), name="served_count_def")

        if objective_mode == "profit":
            # profit_total = price*t - cost_scale*(1/k)*cost_sum*t  (bilinear)
            # Linearize w = cost_sum * t using McCormick envelopes.
            w = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="w_costsum_times_t")

            # bounds for cost_sum
            costs = df["cost_score"].astype(float).tolist()
            La = float(min(costs) * k)
            Ua = float(max(costs) * k)
            Lb = 0.0
            Ub = float(CAPACITY)

            # McCormick constraints for w = a*b, a=cost_sum, b=t
            # w >= La*b + Lb*a - La*Lb  (Lb=0)
            model.addConstr(w >= La * t, name="mcc1")
            # w >= Ua*b + Ub*a - Ua*Ub
            model.addConstr(w >= Ua * t + Ub * cost_sum - Ua * Ub, name="mcc2")
            # w <= Ua*b + Lb*a - Ua*Lb (Lb=0)
            model.addConstr(w <= Ua * t, name="mcc3")
            # w <= La*b + Ub*a - La*Ub
            model.addConstr(w <= La * t + Ub * cost_sum - La * Ub, name="mcc4")

            obj = float(100.0) * t - float(10.0) * (1.0 / k) * w
            # Note: price_per_user and cost_scale are fixed by UI (100 and 10) in this project version.
            model.setObjective(obj, GRB.MAXIMIZE)

        elif objective_mode == "utility":
            # Maximize sum_u x_u * (1/k)*num_u  (bilinear)
            # Linearize m_u = x_u * num_u with bounds on num_u.
            m_vars = {}
            obj_terms = []
            for uid in user_ids:
                num_u = gp.quicksum(coeff[(uid, sid)] * y[sid] for sid in S)

                # bounds on num_u: k * min(coeff_u_s) .. k * max(coeff_u_s)
                vals = [coeff[(uid, sid)] for sid in S]
                L = float(min(vals) * k)
                U = float(max(vals) * k)

                m = model.addVar(lb=min(0.0, L), ub=max(0.0, U), vtype=GRB.CONTINUOUS, name=f"m[{uid}]")
                # Standard linearization for product of binary x and bounded continuous num_u:
                model.addConstr(m <= U * x[uid], name=f"lin1[{uid}]")
                model.addConstr(m >= L * x[uid], name=f"lin2[{uid}]")
                model.addConstr(m <= num_u - L * (1 - x[uid]), name=f"lin3[{uid}]")
                model.addConstr(m >= num_u - U * (1 - x[uid]), name=f"lin4[{uid}]")
                m_vars[uid] = m
                obj_terms.append(m)

            model.setObjective((1.0 / k) * gp.quicksum(obj_terms), GRB.MAXIMIZE)
        else:
            raise ValueError(f"Unknown objective_mode: {objective_mode}")

        model.optimize()

        if model.Status != GRB.OPTIMAL:
            continue

        obj_val = float(model.ObjVal)
        if best_obj is None or obj_val > best_obj + 1e-9:
            chosen = [s for s in S if y[s].X > 0.5]

            sel = df[df["supplier_id"].isin(chosen)]
            avg_env = float(sel["env_risk"].mean())
            avg_soc = float(sel["social_risk"].mean())
            avg_cost = float(sel["cost_score"].mean())
            avg_str = float(sel["strategic"].mean())
            avg_imp = float(sel["improvement"].mean())
            avg_lq = float(sel["low_quality"].mean())

            # Served users
            served_users_list = [u for u in user_ids if x[u].X > 0.5]

            best_obj = obj_val
            best_sol = {
                "k": float(k),
                "chosen_suppliers": chosen,
                "served_user_ids": served_users_list,
                "avg_env": avg_env,
                "avg_social": avg_soc,
                "avg_cost": avg_cost,
                "avg_strategic": avg_str,
                "avg_improvement": avg_imp,
                "avg_low_quality": avg_lq,
                "objective_value": obj_val,
            }

    return best_sol
class MaxProfitAgent:
    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxProfitConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

    def solve(self) -> Dict[str, Any]:
        cfg = self.cfg

        best = _solve_best_over_k(
            self.suppliers,
            self.users,
            self.policy,
            served_users=int(cfg.served_users),
            env_cap=float(cfg.env_cap),
            social_cap=float(cfg.social_cap),
            output_flag=int(cfg.output_flag),
            objective_mode="profit",
        )

        if best is None:
            return {
                "feasible": False,
                "metrics": {
                    "k": 0.0,
                    "avg_env": 0.0,
                    "avg_social": 0.0,
                    "avg_cost": 0.0,
                    "profit_total": 0.0,
                    "utility_total": 0.0,
                },
            }

        # Recompute totals using the official formulas (with customer selection)
        served_user_ids = list(best.get("served_user_ids", []))
        served = int(len(served_user_ids))

        profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * float(best["avg_cost"])
        profit_total = served * profit_per_user

        # utility_total over the served users
        if served and len(self.users):
            util_all = _user_utilities_from_avg(
                self.users,
                self.policy,
                float(best["avg_env"]),
                float(best["avg_social"]),
                float(best["avg_cost"]),
                float(best["avg_strategic"]),
                float(best["avg_improvement"]),
                float(best["avg_low_quality"]),
            )
            tmp = self.users.copy()
            tmp["utility"] = util_all.values
            utility_total = float(tmp[tmp["user_id"].astype(str).isin(set(served_user_ids))]["utility"].sum())
        else:
            utility_total = 0.0

        feasible = (float(best["avg_env"]) <= float(cfg.env_cap) + 1e-12) and (
            float(best["avg_social"]) <= float(cfg.social_cap) + 1e-12
        )

        return {
            "feasible": bool(feasible),
            "served_user_ids": served_user_ids,
            "metrics": {
                "k": float(best["k"]),
                "avg_env": float(best["avg_env"]),
                "avg_social": float(best["avg_social"]),
                "avg_cost": float(best["avg_cost"]),
                "profit_total": float(profit_total),
                "utility_total": float(utility_total),
                "served_users": float(served),
            },
        }


class MaxUtilAgent:
    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxUtilConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available")
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

    def solve(self) -> Dict[str, Any]:
        cfg = self.cfg

        best = _solve_best_over_k(
            self.suppliers,
            self.users,
            self.policy,
            served_users=int(cfg.served_users),
            env_cap=float(cfg.env_cap),
            social_cap=float(cfg.social_cap),
            output_flag=int(cfg.output_flag),
            objective_mode="utility",
        )

        if best is None:
            return {
                "feasible": False,
                "metrics": {
                    "k": 0.0,
                    "avg_env": 0.0,
                    "avg_social": 0.0,
                    "avg_cost": 0.0,
                    "profit_total": 0.0,
                    "utility_total": 0.0,
                },
            }

        served_user_ids = list(best.get("served_user_ids", []))
        served = int(len(served_user_ids))

        profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * float(best["avg_cost"])
        profit_total = served * profit_per_user

        # utility_total over the served users
        if served and len(self.users):
            util_all = _user_utilities_from_avg(
                self.users,
                self.policy,
                float(best["avg_env"]),
                float(best["avg_social"]),
                float(best["avg_cost"]),
                float(best["avg_strategic"]),
                float(best["avg_improvement"]),
                float(best["avg_low_quality"]),
            )
            tmp = self.users.copy()
            tmp["utility"] = util_all.values
            utility_total = float(tmp[tmp["user_id"].astype(str).isin(set(served_user_ids))]["utility"].sum())
        else:
            utility_total = 0.0

        feasible = (float(best["avg_env"]) <= float(cfg.env_cap) + 1e-12) and (
            float(best["avg_social"]) <= float(cfg.social_cap) + 1e-12
        )

        return {
            "feasible": bool(feasible),
            "served_user_ids": served_user_ids,
            "metrics": {
                "k": float(best["k"]),
                "avg_env": float(best["avg_env"]),
                "avg_social": float(best["avg_social"]),
                "avg_cost": float(best["avg_cost"]),
                "profit_total": float(profit_total),
                "utility_total": float(utility_total),
                "served_users": float(served),
            },
        }