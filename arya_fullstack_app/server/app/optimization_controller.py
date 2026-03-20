"""OptimizationController.py

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
    Attributes are transformed directionally before use (scale 1-5):
            bad attrs  (env_risk, social_risk, low_quality): (5 - x)  ->  lower raw = higher utility
      good attrs (strategic, improvement):                         (x - 1)  ->  higher raw = higher utility
            cost_score is intentionally excluded from utility to preserve Profit-vs-Utility trade-off.
    All terms are non-negative. Formula:
    utility_total = sum_{u in selected_users}
        [ w_env*u   * env_mult        * (5 - avg(env_risk))
        + w_social*u * social_mult    * (5 - avg(social_risk))
        + w_strategic*u * strategic_mult * (avg(strategic) - 1)
        + w_improvement*u * improvement_mult * (avg(improvement) - 1)
        + w_low_quality*u * low_quality_mult * (5 - avg(low_quality)) ]

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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_XLSX_PATH = PROJECT_ROOT / "Arya_Phones_Supplier_Selection.xlsx"


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
    served_users: int = 8
    price_per_user: float = 100.0
    cost_scale: float = 10.0
    env_cap: float = 2.75
    social_cap: float = 3.0
    output_flag: int = 0


@dataclass
class MaxUtilConfig:
    served_users: int = 8
    price_per_user: float = 100.0
    cost_scale: float = 10.0
    env_cap: float = 2.75
    social_cap: float = 3.0
    output_flag: int = 0


def manual_metrics(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    policy: Policy,
    cfg: MaxProfitConfig | MaxUtilConfig,
    picks: List[str],
) -> Dict[str, Any]:
    pol = policy.clamp_nonnegative()
    a = _avg_of_selected(suppliers_df, picks)

    feasible = True
    if a["k"] <= 0:
        feasible = False
    if a["avg_env"] > float(cfg.env_cap) + 1e-12:
        feasible = False
    if a["avg_social"] > float(cfg.social_cap) + 1e-12:
        feasible = False

    served = min(int(cfg.served_users), int(len(users_df)))

    profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * a["avg_cost"]
    profit_total = served * profit_per_user

    u = _select_last_n_users(users_df, served)
    if len(u):
        ut_env = 5.0 - a["avg_env"]
        ut_social = 5.0 - a["avg_social"]
        ut_strategic = a["avg_strategic"] - 1.0
        ut_improvement = a["avg_improvement"] - 1.0
        ut_lq = 5.0 - a["avg_low_quality"]
        utility_per_user = (
            u["w_env"] * (pol.env_mult * ut_env)
            + u["w_social"] * (pol.social_mult * ut_social)
            + u["w_strategic"] * (pol.strategic_mult * ut_strategic)
            + u["w_improvement"] * (pol.improvement_mult * ut_improvement)
            + u["w_low_quality"] * (pol.low_quality_mult * ut_lq)
        )
        utility_total = float(utility_per_user.sum())
    else:
        utility_total = 0.0

    return {
        "feasible": bool(feasible),
        "metrics": {
            "k": float(a["k"]),
            "avg_env": float(a["avg_env"]),
            "avg_social": float(a["avg_social"]),
            "avg_cost": float(a["avg_cost"]),
            "avg_strategic": float(a["avg_strategic"]),
            "avg_improvement": float(a["avg_improvement"]),
            "avg_low_quality": float(a["avg_low_quality"]),
            "profit_total": float(profit_total),
            "utility_total": float(utility_total),
        },
    }


from .max_utility_optimizer import MaxUtilAgent, MaxUtilityAgent
from .mincost_optimizer import MaxProfitAgent, MinCostAgent