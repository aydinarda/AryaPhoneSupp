"""MinCostAgent.py

This module now implements the **Profit–Risk supplier selection game**:

- Students choose a **set of suppliers**.
- The delivered product is the **average** of the chosen suppliers.
- Hard risk caps:
  - avg(env_risk) <= ENV_CAP
  - avg(social_risk) <= SOCIAL_CAP
- Profit:
  - profit_per_user = price_per_user - COST_SCALE * avg(cost_score)
  - profit_total = served_users * profit_per_user

Key change vs the older matching-based version:
- There is **no fixed K** constraint in the game.
- For optimization we still solve exact MILPs by enumerating subset sizes k=1..N.
  (This avoids fractional objectives like minimizing avg(cost) directly.)

Exported API used by UI.py:
- load_supplier_user_tables
- ProfitRiskConfig
- ProfitRiskMinRiskConfig
- ProfitRiskMaxProfitAgent
- ProfitRiskMinRiskAgent
- ProfitRiskCurveAgent
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


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_XLSX_PATH = BASE_DIR / "Arya_Phones_Supplier_Selection.xlsx"


# ---------------------------------------------------------------------
# Excel loading / normalization
# ---------------------------------------------------------------------

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

    rename = {}
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
    """Kept for compatibility (UI doesn't use users in the Profit–Risk game)."""
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    # Minimal normalization: ensure a user_id column exists if present.
    if "user_id" not in [c.lower() for c in df2.columns]:
        # if there's no user sheet or it's unused, return empty
        return pd.DataFrame(columns=["user_id"]).copy()

    # Best-effort rename
    rename = {}
    for c in df2.columns:
        if str(c).strip().lower() in {"user", "users", "user id", "user_id"}:
            rename[c] = "user_id"
    df2 = df2.rename(columns=rename)

    if "user_id" not in df2.columns:
        return pd.DataFrame(columns=["user_id"]).copy()

    df2["user_id"] = df2["user_id"].astype(str)
    return df2


def load_supplier_user_tables(xlsx_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_XLSX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found at: {path}")

    suppliers_raw = pd.read_excel(path, sheet_name="Supplier", engine="openpyxl")

    # user sheet is optional for this game
    try:
        users_raw = pd.read_excel(path, sheet_name="User", engine="openpyxl")
    except Exception:
        users_raw = pd.DataFrame()

    return _normalize_supplier_columns(suppliers_raw), _normalize_user_columns(users_raw)


# ---------------------------------------------------------------------
# Profit–Risk game configs
# ---------------------------------------------------------------------

@dataclass
class ProfitRiskConfig:
    served_users: int = 10
    price_per_user: float = 100.0

    env_cap: float = 2.75
    social_cap: float = 3.0
    cost_scale: float = 10.0

    # optional bans (kept for future scenarios)
    ban_child_labor: bool = False
    ban_banned_chem: bool = False

    # subset-size search range ("no K" means we search over k)
    min_k: int = 1
    max_k: Optional[int] = None

    output_flag: int = 0


@dataclass
class ProfitRiskMinRiskConfig(ProfitRiskConfig):
    profit_floor_per_user: float = 0.0


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def _compute_metrics_from_picks(suppliers_df: pd.DataFrame, picks: List[str], cfg: ProfitRiskConfig) -> Dict[str, Any]:
    picks = [str(x) for x in (picks or [])]
    sub = suppliers_df[suppliers_df["supplier_id"].astype(str).isin(picks)].copy()

    k = int(len(sub))
    if k == 0:
        return {
            "chosen_suppliers": [],
            "k": 0,
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "profit_per_user": 0.0,
            "profit_total": 0.0,
            "risk_score": 0.0,
            "feasible": False,
        }

    avg_env = float(sub["env_risk"].mean())
    avg_social = float(sub["social_risk"].mean())
    avg_cost = float(sub["cost_score"].mean())

    profit_per_user = float(cfg.price_per_user) - float(cfg.cost_scale) * avg_cost
    profit_total = float(cfg.served_users) * profit_per_user

    risk_score = 0.5 * ((avg_env / float(cfg.env_cap)) + (avg_social / float(cfg.social_cap)))

    feasible = (avg_env <= float(cfg.env_cap) + 1e-9) and (avg_social <= float(cfg.social_cap) + 1e-9)

    if float(cfg.profit_floor_per_user) > 0.0 if isinstance(cfg, ProfitRiskMinRiskConfig) else False:
        feasible = feasible and (profit_per_user >= float(cfg.profit_floor_per_user) - 1e-9)

    return {
        "chosen_suppliers": sorted([str(x) for x in sub["supplier_id"].tolist()]),
        "k": k,
        "avg_env": avg_env,
        "avg_social": avg_social,
        "avg_cost": avg_cost,
        "profit_per_user": profit_per_user,
        "profit_total": profit_total,
        "risk_score": float(risk_score),
        "feasible": bool(feasible),
    }


def _apply_bans(m: "gp.Model", y, suppliers_df: pd.DataFrame, cfg: ProfitRiskConfig) -> None:
    if not (cfg.ban_child_labor or cfg.ban_banned_chem):
        return

    s_child = dict(zip(suppliers_df["supplier_id"].astype(str), suppliers_df["child_labor"].astype(float)))
    s_ban = dict(zip(suppliers_df["supplier_id"].astype(str), suppliers_df["banned_chem"].astype(float)))

    for sid in suppliers_df["supplier_id"].astype(str).tolist():
        if cfg.ban_child_labor and float(s_child.get(sid, 0.0)) >= 0.5:
            m.addConstr(y[sid] == 0, name=f"ban_child_labor[{sid}]")
        if cfg.ban_banned_chem and float(s_ban.get(sid, 0.0)) >= 0.5:
            m.addConstr(y[sid] == 0, name=f"ban_banned_chem[{sid}]")


# ---------------------------------------------------------------------
# Optimization core (exact, by enumerating k)
# ---------------------------------------------------------------------

def _solve_fixed_k(
    suppliers_df: pd.DataFrame,
    cfg: ProfitRiskConfig,
    k: int,
    *,
    objective: str,
    profit_floor_per_user: Optional[float] = None,
    risk_score_cap: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Solve one MILP for a fixed subset size k.

    objective:
      - "min_total_cost" for max-profit (since profit increases as avg cost decreases)
      - "min_norm_risk_sum" for min-risk

    Returns solution dict or None if infeasible.
    """

    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available.")

    Suppliers = suppliers_df["supplier_id"].astype(str).tolist()

    env = dict(zip(Suppliers, suppliers_df["env_risk"].astype(float)))
    soc = dict(zip(Suppliers, suppliers_df["social_risk"].astype(float)))
    cost = dict(zip(Suppliers, suppliers_df["cost_score"].astype(float)))

    m = gp.Model("ProfitRiskFixedK")
    m.Params.OutputFlag = int(cfg.output_flag)

    y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y")

    # fixed subset size
    m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(k), name="fixed_k")

    _apply_bans(m, y, suppliers_df, cfg)

    # risk caps (average <= cap  <=>  sum <= cap * k)
    m.addConstr(gp.quicksum(env[i] * y[i] for i in Suppliers) <= float(cfg.env_cap) * float(k), name="env_cap")
    m.addConstr(gp.quicksum(soc[i] * y[i] for i in Suppliers) <= float(cfg.social_cap) * float(k), name="social_cap")

    # optional combined risk-score cap:
    # risk_score = 0.5 * (avg_env/env_cap + avg_soc/social_cap)
    # => (sum(env)/env_cap + sum(soc)/social_cap) <= 2 * risk_score_cap * k
    if risk_score_cap is not None:
        lhs = gp.quicksum((env[i] / float(cfg.env_cap) + soc[i] / float(cfg.social_cap)) * y[i] for i in Suppliers)
        m.addConstr(lhs <= (2.0 * float(risk_score_cap) * float(k)), name="risk_score_cap")

    # optional profit floor
    if profit_floor_per_user is not None:
        # price - cost_scale * avg_cost >= floor
        # avg_cost <= (price - floor)/cost_scale
        avg_cost_cap = (float(cfg.price_per_user) - float(profit_floor_per_user)) / max(1e-9, float(cfg.cost_scale))
        m.addConstr(gp.quicksum(cost[i] * y[i] for i in Suppliers) <= float(avg_cost_cap) * float(k), name="profit_floor")

    # objective
    if objective == "min_total_cost":
        m.setObjective(gp.quicksum(cost[i] * y[i] for i in Suppliers), GRB.MINIMIZE)
    elif objective == "min_norm_risk_sum":
        # minimize sum(env)/env_cap + sum(soc)/social_cap
        m.setObjective(
            gp.quicksum((env[i] / float(cfg.env_cap) + soc[i] / float(cfg.social_cap)) * y[i] for i in Suppliers),
            GRB.MINIMIZE,
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        return None

    picks = [i for i in Suppliers if y[i].X > 0.5]
    metrics = _compute_metrics_from_picks(suppliers_df, picks, cfg)
    metrics["_k_fixed"] = int(k)
    metrics["_solver_obj"] = float(m.ObjVal)
    return metrics


# ---------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------

class ProfitRiskMaxProfitAgent:
    """Maximize profit with no fixed K by searching over k."""

    def __init__(self, suppliers_df: pd.DataFrame, cfg: ProfitRiskConfig):
        self.suppliers = suppliers_df.copy()
        self.cfg = cfg

    def solve(self) -> Dict[str, Any]:
        cfg = self.cfg
        n = int(len(self.suppliers))
        min_k = max(1, int(cfg.min_k))
        max_k = int(cfg.max_k) if cfg.max_k is not None else n
        max_k = max(min_k, min(max_k, n))

        best: Optional[Dict[str, Any]] = None

        for k in range(min_k, max_k + 1):
            sol = _solve_fixed_k(self.suppliers, cfg, k, objective="min_total_cost")
            if sol is None or not sol.get("feasible", False):
                continue

            if best is None:
                best = sol
                continue

            # primary: profit_total (higher better)
            if float(sol["profit_total"]) > float(best["profit_total"]) + 1e-9:
                best = sol
                continue

            # tie-break: lower risk_score
            if abs(float(sol["profit_total"]) - float(best["profit_total"])) <= 1e-9:
                if float(sol["risk_score"]) < float(best["risk_score"]) - 1e-9:
                    best = sol
                    continue

                # tie-break: fewer suppliers
                if abs(float(sol["risk_score"]) - float(best["risk_score"])) <= 1e-9:
                    if int(sol["k"]) < int(best["k"]):
                        best = sol

        if best is None:
            raise RuntimeError("No feasible solution found under the risk caps.")

        return best


class ProfitRiskMinRiskAgent:
    """Minimize risk_score with no fixed K by searching over k (optionally with a profit floor)."""

    def __init__(self, suppliers_df: pd.DataFrame, cfg: ProfitRiskMinRiskConfig):
        self.suppliers = suppliers_df.copy()
        self.cfg = cfg

    def solve(self) -> Dict[str, Any]:
        cfg = self.cfg
        n = int(len(self.suppliers))
        min_k = max(1, int(cfg.min_k))
        max_k = int(cfg.max_k) if cfg.max_k is not None else n
        max_k = max(min_k, min(max_k, n))

        best: Optional[Dict[str, Any]] = None

        for k in range(min_k, max_k + 1):
            sol = _solve_fixed_k(
                self.suppliers,
                cfg,
                k,
                objective="min_norm_risk_sum",
                profit_floor_per_user=float(cfg.profit_floor_per_user),
            )
            if sol is None or not sol.get("feasible", False):
                continue

            if best is None:
                best = sol
                continue

            # primary: risk_score (lower better)
            if float(sol["risk_score"]) < float(best["risk_score"]) - 1e-9:
                best = sol
                continue

            # tie-break: higher profit_total
            if abs(float(sol["risk_score"]) - float(best["risk_score"])) <= 1e-9:
                if float(sol["profit_total"]) > float(best["profit_total"]) + 1e-9:
                    best = sol
                    continue

                # tie-break: fewer suppliers
                if abs(float(sol["profit_total"]) - float(best["profit_total"])) <= 1e-9:
                    if int(sol["k"]) < int(best["k"]):
                        best = sol

        if best is None:
            raise RuntimeError("No feasible solution found for min-risk under the current constraints.")

        return best


class ProfitRiskCurveAgent:
    """Generate a profit–risk curve by tightening a combined risk_score cap and maximizing profit."""

    def __init__(self, suppliers_df: pd.DataFrame, cfg: ProfitRiskConfig):
        self.suppliers = suppliers_df.copy()
        self.cfg = cfg

    def compute_curve(self, n_points: int = 9) -> List[Dict[str, Any]]:
        n_points = int(n_points)
        if n_points < 3:
            n_points = 3

        # Find a practical minimum risk_score (best possible under caps)
        min_risk_sol = ProfitRiskMinRiskAgent(self.suppliers, ProfitRiskMinRiskConfig(**self.cfg.__dict__, profit_floor_per_user=0.0)).solve()
        r_min = float(min_risk_sol["risk_score"])
        r_max = 1.0

        # spaced caps
        caps = [r_min + (r_max - r_min) * (i / (n_points - 1)) for i in range(n_points)]

        rows: List[Dict[str, Any]] = []

        n = int(len(self.suppliers))
        min_k = max(1, int(self.cfg.min_k))
        max_k = int(self.cfg.max_k) if self.cfg.max_k is not None else n
        max_k = max(min_k, min(max_k, n))

        for cap in caps:
            best: Optional[Dict[str, Any]] = None
            for k in range(min_k, max_k + 1):
                sol = _solve_fixed_k(
                    self.suppliers,
                    self.cfg,
                    k,
                    objective="min_total_cost",
                    risk_score_cap=float(cap),
                )
                if sol is None or not sol.get("feasible", False):
                    continue

                if best is None:
                    best = sol
                    continue

                if float(sol["profit_total"]) > float(best["profit_total"]) + 1e-9:
                    best = sol

            if best is None:
                rows.append(
                    {
                        "risk_cap": float(cap),
                        "risk_score": None,
                        "profit_total": None,
                        "profit_per_user": None,
                        "avg_env": None,
                        "avg_social": None,
                        "avg_cost": None,
                        "k": None,
                        "suppliers": None,
                    }
                )
            else:
                rows.append(
                    {
                        "risk_cap": float(cap),
                        "risk_score": float(best["risk_score"]),
                        "profit_total": float(best["profit_total"]),
                        "profit_per_user": float(best["profit_per_user"]),
                        "avg_env": float(best["avg_env"]),
                        "avg_social": float(best["avg_social"]),
                        "avg_cost": float(best["avg_cost"]),
                        "k": int(best["k"]),
                        "suppliers": ", ".join(best["chosen_suppliers"]),
                    }
                )

        return rows
