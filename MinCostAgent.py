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
        "user id": "user_id",
        "user_id": "user_id",
        "id": "user_id",
        "w_env": "w_env",
        "env weight": "w_env",
        "w_social": "w_social",
        "social weight": "w_social",
        "w_cost": "w_cost",
        "cost weight": "w_cost",
        "w_strategic": "w_strategic",
        "strategic weight": "w_strategic",
        "w_improvement": "w_improvement",
        "improvement weight": "w_improvement",
        "w_low_quality": "w_low_quality",
        "low quality weight": "w_low_quality",
    }

    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]

    rename: Dict[str, str] = {}
    for c in df2.columns:
        key = str(c).strip().lower()
        if key in col_map:
            rename[c] = col_map[key]
    df2 = df2.rename(columns=rename)

    if "user_id" not in df2.columns:
        raise ValueError("User sheet is missing 'user_id' column.")

    required = [
        "user_id",
        "w_env",
        "w_social",
        "w_cost",
        "w_strategic",
        "w_improvement",
        "w_low_quality",
    ]

    for c in required:
        if c not in df2.columns:
            df2[c] = 0.0

    df2 = df2[required].copy()
    df2["user_id"] = df2["user_id"].astype(str)

    for c in required:
        if c == "user_id":
            continue
        df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0.0).astype(float)

    return df2


def load_supplier_user_tables(xlsx_path: Optional[Union[str, Path]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    path = Path(xlsx_path) if xlsx_path is not None else DEFAULT_XLSX_PATH
    if not path.exists():
        raise FileNotFoundError(f"Excel file not found at: {path}")

    suppliers_raw = pd.read_excel(path, sheet_name="Supplier", engine="openpyxl")
    users_raw = pd.read_excel(path, sheet_name="User", engine="openpyxl")

    return _normalize_supplier_columns(suppliers_raw), _normalize_user_columns(users_raw)


@dataclass
class Policy:
    env_mult: float = 1.0
    social_mult: float = 1.0
    cost_mult: float = 1.0
    strategic_mult: float = 1.0
    improvement_mult: float = 1.0
    low_quality_mult: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "env_mult": float(self.env_mult),
            "social_mult": float(self.social_mult),
            "cost_mult": float(self.cost_mult),
            "strategic_mult": float(self.strategic_mult),
            "improvement_mult": float(self.improvement_mult),
            "low_quality_mult": float(self.low_quality_mult),
        }


def _select_last_n_users(users_df: pd.DataFrame, n: int) -> List[str]:
    u = users_df.copy()
    u["_row"] = range(len(u))
    u = u.sort_values("_row")
    return u["user_id"].tail(int(n)).astype(str).tolist()


def _auto_big_m(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pol: Policy, users: List[str]) -> float:
    s = suppliers_df
    u = users_df.set_index("user_id").loc[users]

    max_w = float(
        u[["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]].abs().max().max()
    )
    max_attr = float(s[["env_risk", "social_risk", "cost_score", "strategic", "improvement", "low_quality"]].abs().max().max())
    mult_max = max(
        abs(float(pol.env_mult)),
        abs(float(pol.social_mult)),
        abs(float(pol.cost_mult)),
        abs(float(pol.strategic_mult)),
        abs(float(pol.improvement_mult)),
        abs(float(pol.low_quality_mult)),
    )
    return max(1.0, 2.0 * max_w * max_attr * mult_max)


def _apply_fixed_suppliers(m: "gp.Model", y: Any, suppliers: List[str], fixed: Optional[List[str]], k: int) -> None:
    if fixed is None:
        return
    fixed_set = [str(x) for x in fixed]
    unknown = [x for x in fixed_set if x not in set(suppliers)]
    if unknown:
        raise ValueError(f"Unknown supplier_id in fixed_suppliers: {unknown}")
    if len(set(fixed_set)) != int(k):
        raise ValueError(f"fixed_suppliers must contain exactly K={int(k)} unique suppliers.")

    fixed_s = set(fixed_set)
    for sid in suppliers:
        m.addConstr(y[sid] == (1 if sid in fixed_s else 0), name=f"fix_y[{sid}]")


@dataclass
class MaxProfitConfig:
    served_users: int = 10
    suppliers_to_select: int = 3

    price_per_match: float = 100.0
    cost_scale: float = 10.0

    env_cap: float = 2.75
    social_cap: float = 3.0

    min_utility: float = 0.0

    output_flag: int = 0
    big_m: Optional[float] = None

    fixed_suppliers: Optional[List[str]] = None


@dataclass
class MaxUtilConfig(MaxProfitConfig):
    pass


class _BaseAgent:
    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxProfitConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available. Install & license Gurobi.")

        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy
        self.cfg = cfg

        self.suppliers["supplier_id"] = self.suppliers["supplier_id"].astype(str)
        self.users["user_id"] = self.users["user_id"].astype(str)

        self._users = _select_last_n_users(self.users, int(cfg.served_users))

        self.model: Optional["gp.Model"] = None
        self.y = None
        self.x = None

    def _build_common(self, name: str) -> Tuple["gp.Model", List[str], List[str], Dict[Tuple[str, str], float]]:
        cfg = self.cfg
        pol = self.policy

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self._users

        s = self.suppliers.set_index("supplier_id")
        u = self.users.set_index("user_id").loc[Users]

        env = s["env_risk"].to_dict()
        soc = s["social_risk"].to_dict()
        cost = s["cost_score"].to_dict()
        strat = s["strategic"].to_dict()
        improv = s["improvement"].to_dict()
        lowq = s["low_quality"].to_dict()

        M = float(cfg.big_m) if cfg.big_m is not None else _auto_big_m(self.suppliers, self.users, pol, Users)

        m = gp.Model(name)
        m.Params.OutputFlag = int(cfg.output_flag)

        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select")
        x = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="x_assign")

        _apply_fixed_suppliers(m, y, Suppliers, cfg.fixed_suppliers, int(cfg.suppliers_to_select))

        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k")

        for uid in Users:
            m.addConstr(gp.quicksum(x[i, uid] for i in Suppliers) == 1, name=f"serve[{uid}]")

        for i in Suppliers:
            for uid in Users:
                m.addConstr(x[i, uid] <= y[i], name=f"link[{i},{uid}]")

        m.addConstr(
            gp.quicksum(float(env[i]) * y[i] for i in Suppliers) <= float(cfg.env_cap) * int(cfg.suppliers_to_select),
            name="avg_env_cap",
        )
        m.addConstr(
            gp.quicksum(float(soc[i]) * y[i] for i in Suppliers) <= float(cfg.social_cap) * int(cfg.suppliers_to_select),
            name="avg_social_cap",
        )

        util: Dict[Tuple[str, str], float] = {}
        for i in Suppliers:
            for uid in Users:
                util_iu = (
                    float(u.loc[uid, "w_env"]) * (float(pol.env_mult) * float(env[i]))
                    + float(u.loc[uid, "w_social"]) * (float(pol.social_mult) * float(soc[i]))
                    + float(u.loc[uid, "w_cost"]) * (float(pol.cost_mult) * float(cost[i]))
                    + float(u.loc[uid, "w_strategic"]) * (float(pol.strategic_mult) * float(strat[i]))
                    + float(u.loc[uid, "w_improvement"]) * (float(pol.improvement_mult) * float(improv[i]))
                    + float(u.loc[uid, "w_low_quality"]) * (float(pol.low_quality_mult) * float(lowq[i]))
                )
                util[(i, uid)] = float(util_iu)
                if float(cfg.min_utility) != 0.0:
                    m.addConstr(util_iu >= float(cfg.min_utility) - M * (1 - x[i, uid]), name=f"minutil[{i},{uid}]")

        self.model, self.y, self.x = m, y, x
        return m, Suppliers, Users, util

    def _extract_solution(self, Suppliers: List[str], Users: List[str], util: Dict[Tuple[str, str], float]) -> Dict[str, Any]:
        assert self.model is not None

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs: List[Tuple[str, str]] = [(u, i) for i in Suppliers for u in Users if self.x[i, u].X > 0.5]
        df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])

        s = self.suppliers.set_index("supplier_id")

        if len(df):
            df["env_risk"] = df["supplier_id"].map(lambda sid: float(s.loc[sid, "env_risk"]))
            df["social_risk"] = df["supplier_id"].map(lambda sid: float(s.loc[sid, "social_risk"]))
            df["cost_score"] = df["supplier_id"].map(lambda sid: float(s.loc[sid, "cost_score"]))
            df["cost_prod"] = df["cost_score"].map(lambda c: float(self.cfg.cost_scale) * float(c))
            df["margin"] = float(self.cfg.price_per_match) - df["cost_prod"]
            df["utility"] = df.apply(lambda r: float(util[(str(r["supplier_id"]), str(r["user_id"]))]), axis=1)
            df = df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)

        if chosen:
            avg_env = float(self.suppliers[self.suppliers["supplier_id"].isin(chosen)]["env_risk"].mean())
            avg_soc = float(self.suppliers[self.suppliers["supplier_id"].isin(chosen)]["social_risk"].mean())
        else:
            avg_env, avg_soc = 0.0, 0.0

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "chosen_suppliers": chosen,
            "selected_users": Users,
            "num_matched": int(len(df)),
            "avg_env_selected": avg_env,
            "avg_social_selected": avg_soc,
            "matches": df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }


class MaxProfitAgent(_BaseAgent):
    def build(self) -> "gp.Model":
        m, Suppliers, Users, util = self._build_common("MaxProfit")

        s = self.suppliers.set_index("supplier_id")
        cost = s["cost_score"].to_dict()
        profit_coef = {i: float(self.cfg.price_per_match) - float(self.cfg.cost_scale) * float(cost[i]) for i in Suppliers}

        m.setObjective(gp.quicksum(profit_coef[i] * self.x[i, u] for i in Suppliers for u in Users), GRB.MAXIMIZE)
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        assert self.model is not None
        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self._users

        s = self.suppliers.set_index("supplier_id")
        u = self.users.set_index("user_id").loc[Users]

        env = s["env_risk"].to_dict()
        soc = s["social_risk"].to_dict()
        cost = s["cost_score"].to_dict()
        strat = s["strategic"].to_dict()
        improv = s["improvement"].to_dict()
        lowq = s["low_quality"].to_dict()

        pol = self.policy
        util: Dict[Tuple[str, str], float] = {}
        for i in Suppliers:
            for uid in Users:
                util[(i, uid)] = float(
                    float(u.loc[uid, "w_env"]) * (float(pol.env_mult) * float(env[i]))
                    + float(u.loc[uid, "w_social"]) * (float(pol.social_mult) * float(soc[i]))
                    + float(u.loc[uid, "w_cost"]) * (float(pol.cost_mult) * float(cost[i]))
                    + float(u.loc[uid, "w_strategic"]) * (float(pol.strategic_mult) * float(strat[i]))
                    + float(u.loc[uid, "w_improvement"]) * (float(pol.improvement_mult) * float(improv[i]))
                    + float(u.loc[uid, "w_low_quality"]) * (float(pol.low_quality_mult) * float(lowq[i]))
                )

        return self._extract_solution(Suppliers, Users, util)


class MaxUtilAgent(_BaseAgent):
    def build(self) -> "gp.Model":
        m, Suppliers, Users, util = self._build_common("MaxUtility")
        m.setObjective(gp.quicksum(float(util[(i, u)]) * self.x[i, u] for i in Suppliers for u in Users), GRB.MAXIMIZE)
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()
        assert self.model is not None
        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self._users

        s = self.suppliers.set_index("supplier_id")
        u = self.users.set_index("user_id").loc[Users]

        env = s["env_risk"].to_dict()
        soc = s["social_risk"].to_dict()
        cost = s["cost_score"].to_dict()
        strat = s["strategic"].to_dict()
        improv = s["improvement"].to_dict()
        lowq = s["low_quality"].to_dict()

        pol = self.policy
        util: Dict[Tuple[str, str], float] = {}
        for i in Suppliers:
            for uid in Users:
                util[(i, uid)] = float(
                    float(u.loc[uid, "w_env"]) * (float(pol.env_mult) * float(env[i]))
                    + float(u.loc[uid, "w_social"]) * (float(pol.social_mult) * float(soc[i]))
                    + float(u.loc[uid, "w_cost"]) * (float(pol.cost_mult) * float(cost[i]))
                    + float(u.loc[uid, "w_strategic"]) * (float(pol.strategic_mult) * float(strat[i]))
                    + float(u.loc[uid, "w_improvement"]) * (float(pol.improvement_mult) * float(improv[i]))
                    + float(u.loc[uid, "w_low_quality"]) * (float(pol.low_quality_mult) * float(lowq[i]))
                )

        return self._extract_solution(Suppliers, Users, util)
