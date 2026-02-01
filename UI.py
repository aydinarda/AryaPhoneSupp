from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# -------------------------
# Optional Gurobi import
# -------------------------
try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False

# -------------------------
# Data loading (Excel)
# -------------------------
DEFAULT_XLSX_PATH = "Arya_Phones_Supplier_Selection.xlsx"
DEFAULT_SHEET = "Max Match Agent"


@st.cache_data(show_spinner=False)
def load_data_from_excel(xlsx_path: str = DEFAULT_XLSX_PATH, sheet_name: str = DEFAULT_SHEET) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads the supplier table and user preference weights from the Excel workbook.

    Expected layout (sheet: "Max Match Agent"):
      - Suppliers: headers at B3:J3, rows 4..10
      - Users: headers at C24:H24, rows 25..34 with user_id in column B
    """
    from openpyxl import load_workbook

    wb = load_workbook(xlsx_path, data_only=True)

    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Sheet '{sheet_name}' not found. Available: {wb.sheetnames}")

    ws = wb[sheet_name]

    # Suppliers (B3:J10)
    supplier_cols = {
        "Supplier": "supplier_id",
        "Environmental Risk": "env_risk",
        "Social Risk": "social_risk",
        "Cost Score": "cost_score",
        "Strategic Importance": "strategic",
        "Improvement Potential": "improvement",
        "Child Labor": "child_labor",
        "Banned Chemicals": "banned_chem",
        "Low Product Quality": "low_quality",
    }

    headers = [ws.cell(3, c).value for c in range(2, 11)]  # B..J
    rows = []
    for r in range(4, 11):  # 4..10
        vals = [ws.cell(r, c).value for c in range(2, 11)]
        if all(v is None for v in vals):
            continue
        row = dict(zip(headers, vals))
        rows.append(row)

    suppliers_df = pd.DataFrame(rows).rename(columns=supplier_cols)
    suppliers_df = suppliers_df[list(supplier_cols.values())].copy()

    # Users (B24:H34, with headers at C24:H24 and user_id at B25..B34)
    user_cols = {
        "Environmental Risk": "w_env",
        "Social Risk": "w_social",
        "Cost Score": "w_cost",
        "Strategic Importance": "w_strategic",
        "Improvement Potential": "w_improvement",
        "Low Product Quality": "w_low_quality",
    }

    u_headers = [ws.cell(24, c).value for c in range(3, 9)]  # C..H
    u_rows = []
    for r in range(25, 35):  # 25..34
        user_id = ws.cell(r, 2).value  # B
        vals = [ws.cell(r, c).value for c in range(3, 9)]  # C..H
        if user_id is None and all(v is None for v in vals):
            continue
        row = {"user_id": str(user_id)}
        row.update(dict(zip(u_headers, vals)))
        u_rows.append(row)

    users_df = pd.DataFrame(u_rows).rename(columns=user_cols)
    users_df = users_df[["user_id"] + list(user_cols.values())].copy()

    # Basic type normalization
    suppliers_df["supplier_id"] = suppliers_df["supplier_id"].astype(str)
    users_df["user_id"] = users_df["user_id"].astype(str)

    for c in ["env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"]:
        suppliers_df[c] = suppliers_df[c].astype(float)

    for c in ["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]:
        users_df[c] = users_df[c].astype(float)

    return suppliers_df, users_df


# -------------------------
# Policy
# -------------------------
@dataclass
class Policy:
    # Multipliers
    env_mult: float = 1.0
    social_mult: float = 1.0
    cost_mult: float = 1.0
    strategic_mult: float = 1.0
    improvement_mult: float = 1.0
    low_quality_mult: float = 1.0

    # Penalties / tariffs
    child_labor_penalty: float = 0.0
    banned_chem_penalty: float = 0.0

    def clamp_nonnegative(self) -> "Policy":
        for k, v in self.__dict__.items():
            setattr(self, k, float(max(0.0, 0.0 if v is None else v)))
        return self


@dataclass
class MaxMarginConfig:
    # Matching structure
    suppliers_to_select: int = 1
    capacity: int = 6
    target_matches: int = 6

    # Economics
    price_per_match: float = 100.0  # P in the objective

    # Feasibility guardrail
    min_utility: float = 0.0

    # Solver
    output_flag: int = 0
    big_m: Optional[float] = None


class MaxMarginAgent:
    """
    Max-Margin Matching Agent (MILP, Gurobi)

    Objective (Z_margin):
        max  Σ_{i,u} ( P - CostProd(i) - T(i) ) * z_{i,u}

    where:
      - P is a fixed selling price per matched unit (price_per_match)
      - CostProd(i) is production cost proxy derived from supplier cost_score
      - T(i) is policy tariff/penalty derived from compliance flags
    """

    def __init__(self, suppliers_df: pd.DataFrame, users_df: pd.DataFrame, policy: Policy, cfg: MaxMarginConfig):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy is not available in this environment.")

        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = policy.clamp_nonnegative()
        self.cfg = cfg

        self.model: Optional["gp.Model"] = None
        self.y = None
        self.z = None

        self._prep()

    def _prep(self) -> None:
        s_req = {"supplier_id", "env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"}
        u_req = {"user_id", "w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"}

        if not s_req.issubset(self.suppliers.columns):
            raise ValueError(f"suppliers_df missing columns: {sorted(s_req - set(self.suppliers.columns))}")
        if not u_req.issubset(self.users.columns):
            raise ValueError(f"users_df missing columns: {sorted(u_req - set(self.users.columns))}")

        self.suppliers["supplier_id"] = self.suppliers["supplier_id"].astype(str)
        self.users["user_id"] = self.users["user_id"].astype(str)

        # Low quality is a "bad" attribute: positive user weight should reduce utility.
        self.users["w_low_quality"] = -self.users["w_low_quality"].astype(float)

    def _auto_big_m(self) -> float:
        p = self.policy
        s = self.suppliers
        u = self.users

        max_s = {
            "env_risk": float((p.env_mult * s["env_risk"]).abs().max()),
            "social_risk": float((p.social_mult * s["social_risk"]).abs().max()),
            "cost_score": float((p.cost_mult * s["cost_score"]).abs().max()),
            "strategic": float((p.strategic_mult * s["strategic"]).abs().max()),
            "improvement": float((p.improvement_mult * s["improvement"]).abs().max()),
            "low_quality": float((p.low_quality_mult * s["low_quality"]).abs().max()),
        }
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
        pen_max = float((p.child_labor_penalty * s["child_labor"] + p.banned_chem_penalty * s["banned_chem"]).max())
        return float(pref_max + pen_max + 10.0)

    def build(self, name: str = "MaxMarginAgent") -> "gp.Model":
        cfg = self.cfg
        p = self.policy

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        # Parameters
        s = {
            "env_risk": dict(zip(self.suppliers["supplier_id"], self.suppliers["env_risk"])),
            "social_risk": dict(zip(self.suppliers["supplier_id"], self.suppliers["social_risk"])),
            "cost_score": dict(zip(self.suppliers["supplier_id"], self.suppliers["cost_score"])),
            "strategic": dict(zip(self.suppliers["supplier_id"], self.suppliers["strategic"])),
            "improvement": dict(zip(self.suppliers["supplier_id"], self.suppliers["improvement"])),
            "low_quality": dict(zip(self.suppliers["supplier_id"], self.suppliers["low_quality"])),
            "child_labor": dict(zip(self.suppliers["supplier_id"], self.suppliers["child_labor"])),
            "banned_chem": dict(zip(self.suppliers["supplier_id"], self.suppliers["banned_chem"])),
        }
        w = {
            "w_env": dict(zip(self.users["user_id"], self.users["w_env"])),
            "w_social": dict(zip(self.users["user_id"], self.users["w_social"])),
            "w_cost": dict(zip(self.users["user_id"], self.users["w_cost"])),
            "w_strategic": dict(zip(self.users["user_id"], self.users["w_strategic"])),
            "w_improvement": dict(zip(self.users["user_id"], self.users["w_improvement"])),
            "w_low_quality": dict(zip(self.users["user_id"], self.users["w_low_quality"])),
        }

        M = cfg.big_m if cfg.big_m is not None else self._auto_big_m()

        m = gp.Model(name)
        m.Params.OutputFlag = int(cfg.output_flag)

        # Decision variables
        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select_supplier")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

        # Selection constraint
        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k_suppliers")

        # Each user matched at most once
        for u_id in Users:
            m.addConstr(gp.quicksum(z[i, u_id] for i in Suppliers) <= 1, name=f"user_once[{u_id}]")

        # Linking: can only match to selected suppliers
        for i in Suppliers:
            for u_id in Users:
                m.addConstr(z[i, u_id] <= y[i], name=f"link[{i},{u_id}]")

        # Capacity and minimum matched
        m.addConstr(gp.quicksum(z[i, u_id] for i in Suppliers for u_id in Users) <= int(cfg.capacity), name="capacity_total")
        m.addConstr(gp.quicksum(z[i, u_id] for i in Suppliers for u_id in Users) >= int(cfg.target_matches), name="target_matches")

        # Utility feasibility: if matched, utility >= min_utility
        for i in Suppliers:
            for u_id in Users:
                user_pref = (
                    (w["w_env"][u_id] * (p.env_mult * s["env_risk"][i]))
                    + (w["w_social"][u_id] * (p.social_mult * s["social_risk"][i]))
                    + (w["w_cost"][u_id] * (p.cost_mult * s["cost_score"][i]))
                    + (w["w_strategic"][u_id] * (p.strategic_mult * s["strategic"][i]))
                    + (w["w_improvement"][u_id] * (p.improvement_mult * s["improvement"][i]))
                    + (w["w_low_quality"][u_id] * (p.low_quality_mult * s["low_quality"][i]))
                )
                tariff = (p.child_labor_penalty * s["child_labor"][i]) + (p.banned_chem_penalty * s["banned_chem"][i])
                util = user_pref - tariff

                m.addConstr(util >= float(cfg.min_utility) - M * (1 - z[i, u_id]), name=f"utility[{i},{u_id}]")

        # Economics: Profit per match
        cost_prod = {i: float(p.cost_mult * s["cost_score"][i]) for i in Suppliers}
        tariff_i = {i: float(p.child_labor_penalty * s["child_labor"][i] + p.banned_chem_penalty * s["banned_chem"][i]) for i in Suppliers}
        margin_i = {i: float(cfg.price_per_match - cost_prod[i] - tariff_i[i]) for i in Suppliers}

        total_margin = gp.quicksum(margin_i[i] * z[i, u_id] for i in Suppliers for u_id in Users)
        m.setObjective(total_margin, GRB.MAXIMIZE)

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        assert self.model is not None
        self.model.optimize()

        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        p = self.policy
        s = {
            "env_risk": dict(zip(self.suppliers["supplier_id"], self.suppliers["env_risk"])),
            "social_risk": dict(zip(self.suppliers["supplier_id"], self.suppliers["social_risk"])),
            "cost_score": dict(zip(self.suppliers["supplier_id"], self.suppliers["cost_score"])),
            "strategic": dict(zip(self.suppliers["supplier_id"], self.suppliers["strategic"])),
            "improvement": dict(zip(self.suppliers["supplier_id"], self.suppliers["improvement"])),
            "low_quality": dict(zip(self.suppliers["supplier_id"], self.suppliers["low_quality"])),
            "child_labor": dict(zip(self.suppliers["supplier_id"], self.suppliers["child_labor"])),
            "banned_chem": dict(zip(self.suppliers["supplier_id"], self.suppliers["banned_chem"])),
        }
        w = {
            "w_env": dict(zip(self.users["user_id"], self.users["w_env"])),
            "w_social": dict(zip(self.users["user_id"], self.users["w_social"])),
            "w_cost": dict(zip(self.users["user_id"], self.users["w_cost"])),
            "w_strategic": dict(zip(self.users["user_id"], self.users["w_strategic"])),
            "w_improvement": dict(zip(self.users["user_id"], self.users["w_improvement"])),
            "w_low_quality": dict(zip(self.users["user_id"], self.users["w_low_quality"])),
        }

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs: List[Tuple[str, str]] = [(u_id, i) for i in Suppliers for u_id in Users if self.z[i, u_id].X > 0.5]

        df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])
        if len(df) > 0:
            tariff = df["supplier_id"].map(lambda sid: float(p.child_labor_penalty * s["child_labor"][sid] + p.banned_chem_penalty * s["banned_chem"][sid]))
            cost_prod = df["supplier_id"].map(lambda sid: float(p.cost_mult * s["cost_score"][sid]))
            df["tariff"] = tariff
            df["cost_prod"] = cost_prod
            df["margin"] = float(self.cfg.price_per_match) - df["cost_prod"] - df["tariff"]

            def _util(row: pd.Series) -> float:
                sid = row["supplier_id"]
                uid = row["user_id"]
                user_pref = (
                    w["w_env"][uid] * (p.env_mult * s["env_risk"][sid])
                    + w["w_social"][uid] * (p.social_mult * s["social_risk"][sid])
                    + w["w_cost"][uid] * (p.cost_mult * s["cost_score"][sid])
                    + w["w_strategic"][uid] * (p.strategic_mult * s["strategic"][sid])
                    + w["w_improvement"][uid] * (p.improvement_mult * s["improvement"][sid])
                    + w["w_low_quality"][uid] * (p.low_quality_mult * s["low_quality"][sid])
                )
                t = float(p.child_labor_penalty * s["child_labor"][sid] + p.banned_chem_penalty * s["banned_chem"][sid])
                return float(user_pref - t)

            df["utility"] = df.apply(_util, axis=1)
            df = df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)
        else:
            df["tariff"] = []
            df["cost_prod"] = []
            df["margin"] = []
            df["utility"] = []

        total_margin = float(df["margin"].sum()) if len(df) else 0.0
        avg_utility = float(df["utility"].mean()) if len(df) else 0.0

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "total_margin": total_margin,
            "avg_utility": avg_utility,
            "chosen_suppliers": chosen,
            "num_matched": int(len(pairs)),
            "matches": df,
        }


# -------------------------
# Streamlit UI (selection-only)
# -------------------------
st.set_page_config(page_title="Max Margin Policy Optimizer", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1200px; }
</style>
""",
    unsafe_allow_html=True,
)

# Load data (Excel in repo)
try:
    suppliers_df, users_df = load_data_from_excel()
    data_ok = True
except Exception:
    suppliers_df, users_df = pd.DataFrame(), pd.DataFrame()
    data_ok = False

# Selection widgets
policy_options = [1, 2, 3, 4, 5]

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("Government policy")

    env = st.selectbox("Environmental multiplier", policy_options, index=0)
    social = st.selectbox("Social multiplier", policy_options, index=0)
    cost = st.selectbox("Cost multiplier", policy_options, index=0)
    strategic = st.selectbox("Strategic multiplier", policy_options, index=0)
    improvement = st.selectbox("Improvement multiplier", policy_options, index=0)
    low_q = st.selectbox("Low-quality multiplier", policy_options, index=0)

    child = st.selectbox("Child labor penalty", policy_options, index=0)
    banned = st.selectbox("Banned chemicals penalty", policy_options, index=0)

    policy = Policy(
        env_mult=float(env),
        social_mult=float(social),
        cost_mult=float(cost),
        strategic_mult=float(strategic),
        improvement_mult=float(improvement),
        low_quality_mult=float(low_q),
        child_labor_penalty=float(child),
        banned_chem_penalty=float(banned),
    )

with col_right:
    st.subheader("Model settings")

    price = st.selectbox("Selling price (P)", [50.0, 75.0, 100.0, 125.0, 150.0], index=2)
    k = st.selectbox("Suppliers to select (K)", [1, 2, 3], index=0)
    capacity = st.selectbox("Total capacity (max matches)", [0, 2, 4, 6, 8, 10], index=3)
    target_matches = st.selectbox("Minimum matches", [0, 2, 4, 6, 8, 10], index=3)
    min_utility = st.selectbox("Minimum utility (if matched)", [0.0, 0.5, 1.0, 1.5, 2.0], index=0)

    show_log = st.selectbox("Solver log", ["Off", "On"], index=0)

    run = st.button("Optimize", type="primary", disabled=(not (GUROBI_AVAILABLE and data_ok)))

if not data_ok:
    st.error(f"Excel file '{DEFAULT_XLSX_PATH}' could not be loaded. Place it next to this app file in your Streamlit repo.")
elif not GUROBI_AVAILABLE:
    st.error("gurobipy is not available in this environment. Install Gurobi + gurobipy to enable optimization.")
elif run:
    cfg = MaxMarginConfig(
        suppliers_to_select=int(k),
        capacity=int(capacity),
        target_matches=int(target_matches),
        price_per_match=float(price),
        min_utility=float(min_utility),
        output_flag=1 if show_log == "On" else 0,
    )

    agent = MaxMarginAgent(suppliers_df, users_df, policy=policy, cfg=cfg)
    agent.build()
    sol = agent.solve()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Matched", sol["num_matched"])
    m2.metric("Total margin", f"{sol['total_margin']:.3f}")
    m3.metric("Avg utility", f"{sol['avg_utility']:.3f}")
    m4.metric("Objective", f"{sol['objective_value']:.3f}")

    st.markdown("**Selected suppliers**")
    st.write(sol["chosen_suppliers"])

    st.markdown("**Matches**")
    st.dataframe(sol["matches"], use_container_width=True)
