# UI.py
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
    gp = None
    GRB = None
    GUROBI_AVAILABLE = False


# =========================
# Default data (EMBEDDED)
#   - extracted from Arya_Phones_Supplier_Selection.xlsx / sheet "Min Cost Agent"
#   - no Excel dependency at runtime
# =========================
DEFAULT_SUPPLIERS: List[Dict[str, Any]] = [
    {"supplier_id": "A", "env_risk": 2, "social_risk": 4, "cost_score": 1, "strategic": 3, "improvement": 4, "child_labor": 1, "banned_chem": 0, "low_quality": 0},
    {"supplier_id": "B", "env_risk": 3, "social_risk": 5, "cost_score": 2, "strategic": 3, "improvement": 3, "child_labor": 0, "banned_chem": 0, "low_quality": 1},
    {"supplier_id": "C", "env_risk": 2, "social_risk": 1, "cost_score": 3, "strategic": 2, "improvement": 3, "child_labor": 0, "banned_chem": 1, "low_quality": 0},
    {"supplier_id": "D", "env_risk": 2, "social_risk": 2, "cost_score": 5, "strategic": 2, "improvement": 2, "child_labor": 0, "banned_chem": 0, "low_quality": 0},
    {"supplier_id": "E", "env_risk": 3, "social_risk": 4, "cost_score": 3, "strategic": 2, "improvement": 5, "child_labor": 0, "banned_chem": 0, "low_quality": 1},
    {"supplier_id": "F", "env_risk": 5, "social_risk": 4, "cost_score": 2, "strategic": 4, "improvement": 4, "child_labor": 1, "banned_chem": 1, "low_quality": 0},
    {"supplier_id": "G", "env_risk": 2, "social_risk": 1, "cost_score": 3, "strategic": 3, "improvement": 3, "child_labor": 0, "banned_chem": 0, "low_quality": 0},
]

DEFAULT_USERS: List[Dict[str, Any]] = [
    {"user_id": "1", "w_env": 0.30, "w_social": 0.20, "w_cost": 0.40, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.20},
    {"user_id": "2", "w_env": 0.25, "w_social": 0.15, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.30},
    {"user_id": "3", "w_env": 0.15, "w_social": 0.30, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.15, "w_low_quality": 0.20},
    {"user_id": "4", "w_env": 0.35, "w_social": 0.10, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.10},
    {"user_id": "5", "w_env": 0.10, "w_social": 0.35, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.10},
    {"user_id": "6", "w_env": 0.10, "w_social": 0.10, "w_cost": 0.35, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.10},
    {"user_id": "7", "w_env": 0.20, "w_social": 0.10, "w_cost": 0.20, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.30},
    {"user_id": "8", "w_env": 0.10, "w_social": 0.20, "w_cost": 0.10, "w_strategic": 0.30, "w_improvement": 0.10, "w_low_quality": 0.20},
    {"user_id": "9", "w_env": 0.10, "w_social": 0.10, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.40, "w_low_quality": 0.20},
    {"user_id": "10", "w_env": 0.20, "w_social": 0.20, "w_cost": 0.10, "w_strategic": 0.10, "w_improvement": 0.10, "w_low_quality": 0.20},
]


# =========================
# Policy (Government-set)
# =========================
@dataclass
class Policy:
    # Multipliers
    env_mult: float = 1.0
    social_mult: float = 1.0
    cost_mult: float = 1.0
    strategic_mult: float = 1.0
    improvement_mult: float = 1.0
    low_quality_mult: float = 1.0

    # Penalties (policy / regulation)
    child_labor_penalty: float = 0.0
    banned_chem_penalty: float = 0.0

    def clamp_nonnegative(self) -> "Policy":
        for k, v in self.__dict__.items():
            if v is None:
                setattr(self, k, 0.0)
            else:
                setattr(self, k, float(max(0.0, v)))
        return self

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items()}


# =========================
# Min-Cost Config
# =========================
@dataclass
class MinCostConfig:
    capacity: int = 6
    suppliers_to_select: int = 1
    target_matches: int = 6
    min_utility: float = 0.0
    big_m: Optional[float] = None
    output_flag: int = 0
    lexicographic_tiebreak: bool = True


# =========================
# MinCostAgent (MILP)
# =========================
class MinCostAgent:
    """
    Minimum-Cost Matching Agent (MILP, Gurobi)

    This file intentionally does NOT import Excel readers (openpyxl).
    Data is provided as pandas DataFrames (embedded defaults or any DF you build).
    """

    def __init__(
        self,
        suppliers_df: pd.DataFrame,
        users_df: pd.DataFrame,
        policy: Optional[Policy] = None,
        cfg: Optional[MinCostConfig] = None,
    ):
        if not GUROBI_AVAILABLE:
            raise RuntimeError("gurobipy mevcut değil. Streamlit UI çalışır ama solver çalışmaz.")

        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = (policy or Policy()).clamp_nonnegative()
        self.cfg = cfg or MinCostConfig()

        self.model: Optional[gp.Model] = None
        self.y = None
        self.z = None

        self._prep()

    def _prep(self) -> None:
        s_req = {"supplier_id", "env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"}
        u_req = {"user_id", "w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"}

        if not s_req.issubset(self.suppliers.columns):
            raise ValueError(f"suppliers_df eksik kolon: {sorted(s_req - set(self.suppliers.columns))}")
        if not u_req.issubset(self.users.columns):
            raise ValueError(f"users_df eksik kolon: {sorted(u_req - set(self.users.columns))}")

        self.suppliers["supplier_id"] = self.suppliers["supplier_id"].astype(str)
        self.users["user_id"] = self.users["user_id"].astype(str)

        # Low quality is a "bad" attribute. In our utility definition it should reduce utility.
        # Excel user weights are positive; we flip sign here (same convention as earlier code).
        self.users["w_low_quality"] = -self.users["w_low_quality"].astype(float)

        num_s = ["env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"]
        for c in num_s:
            self.suppliers[c] = self.suppliers[c].astype(float)

        num_u = ["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]
        for c in num_u:
            self.users[c] = self.users[c].astype(float)

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

    def _utility(self, supplier_id: str, user_id: str, s: Dict[str, Dict[str, float]], w: Dict[str, Dict[str, float]]) -> float:
        p = self.policy
        user_pref = (
            w["w_env"][user_id] * (p.env_mult * s["env_risk"][supplier_id])
            + w["w_social"][user_id] * (p.social_mult * s["social_risk"][supplier_id])
            + w["w_cost"][user_id] * (p.cost_mult * s["cost_score"][supplier_id])
            + w["w_strategic"][user_id] * (p.strategic_mult * s["strategic"][supplier_id])
            + w["w_improvement"][user_id] * (p.improvement_mult * s["improvement"][supplier_id])
            + w["w_low_quality"][user_id] * (p.low_quality_mult * s["low_quality"][supplier_id])
        )
        policy_pen = p.child_labor_penalty * s["child_labor"][supplier_id] + p.banned_chem_penalty * s["banned_chem"][supplier_id]
        return float(user_pref - policy_pen)

    def _cost(self, supplier_id: str, s: Dict[str, Dict[str, float]]) -> float:
        p = self.policy
        base = (
            p.env_mult * s["env_risk"][supplier_id]
            + p.social_mult * s["social_risk"][supplier_id]
            + p.cost_mult * s["cost_score"][supplier_id]
            + p.strategic_mult * s["strategic"][supplier_id]
            + p.improvement_mult * s["improvement"][supplier_id]
            + p.low_quality_mult * s["low_quality"][supplier_id]
        )
        penalties = p.child_labor_penalty * s["child_labor"][supplier_id] + p.banned_chem_penalty * s["banned_chem"][supplier_id]
        return float(base + penalties)

    def build(self, name: str = "MinCostAgent") -> "gp.Model":
        cfg = self.cfg
        p = self.policy

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        # dicts
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

        # Vars
        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select_supplier")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

        # Constraints
        m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(cfg.suppliers_to_select), name="select_k_suppliers")

        for u_id in Users:
            m.addConstr(gp.quicksum(z[i, u_id] for i in Suppliers) <= 1, name=f"user_once[{u_id}]")

        for i in Suppliers:
            for u_id in Users:
                m.addConstr(z[i, u_id] <= y[i], name=f"link[{i},{u_id}]")

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
                pen = (p.child_labor_penalty * s["child_labor"][i]) + (p.banned_chem_penalty * s["banned_chem"][i])
                util = user_pref - pen

                m.addConstr(util >= float(cfg.min_utility) - M * (1 - z[i, u_id]), name=f"utility[{i},{u_id}]")

        # Objective
        cost_i = {i: self._cost(i, s) for i in Suppliers}
        total_cost = gp.quicksum(cost_i[i] * z[i, u_id] for i in Suppliers for u_id in Users)

        if not cfg.lexicographic_tiebreak:
            m.setObjective(total_cost, GRB.MINIMIZE)
        else:
            total_utility = gp.quicksum(
                (
                    (
                        (w["w_env"][u_id] * (p.env_mult * s["env_risk"][i]))
                        + (w["w_social"][u_id] * (p.social_mult * s["social_risk"][i]))
                        + (w["w_cost"][u_id] * (p.cost_mult * s["cost_score"][i]))
                        + (w["w_strategic"][u_id] * (p.strategic_mult * s["strategic"][i]))
                        + (w["w_improvement"][u_id] * (p.improvement_mult * s["improvement"][i]))
                        + (w["w_low_quality"][u_id] * (p.low_quality_mult * s["low_quality"][i]))
                    )
                    - (
                        (p.child_labor_penalty * s["child_labor"][i]) + (p.banned_chem_penalty * s["banned_chem"][i])
                    )
                )
                * z[i, u_id]
                for i in Suppliers
                for u_id in Users
            )
            m.setObjectiveN(total_cost, index=0, priority=2, name="min_total_cost")
            m.setObjectiveN(total_utility, index=1, priority=1, name="max_total_utility")

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"No solution. Status={self.model.Status}")

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        # dicts for reporting
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

        match_df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])
        if len(match_df) > 0:
            match_df["utility"] = match_df.apply(lambda r: self._utility(r["supplier_id"], r["user_id"], s, w), axis=1)
            match_df["unit_cost"] = match_df["supplier_id"].map(lambda sid: self._cost(sid, s))
            match_df = match_df.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)
        else:
            match_df["utility"] = []
            match_df["unit_cost"] = []

        total_cost = float(match_df["unit_cost"].sum()) if len(match_df) else 0.0
        avg_utility = float(match_df["utility"].mean()) if len(match_df) else 0.0

        return {
            "status": int(self.model.Status),
            "objective_value": float(self.model.ObjVal),
            "total_cost": total_cost,
            "avg_utility": avg_utility,
            "chosen_suppliers": chosen,
            "num_matched": int(len(pairs)),
            "matches": match_df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Min-Cost Policy Matching", layout="wide")

st.title("Minimum Cost Matching • Government Policy Selector")

with st.expander("Ne yapıyor?", expanded=False):
    st.markdown(
        """
- Excel import **YOK**: data bu dosyanın içine gömülü (default).
- Hükümet politikası parametrelerini dropdown ile seçtirir.
- Gurobi varsa **minimum-cost** matching çözer.
        """
    )

if not GUROBI_AVAILABLE:
    st.warning(
        "Bu ortamda **gurobipy** bulunamadı. UI açılır ama 'Solve' çalışmaz. "
        "Streamlit Cloud kullanıyorsan, Gurobi'yi/ lisansı environment'a eklemen gerekir."
    )

# ---------- helpers ----------
def _parse_candidates(text: str) -> List[float]:
    vals: List[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return sorted(set(vals))


def _default_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    suppliers_df = pd.DataFrame(DEFAULT_SUPPLIERS)
    users_df = pd.DataFrame(DEFAULT_USERS)
    suppliers_df["supplier_id"] = suppliers_df["supplier_id"].astype(str)
    users_df["user_id"] = users_df["user_id"].astype(str)
    return suppliers_df, users_df


def policy_selector(prefix: str = "pol") -> Policy:
    st.subheader("Government Policy (dropdown ile seç)")

    default_candidates = "0, 0.25, 0.5, 1, 2, 5, 10"
    cand_text = st.text_input(
        "Dropdown seçenekleri (virgülle ayır)",
        value=default_candidates,
        help="Tüm policy parametre dropdown’ları bu listeden değer seçer.",
    )
    candidates = _parse_candidates(cand_text) or [0.0, 1.0]

    # index for 1.0 if exists, else middle
    def _idx(val: float) -> int:
        if val in candidates:
            return candidates.index(val)
        return min(len(candidates) // 2, len(candidates) - 1)

    cols = st.columns(3)
    with cols[0]:
        env = st.selectbox("env_mult", candidates, index=_idx(1.0), key=f"{prefix}_env")
        social = st.selectbox("social_mult", candidates, index=_idx(1.0), key=f"{prefix}_social")
        cost = st.selectbox("cost_mult", candidates, index=_idx(1.0), key=f"{prefix}_cost")
    with cols[1]:
        strategic = st.selectbox("strategic_mult", candidates, index=_idx(1.0), key=f"{prefix}_strategic")
        improvement = st.selectbox("improvement_mult", candidates, index=_idx(1.0), key=f"{prefix}_improvement")
        lq = st.selectbox("low_quality_mult", candidates, index=_idx(1.0), key=f"{prefix}_lq")
    with cols[2]:
        child = st.selectbox("child_labor_penalty", candidates, index=_idx(0.0), key=f"{prefix}_child")
        banned = st.selectbox("banned_chem_penalty", candidates, index=_idx(0.0), key=f"{prefix}_banned")

    pol = Policy(
        env_mult=env,
        social_mult=social,
        cost_mult=cost,
        strategic_mult=strategic,
        improvement_mult=improvement,
        low_quality_mult=lq,
        child_labor_penalty=child,
        banned_chem_penalty=banned,
    ).clamp_nonnegative()

    st.caption("Seçili policy (JSON):")
    st.json(pol.to_dict())
    return pol


# ---------- sidebar controls ----------
st.sidebar.header("Model Ayarları (Excel yok)")

st.sidebar.subheader("Optimization config")
capacity = st.sidebar.number_input("capacity (max matches)", min_value=0, value=6, step=1)
suppliers_to_select = st.sidebar.number_input("K (suppliers_to_select)", min_value=1, value=1, step=1)
target_matches = st.sidebar.number_input("target_matches (en az)", min_value=0, value=6, step=1)
min_utility = st.sidebar.number_input("min_utility (match olunca)", value=0.0, step=0.5)
lexi = st.sidebar.checkbox("Lexicographic tie-break (min cost → max utility)", value=True)
output_flag = st.sidebar.checkbox("Gurobi log göster (OutputFlag=1)", value=False)

st.sidebar.divider()
st.sidebar.caption("Not: Excel okuma (openpyxl) bu UI'dan tamamen çıkarıldı. Data gömülü.")


# ---------- main ----------
colA, colB = st.columns([1.15, 0.85], gap="large")

with colB:
    policy = policy_selector()

with colA:
    st.subheader("Veri & Çözüm")

    suppliers_df, users_df = _default_dfs()

    with st.expander("Suppliers (gömülü tablo)", expanded=False):
        st.dataframe(suppliers_df, use_container_width=True)

    with st.expander("Users (gömülü tablo)", expanded=False):
        st.dataframe(users_df, use_container_width=True)

    cfg = MinCostConfig(
        capacity=int(capacity),
        suppliers_to_select=int(suppliers_to_select),
        target_matches=int(target_matches),
        min_utility=float(min_utility),
        output_flag=1 if output_flag else 0,
        lexicographic_tiebreak=bool(lexi),
    )

    solve_btn = st.button("Solve (Gurobi)", type="primary", disabled=not GUROBI_AVAILABLE)

    if solve_btn:
        try:
            agent = MinCostAgent(suppliers_df, users_df, policy=policy, cfg=cfg)
            agent.build()
            sol = agent.solve()
        except Exception as e:
            st.error("Model solve hatası.")
            st.exception(e)
            st.stop()

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Matched", sol["num_matched"])
        kpi2.metric("Total cost", f"{sol['total_cost']:.3f}")
        kpi3.metric("Avg utility", f"{sol['avg_utility']:.3f}")
        kpi4.metric("Obj (Gurobi)", f"{sol['objective_value']:.3f}")

        st.markdown("### Seçilen supplier(lar)")
        st.write(sol["chosen_suppliers"])

        st.markdown("### Match tablosu (utility & unit_cost)")
        st.dataframe(sol["matches"], use_container_width=True)

        st.markdown("### Hızlı özet")
        if sol["num_matched"] > 0:
            by_supplier = (
                sol["matches"]
                .groupby("supplier_id")
                .agg(
                    matched=("user_id", "count"),
                    avg_utility=("utility", "mean"),
                    avg_unit_cost=("unit_cost", "mean"),
                    total_cost=("unit_cost", "sum"),
                )
                .reset_index()
            )
            st.dataframe(by_supplier, use_container_width=True)
        else:
            st.warning("Hiç match çıkmadı. target_matches / min_utility / capacity / policy değerlerini kontrol et.")
