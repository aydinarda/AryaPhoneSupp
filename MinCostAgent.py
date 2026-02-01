# max_match_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import pandas as pd
import openpyxl

import gurobipy as gp
from gurobipy import GRB

from policy import Policy


@dataclass
class MaxMatchConfig:
    capacity: int = 6
    suppliers_to_select: int = 1
    min_utility: float = 0.0
    big_m: Optional[float] = None
    output_flag: int = 0
    sheet_name: str = "Min Cost Agent"

    # Optional: tie-breaker without "beta"
    # If True: First maximize matches, then maximize total utility (policy-driven)
    lexicographic_tiebreak: bool = False


class MaxMatchAgent:
    """
    Utility-based Max Matching Agent (MILP, Gurobi)
    Objective: MAX number of matches (no beta terms)
    Feasibility: match only if Utility(i,u) >= min_utility
    """

    # -------- Excel loaders --------
    @staticmethod
    def suppliers_from_excel(
        xlsx_path: str,
        sheet_name: str = "Min Cost Agent",
        header_row: int = 3,
        first_row: int = 4,
        last_row: int = 10,
        start_col: int = 2
    ) -> pd.DataFrame:
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb[sheet_name]

        headers = [ws.cell(header_row, c).value for c in range(start_col, start_col + 9)]
        col_map = {
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

        rows = []
        for r in range(first_row, last_row + 1):
            supplier_id = ws.cell(r, start_col).value
            if supplier_id is None:
                continue

            row = {}
            for i, h in enumerate(headers):
                raw = ws.cell(r, start_col + i).value
                row[col_map.get(h, h)] = raw
            rows.append(row)

        df = pd.DataFrame(rows)
        df["supplier_id"] = df["supplier_id"].astype(str)
        for c in ["env_risk","social_risk","cost_score","strategic","improvement","child_labor","banned_chem","low_quality"]:
            df[c] = df[c].astype(float)
        return df

    @staticmethod
    def users_from_excel(
        xlsx_path: str,
        sheet_name: str = "Min Cost Agent",
        header_row: int = 24,
        start_col: int = 2
    ) -> pd.DataFrame:
        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb[sheet_name]

        headers = [ws.cell(header_row, c).value for c in range(start_col, start_col + 7)]
        col_map = {
            "Users": "user_id",
            "Environmental Risk": "w_env",
            "Social Risk": "w_social",
            "Cost Score": "w_cost",
            "Strategic Importance": "w_strategic",
            "Improvement Potential": "w_improvement",
            "Low Product Quality": "w_low_quality",
        }

        rows = []
        r = header_row + 1
        while True:
            uid = ws.cell(r, start_col).value
            if uid is None:
                break
            row = {}
            for i, h in enumerate(headers):
                raw = ws.cell(r, start_col + i).value
                row[col_map.get(h, h)] = raw
            rows.append(row)
            r += 1

        df = pd.DataFrame(rows)
        df["user_id"] = df["user_id"].astype(str)
        for c in ["w_env","w_social","w_cost","w_strategic","w_improvement","w_low_quality"]:
            df[c] = df[c].astype(float)

        # Excel’de low_quality terimi negatifse burada sign çeviriyoruz:
        df["w_low_quality"] = -df["w_low_quality"]
        return df

    # -------- init --------
    def __init__(
        self,
        suppliers_df: pd.DataFrame,
        users_df: pd.DataFrame,
        policy: Optional[Policy] = None,
        cfg: Optional[MaxMatchConfig] = None,
    ):
        self.suppliers = suppliers_df.copy()
        self.users = users_df.copy()
        self.policy = (policy or Policy()).clamp_nonnegative()
        self.cfg = cfg or MaxMatchConfig()

        self.model: Optional[gp.Model] = None
        self.y = None
        self.z = None

        self._prep()

    def _prep(self) -> None:
        s_req = {"supplier_id","env_risk","social_risk","cost_score","strategic","improvement","child_labor","banned_chem","low_quality"}
        u_req = {"user_id","w_env","w_social","w_cost","w_strategic","w_improvement","w_low_quality"}
        if not s_req.issubset(self.suppliers.columns):
            raise ValueError(f"suppliers_df eksik kolon: {s_req - set(self.suppliers.columns)}")
        if not u_req.issubset(self.users.columns):
            raise ValueError(f"users_df eksik kolon: {u_req - set(self.users.columns)}")

        self.suppliers["supplier_id"] = self.suppliers["supplier_id"].astype(str)
        self.users["user_id"] = self.users["user_id"].astype(str)

    def _auto_big_m(self) -> float:
        # conservative: derive from observed magnitudes
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
            max_u["w_env"] * max_s["env_risk"] +
            max_u["w_social"] * max_s["social_risk"] +
            max_u["w_cost"] * max_s["cost_score"] +
            max_u["w_strategic"] * max_s["strategic"] +
            max_u["w_improvement"] * max_s["improvement"] +
            max_u["w_low_quality"] * max_s["low_quality"]
        )
        pen_max = float((p.child_labor_penalty * s["child_labor"] + p.banned_chem_penalty * s["banned_chem"]).max())

        return float(pref_max + pen_max + 10.0)

    # -------- build (screenshot-like style) --------
    def build(self, name: str = "MaxMatchAgent") -> gp.Model:
        cfg = self.cfg
        pol = self.policy

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        # supplier dicts
        s_env = dict(zip(self.suppliers["supplier_id"], self.suppliers["env_risk"]))
        s_social = dict(zip(self.suppliers["supplier_id"], self.suppliers["social_risk"]))
        s_cost = dict(zip(self.suppliers["supplier_id"], self.suppliers["cost_score"]))
        s_str = dict(zip(self.suppliers["supplier_id"], self.suppliers["strategic"]))
        s_imp = dict(zip(self.suppliers["supplier_id"], self.suppliers["improvement"]))
        s_lq = dict(zip(self.suppliers["supplier_id"], self.suppliers["low_quality"]))
        s_child = dict(zip(self.suppliers["supplier_id"], self.suppliers["child_labor"]))
        s_banned = dict(zip(self.suppliers["supplier_id"], self.suppliers["banned_chem"]))

        # user dicts
        u_env = dict(zip(self.users["user_id"], self.users["w_env"]))
        u_soc = dict(zip(self.users["user_id"], self.users["w_social"]))
        u_cost = dict(zip(self.users["user_id"], self.users["w_cost"]))
        u_str = dict(zip(self.users["user_id"], self.users["w_strategic"]))
        u_imp = dict(zip(self.users["user_id"], self.users["w_improvement"]))
        u_lq = dict(zip(self.users["user_id"], self.users["w_low_quality"]))  # already NEG

        M = cfg.big_m if cfg.big_m is not None else self._auto_big_m()

        m = gp.Model(name)
        m.Params.OutputFlag = cfg.output_flag

        # -------------------- Variables --------------------
        y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y")
        z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z")

        # -------------------- Constraints --------------------
        m.addConstr(
            gp.quicksum(y[i] for i in Suppliers) == cfg.suppliers_to_select,
            name="select_k_suppliers"
        )

        for u in Users:
            m.addConstr(
                gp.quicksum(z[i, u] for i in Suppliers) <= 1,
                name=f"user_once[{u}]"
            )

        for i in Suppliers:
            for u in Users:
                m.addConstr(
                    z[i, u] <= y[i],
                    name=f"link[{i},{u}]"
                )

        m.addConstr(
            gp.quicksum(z[i, u] for i in Suppliers for u in Users) <= cfg.capacity,
            name="capacity"
        )

        # -------------------- Utility feasibility --------------------
        for i in Suppliers:
            for u in Users:

                UserPref_i_u = (
                    (u_env[u]  * (pol.env_mult          * s_env[i])) +
                    (u_soc[u]  * (pol.social_mult       * s_social[i])) +
                    (u_cost[u] * (pol.cost_mult         * s_cost[i])) +
                    (u_str[u]  * (pol.strategic_mult    * s_str[i])) +
                    (u_imp[u]  * (pol.improvement_mult  * s_imp[i])) +
                    (u_lq[u]   * (pol.low_quality_mult  * s_lq[i]))
                )

                PolicyPenalty_i = (
                    (pol.child_labor_penalty * s_child[i]) +
                    (pol.banned_chem_penalty * s_banned[i])
                )

                Utility_i_u = UserPref_i_u - PolicyPenalty_i

                m.addConstr(
                    Utility_i_u >= cfg.min_utility - M * (1 - z[i, u]),
                    name=f"utility[{i},{u}]"
                )

        # -------------------- Objective --------------------
        Z_match = gp.quicksum(
            z[i, u]
            for i in Suppliers
            for u in Users
        )

        if not cfg.lexicographic_tiebreak:
            # Pure max-match
            m.setObjective(Z_match, GRB.MAXIMIZE)

        else:
            # Lexicographic (no beta): first maximize matches, then maximize total utility
            # (still fully policy-driven, no extra hyperparameter)
            TotalUtility = gp.quicksum(
                (
                    (
                        (u_env[u]  * (pol.env_mult         * s_env[i])) +
                        (u_soc[u]  * (pol.social_mult      * s_social[i])) +
                        (u_cost[u] * (pol.cost_mult        * s_cost[i])) +
                        (u_str[u]  * (pol.strategic_mult   * s_str[i])) +
                        (u_imp[u]  * (pol.improvement_mult * s_imp[i])) +
                        (u_lq[u]   * (pol.low_quality_mult * s_lq[i]))
                    )
                    - (
                        (pol.child_labor_penalty * s_child[i]) +
                        (pol.banned_chem_penalty * s_banned[i])
                    )
                ) * z[i, u]
                for i in Suppliers
                for u in Users
            )

            m.setObjectiveN(Z_match, index=0, priority=2, name="max_matches")
            m.setObjectiveN(TotalUtility, index=1, priority=1, name="max_total_utility")

        self.model, self.y, self.z = m, y, z
        return m

    def solve(self) -> Dict[str, Any]:
        if self.model is None:
            self.build()

        self.model.optimize()
        if self.model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            raise RuntimeError(f"Gurobi çözüm yok. Status={self.model.Status}")

        Suppliers = self.suppliers["supplier_id"].tolist()
        Users = self.users["user_id"].tolist()

        chosen = [i for i in Suppliers if self.y[i].X > 0.5]
        pairs = [(u, i) for i in Suppliers for u in Users if self.z[i, u].X > 0.5]
        match_df = pd.DataFrame(pairs, columns=["user_id", "supplier_id"]).sort_values(["supplier_id", "user_id"])

        return {
            "objective_value": float(self.model.ObjVal),
            "chosen_suppliers": chosen,
            "num_matched": int(len(pairs)),
            "matches": match_df,
            "policy": self.policy.to_dict(),
            "cfg": self.cfg,
        }


if __name__ == "__main__":
    xlsx = "/mnt/data/Arya_Phones_Supplier_Selection.xlsx"

    suppliers_df = MaxMatchAgent.suppliers_from_excel(xlsx, sheet_name="Min Cost Agent")
    users_df = MaxMatchAgent.users_from_excel(xlsx, sheet_name="Min Cost Agent")

    policy = Policy()  # default all ones
    cfg = MaxMatchConfig(capacity=6, suppliers_to_select=1, min_utility=0.0, output_flag=1)

    agent = MaxMatchAgent(suppliers_df, users_df, policy=policy, cfg=cfg)
    agent.build()
    sol = agent.solve()

    print("Chosen suppliers:", sol["chosen_suppliers"])
    print("Matched:", sol["num_matched"])
    print(sol["matches"])
