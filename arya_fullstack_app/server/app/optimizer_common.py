from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover
    gp = None  # type: ignore
    GRB = None  # type: ignore
    GUROBI_AVAILABLE = False


def select_last_n_users(users_df: pd.DataFrame, n: int) -> pd.DataFrame:
    n = int(n)
    if n <= 0:
        return users_df.iloc[0:0].copy()
    if len(users_df) <= n:
        return users_df.copy()
    return users_df.iloc[-n:].copy()


def apply_policy_bans(model: Any, y, suppliers_df: pd.DataFrame, policy: Any) -> None:
    pol = policy.clamp_nonnegative()
    if float(pol.child_labor_penalty) >= 0.5 and "child_labor" in suppliers_df.columns:
        for _, row in suppliers_df.iterrows():
            if float(row.get("child_labor", 0.0)) >= 0.5:
                model.addConstr(y[str(row["supplier_id"])] == 0)

    if float(pol.banned_chem_penalty) >= 0.5 and "banned_chem" in suppliers_df.columns:
        for _, row in suppliers_df.iterrows():
            if float(row.get("banned_chem", 0.0)) >= 0.5:
                model.addConstr(y[str(row["supplier_id"])] == 0)


def solve_best_over_k(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    policy: Any,
    served_users: int,
    env_cap: float,
    social_cap: float,
    output_flag: int,
    objective_mode: str,
) -> Optional[Dict[str, Any]]:
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    pol = policy.clamp_nonnegative()
    df = suppliers_df.copy()
    df["supplier_id"] = df["supplier_id"].astype(str)

    supplier_ids = df["supplier_id"].tolist()
    supplier_count = len(supplier_ids)
    if supplier_count == 0:
        return None

    selected_users = select_last_n_users(users_df, served_users)
    if len(selected_users):
        w_env = float(selected_users["w_env"].sum())
        w_soc = float(selected_users["w_social"].sum())
        w_str = float(selected_users["w_strategic"].sum())
        w_imp = float(selected_users["w_improvement"].sum())
        w_lq = float(selected_users["w_low_quality"].sum())
    else:
        w_env = w_soc = w_str = w_imp = w_lq = 0.0

    env = dict(zip(df["supplier_id"], df["env_risk"].astype(float)))
    soc = dict(zip(df["supplier_id"], df["social_risk"].astype(float)))
    cost = dict(zip(df["supplier_id"], df["cost_score"].astype(float)))
    strat = dict(zip(df["supplier_id"], df["strategic"].astype(float)))
    imp = dict(zip(df["supplier_id"], df["improvement"].astype(float)))
    lq = dict(zip(df["supplier_id"], df["low_quality"].astype(float)))

    env_ut = {sid: 5.0 - env[sid] for sid in supplier_ids}
    soc_ut = {sid: 5.0 - soc[sid] for sid in supplier_ids}
    strat_ut = {sid: strat[sid] - 1.0 for sid in supplier_ids}
    imp_ut = {sid: imp[sid] - 1.0 for sid in supplier_ids}
    lq_ut = {sid: 5.0 - lq[sid] for sid in supplier_ids}

    util_num_coeff = {
        sid: (
            w_env * (pol.env_mult * env_ut[sid])
            + w_soc * (pol.social_mult * soc_ut[sid])
            + w_str * (pol.strategic_mult * strat_ut[sid])
            + w_imp * (pol.improvement_mult * imp_ut[sid])
            + w_lq * (pol.low_quality_mult * lq_ut[sid])
        )
        for sid in supplier_ids
    }

    best: Optional[Dict[str, Any]] = None

    for k in range(1, supplier_count + 1):
        model = gp.Model(f"avg_game_{objective_mode}_k{k}")
        model.Params.OutputFlag = int(output_flag)

        y = model.addVars(supplier_ids, vtype=GRB.BINARY, name="y")

        model.addConstr(gp.quicksum(y[sid] for sid in supplier_ids) == k, name="choose_k")
        apply_policy_bans(model, y, df, pol)

        model.addConstr(gp.quicksum(env[sid] * y[sid] for sid in supplier_ids) <= float(env_cap) * k, name="env_cap")
        model.addConstr(gp.quicksum(soc[sid] * y[sid] for sid in supplier_ids) <= float(social_cap) * k, name="soc_cap")

        if objective_mode == "profit":
            model.setObjective(-3.0 * gp.quicksum(cost[sid] * y[sid] for sid in supplier_ids), GRB.MAXIMIZE)
        elif objective_mode == "utility":
            model.setObjective(gp.quicksum(util_num_coeff[sid] * y[sid] for sid in supplier_ids), GRB.MAXIMIZE)
        else:
            raise ValueError("objective_mode must be 'profit' or 'utility'")

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            continue

        chosen = [sid for sid in supplier_ids if y[sid].X > 0.5]
        if not chosen:
            continue

        avg_env = sum(env[sid] for sid in chosen) / len(chosen)
        avg_soc = sum(soc[sid] for sid in chosen) / len(chosen)
        avg_cost = sum(cost[sid] for sid in chosen) / len(chosen)
        avg_str = sum(strat[sid] for sid in chosen) / len(chosen)
        avg_imp = sum(imp[sid] for sid in chosen) / len(chosen)
        avg_lq = sum(lq[sid] for sid in chosen) / len(chosen)

        utility_total = (sum(util_num_coeff[sid] for sid in chosen) / len(chosen)) if len(chosen) else 0.0

        candidate = {
            "k": float(len(chosen)),
            "avg_env": float(avg_env),
            "avg_social": float(avg_soc),
            "avg_cost": float(avg_cost),
            "avg_strategic": float(avg_str),
            "avg_improvement": float(avg_imp),
            "avg_low_quality": float(avg_lq),
            "utility_num_over_k": float(utility_total),
            "chosen": chosen,
        }

        if best is None:
            best = candidate
        elif objective_mode == "profit":
            if candidate["avg_cost"] < best["avg_cost"] - 1e-12:
                best = candidate
        else:
            if candidate["utility_num_over_k"] > best["utility_num_over_k"] + 1e-12:
                best = candidate

    return best


def compute_utility_total(users_df: pd.DataFrame, served_users: int, policy: Any, best: Dict[str, Any]) -> float:
    selected_users = select_last_n_users(users_df, served_users)
    if not len(selected_users):
        return 0.0

    pol = policy.clamp_nonnegative()
    ut_env = 5.0 - float(best["avg_env"])
    ut_social = 5.0 - float(best["avg_social"])
    ut_strategic = float(best["avg_strategic"]) - 1.0
    ut_improvement = float(best["avg_improvement"]) - 1.0
    ut_lq = 5.0 - float(best["avg_low_quality"])

    utility_per_user = (
        selected_users["w_env"] * (pol.env_mult * ut_env)
        + selected_users["w_social"] * (pol.social_mult * ut_social)
        + selected_users["w_strategic"] * (pol.strategic_mult * ut_strategic)
        + selected_users["w_improvement"] * (pol.improvement_mult * ut_improvement)
        + selected_users["w_low_quality"] * (pol.low_quality_mult * ut_lq)
    )
    return float(utility_per_user.sum())
