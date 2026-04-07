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


def _density_weighted_sums(
    users_df: pd.DataFrame,
    density_weights: Optional[Dict[str, float]],
) -> tuple:
    """
    Return density-weighted sums of user preference weights.

    density_weights: {user_id -> unnormalised weight}
                     If None, all users are weighted equally.
    Returns: (w_env, w_soc, w_str, w_imp, w_lq, w_cost, total_users)
    """
    df = users_df.copy()
    df["user_id"] = df["user_id"].astype(str)

    if not len(df):
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    if density_weights:
        total_w = sum(density_weights.values())
        if total_w <= 0.0:
            total_w = 1.0
        weights = {uid: w / total_w for uid, w in density_weights.items()}
        w_env = sum(float(row["w_env"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
        w_soc = sum(float(row["w_social"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
        w_str = sum(float(row["w_strategic"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
        w_imp = sum(float(row["w_improvement"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
        w_lq  = sum(float(row["w_low_quality"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
        w_cost = sum(float(row["w_cost"]) * weights.get(str(row["user_id"]), 0.0) for _, row in df.iterrows())
    else:
        n = float(len(df))
        w_env  = float(df["w_env"].sum()) / n
        w_soc  = float(df["w_social"].sum()) / n
        w_str  = float(df["w_strategic"].sum()) / n
        w_imp  = float(df["w_improvement"].sum()) / n
        w_lq   = float(df["w_low_quality"].sum()) / n
        w_cost = float(df["w_cost"].sum()) / n

    return w_env, w_soc, w_str, w_imp, w_lq, w_cost, len(df)


def solve_best_over_k(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    policy: Any,
    env_cap: float,
    social_cap: float,
    output_flag: int,
    objective_mode: str,
    density_weights: Optional[Dict[str, float]] = None,
    # kept for backward compatibility — ignored, use density_weights instead
    served_users: int = 0,
) -> Optional[Dict[str, Any]]:
    """Solve the supplier selection problem.

    If suppliers_df contains a 'category' column, enforces exactly one supplier
    per category (categorical mode).  Otherwise falls back to the original
    k-enumeration over all possible portfolio sizes.
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    pol = policy.clamp_nonnegative()
    df = suppliers_df.copy()
    df["supplier_id"] = df["supplier_id"].astype(str)

    supplier_ids = df["supplier_id"].tolist()
    supplier_count = len(supplier_ids)
    if supplier_count == 0:
        return None

    w_env, w_soc, w_str, w_imp, w_lq, w_cost, _ = _density_weighted_sums(users_df, density_weights)

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

    # ------------------------------------------------------------------
    # Categorical mode: exactly one supplier per category
    # ------------------------------------------------------------------
    has_categories = (
        "category" in df.columns
        and df["category"].notna().any()
    )

    if has_categories:
        category_groups: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            cat = str(row["category"]).strip()
            if cat and cat.lower() != "nan":
                category_groups.setdefault(cat, []).append(str(row["supplier_id"]))

        if not category_groups:
            has_categories = False

    if has_categories:
        k = len(category_groups)

        model = gp.Model(f"avg_game_{objective_mode}_categorical")
        model.Params.OutputFlag = int(output_flag)

        y = model.addVars(supplier_ids, vtype=GRB.BINARY, name="y")

        # Exactly one supplier from each category
        for cat, ids in category_groups.items():
            model.addConstr(
                gp.quicksum(y[sid] for sid in ids) == 1,
                name=f"cat_{cat}",
            )

        apply_policy_bans(model, y, df, pol)

        model.addConstr(
            gp.quicksum(env[sid] * y[sid] for sid in supplier_ids) <= float(env_cap) * k,
            name="env_cap",
        )
        model.addConstr(
            gp.quicksum(soc[sid] * y[sid] for sid in supplier_ids) <= float(social_cap) * k,
            name="soc_cap",
        )

        if objective_mode == "profit":
            model.setObjective(
                -gp.quicksum(cost[sid] * y[sid] for sid in supplier_ids),
                GRB.MAXIMIZE,
            )
        elif objective_mode == "utility":
            model.setObjective(
                gp.quicksum(util_num_coeff[sid] * y[sid] for sid in supplier_ids),
                GRB.MAXIMIZE,
            )
        else:
            raise ValueError("objective_mode must be 'profit' or 'utility'")

        model.optimize()

        if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            return None

        chosen = [sid for sid in supplier_ids if y[sid].X > 0.5]
        if not chosen:
            return None

        n = len(chosen)
        avg_env  = sum(env[sid]  for sid in chosen) / n
        avg_soc  = sum(soc[sid]  for sid in chosen) / n
        avg_cost = sum(cost[sid] for sid in chosen) / n
        avg_str  = sum(strat[sid] for sid in chosen) / n
        avg_imp  = sum(imp[sid]  for sid in chosen) / n
        avg_lq   = sum(lq[sid]   for sid in chosen) / n
        util_over_k = sum(util_num_coeff[sid] for sid in chosen) / n

        return {
            "k": float(n),
            "avg_env": float(avg_env),
            "avg_social": float(avg_soc),
            "avg_cost": float(avg_cost),
            "avg_strategic": float(avg_str),
            "avg_improvement": float(avg_imp),
            "avg_low_quality": float(avg_lq),
            "utility_num_over_k": float(util_over_k),
            "chosen": chosen,
        }

    # ------------------------------------------------------------------
    # Legacy mode: enumerate k = 1 … N
    # ------------------------------------------------------------------
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

        avg_env  = sum(env[sid]  for sid in chosen) / len(chosen)
        avg_soc  = sum(soc[sid]  for sid in chosen) / len(chosen)
        avg_cost = sum(cost[sid] for sid in chosen) / len(chosen)
        avg_str  = sum(strat[sid] for sid in chosen) / len(chosen)
        avg_imp  = sum(imp[sid]  for sid in chosen) / len(chosen)
        avg_lq   = sum(lq[sid]   for sid in chosen) / len(chosen)

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


def compute_utility_total(
    users_df: pd.DataFrame,
    policy: Any,
    best: Dict[str, Any],
    density_weights: Optional[Dict[str, float]] = None,
    # kept for backward compatibility — ignored
    served_users: int = 0,
    price_per_user: float = 0.0,
) -> float:
    """
    Density-weighted total utility across all users.

    Utility = quality dimensions only (env, social, strategic, improvement, low_quality).
    Price is excluded — profit is tracked separately to keep the benchmark bounded.

    Each user's contribution is weighted by their density weight (normalised to sum 1),
    then scaled by N so the result is a total rather than an average.
    """
    if not len(users_df):
        return 0.0

    pol = policy.clamp_nonnegative()
    ut_env         = 5.0 - float(best["avg_env"])
    ut_social      = 5.0 - float(best["avg_social"])
    ut_strategic   = float(best["avg_strategic"]) - 1.0
    ut_improvement = float(best["avg_improvement"]) - 1.0
    ut_lq          = 5.0 - float(best["avg_low_quality"])

    df = users_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    n = float(len(df))

    if density_weights:
        total_w = sum(density_weights.values())
        if total_w <= 0.0:
            total_w = 1.0
        utility_total = 0.0
        for _, row in df.iterrows():
            w = density_weights.get(str(row["user_id"]), 0.0) / total_w * n
            utility_total += w * (
                float(row["w_env"])           * (pol.env_mult         * ut_env)
                + float(row["w_social"])      * (pol.social_mult      * ut_social)
                + float(row["w_strategic"])   * (pol.strategic_mult   * ut_strategic)
                + float(row["w_improvement"]) * (pol.improvement_mult * ut_improvement)
                + float(row["w_low_quality"]) * (pol.low_quality_mult * ut_lq)
            )
        return utility_total
    else:
        utility_per_user = (
            df["w_env"]           * (pol.env_mult         * ut_env)
            + df["w_social"]      * (pol.social_mult      * ut_social)
            + df["w_strategic"]   * (pol.strategic_mult   * ut_strategic)
            + df["w_improvement"] * (pol.improvement_mult * ut_improvement)
            + df["w_low_quality"] * (pol.low_quality_mult * ut_lq)
        )
        return float(utility_per_user.sum())
