from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    MaxProfitConfig,  # kept for compatibility with other modules
    Policy,
    load_supplier_user_tables,
)

try:
    import gurobipy as gp
    from gurobipy import GRB

    _HAS_GUROBI = True
except Exception:
    gp = None  # type: ignore
    GRB = None  # type: ignore
    _HAS_GUROBI = False

# -----------------------------
# Fixed world-goodness weights (used for: goodness score + MILP tie-break)
# Higher "goodness" is better.
# -----------------------------
WORLD_GOODNESS_WEIGHTS: Dict[str, float] = {
    "w_env": 1.0,
    "w_soc": 1.0,
    "w_child": 10.0,
    "w_ban": 10.0,
    "w_lq": 1.0,
}

# -----------------------------
# Student-facing constants
# -----------------------------
TOTAL_USERS = 11

# Candidate generation ranges (hidden from students)
# Multipliers are discrete: low/mid/high -> 1/5/10
MULT_LEVELS = [1, 5, 10]
# Penalties are binary in the policy UI: No=0, Yes=1
PENALTY_LEVELS = [0, 1]
# When using a pre-generated pool, we sample a fixed number of candidates for speed.
POOL_FILENAME = "policy_pool.xlsx"
POOL_SHEET = "policies"  # falls back to first sheet if missing
POOL_SAMPLE_N = 400  # hidden; keep runtime reasonable

# Map multiplier levels to regulation strictness (used INSIDE the retailer MILP)
# Higher strictness => lower allowed average risk.
# This is the missing link that creates an actual trade-off.
_LEVEL_TO_QUANTILE = {1: 0.90, 5: 0.60, 10: 0.30}


@dataclass(frozen=True)
class ScenarioSettings:
    price_per_match: float
    min_utility: float
    suppliers_to_select: int


@dataclass(frozen=True)
class GovLimits:
    # strictness targets (derived from the priority slider)
    env_avg_max: float
    soc_avg_max: float
    child_tot_max: float
    ban_tot_max: float
    min_matches: int = TOTAL_USERS


def render_best_governmental_policy() -> None:
    """Render the student-facing 'Best governmental policy' page.

    Key fixes in this version:
    1) Cards/plots now show BOTH:
         - Total utility (sum of match utilities)
         - Profit (matches*price - cost_mult*cost)
    2) Policy multipliers now actually affect outcomes:
         - env_mult/social_mult/low_quality_mult are interpreted as regulatory strictness levels
           and are enforced as constraints INSIDE the retailer MILP (not only evaluated after).
         - child_labor_penalty/banned_chem_penalty are treated as Yes/No bans (hard constraints).
    Without (2), you can easily get the same solution under all policies (a dominated supplier set).
    """

    st.markdown("### Best governmental policy")
    st.caption(
        "Retailer goal: **maximize profit** under a policy. "
        "Government goal: **select policies that improve world goodness** and enable trade-offs."
    )

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE or not _HAS_GUROBI:
        st.error("gurobipy is not available in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

    # -----------------------------
    # Main student control (Design 1)
    # -----------------------------
    priority = st.slider(
        "Government priority",
        min_value=0,
        max_value=100,
        value=50,
        help="Move left for growth-first regulation, right for stricter sustainability regulation.",
    )
    alpha = float(priority) / 100.0  # 0=growth, 1=sustainability

    # Optional scenario settings (kept out of the way)
    with st.expander("Scenario settings (optional)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="bgp_P")
        with c2:
            min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="bgp_minU")
        with c3:
            suppliers_to_select = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1, key="bgp_K")

    scenario = ScenarioSettings(
        price_per_match=float(price_per_match),
        min_utility=float(min_utility),
        suppliers_to_select=int(suppliers_to_select),
    )

    with st.expander("What does this page do?", expanded=False):
        st.markdown(
            """
- Government **chooses a policy** (rules).
- Retailer then **optimizes under that policy** (inner MILP).
- We evaluate a set of **policy options** (deterministic, not random).
- Government selects a recommended policy using an MILP:
  - First with strict constraints.
  - If that is infeasible, it automatically falls back to a **backup MILP** that allows *minimal* relaxation.
            """
        )

    # -----------------------------
    # Detect if anything important changed -> require re-run
    # -----------------------------
    # IMPORTANT: BestGovPol must read the SAME policy object that the Policy tab writes.
    current_policy_dict = _policy_obj_to_dict(st.session_state.policy) if "policy" in st.session_state else _policy_obj_to_dict(Policy())
    policy_hash = _hash_policy_dict(current_policy_dict)
    sig = (policy_hash, scenario.price_per_match, scenario.min_utility, scenario.suppliers_to_select, float(priority))

    if st.session_state.get("bgp_run_sig") != sig:
        st.session_state.pop("bgp_rows", None)
        st.session_state.pop("bgp_df", None)
        st.session_state["bgp_run_sig"] = sig

    run = st.button("Run policy analysis", type="primary", use_container_width=True)

    if run:
        with st.spinner("Evaluating policy options (solving retailer MILP per option)..."):
            suppliers_df, users_df = load_supplier_user_tables(excel_path)
            options = _make_policy_options(current_policy_dict)
            rows = [_eval_policy(suppliers_df, users_df, opt, scenario) for opt in options]
            df = _rows_to_df(rows)

        st.session_state["bgp_rows"] = rows
        st.session_state["bgp_df"] = df

    # If not yet run, show a friendly hint
    if "bgp_df" not in st.session_state or "bgp_rows" not in st.session_state:
        st.info("Click **Run policy analysis** to see the recommended policy and the trade-off plots.")
        return

    rows: List[Dict[str, Any]] = st.session_state["bgp_rows"]
    df: pd.DataFrame = st.session_state["bgp_df"]

    # Keep only feasible retailer solutions for recommendation / plots.
    df_feas = df[df.get("feasible", 1) == 1].copy()
    infeas_count = int(len(df) - len(df_feas))

    if len(df_feas) == 0:
        st.error(
            "No feasible retailer solution was found for any policy option under the current scenario settings. "
            "Try lowering the minimum utility threshold or changing K."
        )
        st.dataframe(
            df[["__id", "feasible", "error"]].copy(),
            use_container_width=True,
            hide_index=True,
        )
        return

    if infeas_count > 0:
        st.warning(f"{infeas_count} policy option(s) were infeasible for the retailer and were excluded from selection.")

    # -----------------------------
    # Derive hidden regulation targets from the slider (no student knobs)
    # -----------------------------
    limits = _limits_from_priority(df_feas, alpha=alpha)

    # -----------------------------
    # Pick headline policies (now truly different dimensions exist)
    #   - Growth-first: max PROFIT (economic growth)
    #   - Sustainability-first: max GOODNESS (prefer full matching if possible)
    #   - Recommended: government MILP (hard; if infeasible -> backup)
    # Additionally, we compute and display TOTAL UTILITY for transparency/trade-offs.
    # -----------------------------
    best_growth_id = int(df_feas.sort_values(["profit", "matched"], ascending=[False, False]).iloc[0]["__id"])
    best_sust_id = _best_sustainable_id(df_feas)

    recommended_id, used_backup, slack_info = _solve_government_selection_milp(df_feas, limits)

    # -----------------------------
    # Headline cards
    # -----------------------------
    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        _render_policy_card("Growth-first policy", rows[best_growth_id], badge="Max profit")

    with c2:
        badge = "Government-selected" if not used_backup else "Government-selected (backup MILP)"
        _render_policy_card("Recommended policy", rows[recommended_id], badge=badge)

    with c3:
        _render_policy_card("Sustainability-first policy", rows[best_sust_id], badge="Best world goodness")

    # If everything is identical, be explicit (dominance / not enough leverage)
    if len({best_growth_id, recommended_id, best_sust_id}) == 1:
        st.warning(
            "All three headline policies ended up identical. This usually means one policy/matching dominates the others "
            "under the current scenario (K, min utility, price) OR the data does not create a cost↔risk trade-off. "
            "Try increasing K, raising the min-utility threshold, or using a richer policy pool."
        )

    if used_backup:
        st.info(
            "A fully compliant policy was not available at this strictness level. "
            "We selected the *closest* policy using a backup MILP that minimizes constraint relaxations."
        )
        if slack_info:
            st.caption(f"Relaxations used (slacks): {slack_info}")

    # -----------------------------
    # Trade-off plots
    # -----------------------------
    st.divider()
    st.markdown("### Trade-off maps (Policy options)")
    st.caption("Each point is a policy option. Compare different objectives under different regulations.")

    # Plot 1: Profit vs Goodness
    _render_scatter(
        df_feas,
        x_col="profit",
        y_col="goodness",
        x_label="Profit (matches·price − cost_mult·cost)",
        y_label="World goodness (higher is better)",
        highlight_ids=[best_growth_id, recommended_id, best_sust_id],
        labels={best_growth_id: "Growth", recommended_id: "Recommended", best_sust_id: "Sustainable"},
    )

    # Plot 2: Utility vs Profit (what you asked for)
    with st.expander("Show Utility vs Profit (trade-off)", expanded=False):
        _render_scatter(
            df_feas,
            x_col="utility_sum",
            y_col="profit",
            x_label="Total utility (sum over matches)",
            y_label="Profit",
            highlight_ids=[best_growth_id, recommended_id, best_sust_id],
            labels={best_growth_id: "Growth", recommended_id: "Recommended", best_sust_id: "Sustainable"},
        )

    # Exploration widget
    st.markdown("### Explore a policy option")

    def _fmt(opt_id: int) -> str:
        r = rows[opt_id]
        return (
            f"#{opt_id} | Profit={r['profit']:.2f} | Util={r['utility_sum']:.2f} | "
            f"G={r['goodness']:.2f} | matched={r['matched']} | env_avg={r['env_avg']:.2f} | soc_avg={r['soc_avg']:.2f}"
        )

    candidate_ids = list(df_feas.sort_values(["profit"], ascending=False)["__id"].astype(int).tolist())
    default_pick = recommended_id
    picked_id = st.selectbox(
        "Policy option",
        options=candidate_ids,
        index=candidate_ids.index(default_pick) if default_pick in candidate_ids else 0,
        format_func=_fmt,
    )

    _render_policy_details(rows[int(picked_id)], show_policy_json=True)


# ============================================================
# Helpers
# ============================================================

def _hash_policy_dict(p: Dict[str, float]) -> str:
    import hashlib, json
    s = json.dumps({k: float(v) for k, v in sorted(p.items())}, sort_keys=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _policy_obj_to_dict(pol: Policy) -> Dict[str, float]:
    return {
        "env_mult": float(pol.env_mult),
        "social_mult": float(pol.social_mult),
        "cost_mult": float(pol.cost_mult),
        "strategic_mult": float(pol.strategic_mult),
        "improvement_mult": float(pol.improvement_mult),
        "low_quality_mult": float(pol.low_quality_mult),
        "child_labor_penalty": float(pol.child_labor_penalty),
        "banned_chem_penalty": float(pol.banned_chem_penalty),
    }


def _policy_dict_to_obj(pdict: Dict[str, float]) -> Policy:
    pol = Policy()
    for k, v in pdict.items():
        setattr(pol, k, float(v))
    return pol


def _snap_to_levels(v: float, levels: List[int]) -> int:
    """Snap a numeric value to the nearest allowed discrete level."""
    try:
        x = float(v)
    except Exception:
        x = float(levels[0])
    return int(min(levels, key=lambda a: abs(a - x)))


@st.cache_data(show_spinner=False)
def _read_policy_pool(path: str) -> pd.DataFrame:
    """Read a policy pool file (xlsx). Cached for speed."""
    xls = pd.ExcelFile(path)
    sheet = POOL_SHEET if POOL_SHEET in xls.sheet_names else xls.sheet_names[0]
    return pd.read_excel(xls, sheet_name=sheet)


def _pool_path_default() -> Path:
    here = Path(__file__).resolve().parent
    cand = here / POOL_FILENAME
    if cand.exists():
        return cand
    return Path(POOL_FILENAME)


def _canonicalize_policy_dict(p: Dict[str, float]) -> Dict[str, float]:
    """Enforce the intended discrete domain.
    - multipliers: {1, 5, 10}
    - penalties: {0, 1}
    """
    mult_keys = ["env_mult", "social_mult", "cost_mult", "strategic_mult", "improvement_mult", "low_quality_mult"]
    out: Dict[str, float] = {}
    for k in mult_keys:
        out[k] = float(_snap_to_levels(p.get(k, MULT_LEVELS[0]), MULT_LEVELS))
    out["child_labor_penalty"] = float(_snap_to_levels(p.get("child_labor_penalty", 0), PENALTY_LEVELS))
    out["banned_chem_penalty"] = float(_snap_to_levels(p.get("banned_chem_penalty", 0), PENALTY_LEVELS))
    return out


def _make_policy_options(current: Dict[str, float]) -> List[Dict[str, float]]:
    """Policy options for evaluation.

    Preferred: read from a pre-generated pool (policy_pool.xlsx) and sample deterministically.
    Fallback: a small deterministic neighborhood around current.
    """
    cur = _canonicalize_policy_dict(current)

    # --- Try: pool-based options ---
    pool_path = _pool_path_default()
    if pool_path.exists():
        try:
            pool_df = _read_policy_pool(str(pool_path))
            needed = [
                "env_mult","social_mult","cost_mult","strategic_mult","improvement_mult","low_quality_mult",
                "child_labor_penalty","banned_chem_penalty"
            ]
            missing = [c for c in needed if c not in pool_df.columns]
            if not missing:
                pool = pool_df[needed].copy()

                # canonicalize/snap
                for k in needed:
                    pool[k] = pool[k].apply(lambda x: _canonicalize_policy_dict({k: x}).get(k, x))

                # deterministic sample depending on current policy only
                seed = int(_hash_policy_dict(cur)[:8], 16)
                n = int(min(max(1, POOL_SAMPLE_N), len(pool)))
                sampled = pool.sample(n=n, random_state=seed) if len(pool) > n else pool

                options = [ _canonicalize_policy_dict(row.to_dict()) for _, row in sampled.iterrows() ]
                options.append(cur)  # always include current

                # de-dup
                seen = set()
                uniq: List[Dict[str, float]] = []
                for p in options:
                    key = tuple(sorted((k, float(v)) for k, v in p.items()))
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append(p)
                return uniq
        except Exception:
            pass

    # --- Fallback: neighborhood options ---
    growth_anchor = {
        "env_mult": 1, "social_mult": 1, "cost_mult": 1,
        "strategic_mult": 10, "improvement_mult": 5, "low_quality_mult": 1,
        "child_labor_penalty": 0, "banned_chem_penalty": 0,
    }
    sust_anchor = {
        "env_mult": 10, "social_mult": 10, "cost_mult": 1,
        "strategic_mult": 5, "improvement_mult": 5, "low_quality_mult": 10,
        "child_labor_penalty": 1, "banned_chem_penalty": 1,
    }

    options: List[Dict[str, float]] = []
    options.append(_canonicalize_policy_dict(growth_anchor))
    options.append(_canonicalize_policy_dict(sust_anchor))
    options.append(cur)

    mult_keys = ["env_mult", "social_mult", "cost_mult", "strategic_mult", "improvement_mult", "low_quality_mult"]

    for k in mult_keys:
        base = int(cur[k])
        idx = MULT_LEVELS.index(base) if base in MULT_LEVELS else 0
        for di in (-1, 1):
            j = max(0, min(len(MULT_LEVELS) - 1, idx + di))
            p = dict(cur)
            p[k] = float(MULT_LEVELS[j])
            options.append(_canonicalize_policy_dict(p))

    for pk in ("child_labor_penalty", "banned_chem_penalty"):
        p = dict(cur)
        p[pk] = float(1 - int(cur[pk]))
        options.append(_canonicalize_policy_dict(p))

    seen = set()
    uniq: List[Dict[str, float]] = []
    for p in options:
        key = tuple(sorted((k, float(v)) for k, v in p.items()))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _require_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def _level_limit(series: pd.Series, level: float) -> float:
    """Convert discrete level (1/5/10) into a numeric max average risk using data quantiles."""
    lev = int(_snap_to_levels(level, MULT_LEVELS))
    q = _LEVEL_TO_QUANTILE.get(lev, 0.60)
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    return float(s.quantile(q))


def _auto_big_m_for_threshold(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pol: Policy, users: List[str]) -> float:
    """Conservative Big-M for utility threshold constraints."""
    s = suppliers_df
    u = users_df[users_df["user_id"].isin(users)].copy()
    if len(u) == 0 or len(s) == 0:
        return 1.0

    max_s = {
        "env_risk": float((pol.env_mult * s["env_risk"]).abs().max()),
        "social_risk": float((pol.social_mult * s["social_risk"]).abs().max()),
        "cost_score": float((pol.cost_mult * s["cost_score"]).abs().max()),
        "strategic": float((pol.strategic_mult * s["strategic"]).abs().max()),
        "improvement": float((pol.improvement_mult * s["improvement"]).abs().max()),
        "low_quality": float((pol.low_quality_mult * s["low_quality"]).abs().max()),
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
    return float(pref_max + 10.0)


def _solve_retailer_profit_milp(
    suppliers_df: pd.DataFrame,
    users_df: pd.DataFrame,
    pol: Policy,
    scenario: ScenarioSettings,
) -> Dict[str, Any]:
    """Retailer inner MILP:
    - Objective: maximize PROFIT = sum (price - cost_mult*cost_score) * z
    - Hard bans:
        if child_labor_penalty==1 => suppliers with child_labor==1 cannot be selected
        if banned_chem_penalty==1 => suppliers with banned_chem==1 cannot be selected
    - Regulation strictness (policy multipliers) enforced as constraints on average risks:
        env_mult/social_mult/low_quality_mult map to max average risk via quantiles.
    - Utility threshold is still enforced (Big-M) so some policies can become infeasible.
    """
    if gp is None or GRB is None:
        raise RuntimeError("gurobipy is not available.")

    _require_columns(
        suppliers_df,
        ["supplier_id", "env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"],
        "suppliers_df",
    )
    _require_columns(
        users_df,
        ["user_id", "w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"],
        "users_df",
    )

    s = suppliers_df.copy()
    u = users_df.copy()
    s["supplier_id"] = s["supplier_id"].astype(str)
    u["user_id"] = u["user_id"].astype(str)

    # last TOTAL_USERS by row order (same behavior as other tabs)
    users = u["user_id"].tolist()[-TOTAL_USERS:]
    Suppliers = s["supplier_id"].tolist()
    Users = users

    # maps
    s_env = dict(zip(s["supplier_id"], s["env_risk"].astype(float)))
    s_soc = dict(zip(s["supplier_id"], s["social_risk"].astype(float)))
    s_cost = dict(zip(s["supplier_id"], s["cost_score"].astype(float)))
    s_str = dict(zip(s["supplier_id"], s["strategic"].astype(float)))
    s_imp = dict(zip(s["supplier_id"], s["improvement"].astype(float)))
    s_lq = dict(zip(s["supplier_id"], s["low_quality"].astype(float)))
    s_child = dict(zip(s["supplier_id"], s["child_labor"].astype(float)))
    s_ban = dict(zip(s["supplier_id"], s["banned_chem"].astype(float)))

    udf = u[u["user_id"].isin(Users)].copy()
    # NOTE: w_low_quality is already NEG in MinCostAgent loader; but here we use users_df directly.
    # In your pipeline, load_supplier_user_tables() applies the NEG transformation, so this is consistent.
    u_env = dict(zip(udf["user_id"], udf["w_env"].astype(float)))
    u_soc = dict(zip(udf["user_id"], udf["w_social"].astype(float)))
    u_cost = dict(zip(udf["user_id"], udf["w_cost"].astype(float)))
    u_str = dict(zip(udf["user_id"], udf["w_strategic"].astype(float)))
    u_imp = dict(zip(udf["user_id"], udf["w_improvement"].astype(float)))
    u_lq = dict(zip(udf["user_id"], udf["w_low_quality"].astype(float)))

    M = _auto_big_m_for_threshold(s, u, pol, Users)

    m = gp.Model("RetailerProfitWithRegulation")
    m.Params.OutputFlag = 0

    y = m.addVars(Suppliers, vtype=GRB.BINARY, name="y_select")
    z = m.addVars(Suppliers, Users, vtype=GRB.BINARY, name="z_match")

    # select exactly K suppliers
    m.addConstr(gp.quicksum(y[i] for i in Suppliers) == int(scenario.suppliers_to_select), name="select_k")

    # each user at most once
    for uu in Users:
        m.addConstr(gp.quicksum(z[i, uu] for i in Suppliers) <= 1, name=f"user_once[{uu}]")

    # linking
    for i in Suppliers:
        for uu in Users:
            m.addConstr(z[i, uu] <= y[i], name=f"link[{i},{uu}]")

    # capacity upper bound (same as TOTAL_USERS here)
    m.addConstr(gp.quicksum(z[i, uu] for i in Suppliers for uu in Users) <= TOTAL_USERS, name="capacity")

    # hard bans (Yes=1)
    if int(round(pol.child_labor_penalty)) == 1:
        for i in Suppliers:
            if s_child.get(i, 0.0) >= 0.5:
                m.addConstr(y[i] == 0, name=f"ban_child[{i}]")
    if int(round(pol.banned_chem_penalty)) == 1:
        for i in Suppliers:
            if s_ban.get(i, 0.0) >= 0.5:
                m.addConstr(y[i] == 0, name=f"ban_banned[{i}]")

    # regulation strictness: limit average env/social/low-quality risks (policy levels -> data-based limits)
    total_matches = gp.quicksum(z[i, uu] for i in Suppliers for uu in Users)
    env_max = _level_limit(s["env_risk"], pol.env_mult)
    soc_max = _level_limit(s["social_risk"], pol.social_mult)
    lq_max = _level_limit(s["low_quality"], pol.low_quality_mult)

    m.addConstr(
        gp.quicksum(s_env[i] * z[i, uu] for i in Suppliers for uu in Users) <= env_max * total_matches,
        name="reg_env_avg",
    )
    m.addConstr(
        gp.quicksum(s_soc[i] * z[i, uu] for i in Suppliers for uu in Users) <= soc_max * total_matches,
        name="reg_soc_avg",
    )
    m.addConstr(
        gp.quicksum(s_lq[i] * z[i, uu] for i in Suppliers for uu in Users) <= lq_max * total_matches,
        name="reg_lq_avg",
    )

    # utility threshold (kept)
    for i in Suppliers:
        for uu in Users:
            user_pref = (
                (u_env[uu] * (pol.env_mult * s_env[i]))
                + (u_soc[uu] * (pol.social_mult * s_soc[i]))
                + (u_cost[uu] * (pol.cost_mult * s_cost[i]))
                + (u_str[uu] * (pol.strategic_mult * s_str[i]))
                + (u_imp[uu] * (pol.improvement_mult * s_imp[i]))
                + (u_lq[uu] * (pol.low_quality_mult * s_lq[i]))
            )
            m.addConstr(user_pref >= float(scenario.min_utility) - M * (1 - z[i, uu]), name=f"utility[{i},{uu}]")

    # objective: profit (NO penalties in cost; matches*price - cost_mult*cost)
    margin_i = {i: float(scenario.price_per_match - (pol.cost_mult * s_cost[i])) for i in Suppliers}
    m.setObjective(gp.quicksum(margin_i[i] * z[i, uu] for i in Suppliers for uu in Users), GRB.MAXIMIZE)

    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"No solution. Status={m.Status}")

    chosen = [i for i in Suppliers if y[i].X > 0.5]
    pairs = [(uu, i) for i in Suppliers for uu in Users if z[i, uu].X > 0.5]
    matches = pd.DataFrame(pairs, columns=["user_id", "supplier_id"])

    # compute per-match profit and utility for reporting
    if len(matches) > 0:
        matches["cost_prod"] = matches["supplier_id"].map(lambda sid: float(pol.cost_mult * s_cost[sid]))
        matches["profit"] = float(scenario.price_per_match) - matches["cost_prod"]

        def _utility(row: pd.Series) -> float:
            sid = row["supplier_id"]
            uid = row["user_id"]
            return float(
                (u_env[uid] * (pol.env_mult * s_env[sid]))
                + (u_soc[uid] * (pol.social_mult * s_soc[sid]))
                + (u_cost[uid] * (pol.cost_mult * s_cost[sid]))
                + (u_str[uid] * (pol.strategic_mult * s_str[sid]))
                + (u_imp[uid] * (pol.improvement_mult * s_imp[sid]))
                + (u_lq[uid] * (pol.low_quality_mult * s_lq[sid]))
            )

        matches["utility"] = matches.apply(_utility, axis=1)
        matches = matches.sort_values(["supplier_id", "user_id"]).reset_index(drop=True)
    else:
        matches["cost_prod"] = []
        matches["profit"] = []
        matches["utility"] = []

    return {
        "objective_value": float(m.ObjVal),
        "num_matched": int(len(pairs)),
        "chosen_suppliers": chosen,
        "selected_users": Users,
        "matches": matches,
        "env_max": env_max,
        "soc_max": soc_max,
        "lq_max": lq_max,
    }


def _eval_policy(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pdict: Dict[str, float], scenario: ScenarioSettings) -> Dict[str, Any]:
    """Evaluate one policy option by solving the retailer (inner) MILP.

    This is where the previous version often became "flat":
    - If policy parameters do NOT constrain the retailer (or only scale costs uniformly),
      many different policies can yield the exact same matching.
    This version enforces regulation constraints inside the retailer MILP to create real trade-offs.
    """
    pol = _policy_dict_to_obj(pdict)

    feasible = True
    error_msg = ""
    try:
        res = _solve_retailer_profit_milp(suppliers_df, users_df, pol, scenario)
        matches = res["matches"].copy()
    except Exception as e:
        feasible = False
        error_msg = str(e)
        res = {
            "objective_value": -1.0e12,
            "num_matched": 0,
            "chosen_suppliers": [],
            "matches": pd.DataFrame(columns=["supplier_id"]),
        }
        matches = res["matches"].copy()

    sup = suppliers_df.copy()
    sup["supplier_id"] = sup["supplier_id"].astype(str)
    sup = sup.set_index("supplier_id")

    if "supplier_id" in matches.columns:
        matches["supplier_id"] = matches["supplier_id"].astype(str)

    # totals/avgs from realized matching
    env_tot = soc_tot = child_tot = ban_tot = lq_tot = 0.0
    utility_sum = 0.0

    if feasible and len(matches) > 0 and "supplier_id" in matches.columns:
        env_map = sup.get("env_risk", pd.Series(dtype=float)).astype(float).to_dict()
        soc_map = sup.get("social_risk", pd.Series(dtype=float)).astype(float).to_dict()
        child_map = sup.get("child_labor", pd.Series(dtype=float)).astype(float).to_dict()
        ban_map = sup.get("banned_chem", pd.Series(dtype=float)).astype(float).to_dict()
        lq_map = sup.get("low_quality", pd.Series(dtype=float)).astype(float).to_dict()

        matches["env_risk"] = matches["supplier_id"].map(env_map).fillna(0.0)
        matches["social_risk"] = matches["supplier_id"].map(soc_map).fillna(0.0)
        matches["child_labor"] = matches["supplier_id"].map(child_map).fillna(0.0)
        matches["banned_chem"] = matches["supplier_id"].map(ban_map).fillna(0.0)
        matches["low_quality"] = matches["supplier_id"].map(lq_map).fillna(0.0)

        env_tot = float(matches["env_risk"].sum())
        soc_tot = float(matches["social_risk"].sum())
        child_tot = float(matches["child_labor"].sum())
        ban_tot = float(matches["banned_chem"].sum())
        lq_tot = float(matches["low_quality"].sum())

        if "utility" in matches.columns:
            utility_sum = float(pd.to_numeric(matches["utility"], errors="coerce").fillna(0.0).sum())

    mcount = int(res.get("num_matched", 0) or 0)
    env_avg = env_tot / mcount if mcount > 0 else 0.0
    soc_avg = soc_tot / mcount if mcount > 0 else 0.0
    lq_avg = lq_tot / mcount if mcount > 0 else 0.0

    # higher is better
    w = WORLD_GOODNESS_WEIGHTS
    goodness = -(
        w["w_env"] * env_avg
        + w["w_soc"] * soc_avg
        + w["w_child"] * child_tot
        + w["w_ban"] * ban_tot
        + w["w_lq"] * lq_avg
    )

    if not feasible:
        goodness = -1.0e12

    return {
        "policy": pdict,
        "feasible": bool(feasible),
        "error": error_msg,
        "profit": float(res.get("objective_value", -1.0e12)),
        "utility_sum": float(utility_sum),
        "matched": int(mcount),
        "chosen_suppliers": ", ".join(res.get("chosen_suppliers", [])) if feasible else "",
        "env_tot": env_tot,
        "soc_tot": soc_tot,
        "child_tot": child_tot,
        "ban_tot": ban_tot,
        "lq_tot": lq_tot,
        "env_avg": env_avg,
        "soc_avg": soc_avg,
        "lq_avg": lq_avg,
        "goodness": float(goodness),
        "matches_df": matches,
    }


def _rows_to_df(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    out = []
    for j, r in enumerate(rows):
        d: Dict[str, Any] = {"__id": j}
        d["feasible"] = int(bool(r.get("feasible", True)))
        d["error"] = r.get("error", "") or ""
        d.update({k: float(v) for k, v in r["policy"].items()})

        # main metrics
        d["profit"] = float(r["profit"])
        d["utility_sum"] = float(r["utility_sum"])
        d["matched"] = int(r["matched"])

        for k in ["env_avg", "soc_avg", "lq_avg", "child_tot", "ban_tot", "goodness"]:
            d[k] = float(r[k])

        out.append(d)
    return pd.DataFrame(out)


def _limits_from_priority(df: pd.DataFrame, alpha: float) -> GovLimits:
    # alpha=0 => loose (max); alpha=1 => strict (min)
    def lerp(lo: float, hi: float, t: float) -> float:
        return (1.0 - t) * hi + t * lo

    env_lo, env_hi = float(df["env_avg"].min()), float(df["env_avg"].max())
    soc_lo, soc_hi = float(df["soc_avg"].min()), float(df["soc_avg"].max())
    child_lo, child_hi = float(df["child_tot"].min()), float(df["child_tot"].max())
    ban_lo, ban_hi = float(df["ban_tot"].min()), float(df["ban_tot"].max())

    return GovLimits(
        env_avg_max=lerp(env_lo, env_hi, alpha),
        soc_avg_max=lerp(soc_lo, soc_hi, alpha),
        child_tot_max=lerp(child_lo, child_hi, alpha),
        ban_tot_max=lerp(ban_lo, ban_hi, alpha),
        min_matches=TOTAL_USERS,
    )


def _best_sustainable_id(df: pd.DataFrame) -> int:
    full = df[df["matched"] >= TOTAL_USERS]
    if len(full) > 0:
        return int(full.sort_values(["goodness", "profit"], ascending=[False, False]).iloc[0]["__id"])
    return int(df.sort_values(["goodness", "profit"], ascending=[False, False]).iloc[0]["__id"])


def _solve_government_selection_milp(df: pd.DataFrame, limits: GovLimits) -> Tuple[int, bool, str]:
    """Return (recommended_id, used_backup, slack_summary_text)."""
    eps_tie = 1e-3

    rows = df.to_dict(orient="records")
    J = list(range(len(rows)))

    # -----------------------------
    # Attempt 1: hard-constraint MILP
    # Government chooses ONE policy, maximizing PROFIT subject to limits,
    # tie-breaking on goodness.
    # -----------------------------
    hard = gp.Model("GovMILP_hard")
    hard.Params.OutputFlag = 0

    x = hard.addVars(J, vtype=GRB.BINARY, name="x")
    hard.addConstr(gp.quicksum(x[j] for j in J) == 1, name="choose_one")

    hard.addConstr(
        gp.quicksum(rows[j]["matched"] * x[j] for j in J) >= int(limits.min_matches),
        name="min_matches",
    )

    hard.addConstr(
        gp.quicksum(rows[j]["env_avg"] * rows[j]["matched"] * x[j] for j in J)
        <= float(limits.env_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
        name="env_avg",
    )
    hard.addConstr(
        gp.quicksum(rows[j]["soc_avg"] * rows[j]["matched"] * x[j] for j in J)
        <= float(limits.soc_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
        name="soc_avg",
    )
    hard.addConstr(
        gp.quicksum(rows[j]["child_tot"] * x[j] for j in J) <= float(limits.child_tot_max),
        name="child_tot",
    )
    hard.addConstr(
        gp.quicksum(rows[j]["ban_tot"] * x[j] for j in J) <= float(limits.ban_tot_max),
        name="ban_tot",
    )

    hard.setObjective(
        gp.quicksum(rows[j]["profit"] * x[j] for j in J)
        + eps_tie * gp.quicksum(rows[j]["goodness"] * x[j] for j in J),
        GRB.MAXIMIZE,
    )
    hard.optimize()

    if hard.Status == GRB.OPTIMAL:
        chosen = next(j for j in J if x[j].X > 0.5)
        return int(rows[chosen]["__id"]), False, ""

    # -----------------------------
    # Attempt 2: backup MILP with slacks (always feasible)
    # Lexicographic:
    #   (1) minimize total slack
    #   (2) maximize profit (tie-break by goodness)
    # -----------------------------
    backup = gp.Model("GovMILP_backup")
    backup.Params.OutputFlag = 0

    xb = backup.addVars(J, vtype=GRB.BINARY, name="x")
    backup.addConstr(gp.quicksum(xb[j] for j in J) == 1, name="choose_one")

    s_env = backup.addVar(lb=0.0, name="s_env")
    s_soc = backup.addVar(lb=0.0, name="s_soc")
    s_child = backup.addVar(lb=0.0, name="s_child")
    s_ban = backup.addVar(lb=0.0, name="s_ban")
    s_m = backup.addVar(lb=0.0, name="s_min_matches")

    backup.addConstr(
        gp.quicksum(rows[j]["matched"] * xb[j] for j in J) + s_m >= int(limits.min_matches),
        name="min_matches_slack",
    )

    backup.addConstr(
        gp.quicksum(rows[j]["env_avg"] * rows[j]["matched"] * xb[j] for j in J)
        <= float(limits.env_avg_max) * gp.quicksum(rows[j]["matched"] * xb[j] for j in J) + s_env,
        name="env_avg_slack",
    )
    backup.addConstr(
        gp.quicksum(rows[j]["soc_avg"] * rows[j]["matched"] * xb[j] for j in J)
        <= float(limits.soc_avg_max) * gp.quicksum(rows[j]["matched"] * xb[j] for j in J) + s_soc,
        name="soc_avg_slack",
    )

    backup.addConstr(
        gp.quicksum(rows[j]["child_tot"] * xb[j] for j in J) <= float(limits.child_tot_max) + s_child,
        name="child_tot_slack",
    )
    backup.addConstr(
        gp.quicksum(rows[j]["ban_tot"] * xb[j] for j in J) <= float(limits.ban_tot_max) + s_ban,
        name="ban_tot_slack",
    )

    slack_sum = (1000.0 * s_m) + (1.0 * s_env) + (1.0 * s_soc) + (10.0 * s_child) + (10.0 * s_ban)
    backup.setObjective(slack_sum, GRB.MINIMIZE)
    backup.optimize()

    best_slack = float(slack_sum.getValue())
    backup.addConstr(slack_sum <= best_slack + 1e-6, name="fix_slack")
    backup.setObjective(
        gp.quicksum(rows[j]["profit"] * xb[j] for j in J)
        + eps_tie * gp.quicksum(rows[j]["goodness"] * xb[j] for j in J),
        GRB.MAXIMIZE,
    )
    backup.optimize()

    chosen = next(j for j in J if xb[j].X > 0.5)
    slack_info = f"min_matches={s_m.X:.3f}, env={s_env.X:.3f}, soc={s_soc.X:.3f}, child={s_child.X:.3f}, banned={s_ban.X:.3f}"
    return int(rows[chosen]["__id"]), True, slack_info


def _render_policy_card(title: str, r: Dict[str, Any], badge: str) -> None:
    st.markdown(f"**{title}**")
    st.caption(badge)

    st.metric("Profit", f"{r['profit']:.2f}")
    st.metric("Total utility", f"{r['utility_sum']:.2f}")
    st.metric("World goodness", f"{r['goodness']:.2f}")

    st.caption(
        f"Matched: **{r['matched']} / {TOTAL_USERS}**  ·  "
        f"env_avg: **{r['env_avg']:.2f}**  ·  soc_avg: **{r['soc_avg']:.2f}**  ·  "
        f"child_total: **{r['child_tot']:.0f}**  ·  banned_total: **{r['ban_tot']:.0f}**"
    )

    with st.expander("Show details", expanded=False):
        st.write("Chosen suppliers:", r["chosen_suppliers"] or "None")
        st.json(r["policy"])
        st.dataframe(r["matches_df"], use_container_width=True, hide_index=True)


def _render_policy_details(r: Dict[str, Any], show_policy_json: bool = True) -> None:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Profit", f"{r['profit']:.2f}")
        st.metric("Total utility", f"{r['utility_sum']:.2f}")
        st.metric("World goodness", f"{r['goodness']:.2f}")
        st.caption(
            f"Matched: **{r['matched']} / {TOTAL_USERS}**\n\n"
            f"env_avg: **{r['env_avg']:.2f}**\n\n"
            f"soc_avg: **{r['soc_avg']:.2f}**\n\n"
            f"child_total: **{r['child_tot']:.0f}**\n\n"
            f"banned_total: **{r['ban_tot']:.0f}**"
        )
        st.write("Chosen suppliers:", r["chosen_suppliers"] or "None")
        if show_policy_json:
            st.write("Policy parameters:")
            st.json(r["policy"])
    with col2:
        st.dataframe(r["matches_df"], use_container_width=True, hide_index=True)


def _render_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    highlight_ids: List[int],
    labels: Dict[int, str],
) -> None:
    import matplotlib.pyplot as plt

    x = df[x_col].astype(float).values
    y = df[y_col].astype(float).values

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.6)

    marker_map = {0: "X", 1: "o", 2: "^"}
    for k, pid in enumerate(highlight_ids[:3]):
        row = df[df["__id"] == pid].iloc[0]
        ax.scatter([float(row[x_col])], [float(row[y_col])], marker=marker_map.get(k, "o"), s=120)
        ax.annotate(labels.get(pid, str(pid)), (float(row[x_col]), float(row[y_col])))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    st.pyplot(fig, clear_figure=True)
