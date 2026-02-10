from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import streamlit as st

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    MaxProfitAgent,
    MaxProfitConfig,
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
PENALTY_MAX = 100.0
PENALTY_STEP = 25.0  # deterministic "neighborhood" step


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
    """Render the student-facing 'Best governmental policy' page (Design 1 + 2 mix).

    - Student sees ONLY one main slider (growth <-> sustainability).
    - No random seed / no candidate-count knobs.
    - No infeasibility pop-ups:
        1) Try a hard-constraint government MILP.
        2) If infeasible, automatically switch to a slack-based backup MILP (always feasible).
    - Output: 3 cards (Growth / Recommended / Sustainable) + trade-off plot.
    """

    st.markdown("### Best governmental policy")
    st.caption("Retailer goal: **maximize retailer utility**. Government goal: **ensure world goodness while retailer optimizes**.")

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
- We evaluate a small set of **policy options** (deterministic, not random).
- Government selects a recommended policy using an MILP:
  - First with strict constraints.
  - If that is infeasible, it automatically falls back to a **backup MILP** that allows *minimal* relaxation.
            """
        )

    # -----------------------------
    # Detect if anything important changed -> require re-run
    # -----------------------------
    current_policy_dict = _policy_obj_to_dict(st.session_state.policy) if "policy" in st.session_state else _policy_obj_to_dict(Policy())
    policy_hash = _hash_policy_dict(current_policy_dict)
    sig = (policy_hash, scenario.price_per_match, scenario.min_utility, scenario.suppliers_to_select)

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
        st.info("Click **Run policy analysis** to see the recommended policy and the trade-off plot.")
        return

    rows: List[Dict[str, Any]] = st.session_state["bgp_rows"]
    df: pd.DataFrame = st.session_state["bgp_df"]

    # -----------------------------
    # Derive hidden regulation targets from the slider (no student knobs)
    # -----------------------------
    limits = _limits_from_priority(df, alpha=alpha)

    # -----------------------------
    # Pick the three headline policies
    #   - Best Growth: max retailer utility
    #   - Best Sustainable: max goodness (prefer full matching if possible)
    #   - Recommended: government MILP (hard; if infeasible -> backup)
    # -----------------------------
    best_growth_id = int(df.sort_values(["profit_obj", "matched"], ascending=[False, False]).iloc[0]["__id"])
    best_sust_id = _best_sustainable_id(df)

    recommended_id, used_backup, slack_info = _solve_government_selection_milp(df, limits)

    # -----------------------------
    # Headline cards (Design 1)
    # -----------------------------
    st.divider()
    c1, c2, c3 = st.columns(3)

    with c1:
        _render_policy_card("Growth-first policy", rows[best_growth_id], badge="Max retailer utility")

    with c2:
        title = "Recommended policy"
        badge = "Government-selected"
        if used_backup:
            badge = "Government-selected (backup MILP)"
        _render_policy_card(title, rows[recommended_id], badge=badge)

    with c3:
        _render_policy_card("Sustainability-first policy", rows[best_sust_id], badge="Best world goodness")

    if used_backup:
        st.info(
            "A fully compliant policy was not available at this strictness level. "
            "We selected the *closest* policy using a backup MILP that minimizes constraint relaxations."
        )
        if slack_info:
            st.caption(f"Relaxations used (slacks): {slack_info}")

    # -----------------------------
    # Pareto plot (Design 2)
    # -----------------------------
    st.divider()
    st.markdown("### Trade-off map (Policy options)")
    st.caption("Each point is a policy option. X-axis: retailer utility. Y-axis: world goodness.")

    _render_pareto_plot(
        df,
        highlight_ids=[best_growth_id, recommended_id, best_sust_id],
        labels={best_growth_id: "Growth", recommended_id: "Recommended", best_sust_id: "Sustainable"},
    )

    # A light-weight interactive exploration (instead of click-on-plot)
    st.markdown("### Explore a policy option")
    def _fmt(opt_id: int) -> str:
        r = rows[opt_id]
        return f"#{opt_id} | U={r['profit_obj']:.2f} | G={r['goodness']:.2f} | matched={r['matched']} | env_avg={r['env_avg']:.2f} | soc_avg={r['soc_avg']:.2f}"

    candidate_ids = list(df.sort_values(["profit_obj"], ascending=False)["__id"].astype(int).tolist())
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


def _clamp_int(v: float, lo: int, hi: int) -> int:
    return max(lo, min(hi, int(round(v))))


def _clamp_float(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _canonicalize_policy_dict(p: Dict[str, float]) -> Dict[str, float]:
    # enforce the intended discrete-ish domain for multipliers and bounded penalties
    mult_keys = ["env_mult", "social_mult", "cost_mult", "strategic_mult", "improvement_mult", "low_quality_mult"]
    out: Dict[str, float] = {}
    for k in mult_keys:
        out[k] = float(_clamp_int(p.get(k, 3.0), 1, 5))
    out["child_labor_penalty"] = float(_clamp_float(p.get("child_labor_penalty", 0.0), 0.0, PENALTY_MAX))
    out["banned_chem_penalty"] = float(_clamp_float(p.get("banned_chem_penalty", 0.0), 0.0, PENALTY_MAX))
    return out


def _make_policy_options(current: Dict[str, float]) -> List[Dict[str, float]]:
    """Deterministic (student-friendly) policy options:
    - Growth anchor
    - Sustainable anchor
    - Current policy
    - Single-parameter neighbors around current (±1 for multipliers; ±PENALTY_STEP for penalties)
    - A few coupled neighbors (env+social; child+banned)
    """
    cur = _canonicalize_policy_dict(current)

    growth_anchor = {
        "env_mult": 1, "social_mult": 1, "cost_mult": 5,
        "strategic_mult": 5, "improvement_mult": 3, "low_quality_mult": 1,
        "child_labor_penalty": 0, "banned_chem_penalty": 0,
    }
    sust_anchor = {
        "env_mult": 5, "social_mult": 5, "cost_mult": 1,
        "strategic_mult": 2, "improvement_mult": 3, "low_quality_mult": 5,
        "child_labor_penalty": PENALTY_MAX, "banned_chem_penalty": PENALTY_MAX,
    }

    options: List[Dict[str, float]] = []
    options.append(_canonicalize_policy_dict(growth_anchor))
    options.append(_canonicalize_policy_dict(sust_anchor))
    options.append(cur)

    mult_keys = ["env_mult", "social_mult", "cost_mult", "strategic_mult", "improvement_mult", "low_quality_mult"]

    # single-key neighbors
    for k in mult_keys:
        base = int(cur[k])
        for d in (-1, 1):
            v = _clamp_int(base + d, 1, 5)
            p = dict(cur)
            p[k] = float(v)
            options.append(_canonicalize_policy_dict(p))

    # penalty neighbors
    for pk in ("child_labor_penalty", "banned_chem_penalty"):
        base = float(cur[pk])
        for d in (-PENALTY_STEP, PENALTY_STEP):
            p = dict(cur)
            p[pk] = float(_clamp_float(base + d, 0.0, PENALTY_MAX))
            options.append(_canonicalize_policy_dict(p))

    # coupled env/social neighbors
    for d in (-1, 1):
        p = dict(cur)
        p["env_mult"] = float(_clamp_int(int(cur["env_mult"]) + d, 1, 5))
        p["social_mult"] = float(_clamp_int(int(cur["social_mult"]) + d, 1, 5))
        options.append(_canonicalize_policy_dict(p))

    # coupled child/banned neighbors
    for d in (-PENALTY_STEP, PENALTY_STEP):
        p = dict(cur)
        p["child_labor_penalty"] = float(_clamp_float(float(cur["child_labor_penalty"]) + d, 0.0, PENALTY_MAX))
        p["banned_chem_penalty"] = float(_clamp_float(float(cur["banned_chem_penalty"]) + d, 0.0, PENALTY_MAX))
        options.append(_canonicalize_policy_dict(p))

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


def _require_columns(df: pd.DataFrame, cols: List[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} missing required columns: {missing}")


def _eval_policy(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pdict: Dict[str, float], scenario: ScenarioSettings) -> Dict[str, Any]:
    pol = _policy_dict_to_obj(pdict)

    cfg = MaxProfitConfig(
        last_n_users=TOTAL_USERS,
        capacity=TOTAL_USERS,
        suppliers_to_select=int(scenario.suppliers_to_select),
        price_per_match=float(scenario.price_per_match),
        min_utility=float(scenario.min_utility),
        output_flag=0,
    )

    res = MaxProfitAgent(suppliers_df, users_df, pol, cfg).solve()
    matches = res["matches"].copy()

    _require_columns(
        suppliers_df,
        ["supplier_id", "env_risk", "social_risk", "child_labor", "banned_chem", "low_quality"],
        "suppliers_df",
    )
    _require_columns(matches, ["supplier_id"], "matches")

    sup = suppliers_df.copy()
    sup["supplier_id"] = sup["supplier_id"].astype(str)
    sup = sup.set_index("supplier_id")

    matches["supplier_id"] = matches["supplier_id"].astype(str)

    if len(matches) > 0:
        env_map = sup["env_risk"].astype(float).to_dict()
        soc_map = sup["social_risk"].astype(float).to_dict()
        child_map = sup["child_labor"].astype(float).to_dict()
        ban_map = sup["banned_chem"].astype(float).to_dict()
        lq_map = sup["low_quality"].astype(float).to_dict()

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
    else:
        env_tot = soc_tot = child_tot = ban_tot = lq_tot = 0.0

    m = int(res.get("num_matched", 0) or 0)
    env_avg = env_tot / m if m > 0 else 0.0
    soc_avg = soc_tot / m if m > 0 else 0.0
    lq_avg = lq_tot / m if m > 0 else 0.0

    # higher is better
    w = WORLD_GOODNESS_WEIGHTS
    goodness = -(
        w["w_env"] * env_avg
        + w["w_soc"] * soc_avg
        + w["w_child"] * child_tot
        + w["w_ban"] * ban_tot
        + w["w_lq"] * lq_avg
    )

    return {
        "policy": pdict,
        "profit_obj": float(res["objective_value"]),
        "matched": int(m),
        "chosen_suppliers": ", ".join(res.get("chosen_suppliers", [])),
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
        d = {"__id": j}
        d.update({k: float(v) for k, v in r["policy"].items()})
        for k in ["profit_obj", "matched", "env_avg", "soc_avg", "lq_avg", "child_tot", "ban_tot", "goodness"]:
            d[k] = float(r[k]) if k != "matched" else int(r[k])
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
    # Prefer full matching if possible (matched==TOTAL_USERS), else take best goodness overall.
    full = df[df["matched"] >= TOTAL_USERS]
    if len(full) > 0:
        return int(full.sort_values(["goodness", "profit_obj"], ascending=[False, False]).iloc[0]["__id"])
    return int(df.sort_values(["goodness", "profit_obj"], ascending=[False, False]).iloc[0]["__id"])


def _solve_government_selection_milp(df: pd.DataFrame, limits: GovLimits) -> Tuple[int, bool, str]:
    """Return (recommended_id, used_backup, slack_summary_text)."""
    eps_tie = 1e-3

    rows = df.to_dict(orient="records")
    J = list(range(len(rows)))

    # -----------------------------
    # Attempt 1: hard-constraint MILP
    # -----------------------------
    hard = gp.Model("GovMILP_hard")
    hard.Params.OutputFlag = 0

    x = hard.addVars(J, vtype=GRB.BINARY, name="x")
    hard.addConstr(gp.quicksum(x[j] for j in J) == 1, name="choose_one")

    hard.addConstr(
        gp.quicksum(rows[j]["matched"] * x[j] for j in J) >= int(limits.min_matches),
        name="min_matches",
    )

    # avg constraints: total <= avg_max * matches
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
        gp.quicksum(rows[j]["profit_obj"] * x[j] for j in J) + eps_tie * gp.quicksum(rows[j]["goodness"] * x[j] for j in J),
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
    #   (2) maximize retailer utility (tie-break by goodness)
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

    # min matches + slack
    backup.addConstr(
        gp.quicksum(rows[j]["matched"] * xb[j] for j in J) + s_m >= int(limits.min_matches),
        name="min_matches_slack",
    )

    # env/social avg constraints with slack
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

    # child/banned totals with slack
    backup.addConstr(
        gp.quicksum(rows[j]["child_tot"] * xb[j] for j in J) <= float(limits.child_tot_max) + s_child,
        name="child_tot_slack",
    )
    backup.addConstr(
        gp.quicksum(rows[j]["ban_tot"] * xb[j] for j in J) <= float(limits.ban_tot_max) + s_ban,
        name="ban_tot_slack",
    )

    # Stage 1: minimize slack (weights put min-matches slack as most important)
    slack_sum = (1000.0 * s_m) + (1.0 * s_env) + (1.0 * s_soc) + (10.0 * s_child) + (10.0 * s_ban)
    backup.setObjective(slack_sum, GRB.MINIMIZE)
    backup.optimize()

    # Stage 2: maximize retailer utility while keeping slack minimal
    best_slack = float(slack_sum.getValue())
    backup.addConstr(slack_sum <= best_slack + 1e-6, name="fix_slack")
    backup.setObjective(
        gp.quicksum(rows[j]["profit_obj"] * xb[j] for j in J) + eps_tie * gp.quicksum(rows[j]["goodness"] * xb[j] for j in J),
        GRB.MAXIMIZE,
    )
    backup.optimize()

    chosen = next(j for j in J if xb[j].X > 0.5)

    slack_info = f"min_matches={s_m.X:.3f}, env={s_env.X:.3f}, soc={s_soc.X:.3f}, child={s_child.X:.3f}, banned={s_ban.X:.3f}"
    return int(rows[chosen]["__id"]), True, slack_info


def _render_policy_card(title: str, r: Dict[str, Any], badge: str) -> None:
    st.markdown(f"**{title}**")
    st.caption(badge)
    st.metric("Retailer utility", f"{r['profit_obj']:.2f}")
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
        st.metric("Retailer utility", f"{r['profit_obj']:.2f}")
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


def _render_pareto_plot(df: pd.DataFrame, highlight_ids: List[int], labels: Dict[int, str]) -> None:
    import matplotlib.pyplot as plt

    x = df["profit_obj"].astype(float).values
    y = df["goodness"].astype(float).values

    fig, ax = plt.subplots()
    ax.scatter(x, y, alpha=0.6)

    # highlight points using different markers (no explicit colors)
    marker_map = {0: "X", 1: "o", 2: "^"}
    for k, pid in enumerate(highlight_ids[:3]):
        row = df[df["__id"] == pid].iloc[0]
        ax.scatter([float(row["profit_obj"])], [float(row["goodness"])], marker=marker_map.get(k, "o"), s=120)
        ax.annotate(labels.get(pid, str(pid)), (float(row["profit_obj"]), float(row["goodness"])))

    ax.set_xlabel("Retailer utility (objective)")
    ax.set_ylabel("World goodness (higher is better)")
    st.pyplot(fig, clear_figure=True)
