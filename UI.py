import random
from pathlib import Path

import pandas as pd
import streamlit as st

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    MaxProfitAgent,
    MaxProfitConfig,
    MinCostAgent,
    MinCostConfig,
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
# Fixed world-goodness weights (used only for ε tie-break)
# -----------------------------
WORLD_GOODNESS_WEIGHTS = {
    "w_env": 1.0,
    "w_soc": 1.0,
    "w_child": 10.0,
    "w_ban": 10.0,
    "w_lq": 1.0,
}



st.set_page_config(page_title="Arya Case", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stSidebar"] {display: none;}
</style>
""",
    unsafe_allow_html=True,
)

if "policy" not in st.session_state:
    st.session_state.policy = Policy()

tab_policy, tab_profit, tab_mincost, tab_best_policy = st.tabs(
    ["Policy", "Min Cost Agent", "Max Profit Agent", "Best governmental policy"]
)

# -----------------------------
# Policy tab (UNCHANGED)
# -----------------------------
with tab_policy:

    p = st.session_state.policy

    # Discrete policy UI:
    # - multipliers are chosen as Low/Mid/High mapped to 1/5/10
    # - child labor & banned chemicals penalties are Yes/No mapped to 1/0
    LEVELS = [("Low", 1.0), ("Mid", 5.0), ("High", 10.0)]
    LEVEL_LABELS = [x[0] for x in LEVELS]
    LEVEL_MAP = {k: v for k, v in LEVELS}
    LEVEL_VALS = [x[1] for x in LEVELS]

    def _nearest_level_index(v: float) -> int:
        try:
            vv = float(v)
        except Exception:
            vv = 1.0
        return int(min(range(len(LEVEL_VALS)), key=lambda i: abs(vv - LEVEL_VALS[i])))

    def _yesno_index(v: float) -> int:
        try:
            vv = float(v)
        except Exception:
            vv = 0.0
        return 1 if vv > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        env_level = st.selectbox(
            "Environmental multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.env_mult),
            key="policy_env_level",
        )
        soc_level = st.selectbox(
            "Social multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.social_mult),
            key="policy_soc_level",
        )
        cost_level = st.selectbox(
            "Cost multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.cost_mult),
            key="policy_cost_level",
        )
        p.env_mult = LEVEL_MAP[env_level]
        p.social_mult = LEVEL_MAP[soc_level]
        p.cost_mult = LEVEL_MAP[cost_level]

    with c2:
        strat_level = st.selectbox(
            "Strategic multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.strategic_mult),
            key="policy_strat_level",
        )
        improv_level = st.selectbox(
            "Improvement multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.improvement_mult),
            key="policy_improv_level",
        )
        lq_level = st.selectbox(
            "Low-quality multiplier",
            LEVEL_LABELS,
            index=_nearest_level_index(p.low_quality_mult),
            key="policy_lq_level",
        )
        p.strategic_mult = LEVEL_MAP[strat_level]
        p.improvement_mult = LEVEL_MAP[improv_level]
        p.low_quality_mult = LEVEL_MAP[lq_level]

    with c3:
        child_yesno = st.radio(
            "Child labor penalty",
            ["No", "Yes"],
            index=_yesno_index(p.child_labor_penalty),
            horizontal=True,
            key="policy_child_yesno",
        )
        ban_yesno = st.radio(
            "Banned chemicals penalty",
            ["No", "Yes"],
            index=_yesno_index(p.banned_chem_penalty),
            horizontal=True,
            key="policy_ban_yesno",
        )
        p.child_labor_penalty = 1.0 if child_yesno == "Yes" else 0.0
        p.banned_chem_penalty = 1.0 if ban_yesno == "Yes" else 0.0

    st.session_state.policy = p


# -----------------------------
# Max Profit Agent tab (UNCHANGED)
# -----------------------------
with tab_profit:
    excel_path = Path(DEFAULT_XLSX_PATH)

    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0)
    with c2:
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0)
    with c3:
        suppliers_to_select = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1)
    with c4:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1)
    with c5:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1)

    if st.button("Optimize", type="primary", use_container_width=True):
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            cfg = MaxProfitConfig(
                last_n_users=int(last_n_users),
                capacity=int(capacity),
                suppliers_to_select=int(suppliers_to_select),
                price_per_match=float(price_per_match),
                min_utility=float(min_utility),
                output_flag=0,
            )

            agent = MaxProfitAgent(suppliers_df, users_df, st.session_state.policy, cfg)
            res = agent.solve()

            st.metric("Objective value", f"{res['objective_value']:.3f}")
            st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]) if res["chosen_suppliers"] else "None")
            st.write("Matched users:", f"{res['num_matched']} / {len(res['selected_users'])}")
            st.dataframe(res["matches"], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# Min Cost Agent tab (UNCHANGED)
# -----------------------------
with tab_mincost:
    excel_path = Path(DEFAULT_XLSX_PATH)

    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        min_utility_m = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="mincost_minutil")
    with c2:
        suppliers_to_select_m = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1, key="mincost_k")
    with c3:
        last_n_users_m = st.number_input("Last N users", min_value=1, value=6, step=1, key="mincost_lastn")
    with c4:
        capacity_m = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="mincost_cap")
    with c5:
        matches_to_make_m = st.number_input("Matches to make", min_value=0, value=6, step=1, key="mincost_m")

    if st.button("Optimize", type="primary", use_container_width=True, key="mincost_opt"):
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            cfg = MinCostConfig(
                last_n_users=int(last_n_users_m),
                capacity=int(capacity_m),
                matches_to_make=int(matches_to_make_m),
                suppliers_to_select=int(suppliers_to_select_m),
                min_utility=float(min_utility_m),
                output_flag=0,
            )

            agent = MinCostAgent(suppliers_df, users_df, st.session_state.policy, cfg)
            res = agent.solve()

            st.metric("Total cost (objective)", f"{res['objective_value']:.3f}")
            st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]) if res["chosen_suppliers"] else "None")
            st.write("Matched users:", f"{res['num_matched']} / {len(res['selected_users'])}")
            st.dataframe(res["matches"], use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(str(e))


# -----------------------------
# Best governmental policy tab (NEW, MILP)
# -----------------------------
with tab_best_policy:
    st.subheader("Best governmental policy")
    st.markdown(
        "**Maximize retailer utility** (retailer = MaxProfitAgent objective) **while government enforces world-goodness constraints**."
    )

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE or not _HAS_GUROBI:
        st.error("gurobipy is not available in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

    st.markdown(
        r"""
**Government MILP (policy selection):**

- Decision: choose exactly one policy from a candidate set \( \mathcal{P} \)
- \(x_p \in \{0,1\}\) indicates which policy is chosen.

\[
\max \sum_{p\in\mathcal{P}} U_p x_p \;+\; \epsilon \sum_{p\in\mathcal{P}} G_p x_p
\]
\[
\sum_{p} x_p = 1,\quad x_p \in \{0,1\}
\]
\[
\sum_p EnvTot_p x_p \le \overline{EnvAvg}\sum_p Match_p x_p
\]
\[
\sum_p SocTot_p x_p \le \overline{SocAvg}\sum_p Match_p x_p
\]
\[
\sum_p ChildTot_p x_p \le \overline{Child},\quad
\sum_p BanTot_p x_p \le \overline{Ban}
\]
\[
\sum_p Match_p x_p \ge \underline{M}
\]

Where \(U_p\) comes from the **inner retailer MILP** (MaxProfitAgent) solved under policy \(p\).
"""
    )

    st.divider()

    # ---- Controls
    a, b, c, d = st.columns(4)
    with a:
        total_users = st.number_input("Total users considered (fixed)", min_value=2, value=11, step=1, disabled=True)
    with b:
        suppliers_to_select_best = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1, key="bgp_k")
    with c:
        min_utility_best = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="bgp_minutil")
    with d:
        price_per_match_best = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="bgp_price")

    e1, e2, e3 = st.columns(3)
    with e1:
        n_candidates = st.slider("Number of candidate policies", min_value=10, max_value=300, value=80, step=10)
    with e2:
        seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="bgp_seed")
    with e3:
        penalty_max = st.number_input("Penalty max (child labor / banned chem)", min_value=0.0, value=100.0, step=10.0, key="bgp_penmax")

    include_current = st.checkbox("Include current policy from the Policy tab", value=True, key="bgp_inccur")

    st.divider()
    st.markdown("### World-goodness constraints (government)")
    f1, f2, f3 = st.columns(3)
    with f1:
        min_matches = st.number_input("Minimum matched users (M̲)", min_value=0, value=11, step=1, key="bgp_minm")
        eps_tie = st.number_input("Tie-break ε (sustainability)", min_value=0.0, value=0.001, step=0.001, format="%.3f")
    with f2:
        use_env = st.checkbox("Constrain avg environmental risk", value=True)
        env_avg_max = st.number_input("Max avg env risk (Env̄Avg)", value=3.0, step=0.1, disabled=not use_env)
        use_soc = st.checkbox("Constrain avg social risk", value=True)
        soc_avg_max = st.number_input("Max avg social risk (Soc̄Avg)", value=3.0, step=0.1, disabled=not use_soc)
    with f3:
        use_child = st.checkbox("Constrain child labor exposure", value=True)
        child_max = st.number_input("Max total child labor (Child̄)", value=0.0, step=1.0, disabled=not use_child)
        use_ban = st.checkbox("Constrain banned chemicals exposure", value=True)
        ban_max = st.number_input("Max total banned chem (Ban̄)", value=0.0, step=1.0, disabled=not use_ban)

    def make_candidate_policies(n: int, seed_: int, pen_max: float, include_current_: bool) -> list[dict]:
        rng = random.Random(int(seed_))
        mult_vals = [1, 2, 3, 4, 5]

        # anchors
        growth_anchor = {
            "env_mult": 1, "social_mult": 1, "cost_mult": 5,
            "strategic_mult": 5, "improvement_mult": 3, "low_quality_mult": 1,
            "child_labor_penalty": 0, "banned_chem_penalty": 0,
        }
        sustainable_anchor = {
            "env_mult": 5, "social_mult": 5, "cost_mult": 1,
            "strategic_mult": 2, "improvement_mult": 3, "low_quality_mult": 5,
            "child_labor_penalty": float(pen_max), "banned_chem_penalty": float(pen_max),
        }

        candidates = [growth_anchor, sustainable_anchor]

        if include_current_:
            cur = st.session_state.policy
            candidates.append(
                {
                    "env_mult": float(cur.env_mult),
                    "social_mult": float(cur.social_mult),
                    "cost_mult": float(cur.cost_mult),
                    "strategic_mult": float(cur.strategic_mult),
                    "improvement_mult": float(cur.improvement_mult),
                    "low_quality_mult": float(cur.low_quality_mult),
                    "child_labor_penalty": float(cur.child_labor_penalty),
                    "banned_chem_penalty": float(cur.banned_chem_penalty),
                }
            )

        # random
        for _ in range(max(0, int(n) - len(candidates))):
            candidates.append(
                {
                    "env_mult": rng.choice(mult_vals),
                    "social_mult": rng.choice(mult_vals),
                    "cost_mult": rng.choice(mult_vals),
                    "strategic_mult": rng.choice(mult_vals),
                    "improvement_mult": rng.choice(mult_vals),
                    "low_quality_mult": rng.choice(mult_vals),
                    "child_labor_penalty": rng.randint(0, int(pen_max)),
                    "banned_chem_penalty": rng.randint(0, int(pen_max)),
                }
            )

        # de-dup
        seen = set()
        uniq = []
        for d in candidates:
            key = tuple(sorted((k, float(v)) for k, v in d.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        return uniq

    def eval_policy(
        suppliers_df: pd.DataFrame,
        users_df: pd.DataFrame,
        pdict: dict,
        k: int,
        price: float,
        min_util: float,
    ) -> dict:
        pol = Policy()
        for kk, vv in pdict.items():
            setattr(pol, kk, float(vv))

        cfg = MaxProfitConfig(
            last_n_users=11,
            capacity=11,
            suppliers_to_select=int(k),
            price_per_match=float(price),
            min_utility=float(min_util),
            output_flag=0,
        )

        res = MaxProfitAgent(suppliers_df, users_df, pol, cfg).solve()
        matches = res["matches"].copy()

        # exposures from chosen matches
        if len(matches) > 0:
            s = suppliers_df.set_index("supplier_id")
            matches["env_risk"] = matches["supplier_id"].map(lambda sid: float(s.loc[str(sid), "env_risk"]))
            matches["social_risk"] = matches["supplier_id"].map(lambda sid: float(s.loc[str(sid), "social_risk"]))
            matches["child_labor"] = matches["supplier_id"].map(lambda sid: float(s.loc[str(sid), "child_labor"]))
            matches["banned_chem"] = matches["supplier_id"].map(lambda sid: float(s.loc[str(sid), "banned_chem"]))
            matches["low_quality"] = matches["supplier_id"].map(lambda sid: float(s.loc[str(sid), "low_quality"]))

            env_tot = float(matches["env_risk"].sum())
            soc_tot = float(matches["social_risk"].sum())
            child_tot = float(matches["child_labor"].sum())
            ban_tot = float(matches["banned_chem"].sum())
            lq_tot = float(matches["low_quality"].sum())
        else:
            env_tot = soc_tot = child_tot = ban_tot = lq_tot = 0.0

        m = int(res["num_matched"]) if res.get("num_matched") is not None else 0
        env_avg = env_tot / m if m > 0 else 0.0
        soc_avg = soc_tot / m if m > 0 else 0.0
        lq_avg = lq_tot / m if m > 0 else 0.0

        # world goodness score (higher better) — only for tie-break
        goodness = -(
            float(WORLD_GOODNESS_WEIGHTS["w_env"]) * env_avg
            + float(WORLD_GOODNESS_WEIGHTS["w_soc"]) * soc_avg
            + float(WORLD_GOODNESS_WEIGHTS["w_child"]) * child_tot
            + float(WORLD_GOODNESS_WEIGHTS["w_ban"]) * ban_tot
            + float(WORLD_GOODNESS_WEIGHTS["w_lq"]) * lq_avg
        )

        return {
            "policy": pdict,
            "profit_obj": float(res["objective_value"]),
            "matched": m,
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

    if st.button("Solve Government MILP", type="primary", use_container_width=True):
        suppliers_df, users_df = load_supplier_user_tables(excel_path)

        policy_dicts = make_candidate_policies(int(n_candidates), int(seed), float(penalty_max), bool(include_current))

        rows = []
        with st.spinner("Evaluating candidates (inner retailer MILP per policy)..."):
            for pdict in policy_dicts:
                rows.append(
                    eval_policy(
                        suppliers_df,
                        users_df,
                        pdict,
                        k=int(suppliers_to_select_best),
                        price=float(price_per_match_best),
                        min_util=float(min_utility_best),
                    )
                )

        df = pd.DataFrame(
            [
                {
                    **r["policy"],
                    "profit_obj": r["profit_obj"],
                    "matched": r["matched"],
                    "chosen_suppliers": r["chosen_suppliers"],
                    "env_avg": r["env_avg"],
                    "soc_avg": r["soc_avg"],
                    "lq_avg": r["lq_avg"],
                    "child_tot": r["child_tot"],
                    "ban_tot": r["ban_tot"],
                    "goodness": r["goodness"],
                }
                for r in rows
            ]
        )

        # Unconstrained best (pure maximize retailer utility)
        df_growth = df.sort_values(["profit_obj", "matched"], ascending=[False, False]).reset_index(drop=True)
        best_growth_idx = int(df_growth.index[0]) if len(df_growth) else None
        best_growth = rows[df_growth.index[0]] if len(df_growth) else None

        # Government MILP selection among candidates
        m = gp.Model("GovernmentPolicyMILP")
        m.Params.OutputFlag = 0

        J = list(range(len(rows)))
        x = m.addVars(J, vtype=GRB.BINARY, name="x")

        # choose exactly one
        m.addConstr(gp.quicksum(x[j] for j in J) == 1, name="choose_one")

        # minimum matches
        if int(min_matches) > 0:
            m.addConstr(gp.quicksum(rows[j]["matched"] * x[j] for j in J) >= int(min_matches), name="min_matches")

        # avg constraints: total <= max_avg * matches
        if use_env:
            m.addConstr(
                gp.quicksum(rows[j]["env_tot"] * x[j] for j in J)
                <= float(env_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
                name="env_avg",
            )
        if use_soc:
            m.addConstr(
                gp.quicksum(rows[j]["soc_tot"] * x[j] for j in J)
                <= float(soc_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
                name="soc_avg",
            )

        if use_child:
            m.addConstr(gp.quicksum(rows[j]["child_tot"] * x[j] for j in J) <= float(child_max), name="child_tot")
        if use_ban:
            m.addConstr(gp.quicksum(rows[j]["ban_tot"] * x[j] for j in J) <= float(ban_max), name="ban_tot")

        # Objective: maximize retailer utility + epsilon * goodness (tie-break)
        m.setObjective(
            gp.quicksum(rows[j]["profit_obj"] * x[j] for j in J) + float(eps_tie) * gp.quicksum(rows[j]["goodness"] * x[j] for j in J),
            GRB.MAXIMIZE,
        )

        m.optimize()

        if m.Status != GRB.OPTIMAL:
            st.error("Government MILP found no feasible policy under the chosen world constraints. Relax constraints or increase candidates.")
            st.dataframe(df_growth.head(25), use_container_width=True, hide_index=True)
            st.stop()

        chosen_j = max(J, key=lambda j: x[j].X)
        best_gov = rows[chosen_j]

        # ---- Display
        st.markdown("### Results")
        left, right = st.columns(2)

        with left:
            st.markdown("#### Maximize retailer utility (no world constraints)")
            if best_growth is None:
                st.error("No candidates evaluated.")
            else:
                st.metric("Retailer utility (objective)", f"{best_growth['profit_obj']:.3f}")
                st.write("Matched users:", best_growth["matched"])
                st.write("Chosen suppliers:", best_growth["chosen_suppliers"] or "None")
                st.write("Env avg:", f"{best_growth['env_avg']:.3f}", "| Social avg:", f"{best_growth['soc_avg']:.3f}")
                st.write("Child total:", f"{best_growth['child_tot']:.3f}", "| Banned total:", f"{best_growth['ban_tot']:.3f}")
                st.write("Policy:")
                st.json(best_growth["policy"])
                st.dataframe(best_growth["matches_df"], use_container_width=True, hide_index=True)

        with right:
            st.markdown("#### Government MILP: maximize retailer utility + enforce world goodness")
            st.metric("Retailer utility (objective)", f"{best_gov['profit_obj']:.3f}")
            st.write("Matched users:", best_gov["matched"])
            st.write("Chosen suppliers:", best_gov["chosen_suppliers"] or "None")
            st.write("Env avg:", f"{best_gov['env_avg']:.3f}", "| Social avg:", f"{best_gov['soc_avg']:.3f}")
            st.write("Child total:", f"{best_gov['child_tot']:.3f}", "| Banned total:", f"{best_gov['ban_tot']:.3f}")
            st.write("Policy:")
            st.json(best_gov["policy"])
            st.dataframe(best_gov["matches_df"], use_container_width=True, hide_index=True)

        st.markdown("### Evaluated candidates (top by retailer utility)")
        st.dataframe(df_growth.head(50), use_container_width=True, hide_index=True)
