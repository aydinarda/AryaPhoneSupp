import streamlit as st
from pathlib import Path

# For the sandbox downloads, the updated agent module is named `MinCostAgent_updated.py`.
# In your repo, you can either:
#   A) replace MinCostAgent.py with MinCostAgent_updated.py content, OR
#   B) keep the filename MinCostAgent_updated.py and let this import pick it up.
try:
    from MinCostAgent_updated import (
        DEFAULT_XLSX_PATH,
        Policy,
        MaxProfitConfig,
        MaxProfitAgent,
        MinCostConfig,
        MinCostAgent,
        load_supplier_user_tables,
        GUROBI_AVAILABLE,
    )
except Exception:  # pragma: no cover
    from MinCostAgent import (  # type: ignore
        DEFAULT_XLSX_PATH,
        Policy,
        MaxProfitConfig,
        MaxProfitAgent,
        MinCostConfig,
        MinCostAgent,
        load_supplier_user_tables,
        GUROBI_AVAILABLE,
    )

st.set_page_config(page_title="Arya Case — Supplier Selection", layout="wide")

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

# ----------------------------
# Session state
# ----------------------------
if "policy" not in st.session_state:
    st.session_state.policy = Policy()

if "manual_suppliers" not in st.session_state:
    st.session_state.manual_suppliers = []

if "manual_k" not in st.session_state:
    st.session_state.manual_k = 1

# ----------------------------
# Tabs
# ----------------------------
tab_policy, tab_manual, tab_profit, tab_mincost = st.tabs(
    ["Policy", "Manual supplier choice", "Max Profit: manual vs optimal", "Min Cost: manual vs optimal"]
)

# ----------------------------
# Policy (same place as before, but levels are 1/5/10)
# ----------------------------
with tab_policy:
    st.markdown("### Policy (fixed)")
    st.caption("Policy is chosen here. Students then choose suppliers manually and compare against optimal selection.")

    p = st.session_state.policy

    LEVELS = {"Low": 1.0, "Mid": 5.0, "High": 10.0}
    _LEVEL_OPTIONS = ["Low", "Mid", "High"]

    def _level_from_value(v) -> str:
        try:
            v = float(v)
        except Exception:
            return "Mid"
        return min(_LEVEL_OPTIONS, key=lambda k: abs(LEVELS[k] - v))

    def _yn_from_value(v) -> str:
        try:
            return "Yes" if float(v) >= 0.5 else "No"
        except Exception:
            return "No"

    c1, c2, c3 = st.columns(3)

    with c1:
        env_level = st.radio(
            "Environmental multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.env_mult)),
            horizontal=True,
            key="policy_env_level",
        )
        social_level = st.radio(
            "Social multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.social_mult)),
            horizontal=True,
            key="policy_social_level",
        )
        cost_level = st.radio(
            "Cost multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.cost_mult)),
            horizontal=True,
            key="policy_cost_level",
        )

        p.env_mult = LEVELS[env_level]
        p.social_mult = LEVELS[social_level]
        p.cost_mult = LEVELS[cost_level]

    with c2:
        strategic_level = st.radio(
            "Strategic multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.strategic_mult)),
            horizontal=True,
            key="policy_strategic_level",
        )
        improvement_level = st.radio(
            "Improvement multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.improvement_mult)),
            horizontal=True,
            key="policy_improvement_level",
        )
        lowq_level = st.radio(
            "Low-quality multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.low_quality_mult)),
            horizontal=True,
            key="policy_lowq_level",
        )

        p.strategic_mult = LEVELS[strategic_level]
        p.improvement_mult = LEVELS[improvement_level]
        p.low_quality_mult = LEVELS[lowq_level]

    with c3:
        yn_options = ["Yes", "No"]

        child_choice = st.radio(
            "Child labor ban",
            options=yn_options,
            index=yn_options.index(_yn_from_value(p.child_labor_penalty)),
            horizontal=True,
            key="policy_child_labor",
        )
        banned_choice = st.radio(
            "Banned chemicals ban",
            options=yn_options,
            index=yn_options.index(_yn_from_value(p.banned_chem_penalty)),
            horizontal=True,
            key="policy_banned_chem",
        )

        p.child_labor_penalty = 1.0 if child_choice == "Yes" else 0.0
        p.banned_chem_penalty = 1.0 if banned_choice == "Yes" else 0.0

    st.session_state.policy = p

    st.info(
        "Next: go to **Manual supplier choice** and pick suppliers by hand.\n"
        "Then check the two comparison tabs to see how far you are from the optimum."
    )

# ----------------------------
# Manual supplier choice
# ----------------------------
with tab_manual:
    st.markdown("### Manual supplier choice")
    st.caption("Students pick suppliers by hand (no optimization). We'll later compare against the optimal selection.")

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    suppliers_df, _ = load_supplier_user_tables(excel_path)
    suppliers = suppliers_df["supplier_id"].astype(str).tolist()

    # Quick view table (helps manual choice)
    with st.expander("See supplier table", expanded=True):
        show_cols = ["supplier_id", "env_risk", "social_risk", "cost_score", "strategic", "improvement", "child_labor", "banned_chem", "low_quality"]
        st.dataframe(suppliers_df[show_cols], use_container_width=True, hide_index=True)

    k = st.number_input("How many suppliers to pick (K)?", min_value=1, value=int(st.session_state.manual_k), step=1)
    st.session_state.manual_k = int(k)

    picked = st.multiselect(
        "Pick suppliers (exactly K)",
        options=suppliers,
        default=st.session_state.manual_suppliers,
    )

    # Warn if policy bans make some picks illegal
    banned_msgs = []
    if len(picked):
        p = st.session_state.policy
        sdf = suppliers_df.set_index("supplier_id")
        if float(p.child_labor_penalty) >= 0.5:
            bad = [sid for sid in picked if float(sdf.loc[sid, "child_labor"]) >= 0.5]
            if bad:
                banned_msgs.append(f"Child-labor ban active → these picks violate policy: {bad}")
        if float(p.banned_chem_penalty) >= 0.5:
            bad = [sid for sid in picked if float(sdf.loc[sid, "banned_chem"]) >= 0.5]
            if bad:
                banned_msgs.append(f"Banned-chemicals ban active → these picks violate policy: {bad}")

    if banned_msgs:
        st.warning("\n".join(banned_msgs))

    if len(picked) != int(k):
        st.warning(f"Please select exactly K={int(k)} suppliers. Currently selected: {len(picked)}.")
    else:
        st.success(f"Ready: selected {len(picked)} supplier(s). Click **Save selection**.")

    if st.button("Save selection", type="primary", use_container_width=True):
        if len(picked) != int(k):
            st.error(f"Cannot save: selection must contain exactly K={int(k)} suppliers.")
        else:
            st.session_state.manual_suppliers = picked
            st.success("Saved. Now go to the comparison tabs.")

# ----------------------------
# Max Profit comparison
# ----------------------------
with tab_profit:
    st.markdown("### Max Profit: manual vs optimal")
    st.caption("We compute the best matching under the manually chosen suppliers, then compare to the true optimum (Gurobi).")

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0)
    with c2:
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0)
    with c3:
        k = st.number_input("Suppliers to select (K)", min_value=1, value=int(st.session_state.manual_k), step=1, key="profit_k")
    with c4:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="profit_lastn")
    with c5:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="profit_cap")

    st.session_state.manual_k = int(k)

    run = st.button("Run comparison", type="primary", use_container_width=True, key="profit_compare")

    if run:
        suppliers_df, users_df = load_supplier_user_tables(excel_path)

        left, right = st.columns(2)

        # -------- manual evaluation --------
        with left:
            st.markdown("#### Manual selection result")
            manual = list(st.session_state.manual_suppliers or [])
            if len(manual) != int(k):
                st.error("No saved manual selection (or wrong size). Go to **Manual supplier choice** and save exactly K suppliers.")
            else:
                try:
                    cfg_manual = MaxProfitConfig(
                        last_n_users=int(last_n_users),
                        capacity=int(capacity),
                        suppliers_to_select=int(k),
                        price_per_match=float(price_per_match),
                        min_utility=float(min_utility),
                        output_flag=0,
                        fixed_suppliers=manual,
                    )
                    agent_manual = MaxProfitAgent(suppliers_df, users_df, st.session_state.policy, cfg_manual)
                    res_m = agent_manual.solve()

                    st.metric("Manual profit", f"{res_m['objective_value']:.3f}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]) if res_m["chosen_suppliers"] else "None")
                    st.write("Matched users:", f"{res_m['num_matched']} / {len(res_m['selected_users'])}")
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)
                    st.session_state._last_profit_manual = res_m  # for diff
                except Exception as e:
                    st.error(str(e))
                    st.session_state._last_profit_manual = None

        # -------- optimal evaluation --------
        with right:
            st.markdown("#### Optimal selection result (Gurobi)")
            try:
                cfg_opt = MaxProfitConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(k),
                    price_per_match=float(price_per_match),
                    min_utility=float(min_utility),
                    output_flag=0,
                    fixed_suppliers=None,
                )
                agent_opt = MaxProfitAgent(suppliers_df, users_df, st.session_state.policy, cfg_opt)
                res_o = agent_opt.solve()

                st.metric("Optimal profit", f"{res_o['objective_value']:.3f}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]) if res_o["chosen_suppliers"] else "None")
                st.write("Matched users:", f"{res_o['num_matched']} / {len(res_o['selected_users'])}")
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
                st.session_state._last_profit_opt = res_o
            except Exception as e:
                st.error(str(e))
                st.session_state._last_profit_opt = None

        # -------- comparison summary --------
        rm = st.session_state.get("_last_profit_manual")
        ro = st.session_state.get("_last_profit_opt")
        if rm and ro:
            st.divider()
            gap = float(ro["objective_value"]) - float(rm["objective_value"])
            st.markdown("#### Comparison")
            cA, cB, cC = st.columns(3)
            cA.metric("Optimal − Manual", f"{gap:.3f}")
            cB.metric("Manual / Optimal", f"{(float(rm['objective_value'])/float(ro['objective_value'])):.3f}" if float(ro["objective_value"]) != 0 else "—")
            cC.metric("Manual matched", f"{rm['num_matched']}")

# ----------------------------
# Min Cost comparison
# ----------------------------
with tab_mincost:
    st.markdown("### Min Cost: manual vs optimal")
    st.caption("We compute the cheapest matching under the manually chosen suppliers, then compare to the true optimum (Gurobi).")

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="mincost_minutil")
    with c2:
        k = st.number_input("Suppliers to select (K)", min_value=1, value=int(st.session_state.manual_k), step=1, key="mincost_k")
    with c3:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="mincost_lastn")
    with c4:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="mincost_cap")
    with c5:
        matches_to_make = st.number_input("Matches to make (exact)", min_value=0, value=6, step=1, key="mincost_m")
    with c6:
        _ = st.empty()

    st.session_state.manual_k = int(k)

    run = st.button("Run comparison", type="primary", use_container_width=True, key="mincost_compare")

    if run:
        suppliers_df, users_df = load_supplier_user_tables(excel_path)

        left, right = st.columns(2)

        # -------- manual evaluation --------
        with left:
            st.markdown("#### Manual selection result")
            manual = list(st.session_state.manual_suppliers or [])
            if len(manual) != int(k):
                st.error("No saved manual selection (or wrong size). Go to **Manual supplier choice** and save exactly K suppliers.")
            else:
                try:
                    cfg_manual = MinCostConfig(
                        last_n_users=int(last_n_users),
                        capacity=int(capacity),
                        matches_to_make=int(matches_to_make),
                        suppliers_to_select=int(k),
                        min_utility=float(min_utility),
                        output_flag=0,
                        fixed_suppliers=manual,
                    )
                    agent_manual = MinCostAgent(suppliers_df, users_df, st.session_state.policy, cfg_manual)
                    res_m = agent_manual.solve()

                    st.metric("Manual total cost", f"{res_m['objective_value']:.3f}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]) if res_m["chosen_suppliers"] else "None")
                    st.write("Matched users:", f"{res_m['num_matched']} / {len(res_m['selected_users'])}")
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)
                    st.session_state._last_mincost_manual = res_m
                except Exception as e:
                    st.error(str(e))
                    st.session_state._last_mincost_manual = None

        # -------- optimal evaluation --------
        with right:
            st.markdown("#### Optimal selection result (Gurobi)")
            try:
                cfg_opt = MinCostConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    matches_to_make=int(matches_to_make),
                    suppliers_to_select=int(k),
                    min_utility=float(min_utility),
                    output_flag=0,
                    fixed_suppliers=None,
                )
                agent_opt = MinCostAgent(suppliers_df, users_df, st.session_state.policy, cfg_opt)
                res_o = agent_opt.solve()

                st.metric("Optimal total cost", f"{res_o['objective_value']:.3f}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]) if res_o["chosen_suppliers"] else "None")
                st.write("Matched users:", f"{res_o['num_matched']} / {len(res_o['selected_users'])}")
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
                st.session_state._last_mincost_opt = res_o
            except Exception as e:
                st.error(str(e))
                st.session_state._last_mincost_opt = None

        # -------- comparison summary --------
        rm = st.session_state.get("_last_mincost_manual")
        ro = st.session_state.get("_last_mincost_opt")
        if rm and ro:
            st.divider()
            # For cost, lower is better
            gap = float(rm["objective_value"]) - float(ro["objective_value"])  # manual - optimal
            st.markdown("#### Comparison")
            cA, cB, cC = st.columns(3)
            cA.metric("Manual − Optimal (cost)", f"{gap:.3f}")
            cB.metric("Manual / Optimal", f"{(float(rm['objective_value'])/float(ro['objective_value'])):.3f}" if float(ro["objective_value"]) != 0 else "—")
            cC.metric("Manual matched", f"{rm['num_matched']}")
