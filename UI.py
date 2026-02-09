\
import random
from pathlib import Path

import pandas as pd
import streamlit as st

from BestGovPol import render_best_governmental_policy
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
    ["Policy", "Max Profit Agent", "Min Cost Agent", "Best governmental policy"]
)

# -----------------------------
# Policy tab (UNCHANGED)
# -----------------------------
with tab_policy:
    p = st.session_state.policy

    c1, c2, c3 = st.columns(3)
    with c1:
        p.env_mult = st.number_input("Environmental multiplier", min_value=0.0, value=float(p.env_mult), step=0.1)
        p.social_mult = st.number_input("Social multiplier", min_value=0.0, value=float(p.social_mult), step=0.1)
        p.cost_mult = st.number_input("Cost multiplier", min_value=0.0, value=float(p.cost_mult), step=0.1)

    with c2:
        p.strategic_mult = st.number_input("Strategic multiplier", min_value=0.0, value=float(p.strategic_mult), step=0.1)
        p.improvement_mult = st.number_input("Improvement multiplier", min_value=0.0, value=float(p.improvement_mult), step=0.1)
        p.low_quality_mult = st.number_input("Low-quality multiplier", min_value=0.0, value=float(p.low_quality_mult), step=0.1)

    with c3:
        p.child_labor_penalty = st.number_input("Child labor penalty", min_value=0.0, value=float(p.child_labor_penalty), step=1.0)
        p.banned_chem_penalty = st.number_input("Banned chemicals penalty", min_value=0.0, value=float(p.banned_chem_penalty), step=1.0)

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
# Best governmental policy tab (NOW IMPORTED)
# -----------------------------
with tab_best_policy:
    render_best_governmental_policy()
