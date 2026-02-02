from __future__ import annotations

import streamlit as st
import pandas as pd

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    Policy,
    MaxProfitConfig,
    MaxProfitAgent,
    load_supplier_user_tables,
)

st.set_page_config(page_title="Arya Case — Policy Playground", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1200px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def _load_tables(xlsx_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_supplier_user_tables(xlsx_path=xlsx_path)


def _policy_widget() -> Policy:
    policy_options = [0, 1, 2, 3, 4, 5]

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Policy multipliers")
        env = st.selectbox("Environmental multiplier", policy_options, index=1)
        social = st.selectbox("Social multiplier", policy_options, index=1)
        cost = st.selectbox("Cost multiplier", policy_options, index=1)
        strategic = st.selectbox("Strategic multiplier", policy_options, index=1)
        improvement = st.selectbox("Improvement multiplier", policy_options, index=1)
        low_q = st.selectbox("Low-quality multiplier", policy_options, index=1)

    with col2:
        st.subheader("Policy penalties")
        child = st.selectbox("Child labor penalty", policy_options, index=0)
        banned = st.selectbox("Banned chemicals penalty", policy_options, index=0)

    return Policy(
        env_mult=float(env),
        social_mult=float(social),
        cost_mult=float(cost),
        strategic_mult=float(strategic),
        improvement_mult=float(improvement),
        low_quality_mult=float(low_q),
        child_labor_penalty=float(child),
        banned_chem_penalty=float(banned),
    )


tabs = st.tabs(["Policy", "Max Profit Agent"])

with tabs[0]:
    policy = _policy_widget()

with tabs[1]:
    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed in this environment. Add it to requirements.txt (and ensure a valid license).")
        st.stop()

    # Load data
    try:
        suppliers_df, users_df = _load_tables(DEFAULT_XLSX_PATH)
        data_ok = True
    except Exception:
        data_ok = False
        suppliers_df, users_df = pd.DataFrame(), pd.DataFrame()

    if not data_ok:
        st.error(f"Could not load '{DEFAULT_XLSX_PATH}'. Place it next to UI.py in your Streamlit repo.")
        st.stop()

    st.subheader("Optimization settings")

    colA, colB, colC = st.columns([1, 1, 1], gap="large")

    with colA:
        price = st.selectbox("Selling price (P)", [50.0, 75.0, 100.0, 125.0, 150.0], index=2)
        k = st.selectbox("Suppliers to select (K)", [1, 2, 3], index=0)

    with colB:
        last_n = st.selectbox("Optimize for last N users", [6, 8, 10], index=0)
        capacity = st.selectbox("Capacity (max matches)", [2, 4, 6, 8, 10], index=2)

    with colC:
        min_utility = st.selectbox("Minimum utility (if matched)", [0.0, 0.5, 1.0, 1.5, 2.0], index=0)
        show_log = st.selectbox("Solver log", ["Off", "On"], index=0)

    run = st.button("Optimize", type="primary", disabled=not data_ok)

    if run:
        cfg = MaxProfitConfig(
            last_n_users=int(last_n),
            capacity=int(capacity),
            suppliers_to_select=int(k),
            price_per_match=float(price),
            min_utility=float(min_utility),
            output_flag=1 if show_log == "On" else 0,
        )

        agent = MaxProfitAgent(suppliers_df, users_df, policy=policy, cfg=cfg)
        agent.build()
        sol = agent.solve()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Matched users", sol["num_matched"])
        m2.metric("Objective (total margin)", f"{sol['objective_value']:.3f}")
        m3.metric("Selected suppliers", str(len(sol['chosen_suppliers'])))
        m4.metric("Users in scope", str(len(sol["selected_users"])))

        st.markdown("**Selected suppliers**")
        st.write(sol["chosen_suppliers"])

        st.markdown("**Matches**")
        st.dataframe(sol["matches"], use_container_width=True)
