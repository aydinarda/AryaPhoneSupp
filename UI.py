import streamlit as st
from pathlib import Path

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    Policy,
    MaxProfitConfig,
    MaxProfitAgent,
    MinCostConfig,
    MinCostAgent,
    load_supplier_user_tables,
    GUROBI_AVAILABLE,
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

tab_policy, tab_profit, tab_mincost, tab_best_policy = st.tabs(["Policy", "Max Profit Agent", "Min Cost Agent", "Best governmental policy"])

with tab_policy:
    p = st.session_state.policy

    LEVELS = {"Low": 1.0, "Mid": 3.0, "High": 5.0}
    _LEVEL_OPTIONS = ["Low", "Mid", "High"]

    def _level_from_value(v) -> str:
        """Pick the closest level to the existing numeric value (robust to older saves)."""
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
            "Child labor penalty",
            options=yn_options,
            index=yn_options.index(_yn_from_value(p.child_labor_penalty)),
            horizontal=True,
            key="policy_child_labor",
        )
        banned_choice = st.radio(
            "Banned chemicals penalty",
            options=yn_options,
            index=yn_options.index(_yn_from_value(p.banned_chem_penalty)),
            horizontal=True,
            key="policy_banned_chem",
        )

        p.child_labor_penalty = 1.0 if child_choice == "Yes" else 0.0
        p.banned_chem_penalty = 1.0 if banned_choice == "Yes" else 0.0

    st.session_state.policy = p


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
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="mincost_minutil")
    with c2:
        suppliers_to_select = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1, key="mincost_k")
    with c3:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="mincost_lastn")
    with c4:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="mincost_cap")
    with c5:
        matches_to_make = st.number_input("Matches to make", min_value=0, value=6, step=1, key="mincost_m")

    if st.button("Optimize", type="primary", use_container_width=True, key="mincost_opt"):
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            cfg = MinCostConfig(
                last_n_users=int(last_n_users),
                capacity=int(capacity),
                matches_to_make=int(matches_to_make),
                suppliers_to_select=int(suppliers_to_select),
                min_utility=float(min_utility),
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


# -------------------------------
# Best governmental policy (imported)
# -------------------------------
with tab_best_policy:
    from BestGovPol import render_best_governmental_policy
    render_best_governmental_policy()
