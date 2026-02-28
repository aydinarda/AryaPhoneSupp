import streamlit as st
from pathlib import Path

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    Policy,
    MaxProfitConfig,
    MaxProfitAgent,
    MaxUtilConfig,
    MaxUtilAgent,
    MinCostConfig,
    MinCostAgent,
    load_supplier_user_tables,
    GUROBI_AVAILABLE,
)

st.set_page_config(page_title="Arya Case — Supplier Selection", layout="wide")

# Hide Streamlit chrome for classroom use
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

# -------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------
if "policy" not in st.session_state:
    st.session_state.policy = Policy()

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------
tab_policy, tab_select = st.tabs(["Policy (fixed)", "Supplier Selection (Gurobi)"])

# -------------------------------------------------------------------
# Policy tab
# -------------------------------------------------------------------
with tab_policy:
    st.markdown("### Policy (fixed)")
    st.caption("Choose a policy once. All supplier-selection models are solved **under this fixed policy**.")

    p: Policy = st.session_state.policy

    # Discrete policy levels (Low/Mid/High -> 1/5/10)
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

    st.info(
        "**Interpretation in the models**\n\n"
        "- Multipliers scale how much users care about each attribute in utility.\n"
        "- `Cost multiplier` also scales the production cost used in profit / min-cost objectives.\n"
        "- `Child labor` and `Banned chemicals` are **hard bans** when set to Yes.\n"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        env_level = st.radio(
            "Environmental multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.env_mult)),
            horizontal=True,
        )
        social_level = st.radio(
            "Social multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.social_mult)),
            horizontal=True,
        )
        cost_level = st.radio(
            "Cost multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.cost_mult)),
            horizontal=True,
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
        )
        improvement_level = st.radio(
            "Improvement multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.improvement_mult)),
            horizontal=True,
        )
        lowq_level = st.radio(
            "Low-quality multiplier",
            options=_LEVEL_OPTIONS,
            index=_LEVEL_OPTIONS.index(_level_from_value(p.low_quality_mult)),
            horizontal=True,
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
        )
        banned_choice = st.radio(
            "Banned chemicals ban",
            options=yn_options,
            index=yn_options.index(_yn_from_value(p.banned_chem_penalty)),
            horizontal=True,
        )

        p.child_labor_penalty = 1.0 if child_choice == "Yes" else 0.0
        p.banned_chem_penalty = 1.0 if banned_choice == "Yes" else 0.0

    st.session_state.policy = p

    with st.expander("Math definition used in the models", expanded=False):
        st.markdown(
            r"""
We use an assignment-style MILP.

**Sets**
- Suppliers \(i \in S\)
- Users \(u \in U\)

**Decision variables**
- \(y_i \in \{0,1\}\): supplier \(i\) is selected
- \(z_{i,u} \in \{0,1\}\): user \(u\) is matched to supplier \(i\)

**Core constraints**
- Select exactly \(K\) suppliers: \(\sum_i y_i = K\)
- Each user matched at most once: \(\sum_i z_{i,u} \le 1\)
- Linking: \(z_{i,u} \le y_i\)
- Total matches: \(\sum_{i,u} z_{i,u} \le \text{capacity}\)

**Utility threshold**
If \(z_{i,u}=1\), then \(\text{utility}(i,u) \ge \text{min\_utility}\) (implemented with Big‑M).

**Policy hard bans**
If `Child labor ban=Yes` then any supplier with `child_labor=1` must have \(y_i=0\).
Similarly for banned chemicals.

**Objectives (choose one in the next tab)**
- Max Profit: \(\max \sum_{i,u} (P - \text{cost\_mult}\cdot \text{cost\_score}_i)\, z_{i,u}\)
- Max Utility: \(\max \sum_{i,u} \text{utility}(i,u)\, z_{i,u}\)
- Min Cost: \(\min \sum_{i,u} (\text{cost\_mult}\cdot \text{cost\_score}_i)\, z_{i,u}\) with \(\sum z = \text{matches\_to\_make}\)
"""
        )

# -------------------------------------------------------------------
# Supplier Selection tab
# -------------------------------------------------------------------
with tab_select:
    st.markdown("### Supplier Selection (Gurobi)")
    st.caption("Solve a MILP to choose the best supplier(s) under the fixed policy.")

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    objective = st.radio(
        "Objective",
        options=["Max Profit", "Max Utility", "Min Cost"],
        horizontal=True,
        help="Pick what 'best supplier(s)' means for this run.",
    )

    st.divider()

    # Shared controls
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        suppliers_to_select = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1)
    with c2:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1)
    with c3:
        capacity = st.number_input("Capacity (max matches)", min_value=0, value=6, step=1)
    with c4:
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0)

    # Objective-specific controls
    price_per_match = None
    matches_to_make = None
    min_matches = 0

    if objective == "Max Profit":
        c5, c6 = st.columns(2)
        with c5:
            price_per_match = st.number_input("Selling price per match (P)", min_value=0.0, value=100.0, step=5.0)
        with c6:
            min_matches = st.number_input(
                "Minimum matches (optional)",
                min_value=0,
                value=0,
                step=1,
                help="Set 0 to allow the model to match nobody if the threshold is too strict.",
            )

    if objective == "Min Cost":
        matches_to_make = st.number_input(
            "Matches to make (exact)",
            min_value=0,
            value=min(int(capacity), 6),
            step=1,
            help="Min-cost requires an exact number of matches (must be ≤ capacity).",
        )

    run = st.button("Optimize", type="primary", use_container_width=True)

    if run:
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            if objective == "Max Profit":
                cfg = MaxProfitConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(suppliers_to_select),
                    price_per_match=float(price_per_match),
                    min_utility=float(min_utility),
                    min_matches=int(min_matches),
                    output_flag=0,
                )
                agent = MaxProfitAgent(suppliers_df, users_df, st.session_state.policy, cfg)
                res = agent.solve()
                st.metric("Profit (objective)", f"{res['objective_value']:.3f}")

            elif objective == "Max Utility":
                cfg = MaxUtilConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(suppliers_to_select),
                    min_utility=float(min_utility),
                    output_flag=0,
                )
                agent = MaxUtilAgent(suppliers_df, users_df, st.session_state.policy, cfg)
                res = agent.solve()
                st.metric("Total utility (objective)", f"{res['objective_value']:.3f}")

            else:  # Min Cost
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

            chosen = res.get("chosen_suppliers", [])
            st.write("Chosen suppliers:", ", ".join(chosen) if chosen else "None")
            st.write("Matched users:", f"{res.get('num_matched', 0)} / {len(res.get('selected_users', []))}")

            st.dataframe(res.get("matches", []), use_container_width=True, hide_index=True)

            with st.expander("Policy used", expanded=False):
                st.json(res.get("policy", {}))

        except Exception as e:
            st.error(str(e))
