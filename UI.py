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
# Best governmental policy (NEW)
# -------------------------------
with tab_best_policy:
    st.markdown("### Best governmental policy")
    st.caption(
        "This page searches for two *policy presets*:\n"
        "- **Best Growth Policy**: maximizes the Max Profit Agent objective.\n"
        "- **Best Sustainable Policy**: maximizes the Max Utility Agent objective.\n"
        "Both are evaluated on the same user/supplier data and constraints."
    )

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

    # Import MaxUtilAgent lazily so the rest of the app doesn't depend on it unless this tab is used.
    try:
        from MaxUtilAgent import MaxUtilConfig, MaxUtilAgent
        _HAS_MAXUTIL = True
    except Exception:
        _HAS_MAXUTIL = False

    import random
    import pandas as pd

    st.markdown("#### Evaluation setup")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        last_n_users = st.number_input("Last N users (evaluated)", min_value=1, value=11, step=1, key="bgp_lastn")
    with c2:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="bgp_cap")
    with c3:
        suppliers_to_select = st.number_input("Suppliers to select (K)", min_value=1, value=1, step=1, key="bgp_k")
    with c4:
        price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="bgp_price")
    with c5:
        min_utility = st.number_input("Minimum utility threshold", value=0.0, step=1.0, key="bgp_minutil")

    st.markdown("#### Search space")
    s1, s2, s3 = st.columns(3)
    with s1:
        n_candidates = st.number_input("Number of candidate policies", min_value=10, value=200, step=10, key="bgp_ncand")
    with s2:
        seed = st.number_input("Random seed", min_value=0, value=7, step=1, key="bgp_seed")
    with s3:
        penalty_max = st.number_input("Penalty max (child labor / banned chem)", min_value=0.0, value=100.0, step=10.0, key="bgp_penmax")

    include_current = st.checkbox("Include current policy from the Policy tab", value=True, key="bgp_inccur")

    if not _HAS_MAXUTIL:
        st.warning(
            "MaxUtilAgent.py could not be imported. Sustainable policy search will be disabled until the module exists in your repo."
        )

    @st.cache_data(show_spinner=False)
    def _evaluate_policy_batch(
        excel_path_str: str,
        cfg_profit: dict,
        cfg_util: dict,
        policy_dicts: list,
    ) -> list:
        suppliers_df, users_df = load_supplier_user_tables(Path(excel_path_str))

        results = []
        for pdict in policy_dicts:
            pol = Policy()
            for k, v in pdict.items():
                setattr(pol, k, float(v))

            row = {"policy": pdict}

            # Growth eval (Max Profit)
            try:
                cfgp = MaxProfitConfig(
                    last_n_users=int(cfg_profit["last_n_users"]),
                    capacity=int(cfg_profit["capacity"]),
                    suppliers_to_select=int(cfg_profit["suppliers_to_select"]),
                    price_per_match=float(cfg_profit["price_per_match"]),
                    min_utility=float(cfg_profit["min_utility"]),
                    output_flag=0,
                )
                rp = MaxProfitAgent(suppliers_df, users_df, pol, cfgp).solve()
                if int(rp.get("status", -1)) == 2:
                    row["profit_obj"] = float(rp["objective_value"])
                    row["profit_matched"] = int(rp["num_matched"])
                    row["profit_suppliers"] = ", ".join(rp.get("chosen_suppliers", []))
                else:
                    row["profit_obj"] = None
                    row["profit_matched"] = 0
                    row["profit_suppliers"] = ""
            except Exception:
                row["profit_obj"] = None
                row["profit_matched"] = 0
                row["profit_suppliers"] = ""

            # Sustainable eval (Max Utility)
            if cfg_util.get("enabled", False):
                try:
                    cfgu = MaxUtilConfig(
                        last_n_users=int(cfg_util["last_n_users"]),
                        capacity=int(cfg_util["capacity"]),
                        suppliers_to_select=int(cfg_util["suppliers_to_select"]),
                        min_utility=float(cfg_util["min_utility"]),
                        output_flag=0,
                    )
                    ru = MaxUtilAgent(suppliers_df, users_df, pol, cfgu).solve()
                    if int(ru.get("status", -1)) == 2:
                        row["util_obj"] = float(ru["objective_value"])
                        row["util_matched"] = int(ru["num_matched"])
                        row["util_suppliers"] = ", ".join(ru.get("chosen_suppliers", []))
                    else:
                        row["util_obj"] = None
                        row["util_matched"] = 0
                        row["util_suppliers"] = ""
                except Exception:
                    row["util_obj"] = None
                    row["util_matched"] = 0
                    row["util_suppliers"] = ""
            else:
                row["util_obj"] = None
                row["util_matched"] = 0
                row["util_suppliers"] = ""

            results.append(row)

        return results

    def _make_candidate_policies(n: int, seed_: int, pen_max: float, include_current_: bool) -> list:
        rng = random.Random(int(seed_))
        mult_vals = [1, 2, 3, 4, 5]

        # Two anchor presets (always included)
        growth_anchor = {
            "env_mult": 1,
            "social_mult": 1,
            "cost_mult": 5,
            "strategic_mult": 5,
            "improvement_mult": 3,
            "low_quality_mult": 1,
            "child_labor_penalty": 0,
            "banned_chem_penalty": 0,
        }
        sustainable_anchor = {
            "env_mult": 5,
            "social_mult": 5,
            "cost_mult": 1,
            "strategic_mult": 2,
            "improvement_mult": 3,
            "low_quality_mult": 5,
            "child_labor_penalty": float(pen_max),
            "banned_chem_penalty": float(pen_max),
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

        # Random policies
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

        # De-dup (stable)
        seen = set()
        uniq = []
        for d in candidates:
            key = tuple(sorted((k, float(v)) for k, v in d.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(d)
        return uniq

    if st.button("Find best policies", type="primary", use_container_width=True, key="bgp_run"):
        cfg_profit = {
            "last_n_users": int(last_n_users),
            "capacity": int(capacity),
            "suppliers_to_select": int(suppliers_to_select),
            "price_per_match": float(price_per_match),
            "min_utility": float(min_utility),
        }
        cfg_util = {
            "enabled": bool(_HAS_MAXUTIL),
            "last_n_users": int(last_n_users),
            "capacity": int(capacity),
            "suppliers_to_select": int(suppliers_to_select),
            "min_utility": float(min_utility),
        }

        policy_dicts = _make_candidate_policies(int(n_candidates), int(seed), float(penalty_max), bool(include_current))

        with st.spinner("Evaluating candidate policies..."):
            rows = _evaluate_policy_batch(str(excel_path), cfg_profit, cfg_util, policy_dicts)

        df = pd.DataFrame(
            [
                {
                    **r["policy"],
                    "profit_obj": r["profit_obj"],
                    "profit_matched": r["profit_matched"],
                    "profit_suppliers": r["profit_suppliers"],
                    "util_obj": r["util_obj"],
                    "util_matched": r["util_matched"],
                    "util_suppliers": r["util_suppliers"],
                }
                for r in rows
            ]
        )

        # Best growth
        df_growth = df.dropna(subset=["profit_obj"]).sort_values(["profit_obj", "profit_matched"], ascending=[False, False])
        # Best sustainable
        df_sus = df.dropna(subset=["util_obj"]).sort_values(["util_obj", "util_matched"], ascending=[False, False])

        st.markdown("#### Results")
        gcol, scol = st.columns(2)

        with gcol:
            st.markdown("##### Best Growth Policy")
            if len(df_growth) == 0:
                st.error("No feasible/optimal solutions found for growth objective in the candidate set.")
            else:
                best_g = df_growth.iloc[0]
                st.metric("Max Profit objective", f"{best_g['profit_obj']:.3f}")
                st.write("Chosen suppliers:", best_g.get("profit_suppliers", "") or "None")
                st.write("Matched users:", int(best_g.get("profit_matched", 0)))
                st.json({k: best_g[k] for k in [
                    "env_mult","social_mult","cost_mult","strategic_mult","improvement_mult","low_quality_mult","child_labor_penalty","banned_chem_penalty"
                ]})

        with scol:
            st.markdown("##### Best Sustainable Policy")
            if not _HAS_MAXUTIL:
                st.error("MaxUtilAgent module is missing, so sustainable policy cannot be computed.")
            elif len(df_sus) == 0:
                st.error("No feasible/optimal solutions found for sustainable objective in the candidate set.")
            else:
                best_s = df_sus.iloc[0]
                st.metric("Max Utility objective", f"{best_s['util_obj']:.3f}")
                st.write("Chosen suppliers:", best_s.get("util_suppliers", "") or "None")
                st.write("Matched users:", int(best_s.get("util_matched", 0)))
                st.json({k: best_s[k] for k in [
                    "env_mult","social_mult","cost_mult","strategic_mult","improvement_mult","low_quality_mult","child_labor_penalty","banned_chem_penalty"
                ]})

        st.markdown("##### Top candidates (growth)")
        st.dataframe(df_growth.head(10), use_container_width=True, hide_index=True)

        if _HAS_MAXUTIL:
            st.markdown("##### Top candidates (sustainable)")
            st.dataframe(df_sus.head(10), use_container_width=True, hide_index=True)
