import random
from pathlib import Path

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
# Fixed world-goodness weights (used only for ε tie-break)
# -----------------------------
WORLD_GOODNESS_WEIGHTS = {
    "w_env": 1.0,
    "w_soc": 1.0,
    "w_child": 10.0,
    "w_ban": 10.0,
    "w_lq": 1.0,
}



def render_best_governmental_policy() -> None:
    """
    Streamlit page renderer for the "Best governmental policy" tab.

    Usage in UI.py:
        from BestGovPol import render_best_governmental_policy
        ...
        with tab_best_policy:
            render_best_governmental_policy()
    """
    st.subheader("Best governmental policy")

    st.markdown(
        "**Idea:** Government chooses a policy from a **candidate set**. For each candidate policy, we first solve the "
        "**inner retailer MILP** (MaxProfitAgent) to obtain the retailer utility and the world-impact metrics. "
        "Then the government solves a **selection MILP** over candidates."
    )

    excel_path = Path(DEFAULT_XLSX_PATH)
    if not excel_path.exists():
        st.error(f"Excel file not found: {excel_path.name}. Place it next to UI.py in your repo.")
        st.stop()

    if not GUROBI_AVAILABLE or not _HAS_GUROBI:
        st.error("gurobipy is not available in this environment. Add `gurobipy` to requirements.txt.")
        st.stop()

   

    st.divider()

    # -----------------------------
    # Controls
    # -----------------------------
    a, b, c, d = st.columns(4)
    with a:
        st.number_input("Total users considered (fixed)", min_value=2, value=11, step=1, disabled=True)
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
        penalty_max = st.number_input(
            "Penalty max (child labor / banned chem)", min_value=0.0, value=100.0, step=10.0, key="bgp_penmax"
        )

    include_current = st.checkbox("Include current policy from the Policy tab in the candidate set", value=True, key="bgp_inccur")

    st.divider()
    st.markdown("### World-goodness constraints (government)")

    f1, f2, f3 = st.columns(3)
    with f1:
        min_matches = st.number_input("Minimum matched users ", min_value=0, value=11, step=1, key="bgp_minm")
        eps_tie = st.number_input("Tie-break ε (sustainability)", min_value=0.0, value=0.001, step=0.001, format="%.3f", key="bgp_eps")
    with f2:
        use_env = st.checkbox("Constrain avg environmental risk", value=True, key="bgp_use_env")
        env_avg_max = st.number_input("Max avg env risk (EnvAvg)", value=3.0, step=0.1, disabled=not use_env, key="bgp_envavg")
        use_soc = st.checkbox("Constrain avg social risk", value=True, key="bgp_use_soc")
        soc_avg_max = st.number_input("Max avg social risk (SocAvg)", value=3.0, step=0.1, disabled=not use_soc, key="bgp_socavg")
    with f3:
        use_child = st.checkbox("Constrain child labor exposure", value=True, key="bgp_use_child")
        child_max = st.number_input("Max total child labor (Child)", value=0.0, step=1.0, disabled=not use_child, key="bgp_childmax")
        use_ban = st.checkbox("Constrain banned chemicals exposure", value=True, key="bgp_use_ban")
        ban_max = st.number_input("Max total banned chem (Ban)", value=0.0, step=0.1, disabled=not use_ban, key="bgp_banmax")

    # -----------------------------
    # Helpers (kept local)
    # -----------------------------
    def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

    def _policy_obj_to_dict(pol: Policy) -> dict:
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

    def _policy_dict_to_obj(pdict: dict) -> Policy:
        pol = Policy()
        for kk, vv in pdict.items():
            setattr(pol, kk, float(vv))
        return pol

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

        candidates: list[dict] = [growth_anchor, sustainable_anchor]

        if include_current_ and "policy" in st.session_state:
            candidates.append(_policy_obj_to_dict(st.session_state.policy))

        # random
        target = max(0, int(n) - len(candidates))
        for _ in range(target):
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
        seen: set[tuple] = set()
        uniq: list[dict] = []
        for dct in candidates:
            key = tuple(sorted((k, float(v)) for k, v in dct.items()))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(dct)
        return uniq

    def eval_policy(
        suppliers_df: pd.DataFrame,
        users_df: pd.DataFrame,
        pdict: dict,
        k: int,
        price: float,
        min_util: float,
        weights: dict,
    ) -> dict:
        pol = _policy_dict_to_obj(pdict)

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

        # exposures from chosen matches
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

        m = int(res["num_matched"]) if res.get("num_matched") is not None else 0
        env_avg = env_tot / m if m > 0 else 0.0
        soc_avg = soc_tot / m if m > 0 else 0.0
        lq_avg = lq_tot / m if m > 0 else 0.0

        # world goodness score (higher better) — only for tie-break
        goodness = -(
            float(weights["w_env"]) * env_avg
            + float(weights["w_soc"]) * soc_avg
            + float(weights["w_child"]) * child_tot
            + float(weights["w_ban"]) * ban_tot
            + float(weights["w_lq"]) * lq_avg
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

    def _render_one_result(title: str, r: dict) -> None:
        st.markdown(f"#### {title}")
        st.metric("Retailer utility (objective)", f"{r['profit_obj']:.3f}")
        st.write("Matched users:", r["matched"])
        st.write("Chosen suppliers:", r["chosen_suppliers"] or "None")
        st.write("Env avg:", f"{r['env_avg']:.3f}", "| Social avg:", f"{r['soc_avg']:.3f}", "| Low-quality avg:", f"{r['lq_avg']:.3f}")
        st.write("Child total:", f"{r['child_tot']:.3f}", "| Banned total:", f"{r['ban_tot']:.3f}")
        st.write("Policy:")
        st.json(r["policy"])
        st.dataframe(r["matches_df"], use_container_width=True, hide_index=True)

    # -----------------------------
    # Solve button
    # -----------------------------
    if st.button("Evaluate candidates & solve Government MILP", type="primary", use_container_width=True):
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            weights = WORLD_GOODNESS_WEIGHTS

            policy_dicts = make_candidate_policies(int(n_candidates), int(seed), float(penalty_max), bool(include_current))

            rows: list[dict] = []
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
                            weights=weights,
                        )
                    )

            df = pd.DataFrame(
                [
                    {
                        "__id": j,
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
                    for j, r in enumerate(rows)
                ]
            )

            # Best "growth" = max retailer utility (tie-break: matched)
            best_growth_id = int(df.sort_values(["profit_obj", "matched"], ascending=[False, False]).iloc[0]["__id"])

            # Government MILP selection among candidates
            gov = gp.Model("GovernmentPolicyMILP")
            gov.Params.OutputFlag = 0

            J = list(range(len(rows)))
            x = gov.addVars(J, vtype=GRB.BINARY, name="x")

            gov.addConstr(gp.quicksum(x[j] for j in J) == 1, name="choose_one")

            if int(min_matches) > 0:
                gov.addConstr(gp.quicksum(rows[j]["matched"] * x[j] for j in J) >= int(min_matches), name="min_matches")

            if use_env:
                gov.addConstr(
                    gp.quicksum(rows[j]["env_tot"] * x[j] for j in J)
                    <= float(env_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
                    name="env_avg",
                )
            if use_soc:
                gov.addConstr(
                    gp.quicksum(rows[j]["soc_tot"] * x[j] for j in J)
                    <= float(soc_avg_max) * gp.quicksum(rows[j]["matched"] * x[j] for j in J),
                    name="soc_avg",
                )

            if use_child:
                gov.addConstr(gp.quicksum(rows[j]["child_tot"] * x[j] for j in J) <= float(child_max), name="child_tot")
            if use_ban:
                gov.addConstr(gp.quicksum(rows[j]["ban_tot"] * x[j] for j in J) <= float(ban_max), name="ban_tot")

            gov.setObjective(
                gp.quicksum(rows[j]["profit_obj"] * x[j] for j in J)
                + float(eps_tie) * gp.quicksum(rows[j]["goodness"] * x[j] for j in J),
                GRB.MAXIMIZE,
            )

            gov.optimize()

            if gov.Status != GRB.OPTIMAL:
                # Still store candidates so the user can inspect / manually pick
                st.session_state.bgp_rows = rows
                st.session_state.bgp_df = df
                st.session_state.bgp_best_growth_id = best_growth_id
                st.session_state.bgp_best_gov_id = None
                st.error("Government MILP found no feasible policy under the chosen world constraints. Relax constraints or increase candidates.")
            else:
                chosen_id = next(j for j in J if x[j].X > 0.5)
                st.session_state.bgp_rows = rows
                st.session_state.bgp_df = df
                st.session_state.bgp_best_growth_id = best_growth_id
                st.session_state.bgp_best_gov_id = int(chosen_id)

        except Exception as e:
            st.error(str(e))

    # -----------------------------
    # If we have evaluated candidates, show results + manual selection
    # -----------------------------
    if "bgp_rows" in st.session_state and "bgp_df" in st.session_state:
        rows = st.session_state.bgp_rows
        df = st.session_state.bgp_df.copy()

        best_growth_id = st.session_state.get("bgp_best_growth_id", None)
        best_gov_id = st.session_state.get("bgp_best_gov_id", None)

        st.divider()
        st.markdown("### Results")

        left, right = st.columns(2)

        if best_growth_id is not None:
            with left:
                _render_one_result("Maximize retailer utility (no world constraints)", rows[int(best_growth_id)])

        with right:
            if best_gov_id is None:
                st.markdown("#### Government MILP: (infeasible under current constraints)")
                st.info("You can still manually select a candidate below and see its outcomes.")
            else:
                _render_one_result("Government MILP: maximize retailer utility + enforce world goodness", rows[int(best_gov_id)])

        st.markdown("### Evaluated candidates (top by retailer utility)")
        st.dataframe(
            df.sort_values(["profit_obj", "matched"], ascending=[False, False]).head(50),
            use_container_width=True,
            hide_index=True,
        )

        # -----------------------------
        # Manual selection / manual evaluation
        # -----------------------------
        st.divider()
        st.markdown("### Manual policy: see results for an *explicit* choice")

        # 1) Manual pick from evaluated candidates
        cand_ids = list(range(len(rows)))

        def _label(j: int) -> str:
            r = rows[j]
            return f"#{j} | U={r['profit_obj']:.2f} | matched={r['matched']} | env_avg={r['env_avg']:.2f} | soc_avg={r['soc_avg']:.2f}"

        default_idx = 0
        if best_gov_id is not None:
            default_idx = cand_ids.index(int(best_gov_id))
        elif best_growth_id is not None:
            default_idx = cand_ids.index(int(best_growth_id))

        picked_id = st.selectbox(
            "Pick a candidate policy (manual selection)",
            options=cand_ids,
            index=default_idx,
            format_func=_label,
            key="bgp_manual_pick",
        )

        st.markdown("#### Manual selection outcome (picked candidate)")
        _render_one_result(f"Candidate #{picked_id}", rows[int(picked_id)])

        # 2) Evaluate the current policy from the Policy tab (even if it wasn't in candidate set)
        if "policy" in st.session_state:
            with st.expander("Evaluate current Policy-tab settings (manual)", expanded=False):
                if st.button("Evaluate current policy now", key="bgp_eval_current"):
                    try:
                        suppliers_df, users_df = load_supplier_user_tables(excel_path)
                        weights = WORLD_GOODNESS_WEIGHTS
                        cur_dict = _policy_obj_to_dict(st.session_state.policy)
                        cur_res = eval_policy(
                            suppliers_df,
                            users_df,
                            cur_dict,
                            k=int(suppliers_to_select_best),
                            price=float(price_per_match_best),
                            min_util=float(min_utility_best),
                            weights=weights,
                        )
                        _render_one_result("Current Policy-tab settings", cur_res)
                    except Exception as e:
                        st.error(str(e))

        # 3) Fully custom manual policy inputs
        with st.expander("Evaluate a custom policy (manual inputs)", expanded=False):
            base = _policy_obj_to_dict(st.session_state.policy) if "policy" in st.session_state else {
                "env_mult": 3, "social_mult": 3, "cost_mult": 3,
                "strategic_mult": 3, "improvement_mult": 3, "low_quality_mult": 3,
                "child_labor_penalty": 0, "banned_chem_penalty": 0,
            }

            c1, c2, c3 = st.columns(3)
            with c1:
                env_mult_m = st.number_input("env_mult", min_value=0.0, value=float(base["env_mult"]), step=0.5, key="bgp_m_env")
                social_mult_m = st.number_input("social_mult", min_value=0.0, value=float(base["social_mult"]), step=0.5, key="bgp_m_soc")
                cost_mult_m = st.number_input("cost_mult", min_value=0.0, value=float(base["cost_mult"]), step=0.5, key="bgp_m_cost")
            with c2:
                strategic_mult_m = st.number_input("strategic_mult", min_value=0.0, value=float(base["strategic_mult"]), step=0.5, key="bgp_m_str")
                improvement_mult_m = st.number_input("improvement_mult", min_value=0.0, value=float(base["improvement_mult"]), step=0.5, key="bgp_m_imp")
                low_quality_mult_m = st.number_input("low_quality_mult", min_value=0.0, value=float(base["low_quality_mult"]), step=0.5, key="bgp_m_lq")
            with c3:
                child_pen_m = st.number_input("child_labor_penalty", min_value=0.0, value=float(base["child_labor_penalty"]), step=1.0, key="bgp_m_child")
                ban_pen_m = st.number_input("banned_chem_penalty", min_value=0.0, value=float(base["banned_chem_penalty"]), step=1.0, key="bgp_m_ban")

            if st.button("Evaluate custom policy", key="bgp_eval_custom"):
                try:
                    suppliers_df, users_df = load_supplier_user_tables(excel_path)
                    weights = WORLD_GOODNESS_WEIGHTS
                    custom_dict = {
                        "env_mult": float(env_mult_m),
                        "social_mult": float(social_mult_m),
                        "cost_mult": float(cost_mult_m),
                        "strategic_mult": float(strategic_mult_m),
                        "improvement_mult": float(improvement_mult_m),
                        "low_quality_mult": float(low_quality_mult_m),
                        "child_labor_penalty": float(child_pen_m),
                        "banned_chem_penalty": float(ban_pen_m),
                    }
                    custom_res = eval_policy(
                        suppliers_df,
                        users_df,
                        custom_dict,
                        k=int(suppliers_to_select_best),
                        price=float(price_per_match_best),
                        min_util=float(min_utility_best),
                        weights=weights,
                    )
                    _render_one_result("Custom manual policy", custom_res)
                except Exception as e:
                    st.error(str(e))
