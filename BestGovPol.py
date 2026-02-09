\
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


def render_best_governmental_policy() -> None:
    """
    Streamlit page renderer for the "Best governmental policy" tab.

    This file is meant to be imported from UI.py:
        from BestGovPol import render_best_governmental_policy
        ...
        with tab_best_policy:
            render_best_governmental_policy()
    """
    st.subheader("Best governmental policy")
    st.markdown(
        "**Maximize retailer utility** (retailer = MaxProfitAgent objective) "
        "**while government enforces world-goodness constraints**."
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
        ban_max = st.number_input("Max total banned chem (Ban̄)", value=0.0, step=0.1, disabled=not use_ban)

    st.markdown("### World-goodness score weights (for ε tie-break only)")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        w_env = st.number_input("w_env", min_value=0.0, value=1.0, step=0.1)
    with w2:
        w_soc = st.number_input("w_social", min_value=0.0, value=1.0, step=0.1)
    with w3:
        w_child = st.number_input("w_child", min_value=0.0, value=10.0, step=1.0)
    with w4:
        w_ban = st.number_input("w_banned", min_value=0.0, value=10.0, step=1.0)
    with w5:
        w_lq = st.number_input("w_low_quality", min_value=0.0, value=1.0, step=0.1)

    # ---- Helpers (local to avoid polluting UI.py)
    def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{df_name} missing required columns: {missing}")

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

        # required columns
        _require_columns(suppliers_df, ["supplier_id", "env_risk", "social_risk", "child_labor", "banned_chem", "low_quality"], "suppliers_df")
        _require_columns(matches, ["supplier_id"], "matches")

        # normalize ids
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

    if st.button("Solve Government MILP", type="primary", use_container_width=True):
        try:
            suppliers_df, users_df = load_supplier_user_tables(excel_path)

            weights = {
                "w_env": float(w_env),
                "w_soc": float(w_soc),
                "w_child": float(w_child),
                "w_ban": float(w_ban),
                "w_lq": float(w_lq),
            }

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
                            weights=weights,
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
            best_growth = rows[int(df_growth.index[0])] if len(df_growth) else None

            # Government MILP selection among candidates
            gov = gp.Model("GovernmentPolicyMILP")
            gov.Params.OutputFlag = 0

            J = list(range(len(rows)))
            x = gov.addVars(J, vtype=GRB.BINARY, name="x")

            # choose exactly one
            gov.addConstr(gp.quicksum(x[j] for j in J) == 1, name="choose_one")

            # minimum matches
            if int(min_matches) > 0:
                gov.addConstr(gp.quicksum(rows[j]["matched"] * x[j] for j in J) >= int(min_matches), name="min_matches")

            # avg constraints: total <= max_avg * matches
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

            # Objective: maximize retailer utility + epsilon * goodness (tie-break)
            gov.setObjective(
                gp.quicksum(rows[j]["profit_obj"] * x[j] for j in J)
                + float(eps_tie) * gp.quicksum(rows[j]["goodness"] * x[j] for j in J),
                GRB.MAXIMIZE,
            )

            gov.optimize()

            if gov.Status != GRB.OPTIMAL:
                st.error("Government MILP found no feasible policy under the chosen world constraints. Relax constraints or increase candidates.")
                st.dataframe(df_growth.head(25), use_container_width=True, hide_index=True)
                st.stop()

            chosen_j = next(j for j in J if x[j].X > 0.5)
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

        except Exception as e:
            st.error(str(e))
