import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    Policy,
    load_supplier_user_tables,
    MaxProfitAgent,
    MaxProfitConfig,
    MaxUtilAgent,
    MaxUtilConfig,
)

st.set_page_config(page_title="Arya Phones — Supplier Selection Game", layout="wide")

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

SERVED_USERS = 10
K_SUPPLIERS = 3
ENV_CAP = 2.75
SOCIAL_CAP = 3.0
COST_SCALE = 10.0
SUBMISSIONS_PATH = Path(__file__).resolve().parent / "submissions.xlsx"

FIXED_POLICY = Policy(
    env_mult=1.0,
    social_mult=1.0,
    cost_mult=1.0,
    strategic_mult=1.0,
    improvement_mult=1.0,
    low_quality_mult=1.0,
)

if "team_name" not in st.session_state:
    st.session_state.team_name = ""
if "profit_picks" not in st.session_state:
    st.session_state.profit_picks = []
if "util_picks" not in st.session_state:
    st.session_state.util_picks = []


def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def _supplier_overview(suppliers_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    df = suppliers_df.copy()

    df["Env (bad)%"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["Social (bad)%"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["Cost (bad)%"] = (_norm01(df["cost_score"]) * 100).round(1)
    df["LowQ (bad)%"] = (_norm01(df["low_quality"]) * 100).round(1)
    df["Strategic (good)%"] = (_norm01(df["strategic"]) * 100).round(1)
    df["Improvement (good)%"] = (_norm01(df["improvement"]) * 100).round(1)

    if len(users_df):
        uavg = users_df[["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]].mean()
        df["Expected utility"] = (
            float(uavg["w_env"]) * (FIXED_POLICY.env_mult * df["env_risk"])
            + float(uavg["w_social"]) * (FIXED_POLICY.social_mult * df["social_risk"])
            + float(uavg["w_cost"]) * (FIXED_POLICY.cost_mult * df["cost_score"])
            + float(uavg["w_strategic"]) * (FIXED_POLICY.strategic_mult * df["strategic"])
            + float(uavg["w_improvement"]) * (FIXED_POLICY.improvement_mult * df["improvement"])
            + float(uavg["w_low_quality"]) * (FIXED_POLICY.low_quality_mult * df["low_quality"])
        ).astype(float)
    else:
        df["Expected utility"] = 0.0

    df["Profit cost (per match)"] = (COST_SCALE * df["cost_score"]).astype(float)

    cols = [
        "supplier_id",
        "Env (bad)%",
        "Social (bad)%",
        "Cost (bad)%",
        "LowQ (bad)%",
        "Strategic (good)%",
        "Improvement (good)%",
        "Expected utility",
        "Profit cost (per match)",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "low_quality",
        "child_labor",
        "banned_chem",
    ]
    return df[cols].copy()


def _totals(matches_df: pd.DataFrame) -> Dict[str, float]:
    if matches_df is None or not len(matches_df):
        return {"profit": 0.0, "utility": 0.0, "cost": 0.0, "matches": 0.0}

    cost = float(pd.to_numeric(matches_df.get("cost_prod"), errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(matches_df.get("margin"), errors="coerce").fillna(0.0).sum())
    utility = float(pd.to_numeric(matches_df.get("utility"), errors="coerce").fillna(0.0).sum())
    return {"profit": profit, "utility": utility, "cost": cost, "matches": float(len(matches_df))}


def _read_submissions() -> pd.DataFrame:
    cols = [
        "timestamp",
        "name",
        "mode",
        "suppliers",
        "profit",
        "utility",
        "avg_env",
        "avg_social",
        "price",
    ]
    if not SUBMISSIONS_PATH.exists():
        return pd.DataFrame(columns=cols)
    try:
        df = pd.read_excel(SUBMISSIONS_PATH, engine="openpyxl")
        for c in cols:
            if c not in df.columns:
                df[c] = "" if c in {"timestamp", "name", "mode", "suppliers"} else 0.0
        return df[cols].copy()
    except Exception:
        return pd.DataFrame(columns=cols)


def _append_submission(row: Dict[str, Any]) -> None:
    df = _read_submissions()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    tmp = SUBMISSIONS_PATH.with_suffix(".tmp.xlsx")
    df.to_excel(tmp, index=False, engine="openpyxl")
    os.replace(tmp, SUBMISSIONS_PATH)


st.title("Arya Phones — Supplier Selection Game")

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.session_state.team_name = st.text_input("Your name / team", value=st.session_state.team_name)
with c2:
    price_per_match = st.number_input("Selling price per match", min_value=0.0, value=100.0, step=5.0)
with c3:
    st.info(
        f"Served users: {SERVED_USERS} | Select exactly {K_SUPPLIERS} suppliers | "
        f"Risk caps: avg env ≤ {ENV_CAP}, avg social ≤ {SOCIAL_CAP} | Profit cost = {COST_SCALE}×cost_score",
        icon="ℹ️",
    )

excel_path = Path(DEFAULT_XLSX_PATH)
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Place it next to UI.py.")
    st.stop()

suppliers_df, users_df = load_supplier_user_tables(excel_path)

with st.expander("Suppliers overview", expanded=True):
    ov = _supplier_overview(suppliers_df, users_df)
    col_cfg = {
        "Env (bad)%": st.column_config.ProgressColumn("Env (bad)", min_value=0, max_value=100, format="%.1f"),
        "Social (bad)%": st.column_config.ProgressColumn("Social (bad)", min_value=0, max_value=100, format="%.1f"),
        "Cost (bad)%": st.column_config.ProgressColumn("Cost (bad)", min_value=0, max_value=100, format="%.1f"),
        "LowQ (bad)%": st.column_config.ProgressColumn("LowQ (bad)", min_value=0, max_value=100, format="%.1f"),
        "Strategic (good)%": st.column_config.ProgressColumn("Strategic (good)", min_value=0, max_value=100, format="%.1f"),
        "Improvement (good)%": st.column_config.ProgressColumn("Improvement (good)", min_value=0, max_value=100, format="%.1f"),
        "Expected utility": st.column_config.NumberColumn("Expected utility", format="%.3f"),
        "Profit cost (per match)": st.column_config.NumberColumn("Profit cost (per match)", format="%.3f"),
    }
    st.dataframe(ov, use_container_width=True, hide_index=True, column_config=col_cfg)

if not GUROBI_AVAILABLE:
    st.error("gurobipy is not available in this environment.")
    st.stop()

suppliers_list = suppliers_df["supplier_id"].astype(str).tolist()

profit_tab, util_tab, sub_tab = st.tabs(["Max Profit", "Max Utility", "Submissions"])

with profit_tab:
    st.subheader("Max Profit")

    picks = st.multiselect(
        f"Select exactly {K_SUPPLIERS} suppliers",
        options=suppliers_list,
        default=st.session_state.profit_picks,
        key="profit_picks_widget",
    )
    st.session_state.profit_picks = picks

    run = st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="profit_run")

    if run:
        left, right = st.columns(2)

        with left:
            st.markdown("### Manual")
            if len(picks) != K_SUPPLIERS:
                st.error(f"Please select exactly {K_SUPPLIERS} suppliers. Current: {len(picks)}")
            else:
                cfg_m = MaxProfitConfig(
                    served_users=SERVED_USERS,
                    suppliers_to_select=K_SUPPLIERS,
                    price_per_match=float(price_per_match),
                    cost_scale=COST_SCALE,
                    env_cap=ENV_CAP,
                    social_cap=SOCIAL_CAP,
                    fixed_suppliers=list(picks),
                )
                try:
                    res_m = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg_m).solve()
                    tot_m = _totals(res_m["matches"])
                    st.metric("Profit", f"{tot_m['profit']:.3f}")
                    st.metric("Utility", f"{tot_m['utility']:.3f}")
                    st.metric("Avg env (selected)", f"{res_m['avg_env_selected']:.3f}")
                    st.metric("Avg social (selected)", f"{res_m['avg_social_selected']:.3f}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]))
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)

                    if st.button("Submit", use_container_width=True, key="profit_submit"):
                        _append_submission(
                            {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "name": (st.session_state.team_name or "(anonymous)").strip(),
                                "mode": "max_profit",
                                "suppliers": ", ".join(res_m["chosen_suppliers"]),
                                "profit": float(tot_m["profit"]),
                                "utility": float(tot_m["utility"]),
                                "avg_env": float(res_m["avg_env_selected"]),
                                "avg_social": float(res_m["avg_social_selected"]),
                                "price": float(price_per_match),
                            }
                        )
                        st.success("Submitted.")
                except Exception as e:
                    st.error(str(e))

        with right:
            st.markdown("### Optimal")
            cfg_o = MaxProfitConfig(
                served_users=SERVED_USERS,
                suppliers_to_select=K_SUPPLIERS,
                price_per_match=float(price_per_match),
                cost_scale=COST_SCALE,
                env_cap=ENV_CAP,
                social_cap=SOCIAL_CAP,
                fixed_suppliers=None,
            )
            try:
                res_o = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg_o).solve()
                tot_o = _totals(res_o["matches"])
                st.metric("Profit", f"{tot_o['profit']:.3f}")
                st.metric("Utility", f"{tot_o['utility']:.3f}")
                st.metric("Avg env (selected)", f"{res_o['avg_env_selected']:.3f}")
                st.metric("Avg social (selected)", f"{res_o['avg_social_selected']:.3f}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]))
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))

with util_tab:
    st.subheader("Max Utility")

    picks = st.multiselect(
        f"Select exactly {K_SUPPLIERS} suppliers",
        options=suppliers_list,
        default=st.session_state.util_picks,
        key="util_picks_widget",
    )
    st.session_state.util_picks = picks

    run = st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="util_run")

    if run:
        left, right = st.columns(2)

        with left:
            st.markdown("### Manual")
            if len(picks) != K_SUPPLIERS:
                st.error(f"Please select exactly {K_SUPPLIERS} suppliers. Current: {len(picks)}")
            else:
                cfg_m = MaxUtilConfig(
                    served_users=SERVED_USERS,
                    suppliers_to_select=K_SUPPLIERS,
                    price_per_match=float(price_per_match),
                    cost_scale=COST_SCALE,
                    env_cap=ENV_CAP,
                    social_cap=SOCIAL_CAP,
                    fixed_suppliers=list(picks),
                )
                try:
                    res_m = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg_m).solve()
                    tot_m = _totals(res_m["matches"])
                    st.metric("Utility", f"{tot_m['utility']:.3f}")
                    st.metric("Profit", f"{tot_m['profit']:.3f}")
                    st.metric("Avg env (selected)", f"{res_m['avg_env_selected']:.3f}")
                    st.metric("Avg social (selected)", f"{res_m['avg_social_selected']:.3f}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]))
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)

                    if st.button("Submit", use_container_width=True, key="util_submit"):
                        _append_submission(
                            {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "name": (st.session_state.team_name or "(anonymous)").strip(),
                                "mode": "max_utility",
                                "suppliers": ", ".join(res_m["chosen_suppliers"]),
                                "profit": float(tot_m["profit"]),
                                "utility": float(tot_m["utility"]),
                                "avg_env": float(res_m["avg_env_selected"]),
                                "avg_social": float(res_m["avg_social_selected"]),
                                "price": float(price_per_match),
                            }
                        )
                        st.success("Submitted.")
                except Exception as e:
                    st.error(str(e))

        with right:
            st.markdown("### Optimal")
            cfg_o = MaxUtilConfig(
                served_users=SERVED_USERS,
                suppliers_to_select=K_SUPPLIERS,
                price_per_match=float(price_per_match),
                cost_scale=COST_SCALE,
                env_cap=ENV_CAP,
                social_cap=SOCIAL_CAP,
                fixed_suppliers=None,
            )
            try:
                res_o = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg_o).solve()
                tot_o = _totals(res_o["matches"])
                st.metric("Utility", f"{tot_o['utility']:.3f}")
                st.metric("Profit", f"{tot_o['profit']:.3f}")
                st.metric("Avg env (selected)", f"{res_o['avg_env_selected']:.3f}")
                st.metric("Avg social (selected)", f"{res_o['avg_social_selected']:.3f}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]))
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))

with sub_tab:
    st.subheader("Submissions")

    df = _read_submissions()
    if df.empty:
        st.info("No submissions yet.")
    else:
        sort_by = st.selectbox("Sort by", options=["profit", "utility", "timestamp"], index=0)
        ascending = sort_by == "timestamp"
        df2 = df.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
        df2.insert(0, "rank", range(1, len(df2) + 1))
        st.dataframe(df2, use_container_width=True, hide_index=True)

        try:
            payload = SUBMISSIONS_PATH.read_bytes()
            st.download_button("Download submissions.xlsx", data=payload, file_name="submissions.xlsx")
        except Exception:
            pass
