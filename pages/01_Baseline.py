import streamlit as st
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from MinCostAgent import (
    GUROBI_AVAILABLE,
    Policy,
    load_supplier_user_tables,
    MaxProfitAgent,
    MaxProfitConfig,
    MaxUtilAgent,
    MaxUtilConfig,
    manual_metrics,
)

st.set_page_config(page_title="Baseline — Supplier Selection", layout="wide")

st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# Fixed baseline parameters
# ----------------------------
FIXED_PRICE_PER_USER = 100.0      # DO NOT CHANGE (baseline is fixed-price)
SERVED_USERS = 8                 # fixed capacity / served users for baseline
ENV_CAP = 2.75
SOCIAL_CAP = 3.0
COST_SCALE = 10.0

# Penalty settings are handled inside MinCostAgent.manual_metrics / solver
# (child labor and banned chemicals: 20 each, if present)

FIXED_POLICY = Policy(
    env_mult=1.0,
    social_mult=1.0,
    cost_mult=1.0,
    strategic_mult=1.0,
    improvement_mult=1.0,
    low_quality_mult=1.0,
    child_labor_penalty=0.0,
    banned_chem_penalty=0.0,
)

def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)

def supplier_overview(suppliers_df: pd.DataFrame, users_df: pd.DataFrame) -> pd.DataFrame:
    df = suppliers_df.copy()

    df["Env (bad)%"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["Social (bad)%"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["Cost (bad)%"] = (_norm01(df["cost_score"]) * 100).round(1)
    df["LowQ (bad)%"] = (_norm01(df["low_quality"]) * 100).round(1)
    df["Strategic (good)%"] = (_norm01(df["strategic"]) * 100).round(1)
    df["Improvement (good)%"] = (_norm01(df["improvement"]) * 100).round(1)

    if len(users_df):
        uavg = users_df[["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]].mean()
        df["Expected utility (avg user)"] = (
            float(uavg["w_env"]) * (FIXED_POLICY.env_mult * df["env_risk"])
            + float(uavg["w_social"]) * (FIXED_POLICY.social_mult * df["social_risk"])
            + float(uavg["w_cost"]) * (FIXED_POLICY.cost_mult * df["cost_score"])
            + float(uavg["w_strategic"]) * (FIXED_POLICY.strategic_mult * df["strategic"])
            + float(uavg["w_improvement"]) * (FIXED_POLICY.improvement_mult * df["improvement"])
            + float(uavg["w_low_quality"]) * (FIXED_POLICY.low_quality_mult * df["low_quality"])
        ).astype(float)
    else:
        df["Expected utility (avg user)"] = 0.0

    df["Profit cost (=10×cost_score)"] = (COST_SCALE * df["cost_score"]).astype(float)

    cols = [
        "supplier_id",
        "Env (bad)%",
        "Social (bad)%",
        "Cost (bad)%",
        "LowQ (bad)%",
        "Strategic (good)%",
        "Improvement (good)%",
        "Expected utility (avg user)",
        "Profit cost (=10×cost_score)",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "low_quality",
        "child_labor",
        "banned_chem",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()

def _render_overview_with_selection(overview_df: pd.DataFrame, picks: List[str]):
    df = overview_df.copy()
    pick_set = set(str(x) for x in picks)
    df.insert(1, "Selected", df["supplier_id"].astype(str).apply(lambda x: x in pick_set))
    df = df.sort_values(["Selected", "supplier_id"], ascending=[False, True]).reset_index(drop=True)

    col_cfg = {
        "Selected": st.column_config.CheckboxColumn("Selected"),
        "Env (bad)%": st.column_config.ProgressColumn("Env (bad)", min_value=0, max_value=100, format="%.1f"),
        "Social (bad)%": st.column_config.ProgressColumn("Social (bad)", min_value=0, max_value=100, format="%.1f"),
        "Cost (bad)%": st.column_config.ProgressColumn("Cost (bad)", min_value=0, max_value=100, format="%.1f"),
        "LowQ (bad)%": st.column_config.ProgressColumn("LowQ (bad)", min_value=0, max_value=100, format="%.1f"),
        "Strategic (good)%": st.column_config.ProgressColumn("Strategic (good)", min_value=0, max_value=100, format="%.1f"),
        "Improvement (good)%": st.column_config.ProgressColumn("Improvement (good)", min_value=0, max_value=100, format="%.1f"),
        "Expected utility (avg user)": st.column_config.NumberColumn("Expected utility (avg user)", format="%.3f"),
        "Profit cost (=10×cost_score)": st.column_config.NumberColumn("Profit cost (=10×cost_score)", format="%.3f"),
        "env_risk": st.column_config.NumberColumn("env_risk", format="%.3f"),
        "social_risk": st.column_config.NumberColumn("social_risk", format="%.3f"),
        "cost_score": st.column_config.NumberColumn("cost_score", format="%.3f"),
    }
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_cfg)

def _metrics_panel(title: str, m: Dict[str, float], feasible: bool):
    st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Profit (gross)", f"{m.get('profit_total', 0.0):.3f}")
    c2.metric("Penalties", f"{m.get('penalties', 0.0):.3f}")
    c3.metric("Profit (net)", f"{m.get('profit_net', m.get('profit_total', 0.0)):.3f}")
    c4.metric("Total utility", f"{m.get('utility_total', 0.0):.3f}")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Avg env", f"{m.get('avg_env', 0.0):.3f}")
    c6.metric("Avg social", f"{m.get('avg_social', 0.0):.3f}")
    c7.metric("Avg cost", f"{m.get('avg_cost', 0.0):.3f}")
    c8.metric("# suppliers", f"{int(m.get('k', 0))}")

    if int(m.get("k", 0)) <= 0:
        st.warning("Pick at least 1 supplier.")
    elif feasible:
        st.success("Feasible (risk caps satisfied).")
    else:
        st.error("Risk caps violated (infeasible).")

st.title("Baseline — Single Team (No Competition)")

excel_path = Path(__file__).resolve().parents[1] / "Arya_Phones_Supplier_Selection.xlsx"
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Place it next to UI.py (project root).")
    st.stop()

try:
    suppliers_df, users_df = load_supplier_user_tables(excel_path)
except ValueError as e:
    st.error("Excel columns mismatch. The app couldn't find the required columns in your Supplier or User sheet.")
    st.code(str(e))
    st.stop()

# Fixed display (not editable)
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.text_input("Mode", value="Baseline (single team)", disabled=True)
with c2:
    st.number_input("Selling price per user (fixed)", value=float(FIXED_PRICE_PER_USER), disabled=True)
with c3:
    st.info(
        f"Served users: {SERVED_USERS} | Risk caps: avg env ≤ {ENV_CAP}, avg social ≤ {SOCIAL_CAP} | Profit subtracts {COST_SCALE}×avg(cost_score) | Penalties: 20 (CL) + 20 (BC)",
        icon="ℹ️",
    )

overview = supplier_overview(suppliers_df, users_df)

with st.expander("Suppliers overview", expanded=True):
    _render_overview_with_selection(overview, [])

if not GUROBI_AVAILABLE:
    st.error("gurobipy is not available in this environment (baseline requires the solver).")
    st.stop()

suppliers_list = suppliers_df["supplier_id"].astype(str).tolist()

objective = st.radio("Objective (for benchmark)", ["Max Profit", "Max Utility"], horizontal=True)

picks = st.multiselect("Pick any suppliers (1 or more)", options=suppliers_list, default=[])

st.divider()
st.markdown("#### Your selection")
_render_overview_with_selection(overview, picks)

if objective == "Max Profit":
    cfg = MaxProfitConfig(
        served_users=SERVED_USERS,
        price_per_user=float(FIXED_PRICE_PER_USER),
        cost_scale=COST_SCALE,
        env_cap=ENV_CAP,
        social_cap=SOCIAL_CAP,
        output_flag=0,
    )
else:
    cfg = MaxUtilConfig(
        served_users=SERVED_USERS,
        price_per_user=float(FIXED_PRICE_PER_USER),
        cost_scale=COST_SCALE,
        env_cap=ENV_CAP,
        social_cap=SOCIAL_CAP,
        output_flag=0,
    )

man = manual_metrics(suppliers_df, users_df, FIXED_POLICY, cfg, picks)
_metrics_panel("Manual", man["metrics"], man["feasible"])

st.divider()

if st.button("Run benchmark (optimal)", type="primary", use_container_width=True):
    try:
        if objective == "Max Profit":
            res = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
            _metrics_panel("Benchmark (optimal)", res["metrics"], res["feasible"])
            if man["feasible"] and res["feasible"]:
                gap = float(res["metrics"].get("profit_net", res["metrics"].get("profit_total", 0.0)) - man["metrics"].get("profit_net", man["metrics"].get("profit_total", 0.0)))
                st.write(f"Benchmark − Manual net profit gap: **{gap:.3f}**")
        else:
            res = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
            _metrics_panel("Benchmark (optimal)", res["metrics"], res["feasible"])
            if man["feasible"] and res["feasible"]:
                gap = float(res["metrics"]["utility_total"] - man["metrics"]["utility_total"])
                st.write(f"Benchmark − Manual utility gap: **{gap:.3f}**")
    except Exception as e:
        st.error("Benchmark could not be computed.")
        st.code(str(e))
