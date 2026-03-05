import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from supabase_db import insert_submission, fetch_all_submissions
import pandas as pd
from streamlit_autorefresh import st_autorefresh


from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    Policy,
    load_supplier_user_tables,
    MaxProfitAgent,
    MaxProfitConfig,
    MaxUtilAgent,
    MaxUtilConfig,
    manual_metrics,
)

st.set_page_config(page_title="Arya Phones — Supplier Game", layout="wide")

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
ENV_CAP = 2.75
SOCIAL_CAP = 3.0
COST_SCALE = 10.0

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
    return df[cols].copy()


def _metrics_panel(title: str, m: Dict[str, float], feasible: bool):
    st.markdown(f"### {title}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total profit", f"{m['profit_total']:.3f}")
    c2.metric("Total utility", f"{m['utility_total']:.3f}")
    c3.metric("Avg env / avg social", f"{m['avg_env']:.3f} / {m['avg_social']:.3f}")
    c4.metric("# suppliers", f"{int(m['k'])}")

    if int(m["k"]) <= 0:
        st.warning("Pick at least 1 supplier.")
    elif feasible:
        st.success("Feasible (risk caps satisfied).")
    else:
        st.error("Risk caps violated. You can still submit, but it will be marked as infeasible.")


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
    }
    st.dataframe(df, use_container_width=True, hide_index=True, column_config=col_cfg)


st.title("Arya Phones — Supplier Selection Game")

excel_path = Path(__file__).with_name('Arya_Phones_Supplier_Selection.xlsx')
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Place it next to UI.py.")
    st.stop()

try:
    suppliers_df, users_df = load_supplier_user_tables(excel_path)
except ValueError as e:
    st.error("Excel columns mismatch. The app couldn't find the required columns in your Suppliers or Users sheet.")
    st.code(str(e))
    st.markdown(
        """
**Fix:** Open your Excel and ensure the sheet that contains suppliers has (at minimum) these columns (names can vary, but must mean the same thing):
- supplier_id (or Supplier / Supplier ID / Name)
- env_risk (Environmental risk)
- social_risk (Social risk)
- cost_score (Cost / Cost score)
- strategic
- improvement
- low_quality

Optional:
- child_labor
- banned_chem

Then re-run the app.
        """
    )
    st.stop()

c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.session_state.team_name = st.text_input("Name / team", value=st.session_state.team_name)
with c2:
    price_per_user = st.number_input("Selling price per user", min_value=0.0, value=100.0, step=5.0)
with c3:
    st.info(
        f"Served users: {SERVED_USERS} | Risk caps: avg env ≤ {ENV_CAP}, avg social ≤ {SOCIAL_CAP} | Profit subtracts {COST_SCALE}×avg(cost_score)",
        icon="ℹ️",
    )

with st.expander("Suppliers overview", expanded=True):
    overview = supplier_overview(suppliers_df, users_df)
    _render_overview_with_selection(overview, [])

if not GUROBI_AVAILABLE:
    st.error("gurobipy is not available in this environment.")
    st.stop()

suppliers_list = suppliers_df["supplier_id"].astype(str).tolist()

profit_tab, util_tab, lb_tab = st.tabs(["Max Profit", "Max Utility", "Leaderboard"])

with profit_tab:
    st.subheader("Max Profit")

    picks = st.multiselect(
        "Pick any suppliers (1 or more)",
        options=suppliers_list,
        default=st.session_state.profit_picks,
        key="profit_picks_widget",
    )
    st.session_state.profit_picks = picks

    st.divider()
    st.markdown("#### Your selection")
    _render_overview_with_selection(overview, picks)

    cfg = MaxProfitConfig(
        served_users=SERVED_USERS,
        price_per_user=float(price_per_user),
        cost_scale=COST_SCALE,
        env_cap=ENV_CAP,
        social_cap=SOCIAL_CAP,
        output_flag=0,
    )

    man = manual_metrics(suppliers_df, users_df, FIXED_POLICY, cfg, picks)
    _metrics_panel("Manual", man["metrics"], man["feasible"])

    st.divider()

    if st.button("Run benchmark", type="primary", use_container_width=True, key="profit_run"):
        try:
            res = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
            _metrics_panel("Benchmark (optimal)", res["metrics"], res["feasible"])

            if man["feasible"] and res["feasible"]:
                gap = float(res["metrics"]["profit_total"] - man["metrics"]["profit_total"])
                st.write(f"Benchmark − Manual profit gap: **{gap:.3f}**")
        except Exception:
            st.info("Benchmark could not be computed.")

    if st.button("Submit", use_container_width=True, key="profit_submit"):
        if int(man["metrics"]["k"]) <= 0:
            st.warning("Pick at least 1 supplier before submitting.")
        else:
            m = man["metrics"]
            payload = {
                "team": (st.session_state.team_name or "(anonymous)").strip(),
                "player_name": None,
                "selected_suppliers": ",".join([str(x) for x in picks]),
                "objective": "max_profit",
                "comment": None,
                "profit": float(m.get("profit_total", 0.0)),
                "utility": float(m.get("utility_total", 0.0)),
                "env_avg": float(m.get("avg_env", 0.0)),
                "social_avg": float(m.get("avg_social", 0.0)),
                "cost_avg": float(m.get("avg_cost", 0.0)),
                "strategic_avg": float(m.get("avg_strategic", 0.0)),
                "improvement_avg": float(m.get("avg_improvement", 0.0)),
                "low_quality_avg": float(m.get("avg_low_quality", 0.0)),
            }
            try:
                insert_submission(payload)
                st.success("Submitted to live leaderboard.")
            except Exception as e:
                st.error("Submission failed (Supabase).")
                st.code(str(e))

with util_tab:
    st.subheader("Max Utility")

    picks = st.multiselect(
        "Pick any suppliers (1 or more)",
        options=suppliers_list,
        default=st.session_state.util_picks,
        key="util_picks_widget",
    )
    st.session_state.util_picks = picks

    st.divider()
    st.markdown("#### Your selection")
    _render_overview_with_selection(overview, picks)

    cfg = MaxUtilConfig(
        served_users=SERVED_USERS,
        price_per_user=float(price_per_user),
        cost_scale=COST_SCALE,
        env_cap=ENV_CAP,
        social_cap=SOCIAL_CAP,
        output_flag=0,
    )

    man = manual_metrics(suppliers_df, users_df, FIXED_POLICY, cfg, picks)
    _metrics_panel("Manual", man["metrics"], man["feasible"])

    st.divider()

    if st.button("Run benchmark", type="primary", use_container_width=True, key="util_run"):
        try:
            res = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
            _metrics_panel("Benchmark (optimal)", res["metrics"], res["feasible"])

            if man["feasible"] and res["feasible"]:
                gap = float(res["metrics"]["utility_total"] - man["metrics"]["utility_total"])
                st.write(f"Benchmark − Manual utility gap: **{gap:.3f}**")
        except Exception:
            st.info("Benchmark could not be computed.")

    if st.button("Submit", use_container_width=True, key="util_submit"):
        if int(man["metrics"]["k"]) <= 0:
            st.warning("Pick at least 1 supplier before submitting.")
        else:
            m = man["metrics"]
            payload = {
                "team": (st.session_state.team_name or "(anonymous)").strip(),
                "player_name": None,
                "selected_suppliers": ",".join([str(x) for x in picks]),
                "objective": "max_utility",
                "comment": None,
                "profit": float(m.get("profit_total", 0.0)),
                "utility": float(m.get("utility_total", 0.0)),
                "env_avg": float(m.get("avg_env", 0.0)),
                "social_avg": float(m.get("avg_social", 0.0)),
                "cost_avg": float(m.get("avg_cost", 0.0)),
                "strategic_avg": float(m.get("avg_strategic", 0.0)),
                "improvement_avg": float(m.get("avg_improvement", 0.0)),
                "low_quality_avg": float(m.get("avg_low_quality", 0.0)),
            }
            try:
                insert_submission(payload)
                st.success("Submitted to live leaderboard.")
            except Exception as e:
                st.error("Submission failed (Supabase).")
                st.code(str(e))

with lb_tab:
    st.subheader("Live Leaderboard (latest submission per team)")

    # Auto-refresh every 3 seconds
    st_autorefresh(interval=3000, key="lb_refresh")

    # Pull all submissions (latest first)
    try:
        res = fetch_all_submissions(limit=5000)
        rows = getattr(res, "data", None)
        if rows is None and isinstance(res, dict):
            rows = res.get("data", [])
        rows = rows or []
        df_all = pd.DataFrame(rows)
    except Exception as e:
        st.error("Could not load leaderboard from Supabase.")
        st.code(str(e))
        st.stop()

    if df_all.empty:
        st.info("No submissions yet.")
        st.stop()

    # Parse timestamps and keep latest per team
    if "created_at" in df_all.columns:
        df_all["created_at"] = pd.to_datetime(df_all["created_at"], errors="coerce")
    else:
        df_all["created_at"] = pd.NaT

    if "team" not in df_all.columns:
        st.error("Supabase table is missing the 'team' column.")
        st.stop()

    latest = (
        df_all.sort_values("created_at", ascending=True)
        .groupby("team", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    st.caption(f"Teams: {latest['team'].nunique()} | Total submissions stored: {len(df_all)}")

    metrics = [
        "profit",
        "utility",
        "env_avg",
        "social_avg",
        "cost_avg",
        "strategic_avg",
        "improvement_avg",
        "low_quality_avg",
    ]
    metrics = [m for m in metrics if m in latest.columns]

    cA, cB, cC = st.columns([2, 2, 3])
    with cA:
        x = st.selectbox("X axis", metrics, index=0 if metrics else 0)
    with cB:
        y = st.selectbox("Y axis", metrics, index=1 if len(metrics) > 1 else 0)
    with cC:
        st.text_input("Your team name (for visibility)", value=(st.session_state.team_name or ""), disabled=True)

    # Make sure numeric
    for m in metrics:
        latest[m] = pd.to_numeric(latest[m], errors="coerce")

    plot_df = latest[["team", x, y, "created_at", "selected_suppliers", "objective"]].copy()
    plot_df = plot_df.rename(columns={"team": "Team", "selected_suppliers": "Suppliers", "objective": "Mode"})

    st.markdown("#### Scatter")
    st.scatter_chart(plot_df.set_index("Team")[[x, y]])

    st.markdown("#### Latest submissions table")
    show_cols = ["Team", "Mode", "created_at", x, y, "Suppliers"]
    st.dataframe(plot_df[show_cols].sort_values("created_at", ascending=False), use_container_width=True, hide_index=True)

    # Download CSV snapshot
    csv_bytes = plot_df[show_cols].sort_values("created_at", ascending=False).to_csv(index=False).encode("utf-8")
    st.download_button("Download leaderboard CSV", data=csv_bytes, file_name="leaderboard_latest.csv", use_container_width=True)
