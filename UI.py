import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    load_supplier_user_tables,
    ProfitRiskConfig,
    ProfitRiskMinRiskConfig,
    ProfitRiskTradeoffConfig,
    ProfitRiskMaxProfitAgent,
    ProfitRiskMinRiskAgent,
    ProfitRiskTradeoffAgent,
)

# ============================
# App config
# ============================
st.set_page_config(page_title="Supplier Selection Game — Profit & Risk", layout="wide")

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

# ============================
# Fixed game rules (EDIT HERE)
# ============================
SERVED_USERS = 10
ENV_CAP = 2.75
SOCIAL_CAP = 3.0
COST_SCALE = 10.0

# ============================
# Session state
# ============================
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

for key, default in {
    "team_name": "",
    "profit_k": 1,
    "risk_k": 1,
    "trade_k": 1,
    "profit_picks": [],
    "risk_picks": [],
    "trade_picks": [],
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================
# Data loading
# ============================
@st.cache_data(show_spinner=False)
def _load_data(path: Path):
    return load_supplier_user_tables(path)


def _norm01(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def _supplier_overview_table(suppliers_df: pd.DataFrame, price_per_user: float) -> pd.DataFrame:
    df = suppliers_df.copy()

    # Normalized "bad" risk bars (higher = worse)
    df["env_risk_%"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["social_risk_%"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["cost_%"] = (_norm01(df["cost_score"]) * 100).round(1)

    # If you picked ONLY this supplier (K=1), what would profit/risk look like?
    df["profit_per_user_if_single"] = (float(price_per_user) - COST_SCALE * df["cost_score"]).astype(float)
    df["profit_total_if_single"] = float(SERVED_USERS) * df["profit_per_user_if_single"]

    df["risk_score_if_single"] = 0.5 * ((df["env_risk"] / float(ENV_CAP)) + (df["social_risk"] / float(SOCIAL_CAP)))

    show_cols = [
        "supplier_id",
        "env_risk",
        "social_risk",
        "cost_score",
        "profit_per_user_if_single",
        "profit_total_if_single",
        "risk_score_if_single",
        "env_risk_%",
        "social_risk_%",
        "cost_%",
        "child_labor",
        "banned_chem",
        "strategic",
        "improvement",
        "low_quality",
    ]

    return df[show_cols].copy()


def _manual_metrics(suppliers_df: pd.DataFrame, picks: List[str], price_per_user: float) -> Dict[str, float]:
    if not picks:
        return {
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "profit_per_user": 0.0,
            "profit_total": 0.0,
            "risk_score": 0.0,
        }

    sub = suppliers_df[suppliers_df["supplier_id"].astype(str).isin([str(x) for x in picks])].copy()
    avg_env = float(sub["env_risk"].mean()) if len(sub) else 0.0
    avg_social = float(sub["social_risk"].mean()) if len(sub) else 0.0
    avg_cost = float(sub["cost_score"].mean()) if len(sub) else 0.0

    profit_per_user = float(price_per_user) - float(COST_SCALE) * avg_cost
    profit_total = float(SERVED_USERS) * profit_per_user
    risk_score = 0.5 * ((avg_env / float(ENV_CAP)) + (avg_social / float(SOCIAL_CAP)))

    return {
        "avg_env": avg_env,
        "avg_social": avg_social,
        "avg_cost": avg_cost,
        "profit_per_user": profit_per_user,
        "profit_total": profit_total,
        "risk_score": float(risk_score),
    }


def _feasible(avg_env: float, avg_social: float) -> bool:
    return (avg_env <= float(ENV_CAP) + 1e-9) and (avg_social <= float(SOCIAL_CAP) + 1e-9)


def _add_to_leaderboard(
    mode: str,
    team: str,
    k: int,
    picks: List[str],
    metrics: Dict[str, float],
    score: float,
    meta: Dict[str, Any],
):
    st.session_state.leaderboard.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "team": team.strip() or "(anonymous)",
            "mode": mode,
            "K": int(k),
            "suppliers": ", ".join([str(x) for x in picks]),
            "profit_total": float(metrics.get("profit_total", 0.0)),
            "profit_per_user": float(metrics.get("profit_per_user", 0.0)),
            "avg_env": float(metrics.get("avg_env", 0.0)),
            "avg_social": float(metrics.get("avg_social", 0.0)),
            "avg_cost": float(metrics.get("avg_cost", 0.0)),
            "risk_score": float(metrics.get("risk_score", 0.0)),
            "score": float(score),
            **meta,
        }
    )


def _render_leaderboard(title: str, mode_filter: Optional[str] = None):
    st.markdown(f"### {title}")
    df = pd.DataFrame(st.session_state.leaderboard)
    if df.empty:
        st.info("Leaderboard is empty. Add a run to get started.")
        return

    if mode_filter is not None:
        df = df[df["mode"] == mode_filter].copy()

    if df.empty:
        st.info("No entries for this mode yet.")
        return

    df = df.sort_values(["score", "time"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================
# Header
# ============================
st.title("Supplier Selection Game — Profit & Risk")

c1, c2 = st.columns([2, 3])
with c1:
    st.session_state.team_name = st.text_input("Team name (for leaderboard)", value=st.session_state.team_name)
with c2:
    with st.expander("Rules (fixed)", expanded=False):
        st.write(f"Served end-users: **{SERVED_USERS}**")
        st.write(f"Average environmental risk must be **≤ {ENV_CAP}**")
        st.write(f"Average social risk must be **≤ {SOCIAL_CAP}**")
        st.write(f"Profit subtracts **cost_score × {COST_SCALE}** per user")

excel_path = Path(DEFAULT_XLSX_PATH)
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Place it next to UI.py in your repo.")
    st.stop()

suppliers_df, _users_df = _load_data(excel_path)

if not GUROBI_AVAILABLE:
    st.error("gurobipy is not installed / licensed in this environment.")
    st.stop()

# ============================
# Supplier overview (game card)
# ============================
with st.expander("Supplier roster (how good / bad they are)", expanded=True):
    price_preview = st.number_input("Price per user (preview only)", min_value=0.0, value=100.0, step=5.0, key="price_preview")
    ov = _supplier_overview_table(suppliers_df, float(price_preview))

    col_cfg = {
        "env_risk_%": st.column_config.ProgressColumn("Env risk (bad)", min_value=0, max_value=100, format="%.1f"),
        "social_risk_%": st.column_config.ProgressColumn("Social risk (bad)", min_value=0, max_value=100, format="%.1f"),
        "cost_%": st.column_config.ProgressColumn("Cost (bad)", min_value=0, max_value=100, format="%.1f"),
        "profit_per_user_if_single": st.column_config.NumberColumn("Profit/user if single", format="%.3f"),
        "profit_total_if_single": st.column_config.NumberColumn("Total profit if single", format="%.3f"),
        "risk_score_if_single": st.column_config.NumberColumn("Risk score if single", format="%.4f"),
    }

    st.dataframe(ov, use_container_width=True, hide_index=True, column_config=col_cfg)


# ============================
# Tabs
# ============================
tab_profit, tab_risk, tab_trade = st.tabs(["Max Profit", "Min Risk", "Profit–Risk Curve + Leaderboard"])


# ============================
# Max Profit
# ============================
with tab_profit:
    st.header("Max Profit")
    st.caption("Choose K suppliers. The product you deliver is the average of those suppliers. Profit is computed with cost_score × 10.")

    a1, a2, a3 = st.columns(3)
    with a1:
        price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="profit_price")
    with a2:
        K = st.number_input("K (number of suppliers)", min_value=1, value=int(st.session_state.profit_k), step=1, key="profit_K")
    with a3:
        st.write("Served users")
        st.metric("Fixed", str(SERVED_USERS))

    st.session_state.profit_k = int(K)

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.profit_picks[: int(K)]

    picks = st.multiselect("Your supplier set (must be exactly K)", options=suppliers, default=default_picks, key="profit_picks_widget")
    st.session_state.profit_picks = picks

    if st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="profit_run"):
        left, right = st.columns(2)

        # Manual
        with left:
            st.subheader("Manual")
            if len(picks) != int(K):
                st.error(f"Please select exactly K={int(K)} suppliers. Current: {len(picks)}")
            else:
                manual = _manual_metrics(suppliers_df, picks, float(price_per_user))
                ok = _feasible(manual["avg_env"], manual["avg_social"])
                if not ok:
                    st.error("Risk constraints violated (average env/social risk is above the cap).")

                st.metric("Total profit", f"{manual['profit_total']:.3f}")
                st.metric("Profit per user", f"{manual['profit_per_user']:.3f}")
                st.metric("Risk score", f"{manual['risk_score']:.4f}")
                st.write(f"Average env risk: **{manual['avg_env']:.3f}** (cap {ENV_CAP})")
                st.write(f"Average social risk: **{manual['avg_social']:.3f}** (cap {SOCIAL_CAP})")
                st.write(f"Average cost_score: **{manual['avg_cost']:.3f}**")

                if st.button("Add manual to leaderboard", use_container_width=True, key="profit_add_lb"):
                    _add_to_leaderboard(
                        mode="max_profit",
                        team=st.session_state.team_name,
                        k=int(K),
                        picks=list(picks),
                        metrics=manual,
                        score=float(manual["profit_total"]),
                        meta={"price_per_user": float(price_per_user)},
                    )
                    st.success("Added to leaderboard.")

        # Optimal
        with right:
            st.subheader("Optimal benchmark")
            cfg = ProfitRiskConfig(
                suppliers_to_select=int(K),
                served_users=int(SERVED_USERS),
                price_per_user=float(price_per_user),
                cost_scale=float(COST_SCALE),
                env_cap=float(ENV_CAP),
                social_cap=float(SOCIAL_CAP),
                output_flag=0,
                fixed_suppliers=None,
            )
            try:
                res = ProfitRiskMaxProfitAgent(suppliers_df, cfg).solve()
                st.metric("Total profit", f"{res['profit_total']:.3f}")
                st.metric("Profit per user", f"{res['profit_per_user']:.3f}")
                st.metric("Risk score", f"{res['risk_score']:.4f}")
                st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]))
                st.write(f"Average env risk: **{res['avg_env']:.3f}** (cap {ENV_CAP})")
                st.write(f"Average social risk: **{res['avg_social']:.3f}** (cap {SOCIAL_CAP})")
                st.write(f"Average cost_score: **{res['avg_cost']:.3f}**")
            except Exception as e:
                st.error(str(e))


# ============================
# Min Risk
# ============================
with tab_risk:
    st.header("Min Risk")
    st.caption("Minimize a normalized risk score while staying within the risk caps. Optionally require a minimum profit per user.")

    a1, a2, a3, a4 = st.columns(4)
    with a1:
        price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="risk_price")
    with a2:
        K = st.number_input("K (number of suppliers)", min_value=1, value=int(st.session_state.risk_k), step=1, key="risk_K")
    with a3:
        min_profit_per_user = st.number_input(
            "Minimum profit per user (optional)",
            min_value=-1_000_000.0,
            value=0.0,
            step=5.0,
            key="risk_profit_floor",
        )
    with a4:
        use_floor = st.checkbox("Enable profit floor", value=False, key="risk_use_floor")

    st.session_state.risk_k = int(K)

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.risk_picks[: int(K)]
    picks = st.multiselect("Your supplier set (must be exactly K)", options=suppliers, default=default_picks, key="risk_picks_widget")
    st.session_state.risk_picks = picks

    if st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="risk_run"):
        left, right = st.columns(2)

        with left:
            st.subheader("Manual")
            if len(picks) != int(K):
                st.error(f"Please select exactly K={int(K)} suppliers. Current: {len(picks)}")
            else:
                manual = _manual_metrics(suppliers_df, picks, float(price_per_user))
                ok = _feasible(manual["avg_env"], manual["avg_social"])
                if not ok:
                    st.error("Risk constraints violated (average env/social risk is above the cap).")

                st.metric("Risk score", f"{manual['risk_score']:.4f}")
                st.metric("Total profit", f"{manual['profit_total']:.3f}")
                st.metric("Profit per user", f"{manual['profit_per_user']:.3f}")
                st.write(f"Average env risk: **{manual['avg_env']:.3f}** (cap {ENV_CAP})")
                st.write(f"Average social risk: **{manual['avg_social']:.3f}** (cap {SOCIAL_CAP})")
                st.write(f"Average cost_score: **{manual['avg_cost']:.3f}**")

                if st.button("Add manual to leaderboard", use_container_width=True, key="risk_add_lb"):
                    # smaller risk is better -> higher score
                    _add_to_leaderboard(
                        mode="min_risk",
                        team=st.session_state.team_name,
                        k=int(K),
                        picks=list(picks),
                        metrics=manual,
                        score=float(-manual["risk_score"]),
                        meta={"price_per_user": float(price_per_user), "profit_floor_enabled": bool(use_floor)},
                    )
                    st.success("Added to leaderboard.")

        with right:
            st.subheader("Optimal benchmark")
            cfg = ProfitRiskMinRiskConfig(
                suppliers_to_select=int(K),
                served_users=int(SERVED_USERS),
                price_per_user=float(price_per_user),
                cost_scale=float(COST_SCALE),
                env_cap=float(ENV_CAP),
                social_cap=float(SOCIAL_CAP),
                output_flag=0,
                fixed_suppliers=None,
                min_profit_per_user=(float(min_profit_per_user) if use_floor else None),
            )
            try:
                res = ProfitRiskMinRiskAgent(suppliers_df, cfg).solve()
                st.metric("Risk score", f"{res['risk_score']:.4f}")
                st.metric("Total profit", f"{res['profit_total']:.3f}")
                st.metric("Profit per user", f"{res['profit_per_user']:.3f}")
                st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]))
                st.write(f"Average env risk: **{res['avg_env']:.3f}** (cap {ENV_CAP})")
                st.write(f"Average social risk: **{res['avg_social']:.3f}** (cap {SOCIAL_CAP})")
                st.write(f"Average cost_score: **{res['avg_cost']:.3f}**")
            except Exception as e:
                st.error(str(e))


# ============================
# Profit–Risk curve + Leaderboard
# ============================
with tab_trade:
    st.header("Profit–Risk curve")
    st.caption(
        "We generate a curve by solving a weighted trade-off objective: profit vs risk. "
        "All solutions still satisfy the hard risk caps."
    )

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="trade_price")
    with b2:
        K = st.number_input("K (number of suppliers)", min_value=1, value=int(st.session_state.trade_k), step=1, key="trade_K")
    with b3:
        n_points = st.slider("Number of curve points", min_value=3, max_value=25, value=11, step=2, key="trade_n")
    with b4:
        risk_scale = st.number_input(
            "Risk scale (how expensive risk is)",
            min_value=0.0,
            value=float(price_per_user),
            step=10.0,
            key="trade_risk_scale",
        )

    st.session_state.trade_k = int(K)

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.trade_picks[: int(K)]
    picks = st.multiselect(
        "(Optional) Pick exactly K suppliers to score your solution at a chosen weight",
        options=suppliers,
        default=default_picks,
        key="trade_picks_widget",
    )
    st.session_state.trade_picks = picks

    w_profit = st.slider("Profit weight (w)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="trade_w")

    if st.button("Compute curve (optimal)", type="primary", use_container_width=True, key="trade_curve"):
        rows = []
        try:
            for idx in range(int(n_points)):
                w = idx / float(n_points - 1)
                cfg = ProfitRiskTradeoffConfig(
                    suppliers_to_select=int(K),
                    served_users=int(SERVED_USERS),
                    price_per_user=float(price_per_user),
                    cost_scale=float(COST_SCALE),
                    env_cap=float(ENV_CAP),
                    social_cap=float(SOCIAL_CAP),
                    output_flag=0,
                    fixed_suppliers=None,
                    weight_on_profit=float(w),
                    risk_scale=float(risk_scale),
                )
                res = ProfitRiskTradeoffAgent(suppliers_df, cfg).solve()
                rows.append(
                    {
                        "w_profit": w,
                        "profit_total": res["profit_total"],
                        "profit_per_user": res["profit_per_user"],
                        "risk_score": res["risk_score"],
                        "avg_env": res["avg_env"],
                        "avg_social": res["avg_social"],
                        "avg_cost": res["avg_cost"],
                        "suppliers": ", ".join(res["chosen_suppliers"]),
                        "objective": res["objective_value"],
                    }
                )

            curve_df = pd.DataFrame(rows)
            st.session_state._last_curve_df = curve_df

            st.success("Curve computed.")
            st.dataframe(curve_df, use_container_width=True, hide_index=True)
            st.scatter_chart(curve_df, x="profit_total", y="risk_score")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.subheader("Manual vs optimal at your chosen w")

    if st.button("Run score", use_container_width=True, key="trade_run"):
        try:
            # Manual (if provided)
            if len(picks) == int(K):
                cfg_m = ProfitRiskTradeoffConfig(
                    suppliers_to_select=int(K),
                    served_users=int(SERVED_USERS),
                    price_per_user=float(price_per_user),
                    cost_scale=float(COST_SCALE),
                    env_cap=float(ENV_CAP),
                    social_cap=float(SOCIAL_CAP),
                    output_flag=0,
                    fixed_suppliers=list(picks),
                    weight_on_profit=float(w_profit),
                    risk_scale=float(risk_scale),
                )
                man_res = ProfitRiskTradeoffAgent(suppliers_df, cfg_m).solve()
                man_metrics = {
                    "avg_env": man_res["avg_env"],
                    "avg_social": man_res["avg_social"],
                    "avg_cost": man_res["avg_cost"],
                    "profit_per_user": man_res["profit_per_user"],
                    "profit_total": man_res["profit_total"],
                    "risk_score": man_res["risk_score"],
                }
                man_score = float(man_res["objective_value"])
            else:
                man_res, man_metrics, man_score = None, None, None

            # Optimal
            cfg_o = ProfitRiskTradeoffConfig(
                suppliers_to_select=int(K),
                served_users=int(SERVED_USERS),
                price_per_user=float(price_per_user),
                cost_scale=float(COST_SCALE),
                env_cap=float(ENV_CAP),
                social_cap=float(SOCIAL_CAP),
                output_flag=0,
                fixed_suppliers=None,
                weight_on_profit=float(w_profit),
                risk_scale=float(risk_scale),
            )
            opt_res = ProfitRiskTradeoffAgent(suppliers_df, cfg_o).solve()
            opt_metrics = {
                "avg_env": opt_res["avg_env"],
                "avg_social": opt_res["avg_social"],
                "avg_cost": opt_res["avg_cost"],
                "profit_per_user": opt_res["profit_per_user"],
                "profit_total": opt_res["profit_total"],
                "risk_score": opt_res["risk_score"],
            }
            opt_score = float(opt_res["objective_value"])

            L, R = st.columns(2)

            with L:
                st.markdown("#### Manual")
                if man_res is None:
                    st.warning(f"To score manually, select exactly K={int(K)} suppliers.")
                else:
                    st.metric("Objective", f"{man_score:.3f}")
                    st.metric("Total profit", f"{man_metrics['profit_total']:.3f}")
                    st.metric("Risk score", f"{man_metrics['risk_score']:.4f}")
                    st.write("Suppliers:", ", ".join(man_res["chosen_suppliers"]))
                    st.write(f"Avg env: **{man_metrics['avg_env']:.3f}** (cap {ENV_CAP})")
                    st.write(f"Avg social: **{man_metrics['avg_social']:.3f}** (cap {SOCIAL_CAP})")

                    if st.button("Add manual to leaderboard", use_container_width=True, key="trade_add_lb"):
                        _add_to_leaderboard(
                            mode="tradeoff",
                            team=st.session_state.team_name,
                            k=int(K),
                            picks=list(picks),
                            metrics=man_metrics,
                            score=float(man_score),
                            meta={"price_per_user": float(price_per_user), "w_profit": float(w_profit), "risk_scale": float(risk_scale)},
                        )
                        st.success("Added to leaderboard.")

            with R:
                st.markdown("#### Optimal")
                st.metric("Objective", f"{opt_score:.3f}")
                st.metric("Total profit", f"{opt_metrics['profit_total']:.3f}")
                st.metric("Risk score", f"{opt_metrics['risk_score']:.4f}")
                st.write("Suppliers:", ", ".join(opt_res["chosen_suppliers"]))
                st.write(f"Avg env: **{opt_metrics['avg_env']:.3f}** (cap {ENV_CAP})")
                st.write(f"Avg social: **{opt_metrics['avg_social']:.3f}** (cap {SOCIAL_CAP})")

            if man_score is not None:
                st.divider()
                st.write(f"Optimal − Manual objective gap: **{(opt_score - man_score):.3f}**")

        except Exception as e:
            st.error(str(e))

    st.divider()
    _render_leaderboard("Leaderboard (all modes)")
