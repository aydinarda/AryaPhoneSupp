import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from MinCostAgent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    load_supplier_user_tables,
    ProfitRiskConfig,
    ProfitRiskMinRiskConfig,
    ProfitRiskMaxProfitAgent,
    ProfitRiskMinRiskAgent,
    ProfitRiskCurveAgent,
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

if "team_name" not in st.session_state:
    st.session_state.team_name = ""

if "manual_picks" not in st.session_state:
    st.session_state.manual_picks = []


# ============================
# Helpers
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


def _risk_score(avg_env: float, avg_social: float) -> float:
    # normalized score: 1.0 means you're exactly at both caps
    return 0.5 * ((avg_env / float(ENV_CAP)) + (avg_social / float(SOCIAL_CAP)))


def _manual_metrics(suppliers_df: pd.DataFrame, picks: List[str], price_per_user: float) -> Dict[str, float]:
    picks = [str(x) for x in (picks or [])]
    if not picks:
        return {
            "k": 0,
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "profit_per_user": 0.0,
            "profit_total": 0.0,
            "risk_score": 0.0,
            "feasible": False,
        }

    sub = suppliers_df[suppliers_df["supplier_id"].astype(str).isin(picks)].copy()
    k = int(len(sub))
    if k == 0:
        return {
            "k": 0,
            "avg_env": 0.0,
            "avg_social": 0.0,
            "avg_cost": 0.0,
            "profit_per_user": 0.0,
            "profit_total": 0.0,
            "risk_score": 0.0,
            "feasible": False,
        }

    avg_env = float(sub["env_risk"].mean())
    avg_social = float(sub["social_risk"].mean())
    avg_cost = float(sub["cost_score"].mean())

    profit_per_user = float(price_per_user) - float(COST_SCALE) * avg_cost
    profit_total = float(SERVED_USERS) * profit_per_user
    rs = _risk_score(avg_env, avg_social)

    feasible = (avg_env <= float(ENV_CAP) + 1e-9) and (avg_social <= float(SOCIAL_CAP) + 1e-9)

    return {
        "k": k,
        "avg_env": avg_env,
        "avg_social": avg_social,
        "avg_cost": avg_cost,
        "profit_per_user": profit_per_user,
        "profit_total": profit_total,
        "risk_score": float(rs),
        "feasible": bool(feasible),
    }


def _add_to_leaderboard(mode: str, team: str, picks: List[str], metrics: Dict[str, float], score: float, note: str = ""):
    st.session_state.leaderboard.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "team": team.strip() or "(anonymous)",
            "mode": mode,
            "k": int(metrics.get("k", 0)),
            "suppliers": ", ".join([str(x) for x in (picks or [])]),
            "profit_total": float(metrics.get("profit_total", 0.0)),
            "profit_per_user": float(metrics.get("profit_per_user", 0.0)),
            "avg_env": float(metrics.get("avg_env", 0.0)),
            "avg_social": float(metrics.get("avg_social", 0.0)),
            "avg_cost": float(metrics.get("avg_cost", 0.0)),
            "risk_score": float(metrics.get("risk_score", 0.0)),
            "feasible": bool(metrics.get("feasible", False)),
            "score": float(score),
            "note": note.strip(),
        }
    )


def _render_leaderboard(title: str, mode_filter: Optional[str] = None):
    st.markdown(f"### {title}")
    df = pd.DataFrame(st.session_state.leaderboard)
    if df.empty:
        st.info("Leaderboard is empty. Submit a run to get started.")
        return

    if mode_filter is not None:
        df = df[df["mode"] == mode_filter].copy()

    if df.empty:
        st.info("No entries for this mode yet.")
        return

    # score higher is better
    df = df.sort_values(["score", "time"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))
    st.dataframe(df, use_container_width=True, hide_index=True)


# ============================
# Load data
# ============================
excel_path = Path(DEFAULT_XLSX_PATH)
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Put it next to UI.py")
    st.stop()

suppliers_df, _users_df = _load_data(excel_path)

if not GUROBI_AVAILABLE:
    st.error("gurobipy is not installed / licensed in this environment.")
    st.stop()


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
        st.write("The delivered product is the **average** of the selected suppliers.")
        st.write("There is **no fixed K** (number of suppliers).")


# ============================
# Supplier roster (game card)
# ============================
with st.expander("Supplier roster (how good / bad they are)", expanded=True):
    price_preview = st.number_input("Price per user (preview)", min_value=0.0, value=100.0, step=5.0, key="price_preview")

    df = suppliers_df.copy()
    df["env_risk_%"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["social_risk_%"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["cost_%"] = (_norm01(df["cost_score"]) * 100).round(1)

    df["profit_per_user_if_single"] = (float(price_preview) - COST_SCALE * df["cost_score"]).astype(float)
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
    ]

    col_cfg = {
        "env_risk_%": st.column_config.ProgressColumn("Env risk (bad)", min_value=0, max_value=100, format="%.1f"),
        "social_risk_%": st.column_config.ProgressColumn("Social risk (bad)", min_value=0, max_value=100, format="%.1f"),
        "cost_%": st.column_config.ProgressColumn("Cost (bad)", min_value=0, max_value=100, format="%.1f"),
        "profit_per_user_if_single": st.column_config.NumberColumn("Profit/user if single", format="%.3f"),
        "profit_total_if_single": st.column_config.NumberColumn("Total profit if single", format="%.3f"),
        "risk_score_if_single": st.column_config.NumberColumn("Risk score if single", format="%.4f"),
    }

    st.dataframe(df[show_cols], use_container_width=True, hide_index=True, column_config=col_cfg)


# ============================
# Tabs
# ============================
tab_profit, tab_risk, tab_curve = st.tabs(["Max Profit", "Min Risk", "Profit–Risk Curve + Leaderboard"])


# ============================
# Max Profit
# ============================
with tab_profit:
    st.markdown("## Max Profit")

    price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="profit_price")

    picks = st.multiselect(
        "Pick any suppliers (1 or more)",
        options=suppliers_df["supplier_id"].astype(str).tolist(),
        default=st.session_state.manual_picks,
        key="profit_picks",
    )
    st.session_state.manual_picks = picks

    note = st.text_area("Why did you choose these suppliers? (optional)", key="profit_note", height=90)

    manual = _manual_metrics(suppliers_df, picks, float(price_per_user))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Manual total profit", f"{manual['profit_total']:.3f}")
    m2.metric("Manual risk score", f"{manual['risk_score']:.4f}")
    m3.metric("Avg env / avg social", f"{manual['avg_env']:.3f} / {manual['avg_social']:.3f}")
    m4.metric("# suppliers", f"{manual['k']}")

    if manual["k"] == 0:
        st.warning("Pick at least 1 supplier.")
    elif not manual["feasible"]:
        st.error("Manual selection violates the risk caps (avg env and/or avg social).")
    else:
        st.success("Manual selection is feasible under the risk caps.")

    run = st.button("Run optimal benchmark", type="primary", use_container_width=True, key="profit_run")

    if run:
        cfg = ProfitRiskConfig(
            served_users=int(SERVED_USERS),
            price_per_user=float(price_per_user),
            env_cap=float(ENV_CAP),
            social_cap=float(SOCIAL_CAP),
            cost_scale=float(COST_SCALE),
            output_flag=0,
        )
        res = ProfitRiskMaxProfitAgent(suppliers_df, cfg).solve()

        st.divider()
        st.markdown("### Optimal (no fixed K)")

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Optimal total profit", f"{res['profit_total']:.3f}")
        cB.metric("Optimal risk score", f"{res['risk_score']:.4f}")
        cC.metric("Avg env / avg social", f"{res['avg_env']:.3f} / {res['avg_social']:.3f}")
        cD.metric("# suppliers", f"{res['k']}")

        st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]))

        if manual["k"] > 0 and manual["feasible"]:
            st.markdown("#### Gap")
            st.write(f"Optimal − Manual total profit: **{(res['profit_total'] - manual['profit_total']):.3f}**")

    if st.button("Submit manual to leaderboard", use_container_width=True, key="profit_submit"):
        if manual["k"] == 0:
            st.error("Cannot submit: pick at least 1 supplier.")
        else:
            _add_to_leaderboard(
                mode="max_profit",
                team=st.session_state.team_name,
                picks=picks,
                metrics=manual,
                score=float(manual["profit_total"]),
                note=note,
            )
            st.success("Submitted to leaderboard (local session).")


# ============================
# Min Risk
# ============================
with tab_risk:
    st.markdown("## Min Risk")

    price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="risk_price")
    profit_floor = st.number_input(
        "Minimum profit per user (optional)",
        value=0.0,
        step=1.0,
        key="risk_profit_floor",
        help="If set > 0, the optimizer must keep profit per user at least this value.",
    )

    picks = st.multiselect(
        "Pick any suppliers (1 or more)",
        options=suppliers_df["supplier_id"].astype(str).tolist(),
        default=st.session_state.manual_picks,
        key="risk_picks",
    )

    note = st.text_area("Why did you choose these suppliers? (optional)", key="risk_note", height=90)

    manual = _manual_metrics(suppliers_df, picks, float(price_per_user))

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Manual risk score", f"{manual['risk_score']:.4f}")
    r2.metric("Manual total profit", f"{manual['profit_total']:.3f}")
    r3.metric("Avg env / avg social", f"{manual['avg_env']:.3f} / {manual['avg_social']:.3f}")
    r4.metric("# suppliers", f"{manual['k']}")

    if manual["k"] == 0:
        st.warning("Pick at least 1 supplier.")
    elif not manual["feasible"]:
        st.error("Manual selection violates the risk caps (avg env and/or avg social).")
    elif manual["profit_per_user"] < float(profit_floor) - 1e-9:
        st.error("Manual selection does not meet the minimum profit-per-user constraint.")
    else:
        st.success("Manual selection is feasible for the current constraints.")

    run = st.button("Run optimal benchmark", type="primary", use_container_width=True, key="risk_run")

    if run:
        cfg = ProfitRiskMinRiskConfig(
            served_users=int(SERVED_USERS),
            price_per_user=float(price_per_user),
            env_cap=float(ENV_CAP),
            social_cap=float(SOCIAL_CAP),
            cost_scale=float(COST_SCALE),
            profit_floor_per_user=float(profit_floor),
            output_flag=0,
        )
        res = ProfitRiskMinRiskAgent(suppliers_df, cfg).solve()

        st.divider()
        st.markdown("### Optimal (no fixed K)")

        cA, cB, cC, cD = st.columns(4)
        cA.metric("Optimal risk score", f"{res['risk_score']:.4f}")
        cB.metric("Optimal total profit", f"{res['profit_total']:.3f}")
        cC.metric("Avg env / avg social", f"{res['avg_env']:.3f} / {res['avg_social']:.3f}")
        cD.metric("# suppliers", f"{res['k']}")

        st.write("Chosen suppliers:", ", ".join(res["chosen_suppliers"]))

    if st.button("Submit manual to leaderboard", use_container_width=True, key="risk_submit"):
        if manual["k"] == 0:
            st.error("Cannot submit: pick at least 1 supplier.")
        else:
            # score for min-risk: higher is better on leaderboard -> invert risk
            score = 1.0 / max(1e-9, float(manual["risk_score"]))
            _add_to_leaderboard(
                mode="min_risk",
                team=st.session_state.team_name,
                picks=picks,
                metrics=manual,
                score=float(score),
                note=note,
            )
            st.success("Submitted to leaderboard (local session).")


# ============================
# Profit–Risk Curve + Leaderboard
# ============================
with tab_curve:
    st.markdown("## Profit–Risk Curve")

    price_per_user = st.number_input("Price per user", min_value=0.0, value=100.0, step=5.0, key="curve_price")
    n_points = st.slider("Number of curve points", min_value=3, max_value=25, value=9, step=1, key="curve_n")

    st.caption(
        "We generate a profit–risk curve by tightening a **combined risk cap** (risk_score ≤ cap) and maximizing profit. "
        "This does not use a profit-weight slider."
    )

    if st.button("Compute curve", type="primary", use_container_width=True, key="curve_run"):
        cfg = ProfitRiskConfig(
            served_users=int(SERVED_USERS),
            price_per_user=float(price_per_user),
            env_cap=float(ENV_CAP),
            social_cap=float(SOCIAL_CAP),
            cost_scale=float(COST_SCALE),
            output_flag=0,
        )
        curve = ProfitRiskCurveAgent(suppliers_df, cfg).compute_curve(n_points=int(n_points))
        curve_df = pd.DataFrame(curve)

        st.session_state._last_curve = curve_df

        st.dataframe(curve_df, use_container_width=True, hide_index=True)

        # plot profit vs risk
        st.scatter_chart(curve_df, x="risk_score", y="profit_total")

    st.divider()
    _render_leaderboard("Leaderboard (local session)")

    st.divider()
    st.markdown("## Suggested class competition setup (GitHub-friendly)")
    st.markdown(
        """
**Goal:** let students submit a solution + short justification, and maintain a shared leaderboard in the repo.

### Recommended (no Streamlit write access): GitHub Issues → GitHub Action → leaderboard.csv
1. **Students submit** by opening a GitHub Issue using a template (fields: team, supplier_ids, short reasoning).
2. A **GitHub Action** runs on `issues` events (or on a schedule), **parses issues**, **recomputes metrics** using the official Excel, and appends to `leaderboard.csv` (or `.xlsx`).
3. The Action commits the updated leaderboard back to the repo.
4. Streamlit **only reads** `leaderboard.csv` from the repo (easy, stable, no secrets).

Why this works well:
- No auth needed inside Streamlit.
- You avoid cheating: the Action recomputes profit/risk from supplier IDs.
- Issues become the "post" + discussion thread.

### If you want one-click submit inside Streamlit
Still possible, but you need a GitHub token:
- Store a fine‑grained PAT in Streamlit secrets (`.streamlit/secrets.toml` on Streamlit Cloud).
- On submit, Streamlit calls GitHub API to create an Issue/Discussion.
- Action continues to build the leaderboard file.

If you want, I can give you:
- an Issue template (`.github/ISSUE_TEMPLATE/submission.yml`)
- an Action workflow (`.github/workflows/leaderboard.yml`)
- a small parser script that writes `leaderboard.csv`
"""
    )
