import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import pandas as pd

# In your repo you can either:
#   A) replace MinCostAgent.py with this updated version, OR
#   B) keep the filename MinCostAgent_updated.py and let this import pick it up.
try:
    from MinCostAgent_updated import (  # type: ignore
        DEFAULT_XLSX_PATH,
        GUROBI_AVAILABLE,
        Policy,
        load_supplier_user_tables,
        MaxProfitConfig,
        MaxProfitAgent,
        MaxUtilConfig,
        MaxUtilAgent,
        TradeoffConfig,
        TradeoffAgent,
    )
except Exception:  # pragma: no cover
    from MinCostAgent import (  # type: ignore
        DEFAULT_XLSX_PATH,
        GUROBI_AVAILABLE,
        Policy,
        load_supplier_user_tables,
        MaxProfitConfig,
        MaxProfitAgent,
        MaxUtilConfig,
        MaxUtilAgent,
        TradeoffConfig,
        TradeoffAgent,
    )


# ============================
# App config
# ============================
st.set_page_config(page_title="Arya Case — Supplier Selection Game", layout="wide")

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
# Fixed policy (EDIT HERE)
# ============================
# Policy is *hidden and fixed* (no UI controls). If you want a different scenario,
# edit these constants.
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


# ============================
# Session state
# ============================
if "leaderboard" not in st.session_state:
    st.session_state.leaderboard = []

for k in [
    "profit_picks",
    "util_picks",
    "trade_picks",
    "profit_k",
    "util_k",
    "trade_k",
]:
    if k not in st.session_state:
        st.session_state[k] = [] if k.endswith("_picks") else 1

if "team_name" not in st.session_state:
    st.session_state.team_name = ""


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


def _supplier_overview_table(suppliers_df: pd.DataFrame, users_df: pd.DataFrame, pol: Policy) -> pd.DataFrame:
    df = suppliers_df.copy()

    # Normalized "bad" and "good" bars
    df["env_bad_%"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["social_bad_%"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["cost_bad_%"] = (_norm01(df["cost_score"]) * 100).round(1)
    df["lowq_bad_%"] = (_norm01(df["low_quality"]) * 100).round(1)
    df["strategic_good_%"] = (_norm01(df["strategic"]) * 100).round(1)
    df["improve_good_%"] = (_norm01(df["improvement"]) * 100).round(1)

    # Expected utility using *average* user weights (informative, not objective)
    uavg = users_df[["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]].mean()
    df["expected_utility"] = (
        float(uavg["w_env"]) * (pol.env_mult * df["env_risk"])
        + float(uavg["w_social"]) * (pol.social_mult * df["social_risk"])
        + float(uavg["w_cost"]) * (pol.cost_mult * df["cost_score"])
        + float(uavg["w_strategic"]) * (pol.strategic_mult * df["strategic"])
        + float(uavg["w_improvement"]) * (pol.improvement_mult * df["improvement"])
        + float(uavg["w_low_quality"]) * (pol.low_quality_mult * df["low_quality"])
    ).astype(float)

    df["policy_cost"] = (pol.cost_mult * df["cost_score"]).astype(float)

    show_cols = [
        "supplier_id",
        "policy_cost",
        "expected_utility",
        "env_bad_%",
        "social_bad_%",
        "cost_bad_%",
        "lowq_bad_%",
        "strategic_good_%",
        "improve_good_%",
        "child_labor",
        "banned_chem",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "low_quality",
    ]
    return df[show_cols].copy()


def _totals_from_matches(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or not len(df):
        return {"profit": 0.0, "cost": 0.0, "utility": 0.0, "matches": 0.0}

    cost = float(pd.to_numeric(df.get("cost_prod"), errors="coerce").fillna(0.0).sum())
    profit = float(pd.to_numeric(df.get("margin"), errors="coerce").fillna(0.0).sum())
    utility = float(pd.to_numeric(df.get("utility"), errors="coerce").fillna(0.0).sum())
    return {"profit": profit, "cost": cost, "utility": utility, "matches": float(len(df))}


def _add_to_leaderboard(
    mode: str,
    team: str,
    chosen_suppliers: List[str],
    totals: Dict[str, float],
    score: float,
    meta: Dict[str, Any],
):
    st.session_state.leaderboard.append(
        {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "team": team.strip() or "(anonymous)",
            "mode": mode,
            "K": int(meta.get("K", 0)),
            "suppliers": ", ".join(chosen_suppliers),
            "profit": float(totals.get("profit", 0.0)),
            "cost": float(totals.get("cost", 0.0)),
            "utility": float(totals.get("utility", 0.0)),
            "score": float(score),
            **{k: v for k, v in meta.items() if k not in {"K"}},
        }
    )


def _render_leaderboard(title: str, mode_filter: Optional[str] = None):
    st.markdown(f"### {title}")
    df = pd.DataFrame(st.session_state.leaderboard)
    if df.empty:
        st.info("Henüz leaderboard boş. Bir deneme ekleyin.")
        return

    if mode_filter:
        df = df[df["mode"] == mode_filter].copy()

    if df.empty:
        st.info("Bu mod için henüz kayıt yok.")
        return

    df = df.sort_values(["score", "time"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================
# Header
# ============================
st.title("Arya Phones — Supplier Selection Game")

team_col, pol_col = st.columns([2, 3])
with team_col:
    st.session_state.team_name = st.text_input("Takım adı (leaderboard için)", value=st.session_state.team_name)

with pol_col:
    with st.expander("Policy (fixed, read-only)", expanded=False):
        st.json(FIXED_POLICY.to_dict())

excel_path = Path(DEFAULT_XLSX_PATH)
if not excel_path.exists():
    st.error(f"Excel file not found: {excel_path}. Place it next to UI.py in your repo.")
    st.stop()

suppliers_df, users_df = _load_data(excel_path)

st.caption(
    "Bu arayüzde policy değiştirilemez (fixed). Her modda siz supplier setini seçersiniz; sistem de bu set altında en iyi matching'i bulur ve optimumla kıyaslar."
)


# ============================
# Supplier overview
# ============================
with st.expander("Supplier’ları gör (game kartı / tablo)", expanded=True):
    ov = _supplier_overview_table(suppliers_df, users_df, FIXED_POLICY)

    col_cfg = {
        "env_bad_%": st.column_config.ProgressColumn("Env (bad)", min_value=0, max_value=100, format="%.1f"),
        "social_bad_%": st.column_config.ProgressColumn("Social (bad)", min_value=0, max_value=100, format="%.1f"),
        "cost_bad_%": st.column_config.ProgressColumn("Cost (bad)", min_value=0, max_value=100, format="%.1f"),
        "lowq_bad_%": st.column_config.ProgressColumn("Low quality (bad)", min_value=0, max_value=100, format="%.1f"),
        "strategic_good_%": st.column_config.ProgressColumn("Strategic (good)", min_value=0, max_value=100, format="%.1f"),
        "improve_good_%": st.column_config.ProgressColumn("Improvement (good)", min_value=0, max_value=100, format="%.1f"),
        "policy_cost": st.column_config.NumberColumn("Policy cost", format="%.3f"),
        "expected_utility": st.column_config.NumberColumn("Expected utility", format="%.3f"),
    }

    st.dataframe(ov, use_container_width=True, hide_index=True, column_config=col_cfg)


# ============================
# Pages (tabs)
# ============================
tab_profit, tab_util, tab_trade = st.tabs(["Max Profit", "Max Utility", "Trade-off + Leaderboard"])


# ============================
# Max Profit page
# ============================
with tab_profit:
    st.markdown("## Max Profit")

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="profit_price")
    with c2:
        min_utility = st.number_input("Min utility (per match)", value=0.0, step=1.0, key="profit_minutil")
    with c3:
        K = st.number_input("K (suppliers)", min_value=1, value=int(st.session_state.profit_k), step=1, key="profit_K")
    with c4:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="profit_lastn")
    with c5:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="profit_capacity")

    st.session_state.profit_k = int(K)

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.profit_picks

    if len(default_picks) != int(K):
        default_picks = default_picks[: int(K)]

    picks = st.multiselect(
        "Supplier seç (exactly K)",
        options=suppliers,
        default=default_picks,
        key="profit_picks_widget",
    )

    st.session_state.profit_picks = picks

    run = st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="profit_run")

    if run:
        left, right = st.columns(2)

        # Manual
        with left:
            st.markdown("### Manual sonuç")
            if len(picks) != int(K):
                st.error(f"Lütfen tam olarak K={int(K)} supplier seçin. Şu an: {len(picks)}")
            else:
                cfg_m = MaxProfitConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(K),
                    price_per_match=float(price_per_match),
                    min_utility=float(min_utility),
                    output_flag=0,
                    fixed_suppliers=list(picks),
                )
                try:
                    res_m = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg_m).solve()
                    totals_m = _totals_from_matches(res_m["matches"])

                    st.metric("Manual profit", f"{totals_m['profit']:.3f}")
                    st.metric("Manual utility (sum)", f"{totals_m['utility']:.3f}")
                    st.metric("Manual matches", f"{int(totals_m['matches'])}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]))
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)

                    if st.button("Add manual to leaderboard", use_container_width=True, key="profit_add_lb"):
                        _add_to_leaderboard(
                            mode="max_profit",
                            team=st.session_state.team_name,
                            chosen_suppliers=list(picks),
                            totals=totals_m,
                            score=float(totals_m["profit"]),
                            meta={"K": int(K), "P": float(price_per_match), "min_util": float(min_utility)},
                        )
                        st.success("Leaderboard'a eklendi.")
                except Exception as e:
                    st.error(str(e))

        # Optimal
        with right:
            st.markdown("### Optimal benchmark")
            cfg_o = MaxProfitConfig(
                last_n_users=int(last_n_users),
                capacity=int(capacity),
                suppliers_to_select=int(K),
                price_per_match=float(price_per_match),
                min_utility=float(min_utility),
                output_flag=0,
                fixed_suppliers=None,
            )
            try:
                res_o = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg_o).solve()
                totals_o = _totals_from_matches(res_o["matches"])

                st.metric("Optimal profit", f"{totals_o['profit']:.3f}")
                st.metric("Optimal utility (sum)", f"{totals_o['utility']:.3f}")
                st.metric("Optimal matches", f"{int(totals_o['matches'])}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]))
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))


# ============================
# Max Utility page
# ============================
with tab_util:
    st.markdown("## Max Utility")

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        price_per_match = st.number_input("(Reporting) Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="util_price")
    with c2:
        min_utility = st.number_input("Min utility (per match)", value=0.0, step=1.0, key="util_minutil")
    with c3:
        K = st.number_input("K (suppliers)", min_value=1, value=int(st.session_state.util_k), step=1, key="util_K")
    with c4:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="util_lastn")
    with c5:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="util_capacity")
    with c6:
        matches_to_make = st.number_input(
            "Matches to make (exact)",
            min_value=0,
            value=min(int(capacity), int(last_n_users)),
            step=1,
            key="util_matches_exact",
        )

    st.session_state.util_k = int(K)

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.util_picks
    if len(default_picks) != int(K):
        default_picks = default_picks[: int(K)]

    picks = st.multiselect(
        "Supplier seç (exactly K)",
        options=suppliers,
        default=default_picks,
        key="util_picks_widget",
    )
    st.session_state.util_picks = picks

    run = st.button("Run (manual vs optimal)", type="primary", use_container_width=True, key="util_run")

    if run:
        left, right = st.columns(2)

        with left:
            st.markdown("### Manual sonuç")
            if len(picks) != int(K):
                st.error(f"Lütfen tam olarak K={int(K)} supplier seçin. Şu an: {len(picks)}")
            else:
                cfg_m = MaxUtilConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(K),
                    matches_to_make=int(matches_to_make),
                    min_utility=float(min_utility),
                    price_per_match=float(price_per_match),
                    output_flag=0,
                    fixed_suppliers=list(picks),
                )
                try:
                    res_m = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg_m).solve()
                    totals_m = _totals_from_matches(res_m["matches"])

                    st.metric("Manual utility (sum)", f"{totals_m['utility']:.3f}")
                    st.metric("Manual profit (sum)", f"{totals_m['profit']:.3f}")
                    st.metric("Manual matches", f"{int(totals_m['matches'])}")
                    st.write("Chosen suppliers:", ", ".join(res_m["chosen_suppliers"]))
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)

                    if st.button("Add manual to leaderboard", use_container_width=True, key="util_add_lb"):
                        _add_to_leaderboard(
                            mode="max_utility",
                            team=st.session_state.team_name,
                            chosen_suppliers=list(picks),
                            totals=totals_m,
                            score=float(totals_m["utility"]),
                            meta={
                                "K": int(K),
                                "P": float(price_per_match),
                                "min_util": float(min_utility),
                                "matches_exact": int(matches_to_make),
                            },
                        )
                        st.success("Leaderboard'a eklendi.")
                except Exception as e:
                    st.error(str(e))

        with right:
            st.markdown("### Optimal benchmark")
            cfg_o = MaxUtilConfig(
                last_n_users=int(last_n_users),
                capacity=int(capacity),
                suppliers_to_select=int(K),
                matches_to_make=int(matches_to_make),
                min_utility=float(min_utility),
                price_per_match=float(price_per_match),
                output_flag=0,
                fixed_suppliers=None,
            )
            try:
                res_o = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg_o).solve()
                totals_o = _totals_from_matches(res_o["matches"])

                st.metric("Optimal utility (sum)", f"{totals_o['utility']:.3f}")
                st.metric("Optimal profit (sum)", f"{totals_o['profit']:.3f}")
                st.metric("Optimal matches", f"{int(totals_o['matches'])}")
                st.write("Chosen suppliers:", ", ".join(res_o["chosen_suppliers"]))
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e))


# ============================
# Trade-off + Leaderboard
# ============================
with tab_trade:
    st.markdown("## Trade-off")

    if not GUROBI_AVAILABLE:
        st.error("gurobipy is not installed / licensed in this environment.")
        st.stop()

    st.caption(
        "Bu sayfada Profit ↔ Utility arasında trade-off yapıyoruz. "
        "Aşağıdaki slider 'profit ağırlığını' belirler. Ayrıca optimum curve (pareto benzeri) ve leaderboard burada." 
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        price_per_match = st.number_input("Selling price (P)", min_value=0.0, value=100.0, step=5.0, key="trade_price")
    with c2:
        min_utility = st.number_input("Min utility (per match)", value=0.0, step=1.0, key="trade_minutil")
    with c3:
        K = st.number_input("K (suppliers)", min_value=1, value=int(st.session_state.trade_k), step=1, key="trade_K")
    with c4:
        last_n_users = st.number_input("Last N users", min_value=1, value=6, step=1, key="trade_lastn")
    with c5:
        capacity = st.number_input("Capacity (max matches)", min_value=1, value=6, step=1, key="trade_capacity")

    st.session_state.trade_k = int(K)

    w_profit = st.slider("Profit ağırlığı (w)", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="trade_w")
    n_points = st.slider("Curve noktası sayısı", min_value=3, max_value=21, value=11, step=2, key="trade_npts")

    suppliers = suppliers_df["supplier_id"].astype(str).tolist()
    default_picks = st.session_state.trade_picks
    if len(default_picks) != int(K):
        default_picks = default_picks[: int(K)]

    picks = st.multiselect(
        "(Opsiyonel) K supplier seç ve skorunu optimumla kıyasla",
        options=suppliers,
        default=default_picks,
        key="trade_picks_widget",
    )
    st.session_state.trade_picks = picks

    run_curve = st.button("Compute trade-off curve (optimal)", type="primary", use_container_width=True, key="trade_curve")

    # Utility scaling: bring utility close to profit magnitude
    def _compute_scale() -> float:
        # profit optimum
        cfg_p = MaxProfitConfig(
            last_n_users=int(last_n_users),
            capacity=int(capacity),
            suppliers_to_select=int(K),
            price_per_match=float(price_per_match),
            min_utility=float(min_utility),
            output_flag=0,
            fixed_suppliers=None,
        )
        res_p = MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg_p).solve()
        tot_p = _totals_from_matches(res_p["matches"])  # profit

        # utility optimum
        cfg_u = MaxUtilConfig(
            last_n_users=int(last_n_users),
            capacity=int(capacity),
            suppliers_to_select=int(K),
            matches_to_make=min(int(capacity), int(last_n_users)),
            min_utility=float(min_utility),
            price_per_match=float(price_per_match),
            output_flag=0,
            fixed_suppliers=None,
        )
        res_u = MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg_u).solve()
        tot_u = _totals_from_matches(res_u["matches"])  # utility

        denom = max(1e-9, abs(float(tot_u["utility"])))
        return float(abs(float(tot_p["profit"])) / denom)

    if run_curve:
        try:
            util_scale = _compute_scale()
            weights = [i / (n_points - 1) for i in range(n_points)]

            rows = []
            for w in weights:
                cfg = TradeoffConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(K),
                    price_per_match=float(price_per_match),
                    weight_on_profit=float(w),
                    utility_weight=float(util_scale),
                    min_utility=float(min_utility),
                    matches_to_make=None,
                    output_flag=0,
                    fixed_suppliers=None,
                )
                res = TradeoffAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
                tot = _totals_from_matches(res["matches"])
                rows.append(
                    {
                        "w_profit": w,
                        "profit": tot["profit"],
                        "utility": tot["utility"],
                        "matches": int(tot["matches"]),
                        "suppliers": ", ".join(res["chosen_suppliers"]),
                        "objective": float(res["objective_value"]),
                    }
                )

            curve_df = pd.DataFrame(rows)
            st.success(f"Curve hazır. Utility scaling = {util_scale:.4f}")

            st.dataframe(curve_df, use_container_width=True, hide_index=True)
            st.scatter_chart(curve_df, x="profit", y="utility")

            st.session_state._last_curve_df = curve_df
            st.session_state._last_util_scale = util_scale
        except Exception as e:
            st.error(str(e))

    # Manual vs optimal for selected w
    st.divider()
    st.markdown("### Seçtiğiniz w için skor (manual vs optimal)")

    if st.button("Run trade-off score", use_container_width=True, key="trade_run"):
        try:
            util_scale = float(st.session_state.get("_last_util_scale") or _compute_scale())

            # Manual
            if len(picks) == int(K):
                cfg_m = TradeoffConfig(
                    last_n_users=int(last_n_users),
                    capacity=int(capacity),
                    suppliers_to_select=int(K),
                    price_per_match=float(price_per_match),
                    weight_on_profit=float(w_profit),
                    utility_weight=float(util_scale),
                    min_utility=float(min_utility),
                    matches_to_make=None,
                    output_flag=0,
                    fixed_suppliers=list(picks),
                )
                res_m = TradeoffAgent(suppliers_df, users_df, FIXED_POLICY, cfg_m).solve()
                tot_m = _totals_from_matches(res_m["matches"])
                score_m = float(res_m["objective_value"])
            else:
                res_m, tot_m, score_m = None, None, None

            # Optimal
            cfg_o = TradeoffConfig(
                last_n_users=int(last_n_users),
                capacity=int(capacity),
                suppliers_to_select=int(K),
                price_per_match=float(price_per_match),
                weight_on_profit=float(w_profit),
                utility_weight=float(util_scale),
                min_utility=float(min_utility),
                matches_to_make=None,
                output_flag=0,
                fixed_suppliers=None,
            )
            res_o = TradeoffAgent(suppliers_df, users_df, FIXED_POLICY, cfg_o).solve()
            tot_o = _totals_from_matches(res_o["matches"])
            score_o = float(res_o["objective_value"])

            A, B = st.columns(2)

            with A:
                st.markdown("#### Manual")
                if res_m is None:
                    st.warning(f"Manual skor için tam olarak K={int(K)} supplier seçmelisiniz.")
                else:
                    st.metric("Score (objective)", f"{score_m:.3f}")
                    st.metric("Profit", f"{tot_m['profit']:.3f}")
                    st.metric("Utility", f"{tot_m['utility']:.3f}")
                    st.write("Suppliers:", ", ".join(res_m["chosen_suppliers"]))
                    st.dataframe(res_m["matches"], use_container_width=True, hide_index=True)

                    if st.button("Add manual to leaderboard", use_container_width=True, key="trade_add_lb"):
                        _add_to_leaderboard(
                            mode="tradeoff",
                            team=st.session_state.team_name,
                            chosen_suppliers=list(picks),
                            totals=tot_m,
                            score=float(score_m),
                            meta={
                                "K": int(K),
                                "P": float(price_per_match),
                                "w_profit": float(w_profit),
                                "util_scale": float(util_scale),
                            },
                        )
                        st.success("Leaderboard'a eklendi.")

            with B:
                st.markdown("#### Optimal")
                st.metric("Score (objective)", f"{score_o:.3f}")
                st.metric("Profit", f"{tot_o['profit']:.3f}")
                st.metric("Utility", f"{tot_o['utility']:.3f}")
                st.write("Suppliers:", ", ".join(res_o["chosen_suppliers"]))
                st.dataframe(res_o["matches"], use_container_width=True, hide_index=True)

            if score_m is not None:
                st.divider()
                st.markdown("#### Gap")
                st.write(f"Optimal − Manual objective: **{(score_o - score_m):.3f}**")

        except Exception as e:
            st.error(str(e))

    st.divider()
    _render_leaderboard("Leaderboard", mode_filter=None)
