from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from .mincost_agent import (
    DEFAULT_XLSX_PATH,
    GUROBI_AVAILABLE,
    MaxProfitAgent,
    MaxProfitConfig,
    MaxUtilAgent,
    MaxUtilConfig,
    manual_metrics,
    load_supplier_user_tables,
)

from .settings import FIXED_POLICY, GAME_SETTINGS


def _norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


@lru_cache(maxsize=1)
def get_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    xlsx_path = DEFAULT_XLSX_PATH
    return load_supplier_user_tables(xlsx_path)


def get_supplier_overview() -> list[dict[str, Any]]:
    suppliers_df, users_df = get_tables()
    df = suppliers_df.copy()

    df["env_bad_pct"] = (_norm01(df["env_risk"]) * 100).round(1)
    df["social_bad_pct"] = (_norm01(df["social_risk"]) * 100).round(1)
    df["cost_bad_pct"] = (_norm01(df["cost_score"]) * 100).round(1)
    df["low_quality_bad_pct"] = (_norm01(df["low_quality"]) * 100).round(1)
    df["strategic_good_pct"] = (_norm01(df["strategic"]) * 100).round(1)
    df["improvement_good_pct"] = (_norm01(df["improvement"]) * 100).round(1)

    if len(users_df):
        uavg = users_df[["w_env", "w_social", "w_cost", "w_strategic", "w_improvement", "w_low_quality"]].mean()
        df["expected_utility_avg_user"] = (
            float(uavg["w_env"]) * (FIXED_POLICY.env_mult * (5.0 - df["env_risk"]))
            + float(uavg["w_social"]) * (FIXED_POLICY.social_mult * (5.0 - df["social_risk"]))
            + float(uavg["w_strategic"]) * (FIXED_POLICY.strategic_mult * (df["strategic"] - 1.0))
            + float(uavg["w_improvement"]) * (FIXED_POLICY.improvement_mult * (df["improvement"] - 1.0))
            + float(uavg["w_low_quality"]) * (FIXED_POLICY.low_quality_mult * (5.0 - df["low_quality"]))
        ).astype(float)
    else:
        df["expected_utility_avg_user"] = 0.0

    df["profit_cost"] = (GAME_SETTINGS.cost_scale * df["cost_score"]).astype(float)

    cols = [
        "supplier_id",
        "env_bad_pct",
        "social_bad_pct",
        "cost_bad_pct",
        "low_quality_bad_pct",
        "strategic_good_pct",
        "improvement_good_pct",
        "expected_utility_avg_user",
        "profit_cost",
        "env_risk",
        "social_risk",
        "cost_score",
        "strategic",
        "improvement",
        "low_quality",
        "child_labor",
        "banned_chem",
    ]
    out = df[cols].copy()
    out["supplier_id"] = out["supplier_id"].astype(str)
    return out.to_dict(orient="records")


def _build_cfg(objective: str):
    kwargs = dict(
        served_users=GAME_SETTINGS.served_users,
        price_per_user=GAME_SETTINGS.price_per_user,
        cost_scale=GAME_SETTINGS.cost_scale,
        env_cap=GAME_SETTINGS.env_cap,
        social_cap=GAME_SETTINGS.social_cap,
        output_flag=0,
    )
    if objective == "max_profit":
        return MaxProfitConfig(**kwargs)
    return MaxUtilConfig(**kwargs)


def evaluate_manual(objective: str, picks: list[str]) -> dict[str, Any]:
    suppliers_df, users_df = get_tables()
    cfg = _build_cfg(objective)
    return manual_metrics(suppliers_df, users_df, FIXED_POLICY, cfg, [str(x) for x in picks])


def run_benchmark(objective: str) -> dict[str, Any]:
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    suppliers_df, users_df = get_tables()
    cfg = _build_cfg(objective)

    if objective == "max_profit":
        return MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()
    return MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg).solve()


def get_game_constants() -> dict[str, Any]:
    return {
        "served_users": GAME_SETTINGS.served_users,
        "env_cap": GAME_SETTINGS.env_cap,
        "social_cap": GAME_SETTINGS.social_cap,
        "cost_scale": GAME_SETTINGS.cost_scale,
        "price_per_user": GAME_SETTINGS.price_per_user,
        "gurobi_available": GUROBI_AVAILABLE,
    }


@lru_cache(maxsize=1)
def get_both_benchmarks() -> dict[str, Any]:
    results: dict[str, Any] = {}
    for obj in ("max_profit", "max_utility"):
        try:
            r = run_benchmark(obj)
            r["available"] = True
            results[obj] = r
        except Exception as exc:
            results[obj] = {"available": False, "error": str(exc), "metrics": {}, "feasible": False}
    return results
