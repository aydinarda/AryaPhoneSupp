from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import pandas as pd

from .beta_density import BetaDensity
from .optimization_controller import (
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


_CACHE_LOCK = Lock()
_LAST_XLSX_SIGNATURE: tuple[int, int] | None = None


def _norm01(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    mn, mx = float(s.min()), float(s.max())
    if mx - mn <= 1e-12:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def _xlsx_signature(path: Path) -> tuple[int, int]:
    st = path.stat()
    return (int(st.st_mtime_ns), int(st.st_size))


def _maybe_invalidate_caches() -> None:
    global _LAST_XLSX_SIGNATURE
    path = Path(DEFAULT_XLSX_PATH)
    try:
        current_sig = _xlsx_signature(path)
    except Exception:
        # Keep existing caches if file is temporarily unavailable.
        return

    if _LAST_XLSX_SIGNATURE == current_sig:
        return

    with _CACHE_LOCK:
        if _LAST_XLSX_SIGNATURE == current_sig:
            return
        _get_tables_cached.cache_clear()
        _get_both_benchmarks_cached.cache_clear()
        _LAST_XLSX_SIGNATURE = current_sig


@lru_cache(maxsize=1)
def _get_tables_cached() -> tuple[pd.DataFrame, pd.DataFrame]:
    xlsx_path = DEFAULT_XLSX_PATH
    return load_supplier_user_tables(xlsx_path)


def get_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    _maybe_invalidate_caches()
    return _get_tables_cached()


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


def _build_cfg(objective: str, price_per_user: float | None = None):
    price_value = GAME_SETTINGS.price_per_user if price_per_user is None else max(0.0, float(price_per_user))
    kwargs = dict(
        price_per_user=price_value,
        cost_scale=GAME_SETTINGS.cost_scale,
        env_cap=GAME_SETTINGS.env_cap,
        social_cap=GAME_SETTINGS.social_cap,
        output_flag=0,
    )
    if objective == "max_profit":
        return MaxProfitConfig(**kwargs)
    return MaxUtilConfig(**kwargs)


def build_density_weights(users_df: Any) -> dict[str, float]:
    """
    Produce density weights for all users by sorting them by w_cost
    and evaluating a Beta(3, 3) density at each user's position on [0, 1].

    Returns {user_id -> unnormalised weight}.
    The default Beta(3, 3) symmetric bell can be changed once the admin
    configuration panel is wired up.
    """
    df = users_df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df = df.sort_values("w_cost").reset_index(drop=True)
    n = len(df)
    if n == 0:
        return {}
    bd = BetaDensity(alpha=3.0, beta=3.0)
    return {
        str(row["user_id"]): bd.density_at((i + 0.5) / n)
        for i, (_, row) in enumerate(df.iterrows())
    }


def evaluate_manual(objective: str, picks: list[str], price_per_user: float | None = None) -> dict[str, Any]:
    suppliers_df, users_df = get_tables()
    cfg = _build_cfg(objective, price_per_user=price_per_user)
    return manual_metrics(suppliers_df, users_df, FIXED_POLICY, cfg, [str(x) for x in picks])


def run_benchmark(objective: str, density_weights: dict[str, float] | None = None) -> dict[str, Any]:
    if not GUROBI_AVAILABLE:
        raise RuntimeError("gurobipy is not available")

    suppliers_df, users_df = get_tables()
    cfg = _build_cfg(objective)

    if objective == "max_profit":
        return MaxProfitAgent(suppliers_df, users_df, FIXED_POLICY, cfg, density_weights=density_weights).solve()
    return MaxUtilAgent(suppliers_df, users_df, FIXED_POLICY, cfg, density_weights=density_weights).solve()


def get_game_constants() -> dict[str, Any]:
    _, users_df = get_tables()
    return {
        "env_cap": GAME_SETTINGS.env_cap,
        "social_cap": GAME_SETTINGS.social_cap,
        "cost_scale": GAME_SETTINGS.cost_scale,
        "price_per_user": GAME_SETTINGS.price_per_user,
        "gurobi_available": GUROBI_AVAILABLE,
        "num_segments": int(len(users_df)),
    }


@lru_cache(maxsize=1)
def _get_both_benchmarks_cached() -> dict[str, Any]:
    _, users_df = _get_tables_cached()
    density_weights = build_density_weights(users_df)
    results: dict[str, Any] = {}
    for obj in ("max_profit", "max_utility"):
        try:
            result = run_benchmark(obj, density_weights=density_weights)
            result["available"] = True
            results[obj] = result
        except Exception as exc:
            results[obj] = {"available": False, "error": str(exc), "metrics": {}, "feasible": False}
    return results


def get_both_benchmarks() -> dict[str, Any]:
    _maybe_invalidate_caches()
    return _get_both_benchmarks_cached()
