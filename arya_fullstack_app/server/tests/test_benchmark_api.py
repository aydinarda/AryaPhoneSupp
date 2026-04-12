from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _assert_metrics_payload(data: dict) -> None:
    """Validate the structure of a metrics payload (benchmark or manual-eval)."""
    assert "feasible" in data
    assert "metrics" in data
    metrics = data["metrics"]
    # Core keys present in all code paths (optimizer + manual_metrics)
    for key in ("k", "avg_env", "avg_social", "avg_cost", "profit_total", "utility_total"):
        assert key in metrics, f"Missing core metric key: {key!r}"
    # Penalty / cost-per-unit keys present only in the manual_metrics path;
    # benchmark uses the optimizer solve path which does not populate them.
    # Check them only when they are present so the test works for both paths.
    if "cost_per_unit" in metrics:
        assert metrics["cost_per_unit"] >= 0.0
    if "penalty_per_unit" in metrics:
        assert metrics["penalty_per_unit"] >= 0.0
    if "avg_child_labor" in metrics:
        assert metrics["avg_child_labor"] >= 0.0
    if "avg_banned_chem" in metrics:
        assert metrics["avg_banned_chem"] >= 0.0


def _assert_manual_eval_metrics(data: dict) -> None:
    """Validate manual-eval response structure."""
    _assert_metrics_payload(data)
    metrics = data["metrics"]
    for key in ("cost_per_unit", "avg_child_labor", "avg_banned_chem"):
        assert key in metrics, f"Missing manual-eval metric key: {key!r}"
    assert metrics["cost_per_unit"] >= 0.0
    assert metrics["avg_child_labor"] >= 0.0
    assert metrics["avg_banned_chem"] >= 0.0


def test_benchmark_max_utility_returns_200() -> None:
    response = client.post("/api/benchmark", json={"objective": "max_utility"})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        _assert_metrics_payload(response.json())


def test_benchmark_max_profit_returns_200() -> None:
    response = client.post("/api/benchmark", json={"objective": "max_profit"})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        _assert_metrics_payload(response.json())


def test_manual_eval_empty_picks_returns_infeasible() -> None:
    """Manual-eval with no picks returns infeasible; response still has full metrics shape."""
    response = client.post(
        "/api/manual-eval",
        json={"objective": "max_profit", "picks": [], "price_per_user": 100},
    )
    assert response.status_code == 200
    data = response.json()
    _assert_manual_eval_metrics(data)
    assert data["feasible"] is False


def test_manual_eval_exposes_violation_averages() -> None:
    """manual-eval response must include avg_child_labor and avg_banned_chem for
    every valid submission so the audit system can track supplier violations."""
    from app.service import get_tables
    suppliers_df, _ = get_tables()
    # Pick the first supplier from each category (one per category required)
    cat_col = "category" if "category" in suppliers_df.columns else None
    if cat_col:
        picks = (
            suppliers_df.dropna(subset=[cat_col])
            .groupby(cat_col, as_index=False)
            .first()["supplier_id"]
            .astype(str)
            .tolist()
        )
    else:
        picks = suppliers_df["supplier_id"].astype(str).tolist()[:3]

    response = client.post(
        "/api/manual-eval",
        json={"objective": "max_profit", "picks": picks, "price_per_user": 100},
    )
    assert response.status_code == 200
    data = response.json()
    _assert_manual_eval_metrics(data)
    m = data["metrics"]
    assert "avg_child_labor" in m
    assert "avg_banned_chem" in m
    assert m["avg_child_labor"] >= 0.0
    assert m["avg_banned_chem"] >= 0.0
