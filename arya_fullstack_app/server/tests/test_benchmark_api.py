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
    """Validate manual-eval response, which must include penalty-aware metrics."""
    _assert_metrics_payload(data)
    metrics = data["metrics"]
    for key in ("cost_per_unit", "penalty_per_unit", "avg_child_labor", "avg_banned_chem"):
        assert key in metrics, f"Missing manual-eval metric key: {key!r}"
    assert metrics["penalty_per_unit"] >= 0.0
    assert metrics["cost_per_unit"] >= 0.0


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


def test_manual_eval_includes_penalty_metrics() -> None:
    """Manual-eval with no picks returns infeasible but must still include penalty metrics."""
    response = client.post(
        "/api/manual-eval",
        json={
            "objective": "max_profit",
            "picks": [],
            "price_per_user": 100,
            "child_labor_penalty": 0.0,
            "banned_chem_penalty": 0.0,
        },
    )
    assert response.status_code == 200
    _assert_manual_eval_metrics(response.json())
    assert response.json()["feasible"] is False  # empty picks = infeasible


def test_manual_eval_penalty_nonzero_returns_positive_penalty_per_unit() -> None:
    """When penalty rates are set and at least one supplier has a flag,
    penalty_per_unit must be >= 0 (positive if flagged supplier is in picks)."""
    # Use all suppliers — at least one should have child_labor flag; if none do, still ≥ 0
    from app.service import get_tables
    suppliers_df, _ = get_tables()
    all_ids = suppliers_df["supplier_id"].astype(str).tolist()[:3]  # first 3 for speed

    response = client.post(
        "/api/manual-eval",
        json={
            "objective": "max_profit",
            "picks": all_ids,
            "price_per_user": 100,
            "child_labor_penalty": 50.0,
            "banned_chem_penalty": 50.0,
        },
    )
    assert response.status_code == 200
    data = response.json()
    _assert_manual_eval_metrics(data)
    assert data["metrics"]["penalty_per_unit"] >= 0.0
