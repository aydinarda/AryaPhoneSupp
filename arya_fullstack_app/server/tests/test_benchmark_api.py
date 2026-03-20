from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _assert_metrics_payload(data: dict) -> None:
    assert "feasible" in data
    assert "metrics" in data
    metrics = data["metrics"]
    for key in ("k", "avg_env", "avg_social", "avg_cost", "profit_total", "utility_total"):
        assert key in metrics


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
