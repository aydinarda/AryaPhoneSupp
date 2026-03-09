from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def _assert_success_payload(payload: dict) -> None:
    assert "feasible" in payload
    assert "metrics" in payload

    metrics = payload["metrics"]
    assert isinstance(metrics, dict)

    # Core numeric fields returned by solver-backed benchmark.
    assert "profit_total" in metrics
    assert "utility_total" in metrics


def test_benchmark_max_utility_returns_200() -> None:
    response = client.post("/api/benchmark", json={"objective": "max_utility"})

    assert response.status_code == 200
    _assert_success_payload(response.json())


def test_benchmark_max_profit_returns_200() -> None:
    # max_profit path uses the min-cost equivalent optimization in the model logic.
    response = client.post("/api/benchmark", json={"objective": "max_profit"})

    assert response.status_code == 200
    _assert_success_payload(response.json())
