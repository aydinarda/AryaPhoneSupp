from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routers import sessions as sessions_router
from app.settings import GAME_SETTINGS


client = TestClient(app)


def test_admin_can_create_session() -> None:
    response = client.post(
        "/api/sessions",
        json={"game_name": "Round 1", "admin_name": "Instructor"},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["game_name"] == "Round 1"
    assert payload["admin_name"] == "Instructor"
    assert payload["number_of_rounds"] == 5
    assert isinstance(payload["code"], str)
    assert len(payload["code"]) == 6


def test_admin_can_set_number_of_rounds() -> None:
    response = client.post(
        "/api/sessions",
        json={"game_name": "Round Config", "admin_name": "Instructor", "number_of_rounds": 7},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["number_of_rounds"] == 7


def test_player_can_join_existing_session_code() -> None:
    created = client.post(
        "/api/sessions",
        json={"game_name": "Joinable Session", "admin_name": "Host"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    joined = client.get(f"/api/sessions/{code}")
    assert joined.status_code == 200

    payload = joined.json()
    assert payload["code"] == code
    assert payload["game_name"] == "Joinable Session"
    assert payload["admin_name"] == "Host"


def test_player_cannot_join_non_existing_session_code() -> None:
    response = client.get("/api/sessions/ZZZZZZ")

    assert response.status_code == 404
    assert response.json()["detail"] == "Session code not found"


def test_player_can_join_with_team_name() -> None:
    created = client.post(
        "/api/sessions",
        json={"game_name": "Join Test", "admin_name": "Host"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    joined = client.post(f"/api/sessions/{code}/join", json={"team_name": "TeamAlpha"})
    assert joined.status_code == 200
    assert joined.json()["code"] == code


def test_duplicate_team_name_is_rejected() -> None:
    created = client.post(
        "/api/sessions",
        json={"game_name": "Duplicate Test", "admin_name": "Host"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    first = client.post(f"/api/sessions/{code}/join", json={"team_name": "TeamAlpha"})
    assert first.status_code == 200

    second = client.post(f"/api/sessions/{code}/join", json={"team_name": "TeamAlpha"})
    assert second.status_code == 409
    assert second.json()["detail"] == "same username exist"


def test_duplicate_team_name_is_case_insensitive() -> None:
    created = client.post(
        "/api/sessions",
        json={"game_name": "Case Test", "admin_name": "Host"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    client.post(f"/api/sessions/{code}/join", json={"team_name": "TeamBeta"})

    response = client.post(f"/api/sessions/{code}/join", json={"team_name": "teambeta"})
    assert response.status_code == 409
    assert response.json()["detail"] == "same username exist"


def test_join_non_existing_session_returns_404() -> None:
    response = client.post("/api/sessions/ZZZZZZ/join", json={"team_name": "SomeTeam"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Session code not found"


def test_join_requires_team_name() -> None:
    created = client.post(
        "/api/sessions",
        json={"game_name": "Validation Test", "admin_name": "Host"},
    )
    assert created.status_code == 200
    code = created.json()["code"]

    response = client.post(f"/api/sessions/{code}/join", json={"team_name": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "Team name is required"


def _mock_session_round(monkeypatch, submissions: list, suppliers_df, users_df) -> dict:
    """Shared fixture: patch all DB calls and return a dict that insert_matching_result fills."""
    stored: dict = {}

    monkeypatch.setattr(
        sessions_router,
        "fetch_game_session_by_code",
        lambda code: SimpleNamespace(
            data=[{"session_code": code, "session_token": "tok-1", "is_active": True}]
        ),
    )
    monkeypatch.setattr(
        sessions_router,
        "fetch_active_round",
        lambda session_token: SimpleNamespace(
            data=[{
                "round_no": 1,
                "market_capacity": GAME_SETTINGS.default_market_capacity,
                "created_at": "2026-04-01T10:00:00+00:00",
                "is_active": True,
            }]
        ),
    )
    monkeypatch.setattr(sessions_router, "fetch_latest_round", lambda t: SimpleNamespace(data=[]))
    monkeypatch.setattr(sessions_router, "fetch_round_by_number", lambda t, n: SimpleNamespace(data=[]))
    monkeypatch.setattr(
        sessions_router,
        "fetch_submissions_for_round",
        lambda sc, rno: SimpleNamespace(data=submissions),
    )
    monkeypatch.setattr(sessions_router, "get_tables", lambda: (suppliers_df, users_df))
    monkeypatch.setattr(sessions_router, "insert_matching_result", lambda payload: stored.update(payload))

    return stored


def _make_mock_suppliers() -> pd.DataFrame:
    """3-category supplier catalogue: 2 cameras, 1 keyboard, 1 cable."""
    return pd.DataFrame([
        {"supplier_id": "CAM1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "camera"},
        {"supplier_id": "CAM2", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 2.0,
         "strategic": 2.0, "improvement": 2.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "camera"},
        {"supplier_id": "KEY1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "keyboard"},
        {"supplier_id": "CBL1", "env_risk": 1.0, "social_risk": 1.0, "cost_score": 1.0,
         "strategic": 3.0, "improvement": 3.0, "low_quality": 2.0,
         "child_labor": 0.0, "banned_chem": 0.0, "category": "cable"},
    ])


def _make_mock_users(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": f"U{i}", "w_env": 1.0, "w_social": 1.0, "w_cost": 1.0,
         "w_strategic": 1.0, "w_improvement": 1.0, "w_low_quality": 1.0}
        for i in range(1, n + 1)
    ])


def test_round_matching_excludes_risk_infeasible_submissions(monkeypatch) -> None:
    """Teams flagged infeasible in the DB (risk caps) are excluded from MNL matching."""
    stored = _mock_session_round(
        monkeypatch,
        submissions=[
            {
                "team": "FeasibleA",
                "selected_suppliers": "CAM1,KEY1,CBL1",
                "created_at": "2026-04-01T10:00:01+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,
            },
            {
                "team": "InfeasibleB",
                "selected_suppliers": "CAM1,KEY1,CBL1",
                "created_at": "2026-04-01T10:00:02+00:00",
                "env_avg": None, "social_avg": None, "feasible": "false",
            },
            {
                "team": "FeasibleC",
                "selected_suppliers": "CAM2,KEY1,CBL1",
                "created_at": "2026-04-01T10:00:03+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,
            },
        ],
        suppliers_df=_make_mock_suppliers(),
        users_df=_make_mock_users(5),
    )

    response = client.post("/api/sessions/ABC123/match", json={})

    assert response.status_code == 200
    payload = response.json()

    assert payload["eligible_team_count"] == 2
    assert payload["excluded_infeasible_teams"] == ["InfeasibleB"]

    matching = payload["matching"]
    assert matching["meta"]["solver"] == "mnl_v1"
    assert matching["meta"]["user_pool_count"] == 5
    assert matching["meta"]["eligible_team_count"] == 2
    assert matching["meta"]["infeasible_excluded_count"] == 1
    assert set(matching["market_to_users"].keys()) == {"FeasibleA", "FeasibleC"}
    assert "InfeasibleB" not in matching["market_to_users"]
    assert "InfeasibleB" in matching["excluded_infeasible_teams"]

    # Round financials: both feasible teams have entries
    tf = {row["team"]: row for row in matching["round_financials"]["team_financials"]}
    assert set(tf.keys()) == {"FeasibleA", "FeasibleC"}
    for row in tf.values():
        assert row["demand_share"] >= 0.0
        assert "realized_profit" in row

    # Stored result round_no matches
    assert stored["round_no"] == 1
    assert stored["matched_count"] == matching["meta"]["matched_count"]
    assert stored["result"]["round_financials"]["round_profit_total"] == pytest.approx(
        matching["round_financials"]["round_profit_total"], abs=1e-6
    )


def test_round_matching_excludes_category_invalid_submissions(monkeypatch) -> None:
    """Teams whose picks violate category constraints (not 1-per-category) are excluded."""
    _mock_session_round(
        monkeypatch,
        submissions=[
            {
                "team": "GoodTeam",
                "selected_suppliers": "CAM1,KEY1,CBL1",   # 1 per category ✓
                "created_at": "2026-04-01T10:00:01+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,
            },
            {
                "team": "TwoCameras",
                "selected_suppliers": "CAM1,CAM2,CBL1",   # 2 cameras, no keyboard ✗
                "created_at": "2026-04-01T10:00:02+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,  # passes risk check
            },
            {
                "team": "MissingCable",
                "selected_suppliers": "CAM1,KEY1",         # no cable ✗
                "created_at": "2026-04-01T10:00:03+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,
            },
        ],
        suppliers_df=_make_mock_suppliers(),
        users_df=_make_mock_users(3),
    )

    response = client.post("/api/sessions/ABC123/match", json={})

    assert response.status_code == 200
    payload = response.json()

    assert payload["eligible_team_count"] == 1
    excluded = set(payload["excluded_infeasible_teams"])
    assert "TwoCameras" in excluded
    assert "MissingCable" in excluded
    assert "GoodTeam" not in excluded

    matching = payload["matching"]
    assert set(matching["market_to_users"].keys()) == {"GoodTeam"}
    assert "TwoCameras" not in matching["market_to_users"]
    assert "MissingCable" not in matching["market_to_users"]

    tf_teams = {row["team"] for row in matching["round_financials"]["team_financials"]}
    assert tf_teams == {"GoodTeam"}


def test_session_config_patch_stores_and_returns_penalties(monkeypatch) -> None:
    """PATCH /config with penalties stores them and returns them in the response."""
    monkeypatch.setattr(
        sessions_router,
        "fetch_game_session_by_code",
        lambda code: SimpleNamespace(
            data=[{"session_code": code, "session_token": "tok-1",
                   "is_active": True, "number_of_rounds": 5}]
        ),
    )
    monkeypatch.setattr(sessions_router, "fetch_active_round",
                        lambda t: SimpleNamespace(data=[]))

    # Set penalties via PATCH
    patch_resp = client.patch(
        "/api/sessions/PENTEST/config",
        json={"beta_alpha": 3.0, "beta_beta": 3.0, "delta": 0.1,
              "child_labor_penalty": 25.0, "banned_chem_penalty": 50.0},
    )
    assert patch_resp.status_code == 200
    body = patch_resp.json()
    assert body["child_labor_penalty"] == pytest.approx(25.0)
    assert body["banned_chem_penalty"] == pytest.approx(50.0)

    # Verify they come back from GET /rounds/current
    current_resp = client.get("/api/sessions/PENTEST/rounds/current")
    assert current_resp.status_code == 200
    data = current_resp.json()
    assert data["child_labor_penalty"] == pytest.approx(25.0)
    assert data["banned_chem_penalty"] == pytest.approx(50.0)


def test_session_config_without_penalties_defaults_to_zero(monkeypatch) -> None:
    """A freshly configured session with no penalties should return 0.0 for both."""
    monkeypatch.setattr(
        sessions_router,
        "fetch_game_session_by_code",
        lambda code: SimpleNamespace(
            data=[{"session_code": code, "session_token": "tok-2",
                   "is_active": True, "number_of_rounds": 5}]
        ),
    )
    monkeypatch.setattr(sessions_router, "fetch_active_round",
                        lambda t: SimpleNamespace(data=[]))

    client.patch(
        "/api/sessions/NOPEN/config",
        json={"beta_alpha": 2.0, "beta_beta": 2.0},
    )

    resp = client.get("/api/sessions/NOPEN/rounds/current")
    assert resp.status_code == 200
    data = resp.json()
    assert data["child_labor_penalty"] == pytest.approx(0.0)
    assert data["banned_chem_penalty"] == pytest.approx(0.0)


def test_round_matching_penalty_reduces_unit_margin(monkeypatch) -> None:
    """When a session has child_labor_penalty set and a team's supplier has
    child_labor=1.0, the unit_margin in round_financials must be lower than
    without a penalty."""

    # Build a supplier set where CAM1 carries child_labor=1
    suppliers = _make_mock_suppliers().copy()
    suppliers.loc[suppliers["supplier_id"] == "CAM1", "child_labor"] = 1.0

    _mock_session_round(
        monkeypatch,
        submissions=[
            {
                "team": "TeamCL",
                "selected_suppliers": "CAM1,KEY1,CBL1",
                "created_at": "2026-04-01T10:00:01+00:00",
                "env_avg": 1.0, "social_avg": 1.0, "price": 100,
            },
        ],
        suppliers_df=suppliers,
        users_df=_make_mock_users(3),
    )

    # Run without penalty first
    sessions_router._session_penalty.pop("ABC123", None)
    resp_no_pen = client.post("/api/sessions/ABC123/match", json={})
    assert resp_no_pen.status_code == 200
    tf_no_pen = resp_no_pen.json()["matching"]["round_financials"]["team_financials"][0]

    # Set penalty and re-run
    sessions_router._session_penalty["ABC123"] = (30.0, 0.0)
    resp_pen = client.post("/api/sessions/ABC123/match", json={})
    assert resp_pen.status_code == 200
    tf_pen = resp_pen.json()["matching"]["round_financials"]["team_financials"][0]

    # unit_margin should be lower when penalty is active (avg_child_labor = 1/3 → -10)
    assert tf_pen["unit_margin"] < tf_no_pen["unit_margin"]
    # penalty_per_unit should appear and be positive
    assert tf_pen.get("penalty_per_unit", 0.0) > 0.0


def test_start_round_respects_configured_round_limit(monkeypatch) -> None:
    monkeypatch.setattr(
        sessions_router,
        "fetch_game_session_by_code",
        lambda code: SimpleNamespace(
            data=[
                {
                    "session_code": code,
                    "session_token": "token-1",
                    "is_active": True,
                    "number_of_rounds": 2,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        sessions_router,
        "fetch_latest_round",
        lambda session_token: SimpleNamespace(data=[{"round_no": 2}]),
    )
    monkeypatch.setattr(sessions_router, "close_active_rounds", lambda session_token: None)

    called = {"insert": 0}

    def _insert_stub(payload):
        called["insert"] += 1
        return SimpleNamespace(data=[payload])

    monkeypatch.setattr(sessions_router, "insert_game_round", _insert_stub)

    response = client.post(
        "/api/sessions/ABC123/rounds/start",
        json={"duration_seconds": 120, "market_capacity": 8},
    )

    assert response.status_code == 400
    assert "Configured round limit reached" in response.json()["detail"]
    assert called["insert"] == 0
