from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
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


def test_round_matching_excludes_infeasible_submissions(monkeypatch) -> None:
    stored_result: dict[str, object] = {}

    monkeypatch.setattr(
        sessions_router,
        "fetch_game_session_by_code",
        lambda code: SimpleNamespace(
            data=[
                {
                    "session_code": code,
                    "session_token": "token-1",
                    "is_active": True,
                }
            ]
        ),
    )
    monkeypatch.setattr(
        sessions_router,
        "fetch_active_round",
        lambda session_token: SimpleNamespace(
            data=[
                {
                    "round_no": 1,
                    "market_capacity": GAME_SETTINGS.default_market_capacity,
                    "created_at": "2026-03-12T10:00:00+00:00",
                    "is_active": True,
                }
            ]
        ),
    )
    monkeypatch.setattr(sessions_router, "fetch_latest_round", lambda session_token: SimpleNamespace(data=[]))
    monkeypatch.setattr(sessions_router, "fetch_round_by_number", lambda session_token, round_no: SimpleNamespace(data=[]))
    monkeypatch.setattr(
        sessions_router,
        "fetch_submissions_for_round",
        lambda session_code, round_no: SimpleNamespace(
            data=[
                {
                    "team": "FeasibleA",
                    "selected_suppliers": "S1,S2",
                    "created_at": "2026-03-12T10:00:01+00:00",
                    "env_avg": 0.10,
                    "social_avg": 0.10,
                },
                {
                    "team": "InfeasibleB",
                    "selected_suppliers": "S1,S3",
                    "created_at": "2026-03-12T10:00:02+00:00",
                    "env_avg": None,
                    "social_avg": None,
                    "feasible": "false",
                },
                {
                    "team": "FeasibleC",
                    "selected_suppliers": "S1",
                    "created_at": "2026-03-12T10:00:03+00:00",
                    "env_avg": 0.10,
                    "social_avg": 0.10,
                },
            ]
        ),
    )

    # 10 users to verify many-to-one matching with capacity=8
    mock_users = pd.DataFrame(
        [
            {"user_id": f"U{i}", "w_env": round(0.1 * i, 2), "w_social": round(0.2 * (11 - i), 2),
             "w_cost": 1.0, "w_strategic": 1.0, "w_improvement": 1.0, "w_low_quality": 0.5}
            for i in range(1, 11)
        ]
    )

    monkeypatch.setattr(
        sessions_router,
        "get_tables",
        lambda: (
            pd.DataFrame(
                [
                    {
                        "supplier_id": "S1",
                        "env_risk": 0.1,
                        "social_risk": 0.1,
                        "cost_score": 1.0,
                        "strategic": 1.0,
                        "improvement": 1.0,
                        "low_quality": 1.0,
                    },
                    {
                        "supplier_id": "S2",
                        "env_risk": 0.2,
                        "social_risk": 0.2,
                        "cost_score": 1.2,
                        "strategic": 1.1,
                        "improvement": 1.1,
                        "low_quality": 1.1,
                    },
                    {
                        "supplier_id": "S3",
                        "env_risk": 5.0,
                        "social_risk": 5.0,
                        "cost_score": 4.0,
                        "strategic": 0.5,
                        "improvement": 0.5,
                        "low_quality": 4.0,
                    },
                ]
            ),
            mock_users,
        ),
    )
    monkeypatch.setattr(sessions_router, "insert_matching_result", lambda payload: stored_result.update(payload))

    response = client.post("/api/sessions/ABC123/match", json={})

    assert response.status_code == 200
    payload = response.json()
    assert payload["eligible_team_count"] == 2
    assert payload["excluded_infeasible_teams"] == ["InfeasibleB"]
    assert payload["matching"]["excluded_infeasible_users"] == ["InfeasibleB"]
    assert payload["matching"]["meta"]["eligible_team_count"] == 2
    assert payload["matching"]["meta"]["infeasible_excluded_count"] == 1
    assert payload["matching"]["matching_target"] == "team_product"
    assert "InfeasibleB" not in payload["matching"]["market_to_users"]

    # Verify many-to-one: 10 users across 2 feasible teams, each with capacity=8
    # Total capacity = 16 >= 10 users, so all users should be matched
    matching = payload["matching"]
    assert matching["meta"]["user_pool_count"] == 10
    assert matching["meta"]["matched_count"] == 10
    assert len(matching["unmatched_users"]) == 0

    # Each team can have at most default_market_capacity users
    for team_id, users in matching["market_to_users"].items():
        assert len(users) <= GAME_SETTINGS.default_market_capacity, (
            f"{team_id} has {len(users)} users, exceeds capacity {GAME_SETTINGS.default_market_capacity}"
        )

    # All 10 users are distributed across the 2 feasible teams
    total_matched = sum(len(u) for u in matching["market_to_users"].values())
    assert total_matched == 10
    assert set(payload["matching"]["market_to_users"].keys()) == {"FeasibleA", "FeasibleC"}
    expected_user_ids = {f"U{i}" for i in range(1, 11)}
    assert set(payload["matching"]["user_to_market"].keys()) == expected_user_ids
    assert payload["matching"]["market_loads"]["FeasibleA"]["capacity"] == GAME_SETTINGS.default_market_capacity
    assert stored_result["matched_count"] == payload["matching"]["meta"]["matched_count"]
    # Solver must always be reported regardless of which path was taken
    assert payload["matching"]["meta"]["solver"] in {
        "stable_gale_shapley",
        "gurobi_stable_lexicographic",
        "market_demand_v1",
    }

    # Round financials: realized profit must be based on actual matched users.
    round_financials = payload["matching"].get("round_financials") or {}
    team_financials = round_financials.get("team_financials") or []
    assert len(team_financials) == 2

    by_team = {row["team"]: row for row in team_financials}
    assert set(by_team.keys()) == {"FeasibleA", "FeasibleC"}

    for team_id, row in by_team.items():
        matched_users = payload["matching"]["market_to_users"].get(team_id, [])
        assert row["matched_user_count"] == len(matched_users)
        assert row["matched_users"] == matched_users

        expected_realized = row["matched_user_count"] * (
            row["sale_price_per_user"] - row["cost_component_per_user"] - row["penalty_per_user"]
        )
        assert abs(row["realized_profit"] - expected_realized) < 1e-9

    expected_round_total = sum(float(row["realized_profit"]) for row in team_financials)
    assert abs(float(round_financials["round_profit_total"]) - expected_round_total) < 1e-9
    assert abs(float(payload["matching"]["meta"]["round_profit_total"]) - expected_round_total) < 1e-9
    assert abs(float(stored_result["result"]["round_financials"]["round_profit_total"]) - expected_round_total) < 1e-9


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
