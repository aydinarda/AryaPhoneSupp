from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


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
    assert isinstance(payload["code"], str)
    assert len(payload["code"]) == 6


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
    assert "already" in second.json()["detail"].lower()


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


def test_join_non_existing_session_returns_404() -> None:
    response = client.post("/api/sessions/ZZZZZZ/join", json={"team_name": "SomeTeam"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Session code not found"
