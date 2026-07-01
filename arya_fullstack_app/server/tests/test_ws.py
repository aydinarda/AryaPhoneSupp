from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.main import app
from app.routers import sessions as sessions_router
from app.ws_manager import ConnectionManager


client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_session_config():
    """Keep the in-memory session-config dicts isolated between tests."""
    dicts = (
        sessions_router._session_beta,
        sessions_router._session_delta,
        sessions_router._session_quality_sensitivity,
        sessions_router._session_audit,
    )
    for d in dicts:
        d.clear()
    yield
    for d in dicts:
        d.clear()


# --------------------------------------------------------------------------- #
# ConnectionManager unit tests (no network / TestClient)
# --------------------------------------------------------------------------- #

class _FakeWS:
    """Minimal stand-in for a Starlette WebSocket used by ConnectionManager."""

    def __init__(self, *, fail: bool = False) -> None:
        self.sent: list = []
        self.fail = fail

    async def send_json(self, message) -> None:
        if self.fail:
            raise RuntimeError("dead socket")
        self.sent.append(message)


def test_register_and_disconnect_tracks_connections():
    mgr = ConnectionManager()
    ws = _FakeWS()

    mgr.register("ABC123", ws)
    assert ws in mgr._sessions["ABC123"]

    mgr.disconnect("ABC123", ws)
    assert ws not in mgr._sessions.get("ABC123", [])

    # Disconnecting an already-removed / unknown socket is a safe no-op.
    mgr.disconnect("ABC123", ws)
    mgr.disconnect("DOES-NOT-EXIST", ws)


def test_broadcast_delivers_to_live_sockets_and_prunes_dead_ones():
    mgr = ConnectionManager()
    good = _FakeWS()
    dead = _FakeWS(fail=True)
    mgr.register("ABC123", good)
    mgr.register("ABC123", dead)

    asyncio.run(mgr.broadcast("ABC123", {"type": "hello"}))

    assert good.sent == [{"type": "hello"}]
    # A socket that raises on send is pruned; healthy ones stay registered.
    assert dead not in mgr._sessions["ABC123"]
    assert good in mgr._sessions["ABC123"]


def test_broadcast_to_empty_session_is_noop():
    mgr = ConnectionManager()
    asyncio.run(mgr.broadcast("NOBODY", {"type": "hello"}))  # must not raise


def test_broadcast_sync_is_noop_without_running_loop():
    mgr = ConnectionManager()
    ws = _FakeWS()
    mgr.register("ABC123", ws)

    # No loop registered -> broadcast_sync silently does nothing (never raises).
    mgr.broadcast_sync("ABC123", {"type": "x"})
    assert ws.sent == []


# --------------------------------------------------------------------------- #
# WebSocket endpoint integration tests (TestClient.websocket_connect)
# --------------------------------------------------------------------------- #

_SESSION_ROW = {
    "session_code": "WS0001",
    "session_token": "tok-ws-1",
    "game_name": "WS Test",
    "admin_name": "Admin",
    "number_of_rounds": 5,
    "trial_rounds": 2,
    "is_active": True,
}


@pytest.fixture
def _active_session(monkeypatch):
    """Patch the DB reads the WS endpoint + sync builder touch, to a valid session."""
    monkeypatch.setattr(
        sessions_router, "fetch_game_session_by_code",
        lambda code: SimpleNamespace(data=[dict(_SESSION_ROW)]),
    )
    monkeypatch.setattr(sessions_router, "fetch_active_round", lambda token: SimpleNamespace(data=[]))
    monkeypatch.setattr(sessions_router, "fetch_latest_matching_result", lambda token: SimpleNamespace(data=[]))
    monkeypatch.setattr(sessions_router, "list_session_players", lambda code: [])
    return monkeypatch


def test_ws_valid_session_receives_initial_sync(_active_session):
    with client.websocket_connect("/api/sessions/WS0001/ws") as ws:
        msg = ws.receive_json()

    assert msg["type"] == "sync"
    assert msg["total_rounds"] == 5
    assert msg["trial_rounds"] == 2
    assert msg["scheduled_rounds"] == 7
    assert msg["players"] == []
    assert msg["round"] is None
    assert msg["submissions"] == []
    assert msg["match"] is None
    assert msg["match_round_no"] is None
    # Config defaults are present (no admin overrides applied yet).
    for key in ("beta_alpha", "beta_beta", "quality_sensitivity",
                "audit_probability", "catch_probability"):
        assert key in msg


def test_ws_ping_receives_pong(_active_session):
    with client.websocket_connect("/api/sessions/WS0001/ws") as ws:
        ws.receive_json()  # drain the initial sync
        ws.send_json({"type": "ping"})
        reply = ws.receive_json()

    assert reply == {"type": "pong"}


def test_ws_unknown_message_type_is_ignored(_active_session):
    with client.websocket_connect("/api/sessions/WS0001/ws") as ws:
        ws.receive_json()  # initial sync
        ws.send_json({"type": "not-a-real-type"})
        # A ping still works, proving the connection stayed open and healthy.
        ws.send_json({"type": "ping"})
        assert ws.receive_json() == {"type": "pong"}


def test_ws_unknown_session_is_rejected(monkeypatch):
    monkeypatch.setattr(
        sessions_router, "fetch_game_session_by_code",
        lambda code: SimpleNamespace(data=[]),
    )
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/sessions/NOPE00/ws") as ws:
            ws.receive_json()


def test_ws_inactive_session_is_rejected(monkeypatch):
    inactive = dict(_SESSION_ROW)
    inactive["is_active"] = False
    monkeypatch.setattr(
        sessions_router, "fetch_game_session_by_code",
        lambda code: SimpleNamespace(data=[inactive]),
    )
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/api/sessions/WS0001/ws") as ws:
            ws.receive_json()


def test_ws_receives_broadcast_after_config_update(_active_session):
    """A config PATCH must push a fresh `sync` frame to connected clients."""
    # Context-managed client runs the app lifespan so manager.set_loop() fires,
    # enabling broadcast_sync() from the (threadpool) PATCH handler.
    with TestClient(app) as ctx_client:
        with ctx_client.websocket_connect("/api/sessions/WS0001/ws") as ws:
            ws.receive_json()  # initial sync

            resp = ctx_client.patch(
                "/api/sessions/WS0001/config",
                json={
                    "beta_alpha": 2.0,
                    "beta_beta": 5.0,
                    "delta": 0.2,
                    "quality_sensitivity": 1.5,
                },
            )
            assert resp.status_code == 200

            pushed = ws.receive_json()

    assert pushed["type"] == "sync"
    assert pushed["beta_alpha"] == 2.0
    assert pushed["beta_beta"] == 5.0
    assert pushed["quality_sensitivity"] == 1.5
