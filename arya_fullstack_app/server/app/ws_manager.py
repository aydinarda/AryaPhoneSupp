"""
ws_manager.py
-------------
Singleton WebSocket connection manager.

Tracks open connections per session code and lets sync FastAPI
endpoints (running in a thread-pool) push JSON messages to all
connected clients via broadcast_sync().
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, list[WebSocket]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def connect(self, session_code: str, ws: WebSocket) -> None:
        await ws.accept()
        self._sessions.setdefault(session_code, []).append(ws)

    def register(self, session_code: str, ws: WebSocket) -> None:
        """Register an already-accepted WebSocket without calling accept() again."""
        self._sessions.setdefault(session_code, []).append(ws)

    def disconnect(self, session_code: str, ws: WebSocket) -> None:
        conns = self._sessions.get(session_code, [])
        try:
            conns.remove(ws)
        except ValueError:
            pass

    async def broadcast(self, session_code: str, message: dict[str, Any]) -> None:
        conns = list(self._sessions.get(session_code, []))
        dead: list[WebSocket] = []
        for ws in conns:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(session_code, ws)

    def broadcast_sync(self, session_code: str, message: dict[str, Any]) -> None:
        """Call from sync (thread-pool) FastAPI endpoints to push to all WS clients."""
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            self.broadcast(session_code, message), loop
        )


manager = ConnectionManager()
