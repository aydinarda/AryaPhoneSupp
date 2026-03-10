from __future__ import annotations

import random
from threading import Lock
from typing import Any

SESSION_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
SESSION_CODE_LENGTH = 6

_sessions_lock = Lock()
_sessions: dict[str, dict[str, Any]] = {}
_session_players: dict[str, set[str]] = {}  # code -> set of lowercase team names


def _normalize_session_code(raw_code: str) -> str:
    return (raw_code or "").strip().upper()


def _create_session_code() -> str:
    rng = random.SystemRandom()
    for _ in range(100):
        code = "".join(rng.choice(SESSION_CODE_ALPHABET) for _ in range(SESSION_CODE_LENGTH))
        if code not in _sessions:
            return code
    raise RuntimeError("Unable to generate a unique session code")


def create_session(game_name: str, admin_name: str) -> dict[str, Any]:
    cleaned_game_name = (game_name or "").strip()
    cleaned_admin_name = (admin_name or "Admin").strip() or "Admin"

    if not cleaned_game_name:
        raise ValueError("Game name is required")

    with _sessions_lock:
        code = _create_session_code()
        payload = {
            "code": code,
            "game_name": cleaned_game_name,
            "admin_name": cleaned_admin_name,
        }
        _sessions[code] = payload
        return payload


def get_session(code: str) -> dict[str, Any] | None:
    normalized = _normalize_session_code(code)
    if not normalized:
        return None

    with _sessions_lock:
        return _sessions.get(normalized)


def join_session(code: str, team_name: str) -> dict[str, Any] | None:
    """Register a player team name into a session.

    Returns the session dict on success.
    Returns None if the session code does not exist.
    Raises ValueError if the team name is blank or already taken (case-insensitive).
    """
    normalized = _normalize_session_code(code)
    clean_team = (team_name or "").strip()

    if not clean_team:
        raise ValueError("Team name is required")

    with _sessions_lock:
        session = _sessions.get(normalized)
        if session is None:
            return None

        players = _session_players.setdefault(normalized, set())
        if clean_team.lower() in players:
            raise ValueError("A player with that team name has already joined this session")

        players.add(clean_team.lower())
        return session
