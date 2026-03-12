from __future__ import annotations

import random
import secrets
from threading import Lock
from typing import Any

from .db import (
    fetch_game_session_by_code,
    fetch_session_player,
    has_supabase_credentials,
    insert_game_session,
    insert_session_player,
)

SESSION_CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
SESSION_CODE_LENGTH = 6

_sessions_lock = Lock()
_sessions: dict[str, dict[str, Any]] = {}
_session_players: dict[str, set[str]] = {}  # code -> set of lowercase team names


def _use_db_backend() -> bool:
    return has_supabase_credentials()


def _normalize_session_code(raw_code: str) -> str:
    return (raw_code or "").strip().upper()


def _create_session_code() -> str:
    rng = random.SystemRandom()
    for _ in range(100):
        code = "".join(rng.choice(SESSION_CODE_ALPHABET) for _ in range(SESSION_CODE_LENGTH))
        if code not in _sessions:
            return code
    raise RuntimeError("Unable to generate a unique session code")


def _create_session_token() -> str:
    # URL-safe random token used as immutable DB key to separate sessions.
    return secrets.token_urlsafe(18)


def _normalize_team_name(team_name: str) -> str:
    return (team_name or "").strip().lower()


def _from_db_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": row.get("session_code", ""),
        "game_name": row.get("game_name", ""),
        "admin_name": row.get("admin_name", "Admin"),
        "session_token": row.get("session_token", ""),
    }


def _is_duplicate_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "duplicate" in text or "23505" in text or "unique" in text


def _is_missing_table_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "pgrst205" in text or "could not find the table" in text


def _is_recoverable_db_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        _is_missing_table_error(exc)
        or "row-level security" in text
        or "violates row-level security policy" in text
        or "permission denied" in text
        or "42501" in text
    )


def create_session(game_name: str, admin_name: str) -> dict[str, Any]:
    cleaned_game_name = (game_name or "").strip()
    cleaned_admin_name = (admin_name or "Admin").strip() or "Admin"

    if not cleaned_game_name:
        raise ValueError("Game name is required")

    if _use_db_backend():
        for _ in range(100):
            code = _create_session_code()
            token = _create_session_token()
            payload = {
                "session_code": code,
                "session_token": token,
                "game_name": cleaned_game_name,
                "admin_name": cleaned_admin_name,
                "is_active": True,
            }
            try:
                insert_game_session(payload)
                return {
                    "code": code,
                    "game_name": cleaned_game_name,
                    "admin_name": cleaned_admin_name,
                }
            except Exception as exc:
                if _is_recoverable_db_error(exc):
                    raise RuntimeError(f"DB access/config error while creating session: {exc}") from exc
                if not _is_duplicate_db_error(exc):
                    raise RuntimeError(f"Failed to create session: {exc}") from exc
        else:
            raise RuntimeError("Unable to generate a unique session code")

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

    if _use_db_backend():
        try:
            res = fetch_game_session_by_code(normalized)
            rows = getattr(res, "data", None) or []
            if not rows:
                return None
            row = rows[0]
            if row.get("is_active") is False:
                return None
            session = _from_db_row(row)
            return {
                "code": session["code"],
                "game_name": session["game_name"],
                "admin_name": session["admin_name"],
            }
        except Exception as exc:
            if _is_recoverable_db_error(exc):
                raise RuntimeError(f"DB access/config error while fetching session: {exc}") from exc
            raise RuntimeError(f"Failed to fetch session: {exc}") from exc

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
    normalized_team = _normalize_team_name(team_name)

    if not clean_team:
        raise ValueError("Team name is required")

    if _use_db_backend():
        try:
            res = fetch_game_session_by_code(normalized)
            rows = getattr(res, "data", None) or []
            if not rows:
                return None

            row = rows[0]
            if row.get("is_active") is False:
                return None

            token = row.get("session_token")
            if not token:
                raise RuntimeError("Session token is missing in storage")

            existing = fetch_session_player(token, normalized_team)
            existing_rows = getattr(existing, "data", None) or []
            if existing_rows:
                raise ValueError("same username exist")

            try:
                insert_session_player(
                    {
                        "session_token": token,
                        "team_name": clean_team,
                        "team_name_normalized": normalized_team,
                    }
                )
            except Exception as exc:
                if _is_duplicate_db_error(exc):
                    raise ValueError("same username exist") from exc
                raise RuntimeError(f"Failed to join session: {exc}") from exc

            session = _from_db_row(row)
            return {
                "code": session["code"],
                "game_name": session["game_name"],
                "admin_name": session["admin_name"],
            }
        except ValueError:
            raise
        except Exception as exc:
            if _is_recoverable_db_error(exc):
                raise RuntimeError(f"DB access/config error while joining session: {exc}") from exc
            raise RuntimeError(f"Failed to join session: {exc}") from exc

    with _sessions_lock:
        session = _sessions.get(normalized)
        if session is None:
            return None

        players = _session_players.setdefault(normalized, set())
        if normalized_team in players:
            raise ValueError("same username exist")

        players.add(normalized_team)
        return session
