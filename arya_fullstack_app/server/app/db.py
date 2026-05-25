from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import tomllib
from supabase import Client, create_client

_log = logging.getLogger(__name__)


def _run(query: Any) -> Any:
    """Execute a Supabase query and log clearly on failure."""
    try:
        return query.execute()
    except Exception as exc:
        _log.error("[Supabase error] %s: %s", type(exc).__name__, exc)
        raise


def _read_secrets_file() -> Dict[str, Any]:
    root_dir = Path(__file__).resolve().parents[3]
    secrets_path = root_dir / "secrets.toml"
    if not secrets_path.exists():
        return {}
    with secrets_path.open("rb") as f:
        return tomllib.load(f)


def get_supabase_credentials() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")

    if url and key:
        return url, key

    secrets = _read_secrets_file()
    url = url or secrets.get("SUPABASE_URL")
    key = key or secrets.get("SUPABASE_ANON_KEY")

    if not url or not key:
        raise RuntimeError("Missing SUPABASE_URL / SUPABASE_ANON_KEY.")
    return str(url), str(key)


def has_supabase_credentials() -> bool:
    try:
        get_supabase_credentials()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def get_client() -> Client:
    url, key = get_supabase_credentials()
    return create_client(url, key)


def insert_submission(payload: dict):
    return _run(get_client().table("submissions").insert(payload))


def fetch_all_submissions(limit: int = 5000):
    return _run(get_client().table("submissions").select("*").order("created_at", desc=True).limit(limit))


def insert_game_session(payload: dict):
    return _run(get_client().table("game_sessions").insert(payload))


_session_cache: dict[str, Any] = {}
_session_cache_lock = threading.Lock()


def fetch_game_session_by_code(code: str):
    if code in _session_cache:
        return _session_cache[code]
    with _session_cache_lock:
        if code in _session_cache:
            return _session_cache[code]
        result = _run(get_client().table("game_sessions").select("*").eq("session_code", code).limit(1))
        rows = getattr(result, "data", None) or []
        if rows:
            _session_cache[code] = result
        return result


def fetch_session_player(session_token: str, team_name_normalized: str):
    return _run(
        get_client()
        .table("session_players")
        .select("id")
        .eq("session_token", session_token)
        .eq("team_name_normalized", team_name_normalized)
        .limit(1)
    )


def fetch_session_players(session_token: str):
    return _run(
        get_client()
        .table("session_players")
        .select("team_name")
        .eq("session_token", session_token)
    )


def insert_session_player(payload: dict):
    return _run(get_client().table("session_players").insert(payload))


def close_active_rounds(session_token: str):
    return _run(
        get_client()
        .table("game_rounds")
        .update({"is_active": False})
        .eq("session_token", session_token)
        .eq("is_active", True)
    )


def insert_game_round(payload: dict):
    return _run(get_client().table("game_rounds").insert(payload))


def fetch_latest_round(session_token: str):
    return _run(
        get_client()
        .table("game_rounds")
        .select("*")
        .eq("session_token", session_token)
        .order("round_no", desc=True)
        .limit(1)
    )


def fetch_active_round(session_token: str):
    return _run(
        get_client()
        .table("game_rounds")
        .select("*")
        .eq("session_token", session_token)
        .eq("is_active", True)
        .order("round_no", desc=True)
        .limit(1)
    )


def fetch_round_by_number(session_token: str, round_no: int):
    return _run(
        get_client()
        .table("game_rounds")
        .select("*")
        .eq("session_token", session_token)
        .eq("round_no", int(round_no))
        .limit(1)
    )


def fetch_submissions_for_session(session_code: str):
    return _run(
        get_client()
        .table("submissions")
        .select("*")
        .eq("session_code", session_code)
        .order("created_at", desc=False)
    )


def fetch_submissions_for_round(session_code: str, round_no: int):
    return _run(
        get_client()
        .table("submissions")
        .select("*")
        .eq("session_code", session_code)
        .eq("round_no", int(round_no))
        .order("created_at", desc=False)
    )


def insert_matching_result(payload: dict):
    return _run(
        get_client()
        .table("matching_results")
        .upsert(payload, on_conflict="session_token,round_no")
    )


def fetch_latest_matching_result(session_token: str):
    return _run(
        get_client()
        .table("matching_results")
        .select("*")
        .eq("session_token", session_token)
        .order("created_at", desc=True)
        .limit(1)
    )


def fetch_all_matching_results(session_token: str):
    return _run(
        get_client()
        .table("matching_results")
        .select("round_no, result, created_at")
        .eq("session_token", session_token)
        .order("round_no", desc=False)
    )
