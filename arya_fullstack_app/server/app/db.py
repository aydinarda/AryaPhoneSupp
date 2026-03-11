from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import tomllib
from supabase import Client, create_client


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
    return get_client().table("submissions").insert(payload).execute()


def fetch_all_submissions(limit: int = 5000):
    return get_client().table("submissions").select("*").order("created_at", desc=True).limit(limit).execute()


def insert_game_session(payload: dict):
    return get_client().table("game_sessions").insert(payload).execute()


def fetch_game_session_by_code(code: str):
    return get_client().table("game_sessions").select("*").eq("session_code", code).limit(1).execute()


def fetch_session_player(session_token: str, team_name_normalized: str):
    return (
        get_client()
        .table("session_players")
        .select("id")
        .eq("session_token", session_token)
        .eq("team_name_normalized", team_name_normalized)
        .limit(1)
        .execute()
    )


def insert_session_player(payload: dict):
    return get_client().table("session_players").insert(payload).execute()
