from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from typing import Any


_lock = RLock()
_submissions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
_matching_results: dict[str, dict[int, dict[str, Any]]] = {}


def _session_code(value: Any) -> str:
    return str(value or "").strip().upper()


def _round_no(value: Any) -> int:
    try:
        return int(float(value))
    except Exception:
        return 0


def _team(value: Any) -> str:
    return str(value or "(anonymous)").strip() or "(anonymous)"


def upsert_live_submission(row: dict[str, Any]) -> dict[str, Any] | None:
    """Store the latest canonical submission in server memory.

    The live store is the authoritative source for the current classroom round.
    Database writes are still useful for history, but matching should not depend
    on database realtime behaviour.
    """
    session_code = _session_code(row.get("session_code"))
    round_no = _round_no(row.get("round_no"))
    team = _team(row.get("team"))
    if not session_code or round_no <= 0:
        return None

    stored = deepcopy(row)
    stored["session_code"] = session_code
    stored["round_no"] = round_no
    stored["team"] = team
    stored.setdefault("created_at", datetime.now(UTC).isoformat())

    key = (session_code, round_no)
    with _lock:
        _submissions.setdefault(key, {})[team] = stored
        return deepcopy(stored)


def get_live_round_submissions(session_code: str, round_no: int) -> list[dict[str, Any]]:
    key = (_session_code(session_code), _round_no(round_no))
    with _lock:
        rows = list(_submissions.get(key, {}).values())
        return deepcopy(sorted(rows, key=lambda row: str(row.get("created_at") or "")))


def get_live_session_submissions(session_code: str) -> list[dict[str, Any]]:
    normalized = _session_code(session_code)
    with _lock:
        rows: list[dict[str, Any]] = []
        for (code, _round), by_team in _submissions.items():
            if code == normalized:
                rows.extend(by_team.values())
        return deepcopy(sorted(rows, key=lambda row: str(row.get("created_at") or "")))


def _session_token(value: Any) -> str:
    return str(value or "").strip()


def _match_created_at(row: dict[str, Any]) -> str:
    result = row.get("result") if isinstance(row, dict) else {}
    if isinstance(result, dict):
        meta = result.get("meta") or {}
        completed_at = meta.get("completed_at")
        if completed_at:
            return str(completed_at)
    return str(row.get("created_at") or "")


def upsert_live_matching_result(row: dict[str, Any]) -> dict[str, Any] | None:
    """Store the latest matching result per session round in server memory."""
    session_token = _session_token(row.get("session_token"))
    round_no = _round_no(row.get("round_no"))
    if not session_token or round_no <= 0:
        return None

    stored = deepcopy(row)
    stored["session_token"] = session_token
    stored["round_no"] = round_no
    if not stored.get("created_at"):
        result = stored.get("result") if isinstance(stored.get("result"), dict) else {}
        meta = result.get("meta") if isinstance(result, dict) else {}
        stored["created_at"] = str((meta or {}).get("completed_at") or datetime.now(UTC).isoformat())

    with _lock:
        _matching_results.setdefault(session_token, {})[round_no] = stored
        return deepcopy(stored)


def get_live_matching_results(session_token: str) -> list[dict[str, Any]]:
    normalized = _session_token(session_token)
    with _lock:
        rows = list(_matching_results.get(normalized, {}).values())
        return deepcopy(sorted(rows, key=lambda row: (_round_no(row.get("round_no")), _match_created_at(row))))


def get_live_latest_matching_result(session_token: str) -> list[dict[str, Any]]:
    rows = get_live_matching_results(session_token)
    if not rows:
        return []
    return [max(rows, key=lambda row: (_round_no(row.get("round_no")), _match_created_at(row)))]
