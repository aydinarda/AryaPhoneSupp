from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from threading import RLock
from typing import Any


_lock = RLock()
_submissions: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}


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
