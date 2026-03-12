from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException

from ..db import (
    close_active_rounds,
    fetch_active_round,
    fetch_game_session_by_code,
    fetch_latest_matching_result,
    fetch_latest_round,
    fetch_round_by_number,
    fetch_submissions_for_round,
    insert_game_round,
    insert_matching_result,
)
from ..matching_engine import run_market_matching
from ..schemas import MatchRunRequest, PlayerJoinRequest, RoundStartRequest, SessionCreateRequest
from ..session_service import create_session, get_session, join_session

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


def _extract_rows(res: Any) -> list[dict[str, Any]]:
    rows = getattr(res, "data", None)
    if rows is None and isinstance(res, dict):
        rows = res.get("data", [])
    return rows or []


def _get_session_row_or_404(code: str) -> dict[str, Any]:
    normalized = (code or "").strip().upper()
    if not normalized:
        raise HTTPException(status_code=404, detail="Session code not found")

    res = fetch_game_session_by_code(normalized)
    rows = _extract_rows(res)
    if not rows:
        raise HTTPException(status_code=404, detail="Session code not found")
    row = rows[0]
    if row.get("is_active") is False:
        raise HTTPException(status_code=404, detail="Session code not found")
    return row


@router.post("")
def create_game_session(req: SessionCreateRequest) -> dict[str, Any]:
    try:
        return create_session(req.game_name, req.admin_name or "Admin")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/{code}")
def get_game_session(code: str) -> dict[str, Any]:
    try:
        session = get_session(code)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not session:
        raise HTTPException(status_code=404, detail="Session code not found")
    return session


@router.post("/{code}/join")
def join_game_session(code: str, req: PlayerJoinRequest) -> dict[str, Any]:
    try:
        session = join_session(code, req.team_name)
    except ValueError as exc:
        msg = str(exc)
        if "required" in msg.lower():
            raise HTTPException(status_code=400, detail=msg) from exc
        raise HTTPException(status_code=409, detail=msg) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if session is None:
        raise HTTPException(status_code=404, detail="Session code not found")
    return session


@router.post("/{code}/rounds/start")
def start_round(code: str, req: RoundStartRequest) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    duration_seconds = req.duration_seconds if req.duration_seconds and req.duration_seconds > 0 else None
    market_capacity = max(1, int(req.market_capacity or 1))

    latest_rows = _extract_rows(fetch_latest_round(session_token))
    next_round_no = int(latest_rows[0].get("round_no", 0) or 0) + 1 if latest_rows else 1

    now = datetime.now(UTC)
    ends_at = (now + timedelta(seconds=duration_seconds)).isoformat() if duration_seconds else None

    close_active_rounds(session_token)
    payload = {
        "session_token": session_token,
        "round_no": next_round_no,
        "duration_seconds": duration_seconds,
        "market_capacity": market_capacity,
        "is_active": True,
        "ends_at": ends_at,
    }
    insert_game_round(payload)

    return {
        "session_code": str(session_row.get("session_code", "")),
        "round_no": next_round_no,
        "duration_seconds": duration_seconds,
        "market_capacity": market_capacity,
        "started_at": now.isoformat(),
        "ends_at": ends_at,
        "is_active": True,
    }


@router.get("/{code}/rounds/current")
def get_current_round(code: str) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    rows = _extract_rows(fetch_active_round(session_token))
    if not rows:
        return {"round": None}

    row = rows[0]
    return {
        "round": {
            "round_no": int(row.get("round_no", 0) or 0),
            "duration_seconds": row.get("duration_seconds"),
            "market_capacity": int(row.get("market_capacity", 1) or 1),
            "started_at": row.get("created_at"),
            "ends_at": row.get("ends_at"),
            "is_active": bool(row.get("is_active", False)),
        }
    }


@router.post("/{code}/match")
def run_round_matching(code: str, req: MatchRunRequest) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    session_code = str(session_row.get("session_code", "")).strip().upper()
    if not session_token or not session_code:
        raise HTTPException(status_code=400, detail="Session metadata is incomplete")

    target_round: dict[str, Any] | None = None
    if req.round_no is not None:
        rows = _extract_rows(fetch_round_by_number(session_token, int(req.round_no)))
        if rows:
            target_round = rows[0]
    else:
        active_rows = _extract_rows(fetch_active_round(session_token))
        if active_rows:
            target_round = active_rows[0]
        else:
            latest_rows = _extract_rows(fetch_latest_round(session_token))
            if latest_rows:
                target_round = latest_rows[0]

    if not target_round:
        raise HTTPException(status_code=400, detail="No round found for matching")

    round_no = int(target_round.get("round_no", 0) or 0)
    market_capacity = max(1, int(target_round.get("market_capacity", 1) or 1))
    submission_rows = _extract_rows(fetch_submissions_for_round(session_code, round_no))
    if not submission_rows:
        raise HTTPException(status_code=400, detail="No submissions found for this round")

    # Keep latest submission per team within the round.
    by_team: dict[str, dict[str, Any]] = {}
    for row in submission_rows:
        team = str(row.get("team", "")).strip() or "(anonymous)"
        created = str(row.get("created_at") or "")
        existing = by_team.get(team)
        if existing is None or str(existing.get("created_at") or "") <= created:
            by_team[team] = row

    users_payload: list[dict[str, Any]] = []
    market_ids: set[str] = set()

    for team, row in by_team.items():
        selected_raw = str(row.get("selected_suppliers") or "")
        choices = [x.strip() for x in selected_raw.split(",") if x.strip()]
        if not choices:
            continue

        for mid in choices:
            market_ids.add(mid)

        utilities = {mid: float(len(choices) - idx) for idx, mid in enumerate(choices)}
        users_payload.append(
            {
                "user_id": team,
                "choices": choices,
                "utilities": utilities,
            }
        )

    if not users_payload or not market_ids:
        raise HTTPException(status_code=400, detail="Not enough preference data to run matching")

    priority_order = sorted(
        [
            (team, str(by_team[team].get("created_at") or ""))
            for team in by_team.keys()
            if any(u.get("user_id") == team for u in users_payload)
        ],
        key=lambda x: x[1],
    )
    priority_users = [team for team, _ in priority_order]

    market_options = [
        {
            "option_id": mid,
            "capacity": market_capacity,
            "priority": priority_users,
            "request_time": target_round.get("created_at"),
        }
        for mid in sorted(market_ids)
    ]

    result = run_market_matching(users=users_payload, market_options=market_options)

    insert_matching_result(
        {
            "session_token": session_token,
            "round_no": round_no,
            "solver": result.get("meta", {}).get("solver", "unknown"),
            "matched_count": int(result.get("meta", {}).get("matched_count", 0) or 0),
            "result": result,
        }
    )

    return {
        "session_code": session_code,
        "round_no": round_no,
        "market_capacity": market_capacity,
        "matching": result,
    }


@router.get("/{code}/match/latest")
def get_latest_match(code: str) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    rows = _extract_rows(fetch_latest_matching_result(session_token))
    if not rows:
        return {"match": None}

    row = rows[0]
    return {
        "match": {
            "round_no": int(row.get("round_no", 0) or 0),
            "solver": row.get("solver"),
            "matched_count": int(row.get("matched_count", 0) or 0),
            "created_at": row.get("created_at"),
            "result": row.get("result"),
        }
    }
