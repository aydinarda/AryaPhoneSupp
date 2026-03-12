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
from ..service import get_tables
from ..schemas import MatchRunRequest, PlayerJoinRequest, RoundStartRequest, SessionCreateRequest
from ..session_service import create_session, get_session, join_session
from ..settings import FIXED_POLICY, GAME_SETTINGS

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


def _parse_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n", "", "none", "null"}:
        return False
    return False


def _submission_is_feasible(row: dict[str, Any]) -> bool:
    env_avg = row.get("env_avg")
    social_avg = row.get("social_avg")

    try:
        if env_avg is not None and social_avg is not None:
            return (
                float(env_avg) <= float(GAME_SETTINGS.env_cap) + 1e-12
                and float(social_avg) <= float(GAME_SETTINGS.social_cap) + 1e-12
            )
    except Exception:
        pass

    explicit = row.get("feasible")
    if explicit is None:
        return False
    return _parse_bool_flag(explicit)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _build_team_product_profiles(
    team_rows: dict[str, dict[str, Any]],
    suppliers_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    profiles: dict[str, dict[str, Any]] = {}
    excluded: list[str] = []

    for team, row in team_rows.items():
        selected_raw = str(row.get("selected_suppliers") or "")
        picks = [x.strip() for x in selected_raw.split(",") if x.strip()]
        picks = list(dict.fromkeys(picks))
        valid = [pid for pid in picks if pid in suppliers_by_id]
        if not valid:
            excluded.append(team)
            continue

        selected_rows = [suppliers_by_id[pid] for pid in valid]
        count = float(len(selected_rows))
        avg_env = sum(_safe_float(r.get("env_risk")) for r in selected_rows) / count
        avg_social = sum(_safe_float(r.get("social_risk")) for r in selected_rows) / count
        avg_cost = sum(_safe_float(r.get("cost_score")) for r in selected_rows) / count
        avg_strategic = sum(_safe_float(r.get("strategic")) for r in selected_rows) / count
        avg_improvement = sum(_safe_float(r.get("improvement")) for r in selected_rows) / count
        avg_low_quality = sum(_safe_float(r.get("low_quality")) for r in selected_rows) / count

        profile_feasible = (
            avg_env <= float(GAME_SETTINGS.env_cap) + 1e-12
            and avg_social <= float(GAME_SETTINGS.social_cap) + 1e-12
        )
        if not profile_feasible:
            excluded.append(team)
            continue

        profiles[team] = {
            "team": team,
            "created_at": str(row.get("created_at") or ""),
            "picked_suppliers": valid,
            "avg_env": avg_env,
            "avg_social": avg_social,
            "avg_cost": avg_cost,
            "avg_strategic": avg_strategic,
            "avg_improvement": avg_improvement,
            "avg_low_quality": avg_low_quality,
        }

    return profiles, sorted(set(excluded))


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
            "market_capacity": int(row.get("market_capacity", GAME_SETTINGS.default_market_capacity) or GAME_SETTINGS.default_market_capacity),
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
    market_capacity = max(1, int(target_round.get("market_capacity", GAME_SETTINGS.default_market_capacity) or GAME_SETTINGS.default_market_capacity))
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

    eligible_rows = {
        team: row for team, row in by_team.items() if _submission_is_feasible(row)
    }
    excluded_infeasible_teams = sorted(team for team in by_team if team not in eligible_rows)
    if not eligible_rows:
        raise HTTPException(status_code=400, detail="No feasible submissions found for this round")

    suppliers_df, users_df = get_tables()
    suppliers_df = suppliers_df.copy()
    suppliers_df["supplier_id"] = suppliers_df["supplier_id"].astype(str)
    suppliers_by_id = {
        str(row["supplier_id"]): {
            "env_risk": row.get("env_risk"),
            "social_risk": row.get("social_risk"),
            "cost_score": row.get("cost_score"),
            "strategic": row.get("strategic"),
            "improvement": row.get("improvement"),
            "low_quality": row.get("low_quality"),
        }
        for _, row in suppliers_df.iterrows()
    }

    team_profiles, profile_excluded = _build_team_product_profiles(eligible_rows, suppliers_by_id)
    if profile_excluded:
        excluded_infeasible_teams = sorted(set(excluded_infeasible_teams + profile_excluded))

    if not team_profiles:
        raise HTTPException(status_code=400, detail="No feasible team product found for matching")

    if len(users_df) <= 0:
        raise HTTPException(status_code=400, detail="No users available in dataset for matching")

    served_df = users_df.copy()
    team_ids_sorted = sorted(team_profiles.keys())

    user_score_map: dict[str, dict[str, float]] = {}
    users_payload: list[dict[str, Any]] = []

    for _, user_row in served_df.iterrows():
        user_id = str(user_row.get("user_id", "")).strip()
        if not user_id:
            continue

        w_env = _safe_float(user_row.get("w_env"))
        w_social = _safe_float(user_row.get("w_social"))
        w_cost = _safe_float(user_row.get("w_cost"))
        w_str = _safe_float(user_row.get("w_strategic"))
        w_imp = _safe_float(user_row.get("w_improvement"))
        w_lq = _safe_float(user_row.get("w_low_quality"))

        utilities: dict[str, float] = {}
        for team_id in team_ids_sorted:
            profile = team_profiles[team_id]
            score = (
                w_env * (float(FIXED_POLICY.env_mult) * float(profile["avg_env"]))
                + w_social * (float(FIXED_POLICY.social_mult) * float(profile["avg_social"]))
                + w_cost * (float(FIXED_POLICY.cost_mult) * float(profile["avg_cost"]))
                + w_str * (float(FIXED_POLICY.strategic_mult) * float(profile["avg_strategic"]))
                + w_imp * (float(FIXED_POLICY.improvement_mult) * float(profile["avg_improvement"]))
                + w_lq * (float(FIXED_POLICY.low_quality_mult) * float(profile["avg_low_quality"]))
            )
            utilities[team_id] = float(score)

        ordered = sorted(team_ids_sorted, key=lambda tid: (-utilities[tid], tid))
        users_payload.append(
            {
                "user_id": user_id,
                "choices": ordered,
                "utilities": utilities,
            }
        )
        user_score_map[user_id] = utilities

    if not users_payload:
        raise HTTPException(status_code=400, detail="No valid users available for matching")

    market_options = []
    for team_id in team_ids_sorted:
        ranked_users = sorted(
            [str(u.get("user_id", "")) for u in users_payload],
            key=lambda uid: (-user_score_map.get(uid, {}).get(team_id, 0.0), uid),
        )
        market_options.append(
            {
                "option_id": team_id,
                "capacity": market_capacity,
                "priority": ranked_users,
                "request_time": team_profiles[team_id].get("created_at") or target_round.get("created_at"),
            }
        )

    result = run_market_matching(users=users_payload, market_options=market_options)
    result["excluded_infeasible_users"] = excluded_infeasible_teams
    result["excluded_infeasible_teams"] = excluded_infeasible_teams
    result["matching_target"] = "team_product"
    result_meta = result.setdefault("meta", {})
    result_meta["submitted_team_count"] = len(by_team)
    result_meta["eligible_team_count"] = len(team_profiles)
    result_meta["user_pool_count"] = len(users_payload)
    result_meta["infeasible_excluded_count"] = len(excluded_infeasible_teams)

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
        "eligible_team_count": len(team_profiles),
        "excluded_infeasible_teams": excluded_infeasible_teams,
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
