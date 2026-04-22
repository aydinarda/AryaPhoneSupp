from __future__ import annotations

import json
import hashlib
import random
import threading
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from ..db import (
    close_active_rounds,
    fetch_active_round,
    fetch_all_matching_results,
    fetch_game_session_by_code,
    fetch_latest_matching_result,
    fetch_latest_round,
    fetch_round_by_number,
    fetch_submissions_for_round,
    fetch_submissions_for_session,
    insert_game_round,
    insert_matching_result,
)
from ..live_state import (
    get_live_latest_matching_result,
    get_live_matching_results,
    get_live_round_submissions,
    get_live_session_submissions,
    upsert_live_matching_result,
)
from ..service import get_tables
from ..schemas import MatchRunRequest, PlayerJoinRequest, RoundStartRequest, SessionConfigRequest, SessionCreateRequest
from ..ws_manager import manager
from ..session_service import create_session, get_session, join_session, list_session_players
from ..settings import GAME_SETTINGS

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# In-memory session config per session code (beta distribution + delta + audit params).
# Resets on server restart (acceptable for a classroom game).
_session_beta: dict[str, tuple[float, float]] = {}
_session_delta: dict[str, float] = {}
_session_quality_sensitivity: dict[str, float] = {}
_session_audit: dict[str, tuple[float, float]] = {}  # (audit_probability, catch_probability)

_DEFAULT_ALPHA = 3.0
_DEFAULT_BETA = 3.0


def _stable_seed(payload: dict[str, Any]) -> int:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return int(hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16], 16)


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


def _resolve_session_int(
    session_row: dict[str, Any],
    key: str,
    default: int,
    minimum: int,
) -> int:
    raw = session_row.get(key, None)
    if raw is None:
        code = str(session_row.get("session_code", "")).strip().upper()
        if code:
            try:
                session = get_session(code)
                if session is not None:
                    raw = session.get(key, None)
            except Exception:
                raw = None

    try:
        return max(minimum, int(raw))
    except Exception:
        return default


def _resolve_total_rounds(session_row: dict[str, Any]) -> int:
    return _resolve_session_int(session_row, "number_of_rounds", default=5, minimum=1)


def _resolve_trial_rounds(session_row: dict[str, Any]) -> int:
    return _resolve_session_int(session_row, "trial_rounds", default=2, minimum=0)


def _resolve_scheduled_rounds(session_row: dict[str, Any]) -> int:
    return _resolve_total_rounds(session_row) + _resolve_trial_rounds(session_row)


def _build_round_phase(round_no: int, trial_rounds: int) -> dict[str, Any]:
    round_index = max(0, int(round_no))
    practice_rounds = max(0, int(trial_rounds))
    is_trial_round = practice_rounds > 0 and 0 < round_index <= practice_rounds
    return {
        "is_trial_round": is_trial_round,
        "trial_round_no": round_index if is_trial_round else None,
        "game_round_no": max(0, round_index - practice_rounds),
    }


def _build_game_finish_url(session_code: str) -> str:
    normalized = (session_code or "").strip().upper()
    if not normalized:
        return "/game-finish"
    return f"/game-finish?code={normalized}"


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


def _safe_int(value: Any, default: int = 0) -> int:
    """Convert value to int safely, returning default for None/NaN/inf."""
    try:
        import math as _math
        v = float(value)
        if _math.isnan(v) or _math.isinf(v):
            return default
        return int(v)
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
        avg_child_labor = sum(_safe_float(r.get("child_labor")) for r in selected_rows) / count
        avg_banned_chem = sum(_safe_float(r.get("banned_chem")) for r in selected_rows) / count

        # Category constraint: require exactly 1 supplier from each category
        all_cats_in_catalog = {
            (v.get("category") or "").strip()
            for v in suppliers_by_id.values()
            if (v.get("category") or "").strip()
        }
        if all_cats_in_catalog:
            cat_counts: dict[str, int] = {}
            for pid in valid:
                cat = (suppliers_by_id[pid].get("category") or "").strip()
                if cat:
                    cat_counts[cat] = cat_counts.get(cat, 0) + 1
            if any(cat_counts.get(cat, 0) != 1 for cat in all_cats_in_catalog):
                excluded.append(team)
                continue

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
            "price_per_user": _safe_float(
                row.get("price") if row.get("price") is not None else row.get("price_per_user"),
                float(GAME_SETTINGS.price_per_user),
            ),
            "picked_suppliers": valid,
            "avg_env": avg_env,
            "avg_social": avg_social,
            "avg_cost": avg_cost,
            "avg_strategic": avg_strategic,
            "avg_child_labor": avg_child_labor,
            "avg_banned_chem": avg_banned_chem,
        }

    return profiles, sorted(set(excluded))


@router.post("")
def create_game_session(req: SessionCreateRequest) -> dict[str, Any]:
    try:
        return create_session(req.game_name, req.admin_name or "Admin", req.number_of_rounds, req.trial_rounds)
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
    session_code = str(session.get("code", code)).strip().upper()
    try:
        manager.broadcast_sync(session_code, {
            "type": "player_joined",
            "team_name": (req.team_name or "").strip(),
            "players": list_session_players(session_code),
        })
    except Exception:
        pass
    return session


@router.post("/{code}/rounds/start")
def start_round(code: str, req: RoundStartRequest) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    total_rounds = _resolve_total_rounds(session_row)
    trial_rounds = _resolve_trial_rounds(session_row)
    scheduled_rounds = _resolve_scheduled_rounds(session_row)

    duration_seconds = req.duration_seconds if req.duration_seconds and req.duration_seconds > 0 else None
    market_capacity = max(1, _safe_int(req.market_capacity, 1))

    latest_rows = _extract_rows(fetch_latest_round(session_token))
    next_round_no = _safe_int(latest_rows[0].get("round_no"), 0) + 1 if latest_rows else 1

    if next_round_no > scheduled_rounds:
        raise HTTPException(
            status_code=400,
            detail=f"Configured round limit reached ({scheduled_rounds}). No more rounds can be started.",
        )
    phase = _build_round_phase(next_round_no, trial_rounds)

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

    session_code = str(session_row.get("session_code", code)).strip().upper()
    manager.broadcast_sync(session_code, {
        "type": "round_started",
        "round_no": next_round_no,
        "total_rounds": total_rounds,
        "trial_rounds": trial_rounds,
        "scheduled_rounds": scheduled_rounds,
        **phase,
        "duration_seconds": duration_seconds,
        "ends_at": ends_at,
    })

    return {
        "session_code": session_code,
        "round_no": next_round_no,
        "total_rounds": total_rounds,
        "trial_rounds": trial_rounds,
        "scheduled_rounds": scheduled_rounds,
        "remaining_rounds": max(0, scheduled_rounds - next_round_no),
        "remaining_game_rounds": max(0, total_rounds - phase["game_round_no"]),
        "remaining_trial_rounds": max(0, trial_rounds - (phase["trial_round_no"] or 0)),
        **phase,
        "duration_seconds": duration_seconds,
        "market_capacity": market_capacity,
        "started_at": now.isoformat(),
        "ends_at": ends_at,
        "is_active": True,
    }


@router.get("/{code}/rounds/current")
def get_current_round(code: str, include_delta: bool = False) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    total_rounds = _resolve_total_rounds(session_row)
    trial_rounds = _resolve_trial_rounds(session_row)
    scheduled_rounds = _resolve_scheduled_rounds(session_row)

    rows = _extract_rows(fetch_active_round(session_token))
    normalized = (code or "").strip().upper()
    beta_alpha, beta_beta = _session_beta.get(normalized, (_DEFAULT_ALPHA, _DEFAULT_BETA))
    quality_sensitivity = _session_quality_sensitivity.get(normalized, float(GAME_SETTINGS.quality_sensitivity))
    audit_probability, catch_probability = _session_audit.get(
        normalized, (float(GAME_SETTINGS.audit_probability), float(GAME_SETTINGS.catch_probability))
    )

    payload: dict[str, Any] = {
        "total_rounds": total_rounds,
        "trial_rounds": trial_rounds,
        "scheduled_rounds": scheduled_rounds,
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
        "quality_sensitivity": quality_sensitivity,
        "audit_probability": audit_probability,
        "catch_probability": catch_probability,
    }
    if include_delta:
        payload["delta"] = _session_delta.get(normalized, float(GAME_SETTINGS.price_sensitivity_delta))

    if not rows:
        return {"round": None, **payload}

    row = rows[0]
    phase = _build_round_phase(_safe_int(row.get("round_no")), trial_rounds)
    return {
        "round": {
            "round_no": _safe_int(row.get("round_no")),
            **phase,
            "duration_seconds": row.get("duration_seconds"),
            "market_capacity": max(1, _safe_int(row.get("market_capacity"), GAME_SETTINGS.default_market_capacity)),
            "started_at": row.get("created_at"),
            "ends_at": row.get("ends_at"),
            "is_active": bool(row.get("is_active", False)),
        },
        **payload,
    }


@router.patch("/{code}/config")
def update_session_config(code: str, req: SessionConfigRequest) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_code = str(session_row.get("session_code", "")).strip().upper()
    if not session_code:
        raise HTTPException(status_code=400, detail="Session code not found")

    _session_beta[session_code] = (float(req.beta_alpha), float(req.beta_beta))
    if req.delta is not None:
        _session_delta[session_code] = float(req.delta)
    if req.quality_sensitivity is not None:
        _session_quality_sensitivity[session_code] = float(req.quality_sensitivity)
    if req.audit_probability is not None or req.catch_probability is not None:
        old_ap, old_cp = _session_audit.get(
            session_code,
            (float(GAME_SETTINGS.audit_probability), float(GAME_SETTINGS.catch_probability)),
        )
        new_ap = float(req.audit_probability) if req.audit_probability is not None else old_ap
        new_cp = float(req.catch_probability) if req.catch_probability is not None else old_cp
        _session_audit[session_code] = (new_ap, new_cp)
    current_delta = _session_delta.get(session_code, float(GAME_SETTINGS.price_sensitivity_delta))
    current_quality_sensitivity = _session_quality_sensitivity.get(session_code, float(GAME_SETTINGS.quality_sensitivity))
    current_ap, current_cp = _session_audit.get(
        session_code,
        (float(GAME_SETTINGS.audit_probability), float(GAME_SETTINGS.catch_probability)),
    )
    session_token = str(session_row.get("session_token", "")).strip()
    if session_token:
        manager.broadcast_sync(
            session_code,
            _build_sync_message(session_code, session_token, session_row),
        )
    return {
        "ok": True,
        "session_code": session_code,
        "beta_alpha": req.beta_alpha,
        "beta_beta": req.beta_beta,
        "delta": current_delta,
        "quality_sensitivity": current_quality_sensitivity,
        "audit_probability": current_ap,
        "catch_probability": current_cp,
    }


@router.post("/{code}/match")
def run_round_matching(code: str, req: MatchRunRequest) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    session_code = str(session_row.get("session_code", "")).strip().upper()
    if not session_token or not session_code:
        raise HTTPException(status_code=400, detail="Session metadata is incomplete")
    scheduled_rounds = _resolve_scheduled_rounds(session_row)

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

    round_no = _safe_int(target_round.get("round_no"))
    market_capacity = max(1, _safe_int(target_round.get("market_capacity"), GAME_SETTINGS.default_market_capacity))
    live_submission_rows = get_live_round_submissions(session_code, round_no)
    submission_source = "live"
    submission_rows = live_submission_rows
    if not submission_rows:
        submission_source = "database"
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

    suppliers_df, users_df = get_tables()
    suppliers_df = suppliers_df.copy()
    suppliers_df["supplier_id"] = suppliers_df["supplier_id"].astype(str)
    suppliers_by_id = {
        str(row["supplier_id"]): {k: row.get(k) for k in ("env_risk", "social_risk", "cost_score", "strategic", "child_labor", "banned_chem", "category")}
        for _, row in suppliers_df.iterrows()
    }

    team_profiles, profile_excluded = _build_team_product_profiles(eligible_rows, suppliers_by_id) if eligible_rows else ({}, [])
    if profile_excluded:
        excluded_infeasible_teams = sorted(set(excluded_infeasible_teams + profile_excluded))

    N = len(users_df)
    if N <= 0:
        raise HTTPException(status_code=400, detail="No users available in dataset for matching")

    normalized_code = (code or "").strip().upper()
    beta_alpha, beta_beta = _session_beta.get(normalized_code, (_DEFAULT_ALPHA, _DEFAULT_BETA))
    delta = _session_delta.get(normalized_code, float(GAME_SETTINGS.price_sensitivity_delta))
    quality_sensitivity = _session_quality_sensitivity.get(normalized_code, float(GAME_SETTINGS.quality_sensitivity))

    # --- Audit phase (runs before MNL; caught teams receive a utility penalty) ---
    from ..audit import run_audit

    audit_ap, audit_cp = _session_audit.get(
        session_code,
        (float(GAME_SETTINGS.audit_probability), float(GAME_SETTINGS.catch_probability)),
    )
    match_input = {
        "session_code": session_code,
        "round_no": round_no,
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
        "delta": delta,
        "quality_sensitivity": quality_sensitivity,
        "audit_probability": audit_ap,
        "catch_probability": audit_cp,
        "teams": [
            {
                "team": team,
                "selected_suppliers": str(row.get("selected_suppliers") or ""),
                "price": _safe_float(row.get("price"), GAME_SETTINGS.price_per_user),
                "feasible": _submission_is_feasible(row),
            }
            for team, row in sorted(by_team.items())
        ],
    }
    match_seed = _stable_seed(match_input)
    pre_audit_team_profiles = {team: dict(profile) for team, profile in team_profiles.items()}
    audit_result = run_audit(
        team_profiles=team_profiles,
        suppliers_df=suppliers_df,
        audit_probability=audit_ap,
        catch_probability=audit_cp,
        rng=random.Random(match_seed),
    )
    audit_penalized_teams = sorted(audit_result.penalized_teams)
    audit_penalties = {str(team): _safe_float(penalty) for team, penalty in audit_result.team_penalties.items()}

    # --- MNL demand model ---
    from ..beta_density import BetaDensity
    from ..mnl_market import BuyerProfile, run_mnl_market
    from ..customer_segment import CustomerSegment

    bd = BetaDensity(alpha=max(0.01, beta_alpha), beta=max(0.01, beta_beta))
    users_sorted = users_df.sort_values("w_cost").reset_index(drop=True)
    segments = [
        CustomerSegment(
            segment_id=str(row["user_id"]),
            density=float(bd.density_at((i + 0.5) / N)),
            w_env=_safe_float(row.get("w_env")),
            w_social=_safe_float(row.get("w_social")),
            w_cost=_safe_float(row.get("w_cost"), 1.0),
            w_low_quality=_safe_float(row.get("w_low_quality")),
        )
        for i, (_, row) in enumerate(users_sorted.iterrows())
    ]

    team_ids_sorted = sorted(team_profiles.keys())
    profiles = [
        BuyerProfile(
            team_name=tid,
            price_per_user=_safe_float(team_profiles[tid]["price_per_user"], GAME_SETTINGS.price_per_user),
            avg_env=_safe_float(team_profiles[tid]["avg_env"]),
            avg_social=_safe_float(team_profiles[tid]["avg_social"]),
            utility_adjustment=_safe_float(audit_penalties.get(tid, 0.0)),
        )
        for tid in team_ids_sorted
    ]

    mnl_result = run_mnl_market(
        profiles,
        segments,
        delta=delta,
        quality_sensitivity=quality_sensitivity,
        u_outside=None,
    ) if profiles else None

    market_to_users: dict[str, Any] = {}
    market_loads: dict[str, Any] = {}
    team_round_financials: list[dict[str, Any]] = []
    round_profit_total = 0.0

    for tid in team_ids_sorted:
        profile = team_profiles[tid]
        br = mnl_result.buyer_results.get(tid) if mnl_result else None
        demand_share = br.total_demand if br else 0.0
        effective_users = round(demand_share * N, 3)
        price = _safe_float(profile["price_per_user"], GAME_SETTINGS.price_per_user)
        avg_cost = _safe_float(profile["avg_cost"])
        unit_margin = price - float(GAME_SETTINGS.cost_scale) * avg_cost
        realized_profit = effective_users * unit_margin
        realized_utility = round((br.realized_utility * N) if br else 0.0, 3)
        audit_penalty = _safe_float(audit_penalties.get(tid, 0.0))
        round_profit_total += realized_profit

        avg_env       = _safe_float(profile.get("avg_env"),       0.0)
        avg_social    = _safe_float(profile.get("avg_social"),    0.0)
        avg_strategic = _safe_float(profile.get("avg_strategic"), 0.0)
        buyer_utility = round(
            0.1 * realized_profit
            + (5.0 - avg_env)
            + (5.0 - avg_social)
            + (5.0 - avg_strategic),
            4,
        )

        market_to_users[tid] = effective_users
        market_loads[tid] = {
            "demand_share": round(demand_share, 4),
            "effective_users": effective_users,
            "capacity": N,
            "assigned_count": effective_users,
        }
        team_round_financials.append({
            "team": tid,
            "demand_share": round(demand_share, 4),
            "effective_users": effective_users,
            "realized_profit": round(realized_profit, 2),
            "realized_utility": realized_utility,
            "buyer_utility": buyer_utility,
            "price_per_user": price,
            "avg_cost_score": avg_cost,
            "unit_margin": round(unit_margin, 2),
            "avg_strategic": round(avg_strategic, 3),
            "audit_penalty": round(audit_penalty, 3),
            "penalized_by_audit": audit_penalty != 0.0,
            "excluded_by_audit": False,
            "excluded_by_infeasible": False,
        })

    zero_team_ids = sorted(set(excluded_infeasible_teams))
    for tid in zero_team_ids:
        if tid in {row["team"] for row in team_round_financials}:
            continue
        original_profile = pre_audit_team_profiles.get(tid, {})
        original_row = by_team.get(tid, {})
        team_round_financials.append({
            "team": tid,
            "demand_share": 0.0,
            "effective_users": 0.0,
            "realized_profit": 0.0,
            "realized_utility": 0.0,
            "buyer_utility": 0.0,
            "price_per_user": _safe_float(
                original_profile.get("price_per_user", original_row.get("price")),
                GAME_SETTINGS.price_per_user,
            ),
            "avg_cost_score": _safe_float(
                original_profile.get("avg_cost", original_row.get("cost_avg")),
                0.0,
            ),
            "unit_margin": 0.0,
            "avg_strategic": _safe_float(
                original_profile.get("avg_strategic", original_row.get("strategic_avg")),
                0.0,
            ),
            "audit_penalty": 0.0,
            "penalized_by_audit": False,
            "excluded_by_audit": False,
            "excluded_by_infeasible": True,
        })

    team_round_financials.sort(key=lambda row: str(row.get("team") or ""))

    # Build per-segment share breakdown (sorted by segment index = w_cost order)
    segment_shares: list[dict[str, Any]] = []
    for idx, alloc in enumerate(mnl_result.segment_allocations if mnl_result else []):
        entry: dict[str, Any] = {
            "segment_index": idx + 1,
            "segment_id": alloc.segment_id,
            "density": round(alloc.density, 4),
            "shares": {t: round(s * 100, 2) for t, s in alloc.shares.items()},
        }
        segment_shares.append(entry)
    segment_shares.sort(key=lambda x: x["segment_index"])

    completed_at = datetime.now(UTC).isoformat()
    result = {
        "meta": {
            "solver": "mnl_v1",
            "completed_at": completed_at,
            "input_fingerprint": f"{match_seed:016x}",
            "user_pool_count": N,
            "eligible_team_count": len(profiles),
            "matched_count": N,
            "submitted_team_count": len(by_team),
            "submission_source": submission_source,
            "infeasible_excluded_count": len(excluded_infeasible_teams),
            "round_profit_total": round(round_profit_total, 2),
        },
        "market_to_users": market_to_users,
        "market_loads": market_loads,
        "excluded_infeasible_users": excluded_infeasible_teams,
        "excluded_infeasible_teams": excluded_infeasible_teams,
        "segment_shares": segment_shares,
        "audit": audit_result.to_dict(),
        "round_financials": {
            "formula": "realized_profit = effective_users(MNL) x unit_margin",
            "delta": round(delta, 4),
            "quality_sensitivity": round(quality_sensitivity, 4),
            "audit_utility_penalty": float(audit_result.utility_penalty),
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "cost_scale": float(GAME_SETTINGS.cost_scale),
            "team_financials": team_round_financials,
            "round_profit_total": round(round_profit_total, 2),
        },
    }

    _db_payload = {
        "session_token": session_token,
        "round_no": round_no,
        "solver": result.get("meta", {}).get("solver", "unknown"),
        "matched_count": _safe_int(result.get("meta", {}).get("matched_count")),
        "result": result,
    }
    upsert_live_matching_result(_db_payload)
    threading.Thread(target=insert_matching_result, args=(_db_payload.copy(),), daemon=True).start()

    manager.broadcast_sync(session_code, {
        "type": "match_result",
        "round_no": round_no,
        "matching": result,
    })
    game_finished = round_no >= scheduled_rounds
    finish_url = _build_game_finish_url(session_code)
    if game_finished:
        manager.broadcast_sync(session_code, {
            "type": "game_finished",
            "round_no": round_no,
            "session_code": session_code,
            "redirect_url": finish_url,
        })

    return {
        "session_code": session_code,
        "round_no": round_no,
        "market_capacity": market_capacity,
        "eligible_team_count": len(team_profiles),
        "excluded_infeasible_teams": excluded_infeasible_teams,
        "matching": result,
        "game_finished": game_finished,
        "redirect_url": finish_url if game_finished else None,
    }


@router.get("/{code}/match/latest")
def get_latest_match(code: str) -> dict[str, Any]:
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    if not session_token:
        raise HTTPException(status_code=400, detail="Session token not found")

    rows = get_live_latest_matching_result(session_token)
    if not rows:
        rows = _extract_rows(fetch_latest_matching_result(session_token))
    if not rows:
        return {"match": None}

    row = rows[0]
    return {
        "match": {
            "round_no": _safe_int(row.get("round_no")),
            "solver": row.get("solver"),
            "matched_count": _safe_int(row.get("matched_count")),
            "created_at": row.get("created_at"),
            "result": row.get("result"),
        }
    }


@router.get("/{code}/rounds/history")
def get_round_history(code: str) -> dict[str, Any]:
    """Return per-round profit/utility for every team in this session."""
    session_row = _get_session_row_or_404(code)
    session_code = str(session_row.get("session_code", "")).strip().upper()

    rows = get_live_session_submissions(session_code) or _extract_rows(fetch_submissions_for_session(session_code))
    if not rows:
        return {"teams": [], "rounds": []}

    # Latest submission per (team, round_no)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        team = str(row.get("team") or "(anonymous)").strip()
        rno = _safe_int(row.get("round_no"))
        key = (team, rno)
        existing = latest.get(key)
        if existing is None or str(existing.get("created_at") or "") <= str(row.get("created_at") or ""):
            latest[key] = row

    teams: list[str] = sorted({k[0] for k in latest})
    rounds: list[int] = sorted({k[1] for k in latest if k[1] > 0})

    series: list[dict[str, Any]] = [
        {
            "team": team,
            "data": [
                {
                    "round_no": rno,
                    "profit": float(latest[(team, rno)].get("profit") or 0),
                    "utility": float(latest[(team, rno)].get("utility") or 0),
                }
                for rno in rounds
                if (team, rno) in latest
            ],
        }
        for team in teams
    ]

    return {"teams": teams, "rounds": rounds, "series": series}


# ---------------------------------------------------------------------------
# Helper: parse JSON result blob stored in matching_results table
# ---------------------------------------------------------------------------

def _parse_result_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        import json as _json
        try:
            return _json.loads(raw)
        except Exception:
            return {}
    return {}


def _match_row_completed_at(row: dict[str, Any]) -> str:
    result = _parse_result_json(row.get("result"))
    meta = result.get("meta") or {}
    return str(meta.get("completed_at") or row.get("created_at") or "")


def _latest_match_rows_by_round(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    for row in rows:
        rno = _safe_int(row.get("round_no"))
        if rno <= 0:
            continue
        existing = latest.get(rno)
        if existing is None or _match_row_completed_at(existing) <= _match_row_completed_at(row):
            latest[rno] = row
    return [latest[rno] for rno in sorted(latest)]


def _load_matching_rows(session_token: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        rows.extend(_extract_rows(fetch_all_matching_results(session_token)))
    except Exception:
        pass
    rows.extend(get_live_matching_results(session_token))
    return _latest_match_rows_by_round(rows)


_LEADERBOARD_METRICS = {"buyer_utility", "profit", "market_share", "market_utility"}

_METRIC_FIELD: dict[str, str] = {
    "buyer_utility":  "buyer_utility",
    "profit":         "realized_profit",
    "market_share":   "market_share_pct",
    "market_utility": "realized_utility",
}

_CUMULATIVE_FIELD: dict[str, str] = {
    "buyer_utility":  "total_buyer_utility",
    "profit":         "total_profit",
    "market_share":   "total_market_share_pct",
    "market_utility": "total_realized_utility",
}


@router.get("/{code}/leaderboard")
def get_session_leaderboard(
    code: str,
    x_metric: str = "market_utility",
    y_metric: str = "buyer_utility",
) -> dict[str, Any]:
    """Return match leaderboard metrics: demand share, profit, market/customer utility, and buyer utility."""
    session_row = _get_session_row_or_404(code)
    session_token = str(session_row.get("session_token", "")).strip()
    session_code = str(session_row.get("session_code", "")).strip().upper()
    if not session_token or not session_code:
        raise HTTPException(status_code=400, detail="Session metadata is incomplete")

    x_metric = x_metric if x_metric in _LEADERBOARD_METRICS else "market_utility"
    y_metric = y_metric if y_metric in _LEADERBOARD_METRICS else "buyer_utility"

    # --- Load matching results (team financials per round) ---
    match_rows = _load_matching_rows(session_token)

    # round_no → { team → {demand_share, realized_profit, realized_utility, buyer_utility} }
    round_financials: dict[int, dict[str, dict[str, float]]] = {}
    for mr in match_rows:
        rno = _safe_int(mr.get("round_no"))
        result = _parse_result_json(mr.get("result"))
        team_fins = (result.get("round_financials") or {}).get("team_financials") or []
        round_financials[rno] = {}
        for tf in team_fins:
            team = str(tf.get("team") or "").strip()
            if team:
                round_financials[rno][team] = {
                    "demand_share": float(tf.get("demand_share") or 0.0),
                    "realized_profit": float(tf.get("realized_profit") or 0.0),
                    "realized_utility": float(tf.get("realized_utility") or 0.0),
                    "buyer_utility": float(tf.get("buyer_utility") or 0.0),
                    "price_per_user": float(tf.get("price_per_user") or 0.0),
                    "audit_penalty": float(tf.get("audit_penalty") or 0.0),
                    "penalized_by_audit": bool(tf.get("penalized_by_audit")),
                    "excluded_by_audit": bool(tf.get("excluded_by_audit")),
                    "excluded_by_infeasible": bool(tf.get("excluded_by_infeasible")),
                }

    trial_rounds = _resolve_trial_rounds(session_row)

    # --- Build per-round entries ---
    all_rounds = sorted(rno for rno in round_financials.keys() if rno > trial_rounds)
    turn_leaderboard: list[dict[str, Any]] = []

    for rno in all_rounds:
        teams_fin = round_financials[rno]
        if not teams_fin:
            continue

        round_entries: list[dict[str, Any]] = []
        for team, m in teams_fin.items():
            entry = {
                "team": team,
                "round_no": rno,
                "game_round_no": max(1, rno - trial_rounds),
                "price_per_user": round(m["price_per_user"], 2),
                "realized_profit": round(m["realized_profit"], 2),
                "market_share_pct": round(m["demand_share"] * 100.0, 2),
                "realized_utility": round(m["realized_utility"], 4),
                "buyer_utility": round(m["buyer_utility"], 4),
            }
            entry["x_value"] = entry[_METRIC_FIELD[x_metric]]
            entry["y_value"] = entry[_METRIC_FIELD[y_metric]]
            round_entries.append(entry)

        round_entries.sort(key=lambda x: x["realized_profit"], reverse=True)
        turn_leaderboard.extend(round_entries)

    # --- Build cumulative leaderboard (plain sum across rounds) ---
    completed_round_count = len(all_rounds)
    cumulative: dict[str, dict[str, Any]] = {}
    for entry in turn_leaderboard:
        team = entry["team"]
        if team not in cumulative:
            cumulative[team] = {
                "team": team,
                "rounds_played": completed_round_count,
                "total_profit": 0.0,
                "total_market_share_pct": 0.0,
                "total_realized_utility": 0.0,
                "total_buyer_utility": 0.0,
            }
        c = cumulative[team]
        c["total_profit"]            += entry["realized_profit"]
        c["total_market_share_pct"]  += entry["market_share_pct"]
        c["total_realized_utility"]  += entry["realized_utility"]
        c["total_buyer_utility"]     += entry["buyer_utility"]

    cumulative_list = list(cumulative.values())
    for c in cumulative_list:
        c["total_profit"]           = round(c["total_profit"], 2)
        c["total_market_share_pct"] = round(c["total_market_share_pct"], 2)
        c["total_realized_utility"] = round(c["total_realized_utility"], 4)
        c["total_buyer_utility"]    = round(c["total_buyer_utility"], 4)
        c["x_value"] = c[_CUMULATIVE_FIELD[x_metric]]
        c["y_value"] = c[_CUMULATIVE_FIELD[y_metric]]

    cumulative_list.sort(
        key=lambda x: (x["total_profit"] / max(1, x["rounds_played"])),
        reverse=True,
    )

    return {
        "rounds": all_rounds,
        "trial_rounds": trial_rounds,
        "available_metrics": sorted(_LEADERBOARD_METRICS),
        "x_metric": x_metric,
        "y_metric": y_metric,
        "turn_leaderboard": turn_leaderboard,
        "cumulative_leaderboard": cumulative_list,
    }


# ---------------------------------------------------------------------------
# WebSocket helpers + endpoint
# ---------------------------------------------------------------------------

def _build_sync_message(
    session_code: str,
    session_token: str,
    session_row: dict[str, Any],
) -> dict[str, Any]:
    total_rounds = _resolve_total_rounds(session_row)
    trial_rounds = _resolve_trial_rounds(session_row)
    scheduled_rounds = _resolve_scheduled_rounds(session_row)
    beta_alpha, beta_beta = _session_beta.get(session_code, (_DEFAULT_ALPHA, _DEFAULT_BETA))
    quality_sensitivity = _session_quality_sensitivity.get(session_code, float(GAME_SETTINGS.quality_sensitivity))
    audit_probability, catch_probability = _session_audit.get(
        session_code, (float(GAME_SETTINGS.audit_probability), float(GAME_SETTINGS.catch_probability))
    )

    round_data = None
    live_submissions: list[dict[str, Any]] = []
    try:
        active_rows = _extract_rows(fetch_active_round(session_token))
        if active_rows:
            r = active_rows[0]
            round_no = _safe_int(r.get("round_no"))
            phase = _build_round_phase(round_no, trial_rounds)
            round_data = {
                "round_no": round_no,
                **phase,
                "duration_seconds": r.get("duration_seconds"),
                "market_capacity": max(1, _safe_int(r.get("market_capacity"), GAME_SETTINGS.default_market_capacity)),
                "started_at": r.get("created_at"),
                "ends_at": r.get("ends_at"),
                "is_active": bool(r.get("is_active", False)),
            }
            live_submissions = get_live_round_submissions(session_code, round_no)
    except Exception:
        pass

    match_data = None
    try:
        match_rows = _extract_rows(fetch_latest_matching_result(session_token))
        if match_rows:
            match_data = _parse_result_json(match_rows[0].get("result"))
    except Exception:
        pass

    return {
        "type": "sync",
        "total_rounds": total_rounds,
        "trial_rounds": trial_rounds,
        "scheduled_rounds": scheduled_rounds,
        "players": list_session_players(session_code),
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
        "quality_sensitivity": quality_sensitivity,
        "audit_probability": audit_probability,
        "catch_probability": catch_probability,
        "round": round_data,
        "submissions": live_submissions,
        "match": match_data,
    }


@router.websocket("/{code}/ws")
async def websocket_session(code: str, websocket: WebSocket) -> None:
    await websocket.accept()

    normalized = (code or "").strip().upper()
    rows: list[dict[str, Any]] = []
    try:
        rows = _extract_rows(fetch_game_session_by_code(normalized))
    except Exception:
        pass

    if not rows or rows[0].get("is_active") is False:
        await websocket.close(code=4004)
        return

    session_row = rows[0]
    session_code = str(session_row.get("session_code", normalized)).strip().upper()
    session_token = str(session_row.get("session_token", "")).strip()

    manager.register(session_code, websocket)
    try:
        await websocket.send_json(_build_sync_message(session_code, session_token, session_row))
        async for text in websocket.iter_text():
            try:
                msg = json.loads(text)
                if isinstance(msg, dict) and msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except Exception:
                pass
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        manager.disconnect(session_code, websocket)
