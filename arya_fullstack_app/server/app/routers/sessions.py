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
    fetch_submissions_for_session,
    insert_game_round,
    insert_matching_result,
)
from ..matching_engine import run_market_matching
from ..service import get_tables
from ..schemas import MatchRunRequest, PlayerJoinRequest, RoundStartRequest, SessionConfigRequest, SessionCreateRequest
from ..session_service import create_session, get_session, join_session
from ..settings import FIXED_POLICY, GAME_SETTINGS

router = APIRouter(prefix="/api/sessions", tags=["sessions"])

# In-memory beta distribution config per session code.
# Resets on server restart (acceptable for a classroom game).
_session_beta: dict[str, tuple[float, float]] = {}

_DEFAULT_ALPHA = 3.0
_DEFAULT_BETA = 3.0


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


def _resolve_total_rounds(session_row: dict[str, Any]) -> int:
    raw = session_row.get("number_of_rounds", None)
    if raw is None:
        code = str(session_row.get("session_code", "")).strip().upper()
        if code:
            try:
                session = get_session(code)
                if session is not None:
                    raw = session.get("number_of_rounds", None)
            except Exception:
                raw = None

    try:
        return max(1, int(raw))
    except Exception:
        return 5


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
        avg_child_labor = sum(_safe_float(r.get("child_labor")) for r in selected_rows) / count
        avg_banned_chem = sum(_safe_float(r.get("banned_chem")) for r in selected_rows) / count

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
            "avg_improvement": avg_improvement,
            "avg_low_quality": avg_low_quality,
            "avg_child_labor": avg_child_labor,
            "avg_banned_chem": avg_banned_chem,
        }

    return profiles, sorted(set(excluded))


@router.post("")
def create_game_session(req: SessionCreateRequest) -> dict[str, Any]:
    try:
        return create_session(req.game_name, req.admin_name or "Admin", req.number_of_rounds)
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

    total_rounds = _resolve_total_rounds(session_row)

    duration_seconds = req.duration_seconds if req.duration_seconds and req.duration_seconds > 0 else None
    market_capacity = max(1, int(req.market_capacity or 1))

    latest_rows = _extract_rows(fetch_latest_round(session_token))
    next_round_no = int(latest_rows[0].get("round_no", 0) or 0) + 1 if latest_rows else 1

    if next_round_no > total_rounds:
        raise HTTPException(
            status_code=400,
            detail=f"Configured round limit reached ({total_rounds}). No more rounds can be started.",
        )

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
        "total_rounds": total_rounds,
        "remaining_rounds": max(0, total_rounds - next_round_no),
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

    total_rounds = _resolve_total_rounds(session_row)

    rows = _extract_rows(fetch_active_round(session_token))
    normalized = (code or "").strip().upper()
    beta_alpha, beta_beta = _session_beta.get(normalized, (_DEFAULT_ALPHA, _DEFAULT_BETA))

    if not rows:
        return {"round": None, "total_rounds": total_rounds, "beta_alpha": beta_alpha, "beta_beta": beta_beta}

    row = rows[0]
    return {
        "round": {
            "round_no": int(row.get("round_no", 0) or 0),
            "duration_seconds": row.get("duration_seconds"),
            "market_capacity": int(row.get("market_capacity", GAME_SETTINGS.default_market_capacity) or GAME_SETTINGS.default_market_capacity),
            "started_at": row.get("created_at"),
            "ends_at": row.get("ends_at"),
            "is_active": bool(row.get("is_active", False)),
        },
        "total_rounds": total_rounds,
        "beta_alpha": beta_alpha,
        "beta_beta": beta_beta,
    }


@router.patch("/{code}/config")
def update_session_config(code: str, req: SessionConfigRequest) -> dict[str, Any]:
    normalized = (code or "").strip().upper()
    _session_beta[normalized] = (float(req.beta_alpha), float(req.beta_beta))
    return {"ok": True, "beta_alpha": req.beta_alpha, "beta_beta": req.beta_beta}


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
        str(row["supplier_id"]): {k: row.get(k) for k in ("env_risk", "social_risk", "cost_score", "strategic", "improvement", "low_quality", "child_labor", "banned_chem")}
        for _, row in suppliers_df.iterrows()
    }

    team_profiles, profile_excluded = _build_team_product_profiles(eligible_rows, suppliers_by_id)
    if profile_excluded:
        excluded_infeasible_teams = sorted(set(excluded_infeasible_teams + profile_excluded))
    if not team_profiles:
        raise HTTPException(status_code=400, detail="No feasible team product found for matching")

    N = len(users_df)
    if N <= 0:
        raise HTTPException(status_code=400, detail="No users available in dataset for matching")

    # --- MNL demand model ---
    from ..beta_density import BetaDensity
    from ..mnl_market import BuyerProfile, run_mnl_market
    from ..customer_segment import CustomerSegment

    normalized_code = (code or "").strip().upper()
    beta_alpha, beta_beta = _session_beta.get(normalized_code, (_DEFAULT_ALPHA, _DEFAULT_BETA))
    bd = BetaDensity(alpha=max(0.01, beta_alpha), beta=max(0.01, beta_beta))
    users_sorted = users_df.sort_values("w_cost").reset_index(drop=True)
    segments = [
        CustomerSegment(
            segment_id=str(row["user_id"]),
            density=float(bd.density_at((i + 0.5) / N)),
            w_env=_safe_float(row.get("w_env")),
            w_social=_safe_float(row.get("w_social")),
            w_cost=_safe_float(row.get("w_cost"), 1.0),
            w_strategic=_safe_float(row.get("w_strategic")),
            w_improvement=_safe_float(row.get("w_improvement")),
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
            avg_strategic=_safe_float(team_profiles[tid]["avg_strategic"]),
            avg_improvement=_safe_float(team_profiles[tid]["avg_improvement"]),
            avg_low_quality=_safe_float(team_profiles[tid]["avg_low_quality"]),
        )
        for tid in team_ids_sorted
    ]

    avg_price = sum(p.price_per_user for p in profiles) / max(len(profiles), 1)
    delta = float(GAME_SETTINGS.cost_scale) / max(avg_price, 1.0)
    mnl_result = run_mnl_market(profiles, segments, delta=delta, u_outside=None)

    market_to_users: dict[str, Any] = {}
    market_loads: dict[str, Any] = {}
    team_round_financials: list[dict[str, Any]] = []
    round_profit_total = 0.0

    for tid in team_ids_sorted:
        profile = team_profiles[tid]
        br = mnl_result.buyer_results.get(tid)
        demand_share = br.total_demand if br else 0.0
        effective_users = round(demand_share * N, 3)
        price = _safe_float(profile["price_per_user"], GAME_SETTINGS.price_per_user)
        avg_cost = _safe_float(profile["avg_cost"])
        unit_margin = price - float(GAME_SETTINGS.cost_scale) * avg_cost
        realized_profit = effective_users * unit_margin
        realized_utility = round((br.realized_utility * N) if br else 0.0, 3)
        round_profit_total += realized_profit

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
            "price_per_user": price,
            "avg_cost_score": avg_cost,
            "unit_margin": round(unit_margin, 2),
        })

    result = {
        "meta": {
            "solver": "mnl_v1",
            "user_pool_count": N,
            "eligible_team_count": len(profiles),
            "matched_count": N,
            "submitted_team_count": len(by_team),
            "infeasible_excluded_count": len(excluded_infeasible_teams),
            "round_profit_total": round(round_profit_total, 2),
        },
        "market_to_users": market_to_users,
        "market_loads": market_loads,
        "excluded_infeasible_users": excluded_infeasible_teams,
        "excluded_infeasible_teams": excluded_infeasible_teams,
        "round_financials": {
            "formula": "realized_profit = effective_users(MNL) x unit_margin",
            "delta": round(delta, 4),
            "beta_alpha": beta_alpha,
            "beta_beta": beta_beta,
            "cost_scale": float(GAME_SETTINGS.cost_scale),
            "team_financials": team_round_financials,
            "round_profit_total": round(round_profit_total, 2),
        },
    }

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


@router.get("/{code}/rounds/history")
def get_round_history(code: str) -> dict[str, Any]:
    """Return per-round profit/utility for every team in this session."""
    session_row = _get_session_row_or_404(code)
    session_code = str(session_row.get("session_code", "")).strip().upper()

    rows = _extract_rows(fetch_submissions_for_session(session_code))
    if not rows:
        return {"teams": [], "rounds": []}

    # Latest submission per (team, round_no)
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        team = str(row.get("team") or "(anonymous)").strip()
        rno = int(row.get("round_no") or 0)
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
