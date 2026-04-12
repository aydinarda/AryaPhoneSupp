from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from starlette.types import Scope, Receive, Send

from .db import fetch_all_submissions, insert_submission
from .matching_engine import run_market_matching
from .routers.sessions import router as sessions_router
from .schemas import BenchmarkRequest, EvalRequest, MatchingRequest, SubmitRequest
from .service import evaluate_manual, get_both_benchmarks, get_game_constants, get_supplier_overview, run_benchmark

app = FastAPI(title="Arya Phone Game API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return get_game_constants()


@app.get("/api/benchmarks/both")
def benchmarks_both() -> dict[str, Any]:
    return get_both_benchmarks()


@app.get("/api/suppliers")
def suppliers() -> list[dict[str, Any]]:
    try:
        return get_supplier_overview()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/manual-eval")
def manual_eval(req: EvalRequest) -> dict[str, Any]:
    try:
        return evaluate_manual(
            req.objective, req.picks,
            price_per_user=req.price_per_user,
            beta_alpha=req.beta_alpha,
            beta_beta=req.beta_beta,
            delta=req.delta,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/benchmark")
def benchmark(req: BenchmarkRequest) -> dict[str, Any]:
    try:
        return run_benchmark(req.objective)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/submit")
def submit(req: SubmitRequest) -> dict[str, Any]:
    try:
        result = evaluate_manual(
            req.objective, req.picks,
            price_per_user=req.price_per_user,
            delta=req.delta,
        )
        metrics = result["metrics"]
        feasible = result.get("feasible", False)

        payload = {
            "team": (req.team or "(anonymous)").strip() or "(anonymous)",
            "player_name": (req.player_name or "(anonymous)").strip() or "(anonymous)",
            "selected_suppliers": ",".join([str(x) for x in req.picks]),
            "objective": req.objective,
            "comment": req.comment,
            "session_code": ((req.session_code or "").strip().upper() or None),
            "round_no": int(req.round_no) if req.round_no is not None else None,
            "price": float(req.price_per_user) if req.price_per_user is not None else None,
            "profit": float(metrics.get("profit_total", 0.0)),
            "utility": float(metrics.get("utility_total", 0.0)),
            "env_avg": float(metrics.get("avg_env", 0.0)),
            "social_avg": float(metrics.get("avg_social", 0.0)),
            "cost_avg": float(metrics.get("avg_cost", 0.0)),
            "strategic_avg": float(metrics.get("avg_strategic", 0.0)),
        }

        insert_submission(payload)
        return {"ok": True, "manual": result}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/leaderboard")
def leaderboard(limit: int = 5000, sort_by: str = "profit", feasible_only: bool = False) -> dict[str, Any]:
    try:
        res = fetch_all_submissions(limit=limit)
        rows = getattr(res, "data", None)
        if rows is None and isinstance(res, dict):
            rows = res.get("data", [])
        rows = rows or []

        if not rows:
            return {"rows": [], "latest": [], "top_profit": [], "top_utility": []}

        df_all = pd.DataFrame(rows)
        if "created_at" in df_all.columns:
            df_all["created_at"] = pd.to_datetime(df_all["created_at"], errors="coerce")
        else:
            df_all["created_at"] = pd.NaT

        if "team" not in df_all.columns:
            raise RuntimeError("Supabase table is missing the 'team' column.")

        # Feasibility filter
        if feasible_only and "feasible" in df_all.columns:
            df_all = df_all[df_all["feasible"] == True].copy()

        # Latest submission per team
        latest = (
            df_all.sort_values("created_at", ascending=True)
            .groupby("team", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )

        # Top 10 by profit
        top_profit = latest.nlargest(10, "profit") if "profit" in latest.columns else pd.DataFrame()

        # Top 10 by utility
        top_utility = latest.nlargest(10, "utility") if "utility" in latest.columns else pd.DataFrame()

        return {
            "rows": df_all.sort_values("created_at", ascending=False).to_dict(orient="records"),
            "latest": latest.sort_values("created_at", ascending=False).to_dict(orient="records"),
            "top_profit": top_profit.to_dict(orient="records"),
            "top_utility": top_utility.to_dict(orient="records"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/matching")
def matching(req: MatchingRequest) -> dict[str, Any]:
    try:
        return run_market_matching(
            users=[u.model_dump() for u in req.users],
            market_options=[m.model_dump() for m in req.market_options],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


CLIENT_DIR = Path(__file__).resolve().parents[2] / "client"

class NoCacheJSFiles(StaticFiles):
    """StaticFiles that disables browser caching for JS modules."""
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope.get("path", "").endswith(".js"):
            async def send_no_cache(message: Any) -> None:
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))
                    headers[b"cache-control"] = b"no-store, no-cache, must-revalidate"
                    message = {**message, "headers": list(headers.items())}
                await send(message)
            await super().__call__(scope, receive, send_no_cache)
        else:
            await super().__call__(scope, receive, send)


if CLIENT_DIR.exists():
    app.mount("/assets", NoCacheJSFiles(directory=CLIENT_DIR), name="assets")

app.include_router(sessions_router)


@app.get("/")
def home() -> FileResponse:
    index_file = CLIENT_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Client index.html not found")
    return FileResponse(index_file)
