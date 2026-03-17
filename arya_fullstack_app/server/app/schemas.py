from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Objective = Literal["max_profit", "max_utility"]


class EvalRequest(BaseModel):
    objective: Objective
    picks: list[str] = Field(default_factory=list)
    price_per_user: Optional[float] = None


class BenchmarkRequest(BaseModel):
    objective: Objective


class SessionCreateRequest(BaseModel):
    game_name: str
    admin_name: Optional[str] = None
    number_of_rounds: int = Field(default=5, ge=1, le=100)


class PlayerJoinRequest(BaseModel):
    team_name: str


class RoundStartRequest(BaseModel):
    duration_seconds: Optional[int] = None
    market_capacity: int = 8


class MatchRunRequest(BaseModel):
    round_no: Optional[int] = None


class SubmitRequest(BaseModel):
    team: str = "(anonymous)"
    objective: Objective
    picks: list[str] = Field(default_factory=list)
    price_per_user: Optional[float] = None
    comment: Optional[str] = None
    player_name: Optional[str] = None
    session_code: Optional[str] = None
    round_no: Optional[int] = None


class MatchingUserRequest(BaseModel):
    user_id: str
    choices: list[str] = Field(default_factory=list)
    utilities: dict[str, float] = Field(default_factory=dict)
    price_sensitivity: Optional[float] = None
    sustainability_sensitivity: Optional[float] = None


class MatchingMarketOptionRequest(BaseModel):
    option_id: str
    capacity: int = 0
    priority: list[str] = Field(default_factory=list)
    request_time: Optional[str] = None
    price: Optional[float] = None
    sustainability: Optional[float] = None


class MatchingRequest(BaseModel):
    users: list[MatchingUserRequest] = Field(default_factory=list)
    market_options: list[MatchingMarketOptionRequest] = Field(default_factory=list)
