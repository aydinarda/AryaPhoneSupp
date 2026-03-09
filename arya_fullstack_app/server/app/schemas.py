from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


Objective = Literal["max_profit", "max_utility"]


class EvalRequest(BaseModel):
    objective: Objective
    picks: list[str] = Field(default_factory=list)


class BenchmarkRequest(BaseModel):
    objective: Objective


class SubmitRequest(BaseModel):
    team: str = "(anonymous)"
    objective: Objective
    picks: list[str] = Field(default_factory=list)
    comment: Optional[str] = None
    player_name: Optional[str] = None
