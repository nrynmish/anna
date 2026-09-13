from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    AUTO_HANDLE = "auto_handle"
    ESCALATE = "escalate"


class DecisionInput(BaseModel):
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    top_similarity: float = Field(ge=0.0, le=1.0)
    evidence_agreement: float = Field(ge=0.0, le=1.0)
    resolution_agreement: float = Field(ge=0.0, le=1.0)
    retrieved_case_count: int = Field(ge=0)
    generated_confidence: float = Field(ge=0.0, le=1.0)
    generated_grounded: bool
    generated_escalate: bool
    customer_message: str


class DecisionResult(BaseModel):
    decision: Decision
    auto_handle: bool
    reason: str
    risk_flags: list[str]
    factors: dict[str, float | bool | int | str]
