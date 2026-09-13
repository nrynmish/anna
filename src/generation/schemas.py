from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedCase(BaseModel):
    case_id: str
    customer_text: str
    uber_response: str
    resolution_type: str
    similarity: float = Field(ge=0.0, le=1.0)
    intent: str | None = None


class GenerationRequest(BaseModel):
    customer_message: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    retrieved_cases: list[RetrievedCase]
    evidence_agreement: float = Field(ge=0.0, le=1.0)


class GenerationResponse(BaseModel):
    reply: str
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool
    escalate: bool
    reason: str
