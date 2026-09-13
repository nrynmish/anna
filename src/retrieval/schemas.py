from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedCase(BaseModel):
    case_id: str
    customer_text: str
    uber_response: str
    final_uber_response: str
    resolution_type: str
    intent: str
    intent_confidence: float = Field(ge=0.0, le=1.0)
    similarity: float = Field(ge=-1.0, le=1.0)
    created_at: str | None = None


class RetrievalResult(BaseModel):
    query: str
    results: list[RetrievedCase]
