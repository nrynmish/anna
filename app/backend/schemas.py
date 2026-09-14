from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)


class AnalyzeResponse(BaseModel):
    customer_message: str
    intent: str
    intent_confidence: float
    retrieved_cases: list[dict[str, Any]]
    evidence: dict[str, Any]
    generation: dict[str, Any]
    decision: dict[str, Any]
