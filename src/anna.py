from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.intent.labeler import label_case
from src.retrieval import BGERetriever, assess_evidence
from src.generation import GenerationRequest, QwenClient
from src.decision import DecisionInput, DecisionResult, decide


@dataclass
class ANNAResult:
    customer_message: str
    intent: str
    intent_confidence: float

    retrieved_cases: list[dict[str, Any]]

    evidence_score: float
    intent_agreement: float
    resolution_agreement: float
    similarity_strength: float
    strong_precedent_ratio: float
    strong_precedent_count: int
    usable_case_count: int

    reply: str
    generation_confidence: float
    grounded: bool
    generation_escalate: bool
    generation_reason: str

    decision: str
    auto_handle: bool
    decision_reason: str
    risk_flags: list[str]
    decision_factors: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "customer_message": self.customer_message,
            "intent": self.intent,
            "intent_confidence": self.intent_confidence,
            "retrieved_cases": self.retrieved_cases,
            "evidence": {
                "score": self.evidence_score,
                "intent_agreement": self.intent_agreement,
                "resolution_agreement": self.resolution_agreement,
                "similarity_strength": self.similarity_strength,
                "strong_precedent_ratio": self.strong_precedent_ratio,
                "strong_precedent_count": self.strong_precedent_count,
                "usable_case_count": self.usable_case_count,
            },
            "generation": {
                "reply": self.reply,
                "confidence": self.generation_confidence,
                "grounded": self.grounded,
                "escalate": self.generation_escalate,
                "reason": self.generation_reason,
            },
            "decision": {
                "decision": self.decision,
                "auto_handle": self.auto_handle,
                "reason": self.decision_reason,
                "risk_flags": self.risk_flags,
                "factors": self.decision_factors,
            },
        }


class ANNA:
    """
    End-to-end ANNA support pipeline.

    customer message
        -> intent classification
        -> BGE historical retrieval
        -> evidence assessment
        -> Qwen grounded draft
        -> deterministic safety/automation decision
    """

    def __init__(
        self,
        *,
        retriever: BGERetriever | None = None,
        generator: QwenClient | None = None,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        self.retriever = retriever or BGERetriever()
        self.generator = generator or QwenClient()
        self.top_k = top_k

    def classify(self, customer_message: str) -> dict[str, Any]:
        if not customer_message.strip():
            raise ValueError("customer_message must not be empty")

        case = {
            "case_id": "live_message",
            "customer_id": "live_customer",
            "customer_messages": [{"text": customer_message}],
        }

        return label_case(case)

    def analyze(self, customer_message: str) -> ANNAResult:
        if not customer_message.strip():
            raise ValueError("customer_message must not be empty")

        # ------------------------------------------------------------
        # 1. Intent
        # ------------------------------------------------------------
        label = self.classify(customer_message)

        intent = str(label["intent"])
        intent_confidence = float(label["labeling_confidence"])

        # ------------------------------------------------------------
        # 2. Historical retrieval
        # ------------------------------------------------------------
        retrieval = self.retriever.retrieve(
            customer_message,
            top_k=self.top_k,
        )

        # ------------------------------------------------------------
        # 3. Evidence assessment
        # ------------------------------------------------------------
        evidence = assess_evidence(
            retrieval.results,
            intent=intent,
        )

        # ------------------------------------------------------------
        # 4. Grounded response generation
        # ------------------------------------------------------------
        generation_cases = [
            {
                "case_id": case.case_id,
                "customer_text": case.customer_text,
                "uber_response": case.final_uber_response or case.uber_response,
                "resolution_type": case.resolution_type,
                "similarity": case.similarity,
                "intent": case.intent,
            }
            for case in retrieval.results
        ]

        generation_request = GenerationRequest(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            retrieved_cases=generation_cases,
            evidence_agreement=evidence.score,
        )

        generation = self.generator.generate(generation_request)

        # ------------------------------------------------------------
        # 5. Deterministic automation decision
        # ------------------------------------------------------------
        top_similarity = (
            retrieval.results[0].similarity
            if retrieval.results
            else 0.0
        )

        decision_input = DecisionInput(
            intent=intent,
            intent_confidence=intent_confidence,
            top_similarity=top_similarity,
            evidence_agreement=evidence.score,
            resolution_agreement=evidence.resolution_agreement,
            retrieved_case_count=len(retrieval.results),
            generated_confidence=generation.confidence,
            generated_grounded=generation.grounded,
            generated_escalate=generation.escalate,
            customer_message=customer_message,
        )

        decision: DecisionResult = decide(decision_input)

        # ------------------------------------------------------------
        # 6. Structured result
        # ------------------------------------------------------------
        retrieved_cases = [
            {
                "case_id": case.case_id,
                "customer_text": case.customer_text,
                "historical_response": case.final_uber_response
                or case.uber_response,
                "resolution_type": case.resolution_type,
                "intent": case.intent,
                "similarity": case.similarity,
                "created_at": case.created_at,
            }
            for case in retrieval.results
        ]

        return ANNAResult(
            customer_message=customer_message,
            intent=intent,
            intent_confidence=intent_confidence,
            retrieved_cases=retrieved_cases,
            evidence_score=evidence.score,
            intent_agreement=evidence.intent_agreement,
            resolution_agreement=evidence.resolution_agreement,
            similarity_strength=evidence.similarity_strength,
            strong_precedent_ratio=evidence.strong_precedent_ratio,
            strong_precedent_count=evidence.strong_precedent_count,
            usable_case_count=evidence.usable_case_count,
            reply=generation.reply,
            generation_confidence=generation.confidence,
            grounded=generation.grounded,
            generation_escalate=generation.escalate,
            generation_reason=generation.reason,
            decision=decision.decision.value,
            auto_handle=decision.auto_handle,
            decision_reason=decision.reason,
            risk_flags=decision.risk_flags,
            decision_factors=decision.factors,
        )


__all__ = ["ANNA", "ANNAResult"]
