from __future__ import annotations

import re

from .schemas import Decision, DecisionInput, DecisionResult


# These are configuration values rather than claimed optimal thresholds.
# They will be tuned against the golden evaluation.
DEFAULT_MIN_INTENT_CONFIDENCE = 0.60
DEFAULT_MIN_SIMILARITY = 0.80
DEFAULT_MIN_EVIDENCE_AGREEMENT = 0.70
DEFAULT_MIN_GENERATED_CONFIDENCE = 0.70


HIGH_RISK_PATTERNS: dict[str, tuple[str, ...]] = {
    "account_compromise": (
        "hacked my account",
        "account was hacked",
        "someone accessed my account",
        "someone used my account",
        "account compromised",
        "account may be compromised",
    ),
    "unauthorized_charge": (
        "unauthorized charge",
        "unauthorised charge",
        "charge i never authorized",
        "charge i never authorised",
        "charged my card for something i never",
        "card was charged without",
    ),
    "fraud": (
        "fraud",
        "fraudulent",
        "stolen from me",
        "stole my money",
        "scam",
    ),
    "legal_threat": (
        "sue you",
        "lawsuit",
        "legal action",
        "lawyer",
        "police",
        "court",
    ),
    "safety_incident": (
        "assault",
        "attacked",
        "injured",
        "accident",
        "pain and suffering",
        "unsafe",
        "threatened",
    ),
}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_risk_flags(customer_message: str) -> list[str]:
    text = _normalize(customer_message)
    flags: list[str] = []

    for flag, patterns in HIGH_RISK_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            flags.append(flag)

    return flags


def decide(
    decision_input: DecisionInput,
    *,
    min_intent_confidence: float = DEFAULT_MIN_INTENT_CONFIDENCE,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_evidence_agreement: float = DEFAULT_MIN_EVIDENCE_AGREEMENT,
    min_generated_confidence: float = DEFAULT_MIN_GENERATED_CONFIDENCE,
) -> DecisionResult:
    risk_flags = detect_risk_flags(decision_input.customer_message)

    factors: dict[str, float | bool | int | str] = {
        "intent": decision_input.intent,
        "intent_confidence": decision_input.intent_confidence,
        "top_similarity": decision_input.top_similarity,
        "evidence_agreement": decision_input.evidence_agreement,
        "resolution_agreement": decision_input.resolution_agreement,
        "retrieved_case_count": decision_input.retrieved_case_count,
        "generated_confidence": decision_input.generated_confidence,
        "generated_grounded": decision_input.generated_grounded,
        "generated_escalate": decision_input.generated_escalate,
        "high_risk": bool(risk_flags),
    }

    if risk_flags:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason=f"High-risk issue detected: {', '.join(risk_flags)}.",
            risk_flags=risk_flags,
            factors=factors,
        )

    if decision_input.retrieved_case_count == 0:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason="No historical support evidence was retrieved.",
            risk_flags=[],
            factors=factors,
        )

    if decision_input.intent_confidence < min_intent_confidence:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason="Intent confidence is below the auto-handle threshold.",
            risk_flags=[],
            factors=factors,
        )

    if decision_input.top_similarity < min_similarity:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason=(
                "Historical evidence similarity is below the "
                "auto-handle threshold."
            ),
            risk_flags=[],
            factors=factors,
        )

    if decision_input.evidence_agreement < min_evidence_agreement:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason=(
                "Retrieved historical cases do not provide sufficiently "
                "consistent evidence."
            ),
            risk_flags=[],
            factors=factors,
        )

    if decision_input.intent in {"payments_charges", "refunds_adjustments"} and decision_input.resolution_agreement < min_evidence_agreement:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason=(
                "Financial or refund issue lacks sufficiently consistent "
                "historical resolution evidence."
            ),
            risk_flags=[],
            factors=factors,
        )

    if not decision_input.generated_grounded:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason=(
                "The generated response was not judged grounded in the "
                "supplied evidence."
            ),
            risk_flags=[],
            factors=factors,
        )

    if decision_input.generated_confidence < min_generated_confidence:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason="Generation confidence is below the auto-handle threshold.",
            risk_flags=[],
            factors=factors,
        )

    if decision_input.generated_escalate:
        return DecisionResult(
            decision=Decision.ESCALATE,
            auto_handle=False,
            reason="The generation layer requested escalation.",
            risk_flags=[],
            factors=factors,
        )

    return DecisionResult(
        decision=Decision.AUTO_HANDLE,
        auto_handle=True,
        reason=(
            "Intent, historical evidence, grounding, and risk checks "
            "all passed."
        ),
        risk_flags=[],
        factors=factors,
    )

