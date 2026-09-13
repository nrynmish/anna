from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

from .schemas import RetrievedCase


@dataclass(frozen=True)
class EvidenceAssessment:
    """Explainable assessment of historical evidence strength."""

    score: float
    intent_agreement: float
    resolution_agreement: float
    similarity_strength: float
    strong_precedent_ratio: float
    strong_precedent_count: int
    usable_case_count: int


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def assess_evidence(
    cases: list[RetrievedCase],
    *,
    intent: str,
    similarity_threshold: float = 0.75,
) -> EvidenceAssessment:
    """
    Produce an explainable evidence-strength score.

    Historical intent labels are weak labels, so intent agreement is only one
    component. Strong semantic matches and convergence on a resolution carry
    substantial weight.
    """
    if not cases:
        return EvidenceAssessment(
            score=0.0,
            intent_agreement=0.0,
            resolution_agreement=0.0,
            similarity_strength=0.0,
            strong_precedent_ratio=0.0,
            strong_precedent_count=0,
            usable_case_count=0,
        )

    similarities = [_clip(case.similarity) for case in cases]

    # Strong semantic precedents are cases above the configured similarity
    # threshold. This avoids letting weak tail results dominate the score.
    strong_cases = [
        case
        for case in cases
        if _clip(case.similarity) >= similarity_threshold
    ]

    usable_cases = [
        case
        for case in cases
        if case.intent not in {"other", "unclear", ""}
    ]

    # Intent agreement is deliberately not the primary signal because the
    # historical labels themselves are heuristic/weak labels.
    intent_agreement = (
        sum(case.intent == intent for case in usable_cases)
        / len(usable_cases)
        if usable_cases
        else 0.0
    )

    # Among strong precedents, measure whether the historical support team
    # converged on the same resolution type.
    strong_resolutions = [
        case.resolution_type
        for case in strong_cases
        if case.resolution_type
    ]

    if strong_resolutions:
        counts = Counter(strong_resolutions)
        resolution_agreement = max(counts.values()) / len(strong_resolutions)
    else:
        resolution_agreement = 0.0

    # Mean similarity of the strongest retrieved evidence.
    strong_similarities = [_clip(case.similarity) for case in strong_cases]
    similarity_strength = (
        sum(strong_similarities) / len(strong_similarities)
        if strong_similarities
        else sum(similarities) / len(similarities)
    )

    strong_precedent_count = len(strong_cases)
    strong_precedent_ratio = strong_precedent_count / len(cases)

    # Weighted toward the evidence that is least dependent on noisy historical
    # labels: semantic similarity and resolution convergence.
    score = (
        0.45 * similarity_strength
        + 0.30 * resolution_agreement
        + 0.15 * strong_precedent_ratio
        + 0.10 * intent_agreement
    )

    return EvidenceAssessment(
        score=_clip(score),
        intent_agreement=_clip(intent_agreement),
        resolution_agreement=_clip(resolution_agreement),
        similarity_strength=_clip(similarity_strength),
        strong_precedent_ratio=_clip(strong_precedent_ratio),
        strong_precedent_count=strong_precedent_count,
        usable_case_count=len(usable_cases),
    )


def sanitize_evidence_text(text: str) -> str:
    """
    Remove case-specific URLs/mentions before evidence is passed to generation.

    Historical responses are evidence of resolution patterns, not text to copy
    verbatim into a new customer reply.
    """
    text = re.sub(r"https?://\S+|www\.\S+", "[link removed]", text)
    text = re.sub(r"@\w+", "[account removed]", text)
    return text.strip()
