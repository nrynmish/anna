from src.retrieval.evidence import assess_evidence, sanitize_evidence_text
from src.retrieval.schemas import RetrievedCase


def make_case(
    case_id: str,
    similarity: float,
    intent: str,
    resolution_type: str = "redirected_to_dm",
) -> RetrievedCase:
    return RetrievedCase(
        case_id=case_id,
        customer_text="driver was rude and cancelled",
        uber_response="Send us a note here.",
        final_uber_response="Send us a note here.",
        resolution_type=resolution_type,
        intent=intent,
        intent_confidence=0.8,
        similarity=similarity,
    )


def test_strong_semantic_cases_can_produce_strong_evidence_despite_noisy_labels():
    cases = [
        make_case("1", 0.836, "unclear"),
        make_case("2", 0.831, "driver_behavior"),
        make_case("3", 0.825, "unclear"),
        make_case("4", 0.819, "refunds_adjustments"),
        make_case("5", 0.815, "driver_behavior"),
    ]

    assessment = assess_evidence(cases, intent="driver_behavior")

    assert assessment.strong_precedent_count == 5
    assert assessment.similarity_strength > 0.80
    assert assessment.resolution_agreement == 1.0
    assert assessment.score > 0.70


def test_empty_evidence_scores_zero():
    assessment = assess_evidence([], intent="driver_behavior")

    assert assessment.score == 0.0
    assert assessment.strong_precedent_count == 0


def test_sanitization_removes_case_specific_identifiers():
    text = (
        "Send us a note here: https://t.co/example "
        "@Uber_Support and contact @12345."
    )

    sanitized = sanitize_evidence_text(text)

    assert "https://" not in sanitized
    assert "@Uber_Support" not in sanitized
    assert "@12345" not in sanitized
    assert "[link removed]" in sanitized
    assert "[account removed]" in sanitized
