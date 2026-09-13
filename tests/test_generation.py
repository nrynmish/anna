import pytest
from pydantic import ValidationError

from src.generation.prompts import build_messages, build_user_prompt
from src.generation.schemas import GenerationRequest, RetrievedCase


def make_request() -> GenerationRequest:
    return GenerationRequest(
        customer_message="My driver cancelled my ride.",
        intent="driver_behavior",
        intent_confidence=0.91,
        evidence_agreement=0.84,
        retrieved_cases=[
            RetrievedCase(
                case_id="uber_case_001",
                customer_text="My driver cancelled my trip.",
                uber_response="Please send us a DM so we can assist further.",
                resolution_type="redirected_to_dm",
                similarity=0.91,
                intent="driver_behavior",
            )
        ],
    )


def test_generation_request_validates():
    request = make_request()
    assert request.intent == "driver_behavior"
    assert request.retrieved_cases[0].similarity == pytest.approx(0.91)


def test_generation_request_rejects_invalid_similarity():
    with pytest.raises(ValidationError):
        RetrievedCase(
            case_id="x",
            customer_text="issue",
            uber_response="response",
            resolution_type="unclear",
            similarity=1.5,
        )


def test_prompt_contains_customer_and_evidence():
    prompt = build_user_prompt(make_request())

    assert "My driver cancelled my ride." in prompt
    assert "My driver cancelled my trip." in prompt
    assert "Please send us a DM" in prompt
    assert "driver_behavior" in prompt


def test_messages_have_system_and_user_roles():
    messages = build_messages(make_request())

    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Do not invent Uber policies" in messages[0]["content"]


def test_generation_response_schema():
    from src.generation.schemas import GenerationResponse

    result = GenerationResponse(
        reply="Please send us a DM so we can assist.",
        confidence=0.88,
        grounded=True,
        escalate=False,
        reason="Strong historical precedent with consistent resolution.",
    )

    assert result.grounded is True
    assert result.escalate is False
