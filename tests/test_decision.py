from src.decision import Decision, DecisionInput, decide, detect_risk_flags


def make_input(**overrides):
    values = {
        "intent": "driver_behavior",
        "intent_confidence": 0.91,
        "top_similarity": 0.92,
        "evidence_agreement": 0.88,
        "resolution_agreement": 0.88,
        "retrieved_case_count": 2,
        "generated_confidence": 0.88,
        "generated_grounded": True,
        "generated_escalate": False,
        "customer_message": "My driver was rude and cancelled my ride.",
    }
    values.update(overrides)
    return DecisionInput(**values)


def test_strong_evidence_auto_handles():
    result = decide(make_input())

    assert result.decision == Decision.AUTO_HANDLE
    assert result.auto_handle is True
    assert result.risk_flags == []


def test_low_evidence_agreement_escalates():
    result = decide(make_input(evidence_agreement=0.25))

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "consistent evidence" in result.reason


def test_unauthorized_charge_escalates_even_when_model_says_auto_handle():
    result = decide(
        make_input(
            intent="payments_charges",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            customer_message=(
                "Uber has charged my card for something I never authorized "
                "and I believe my account may be compromised."
            ),
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "unauthorized_charge" in result.risk_flags
    assert "account_compromise" in result.risk_flags


def test_no_retrieval_escalates():
    result = decide(make_input(retrieved_case_count=0))

    assert result.decision == Decision.ESCALATE


def test_low_resolution_agreement_does_not_block_strong_intent():
    result = decide(
        make_input(
            intent="account_access",
            intent_confidence=0.90,
            top_similarity=0.90,
            evidence_agreement=0.78,
            resolution_agreement=0.40,
        )
    )

    assert result.decision == Decision.AUTO_HANDLE
    assert result.auto_handle is True


def test_low_intent_confidence_escalates():
    result = decide(
        make_input(
            intent="driver_behavior",
            intent_confidence=0.59,
            top_similarity=0.82,
            evidence_agreement=0.72,
            resolution_agreement=0.60,
        )
    )

    assert result.decision == Decision.ESCALATE
    assert "Intent confidence" in result.reason


def test_moderate_other_with_strong_evidence_still_escalates():
    result = decide(
        make_input(
            intent="other",
            intent_confidence=0.50,
            top_similarity=0.86,
            evidence_agreement=0.75,
            resolution_agreement=1.00,
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "Intent confidence" in result.reason


def test_unclear_intent_still_escalates():
    result = decide(
        make_input(
            intent="unclear",
            intent_confidence=0.0,
            top_similarity=0.93,
            evidence_agreement=0.75,
            resolution_agreement=1.00,
        )
    )

    assert result.decision == Decision.ESCALATE


def test_low_similarity_escalates():
    result = decide(make_input(top_similarity=0.42))

    assert result.decision == Decision.ESCALATE


def test_low_intent_threshold_allows_strong_supported_request():
    result = decide(
        make_input(
            intent="driver_behavior",
            intent_confidence=0.60,
            top_similarity=0.90,
            evidence_agreement=0.80,
            resolution_agreement=0.40,
        )
    )

    assert result.decision == Decision.AUTO_HANDLE
    assert result.auto_handle is True


def test_financial_issue_with_weak_resolution_evidence_escalates():
    result = decide(
        make_input(
            intent="payments_charges",
            intent_confidence=0.80,
            top_similarity=0.86,
            evidence_agreement=0.75,
            resolution_agreement=0.40,
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "resolution evidence" in result.reason


def test_ungrounded_generation_escalates():
    result = decide(make_input(generated_grounded=False))

    assert result.decision == Decision.ESCALATE


def test_generation_escalation_request_is_respected():
    result = decide(make_input(generated_escalate=True))

    assert result.decision == Decision.ESCALATE


def test_risk_detection():
    flags = detect_risk_flags(
        "Someone hacked my account and stole my money. I am contacting the police."
    )

    assert "account_compromise" in flags
    assert "fraud" in flags
    assert "legal_threat" in flags


def test_racist_driver_attack_escalates():
    result = decide(
        make_input(
            intent="driver_behavior",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            resolution_agreement=0.95,
            customer_message="I have just faced the second racist attack from a driver.",
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "safety_incident" in result.risk_flags


def test_hacked_disabled_account_escalates():
    result = decide(
        make_input(
            intent="account_access",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            resolution_agreement=0.95,
            customer_message="My hacked/disabled account is preventing me from taking rides.",
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "account_compromise" in result.risk_flags


def test_account_specific_driver_contact_escalates():
    result = decide(
        make_input(
            intent="lost_and_found",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            resolution_agreement=0.95,
            customer_message="I lost my mobile in the cab and need the driver's contact number.",
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False


def test_account_specific_refund_escalates():
    result = decide(
        make_input(
            intent="refunds_adjustments",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            resolution_agreement=0.95,
            customer_message="I would like to be refunded for that cancellation.",
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "account-specific" in result.reason


def test_account_specific_lost_item_request_escalates():
    result = decide(
        make_input(
            intent="lost_and_found",
            intent_confidence=0.95,
            top_similarity=0.95,
            evidence_agreement=0.95,
            resolution_agreement=0.95,
            customer_message="I lost my mobile in the cab. Please contact my driver.",
        )
    )

    assert result.decision == Decision.ESCALATE
    assert result.auto_handle is False
    assert "account-specific" in result.reason
