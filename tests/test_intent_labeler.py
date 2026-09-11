import copy

from src.intent import labeler, taxonomy


def test_normalize_customer_text():
    text = "@Uber_Support I'm stuck! https://t.co/abc123 Please help."
    norm = labeler.normalize_customer_text(text)
    assert "@" not in norm
    assert "http" not in norm
    assert "stuck" in norm


def test_deterministic_labeling_and_confidence():
    case = {"case_id": "c1", "customer_id": "u1", "customer_messages": ["I left my bag in an Uber."]}
    a = labeler.label_case(case)
    b = labeler.label_case(copy.deepcopy(case))
    assert a == b
    assert 0.0 <= a["labeling_confidence"] <= 1.0
    # taxonomy version attached
    assert a["taxonomy_version"] == taxonomy.TAXONOMY_VERSION


def test_matching_example_lost_and_found():
    case = {"case_id": "lost1", "customer_id": "u2", "customer_messages": ["I just left my bag in an Uber. How do you contact the driver?"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "lost_and_found"


def test_ambiguous_multi_intent_becomes_unclear():
    # contains keywords for lost_and_found and payments_charges
    text = "I left my bag and I was also charged a cancellation fee for that trip"
    case = {"case_id": "amb1", "customer_id": "u3", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "unclear"


def test_unsupported_but_understandable_becomes_other():
    text = "When will UberKITTENS be back in the city?"
    case = {"case_id": "other1", "customer_id": "u4", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] in {"other", "unclear"}


def test_valid_taxonomy_ids_only():
    # run labeler on several cases and ensure intents are valid taxonomy ids or special buckets
    cases = [
        {"case_id": "c2", "customer_id": "u5", "customer_messages": ["I can't login to my account"]},
        {"case_id": "c3", "customer_id": "u6", "customer_messages": ["Why am I charged twice?"]},
    ]
    labels = labeler.label_cases(cases)
    valid_ids = {i["id"] for i in taxonomy.TAXONOMY} | {"other", "unclear"}
    for l in labels:
        assert l["intent"] in valid_ids


def test_explain_label_contains_matches():
    case = {"case_id": "c4", "customer_id": "u7", "customer_messages": ["My app is not working and shows an error"]}
    expl = labeler.explain_label(case)
    assert isinstance(expl, dict)
    assert "matches" in expl


def test_account_disabled_is_account_access():
    text = "I used my acct 4 hours ago, now it's disabled. I submitted the form, confirmed my email. How long does it take to get a reply?"
    case = {"case_id": "acc1", "customer_id": "u8", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "account_access"


def test_multiple_drivers_cancelled_maps_to_trip_or_driver():
    text = "three cars have canceled on me and now I have to incur the cost"
    case = {"case_id": "drv1", "customer_id": "u9", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] in {"rides_and_trips", "driver_behavior"}


def test_app_alone_not_app_technical():
    case = {"case_id": "app1", "customer_id": "u10", "customer_messages": ["app"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] != "app_technical"


def test_generic_ride_not_decisive():
    case = {"case_id": "ride1", "customer_id": "u11", "customer_messages": ["ride"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] != "rides_and_trips"


def test_payment_beats_generic_ride_context():
    case = {"case_id": "pay1", "customer_id": "u12", "customer_messages": ["My ride was fine but I was charged a cancellation fee"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "payments_charges"


def test_driver_behavior_beats_generic_payment_when_specific():
    case = {"case_id": "drv2", "customer_id": "u13", "customer_messages": ["Driver was rude and I got charged"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "driver_behavior"


def test_ambiguous_specific_intents_become_unclear():
    case = {"case_id": "amb2", "customer_id": "u14", "customer_messages": ["I left my bag and I was attacked by someone"]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "unclear"


def test_deterministic_labeling_remains():
    case = {"case_id": "c1", "customer_id": "u1", "customer_messages": ["I left my bag in an Uber."]}
    a = labeler.label_case(case)
    b = labeler.label_case(case)
    assert a == b
    assert 0.0 <= a["labeling_confidence"] <= 1.0


def test_charge_amount_prefers_payments():
    text = "Charged me $140 for a $90 ride. This was after quoting me $65"
    case = {"case_id": "pay_amt", "customer_id": "u20", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "payments_charges"


def test_call_mom_lost_and_found():
    text = "I need someone to call my mom back she left her bag in an Uber"
    case = {"case_id": "lf1", "customer_id": "u21", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "lost_and_found"


def test_driver_unprofessional_prefers_driver_over_fee():
    text = "Driver unprofessional, cancelled my trip and they charged me a cancellation fee"
    case = {"case_id": "db1", "customer_id": "u22", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "driver_behavior"


def test_refund_request_prefers_refunds():
    text = "I was charged twice, please refund me the extra charge"
    case = {"case_id": "ref1", "customer_id": "u23", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "refunds_adjustments"


def test_driver_cancel_prefers_driver_behavior_over_trip():
    text = "Driver canceled on me for my scheduled airport pickup"
    case = {"case_id": "drv3", "customer_id": "u24", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "driver_behavior"


def test_account_access_wins_over_app_technical():
    text = "I can't sign in to my account, the app shows an error"
    case = {"case_id": "acc2", "customer_id": "u25", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "account_access"


def test_lost_and_found_wins_over_ride_context():
    text = "I left my wallet during the trip to the airport"
    case = {"case_id": "lf2", "customer_id": "u26", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "lost_and_found"


def test_strong_payments_and_strong_lost_become_unclear():
    text = "I left my bag and I was charged $50 for it"
    case = {"case_id": "comb1", "customer_id": "u30", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "unclear"


def test_explicit_refund_phrase_prefers_refund():
    text = "I was charged twice, please refund the extra charge"
    case = {"case_id": "ref2", "customer_id": "u31", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "refunds_adjustments"


def test_payments_and_lost_with_cancellation_fee_are_unclear():
    text = "I left my bag and I was also charged a cancellation fee for that trip"
    case = {"case_id": "comb2", "customer_id": "u32", "customer_messages": [text]}
    lbl = labeler.label_case(case)
    assert lbl["intent"] == "unclear"
