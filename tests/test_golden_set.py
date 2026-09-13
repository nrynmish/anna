import json
from pathlib import Path

from src.evaluation.golden_set import (
    build_case_index,
    build_labels_by_case,
    compute_intent_metrics,
    compute_routing_metrics,
    map_golden_examples,
)


def test_exact_message_maps_to_case(tmp_path: Path):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        json.dumps(
            {
                "case_id": "uber_case_0001",
                "customer_messages": [
                    {"text": "hello"},
                    {"text": "second message"},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    index, case_texts, case_count = build_case_index(cases)

    assert index["hello"] == ["uber_case_0001"]
    assert case_texts["uber_case_0001"] == "hello second message"
    assert case_count == 1
    assert index["second message"] == ["uber_case_0001"]


def test_golden_message_inside_multiturn_case_maps():
    golden = [
        {
            "tweet": "second message",
            "intent": "account_access",
            "routable": "yes",
            "resolution": "fix account",
            "difficulty": "medium",
        }
    ]

    cases_index = {
        "second message": ["uber_case_0001"],
    }

    labels = {
        "uber_case_0001": {
            "case_id": "uber_case_0001",
            "intent": "account_access",
        }
    }

    records = map_golden_examples(
        golden,
        cases_index,
        labels,
    )

    assert records[0]["mapping_status"] == "unique"
    assert records[0]["case_id"] == "uber_case_0001"
    assert records[0]["predicted_intent"] == "account_access"


def test_unmatched_mapping():
    golden = [
        {
            "tweet": "missing",
            "intent": "other",
            "routable": "no",
            "resolution": "none",
            "difficulty": "easy",
        }
    ]

    records = map_golden_examples(
        golden,
        {},
        {},
    )

    assert records[0]["mapping_status"] == "unmatched"
    assert records[0]["case_id"] is None
    assert records[0]["predicted_intent"] is None


def test_ambiguous_mapping():
    golden = [
        {
            "tweet": "duplicate",
            "intent": "other",
            "routable": "yes",
            "resolution": "none",
            "difficulty": "hard",
        }
    ]

    records = map_golden_examples(
        golden,
        {
            "duplicate": [
                "uber_case_0001",
                "uber_case_0002",
            ]
        },
        {},
    )

    assert records[0]["mapping_status"] == "ambiguous"
    assert records[0]["case_id"] is None
    assert len(records[0]["mapping_candidates"]) == 2


def test_case_id_join_and_no_routing_prediction():
    golden = [
        {
            "tweet": "hello",
            "intent": "other",
            "routable": "yes",
            "resolution": "none",
            "difficulty": "easy",
        }
    ]

    records = map_golden_examples(
        golden,
        {"hello": ["uber_case_0001"]},
        {
            "uber_case_0001": {
                "case_id": "uber_case_0001",
                "intent": "other",
            }
        },
    )

    assert records[0]["predicted_intent"] == "other"
    assert records[0]["gold_should_auto_handle"] is True
    assert records[0]["predicted_should_auto_handle"] is None

    routing = compute_routing_metrics(records)
    assert routing["available"] is False


def test_intent_metrics_include_weighted_f1():
    records = [
        {
            "mapping_status": "unique",
            "prediction_status": "available",
            "gold_intent": "other",
            "predicted_intent": "other",
        },
        {
            "mapping_status": "unique",
            "prediction_status": "available",
            "gold_intent": "account_access",
            "predicted_intent": "other",
        },
    ]

    metrics = compute_intent_metrics(records)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["counts"]["predictions_available"] == 2

def test_short_case_text_does_not_create_false_containment(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text(
        '{"case_id":"uber_case_short","customer_messages":'
        '[{"text":"@Uber_Support no!"}]}\n'
    )

    index, case_texts, _ = build_case_index(cases_path)

    records = map_golden_examples(
        golden=[
            {
                "tweet": (
                    "Driver was unprofessional and customer was charged "
                    "a cancellation fee with no resolution."
                ),
                "intent": "driver_behavior",
                "routable": "yes",
                "resolution": "review driver",
                "difficulty": "medium",
            }
        ],
        cases_index=index,
        labels_by_case={},
        case_texts=case_texts,
    )

    assert records[0]["case_id"] != "uber_case_short"
    assert records[0]["mapping_method"] != "normalized_containment"


def test_token_containment_maps_real_long_message(tmp_path):
    cases_path = tmp_path / "cases.jsonl"

    cases_path.write_text(
        '{"case_id":"uber_case_real","customer_messages":'
        '[{"text":"I cannot access my account after changing my phone number"}]}'
        + "\n"
    )

    index, case_texts, _ = build_case_index(cases_path)

    records = map_golden_examples(
        golden=[
            {
                "tweet": (
                    "I cannot access my account after changing my phone number "
                    "and support has not replied."
                ),
                "intent": "account_access",
                "routable": "yes",
                "resolution": "restore access",
                "difficulty": "medium",
            }
        ],
        cases_index=index,
        labels_by_case={
            "uber_case_real": {"intent": "account_access"}
        },
        case_texts=case_texts,
    )

    assert records[0]["case_id"] == "uber_case_real"
    assert records[0]["mapping_method"] == "normalized_containment"
