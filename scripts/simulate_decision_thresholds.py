from __future__ import annotations

import json
from pathlib import Path

from src.decision import DecisionInput, decide

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "artifacts" / "anna_golden_evaluation.jsonl"
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def load_records():
    return [
        json.loads(line)
        for line in INPUT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_input(record):
    return DecisionInput(
        intent=record["predicted_intent"] or "unclear",
        intent_confidence=record["intent_confidence"],
        top_similarity=record["top_similarity"],
        evidence_agreement=record["evidence_score"],
        resolution_agreement=record["resolution_agreement"],
        retrieved_case_count=record["retrieved_case_count"],
        generated_confidence=record["generated_confidence"],
        generated_grounded=record["generated_grounded"],
        generated_escalate=record["generated_escalate"],
        customer_message=record["customer_message"],
    )


def main():
    records = load_records()

    print("THRESHOLD SIMULATION")
    print("threshold | auto | coverage | unsafe_auto | gold_auto_captured")
    print("-" * 70)

    for threshold in THRESHOLDS:
        results = [
            decide(
                make_input(record),
                min_intent_confidence=threshold,
            )
            for record in records
        ]

        auto_count = sum(result.auto_handle for result in results)
        unsafe_count = sum(
            result.auto_handle and not record["gold_should_auto_handle"]
            for result, record in zip(results, records)
        )
        gold_auto_captured = sum(
            result.auto_handle and record["gold_should_auto_handle"]
            for result, record in zip(results, records)
        )

        print(
            f"{threshold:.2f}      | "
            f"{auto_count:3d}  | "
            f"{auto_count / len(records):7.2%}  | "
            f"{unsafe_count:11d} | "
            f"{gold_auto_captured:16d}"
        )


if __name__ == "__main__":
    main()
