"""Build the case-level retrieval corpus for ANNA."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CASES_PATH = Path("artifacts/uber_support_cases.jsonl")
LABELS_PATH = Path("artifacts/uber_case_intent_labels.jsonl")
OUTPUT_PATH = Path("artifacts/uber_retrieval_corpus.jsonl")


def clean(text: str) -> str:
    return " ".join(str(text or "").split())


def main() -> None:
    labels = {}

    with LABELS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                labels[record["case_id"]] = record

    resolution_counts = Counter()
    intent_counts = Counter()
    empty_customer = 0
    empty_response = 0
    records_written = 0

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with (
        CASES_PATH.open("r", encoding="utf-8") as cases_fh,
        OUTPUT_PATH.open("w", encoding="utf-8") as out_fh,
    ):
        for line in cases_fh:
            if not line.strip():
                continue

            case = json.loads(line)
            case_id = case["case_id"]
            label = labels.get(case_id)

            if label is None:
                continue

            customer_text = clean(
                " ".join(
                    clean(message.get("text", ""))
                    for message in case.get("customer_messages", [])
                )
            )

            response_text = clean(
                " ".join(
                    clean(message.get("text", ""))
                    for message in case.get("uber_messages", [])
                )
            )

            final_response = clean(
                (case.get("final_uber_response") or {}).get("text", "")
            )

            if not customer_text:
                empty_customer += 1

            if not response_text:
                empty_response += 1

            intent = label.get("intent", "unclear")
            resolution_type = case.get("resolution_type", "unclear")

            # Retrieval text prioritizes the customer's problem, then the
            # historical support response. Metadata is stored separately.
            retrieval_text = clean(
                f"Customer issue: {customer_text} "
                f"Historical support response: {response_text}"
            )

            record = {
                "case_id": case_id,
                "customer_id": case.get("customer_id"),
                "intent": intent,
                "intent_confidence": label.get("labeling_confidence", 0.0),
                "labeling_method": label.get("labeling_method"),
                "customer_text": customer_text,
                "uber_response": response_text,
                "final_uber_response": final_response,
                "resolution_type": resolution_type,
                "created_at": case.get("created_at"),
                "retrieval_text": retrieval_text,
            }

            out_fh.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            records_written += 1
            intent_counts[intent] += 1
            resolution_counts[resolution_type] += 1

    print("===== RETRIEVAL CORPUS =====")
    print(f"Records written: {records_written}")
    print(f"Empty customer text: {empty_customer}")
    print(f"Empty Uber response: {empty_response}")
    print()

    print("===== INTENTS =====")
    for intent, count in intent_counts.most_common():
        print(f"{intent:30s} {count:6d}")

    print()
    print("===== RESOLUTION TYPES =====")
    for resolution, count in resolution_counts.most_common():
        print(f"{resolution:30s} {count:6d}")

    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
