"""Sample 61 additional Uber cases for the 200-example golden evaluation set.

The existing 139 golden examples are left untouched.
This script selects real Uber support cases using the existing heuristic labels,
with quotas designed to improve intent and routing coverage.

Output:
    data/golden/golden_additions_candidates.json
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.intent.labeler import label_case


CASES_PATH = ROOT / "artifacts" / "uber_support_cases.jsonl"
OUTPUT_PATH = ROOT / "data" / "golden" / "golden_additions_candidates.json"

SEED = 200
TARGET = 61

# Deliberately emphasize useful/underrepresented intents while retaining
# difficult and ambiguous cases.
QUOTAS = {
    "payments_charges": 8,
    "refunds_adjustments": 8,
    "rides_and_trips": 6,
    "driver_behavior": 6,
    "account_access": 5,
    "lost_and_found": 5,
    "promotions_discounts": 5,
    "app_technical": 4,
    "signup_onboarding": 4,
    "payouts_and_earnings": 4,
    "safety_incident": 3,
    "other": 1,
    "unclear": 2,
}

assert sum(QUOTAS.values()) == TARGET


def load_cases():
    cases = []
    with CASES_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def customer_text(case):
    messages = case.get("customer_messages") or []
    return " ".join(
        m.get("text", "").strip()
        for m in messages
        if isinstance(m, dict) and m.get("text")
    ).strip()


def usable(case):
    text = customer_text(case)

    if not text:
        return False

    # Avoid cases dominated by personal data or unusably short fragments.
    if len(text) < 20:
        return False

    # Keep cases with an actual Uber response, since the golden set evaluates
    # support behavior grounded in historical resolutions.
    if not case.get("uber_messages"):
        return False

    return True


def label(case):
    return label_case(
        {
            "case_id": case.get("case_id"),
            "customer_id": case.get("customer_id"),
            "customer_messages": case.get("customer_messages") or [],
        }
    )


def difficulty(case, predicted_intent, confidence):
    text = customer_text(case).lower()
    msg_count = case.get("customer_message_count", 1)

    risk_terms = [
        "fraud",
        "stolen",
        "police",
        "lawyer",
        "legal",
        "unsafe",
        "assault",
        "threat",
        "harass",
        "charged",
        "refund",
        "scam",
        "account hacked",
        "unauthorized",
    ]

    ambiguity_terms = [
        "don't know",
        "not sure",
        "something",
        "everything",
        "help me",
        "what happened",
        "issue",
        "problem",
    ]

    if any(term in text for term in risk_terms):
        return "hard"

    if predicted_intent in {"other", "unclear"}:
        return "hard"

    if confidence < 0.50 or msg_count >= 3:
        return "hard"

    if confidence < 0.75 or msg_count == 2 or any(
        term in text for term in ambiguity_terms
    ):
        return "medium"

    return "easy"


def main():
    random.seed(SEED)

    cases = load_cases()

    buckets = {intent: [] for intent in QUOTAS}

    for case in cases:
        if not usable(case):
            continue

        lbl = label(case)
        intent = lbl.get("intent")

        if intent not in buckets:
            continue

        buckets[intent].append(
            {
                "case": case,
                "label": lbl,
            }
        )

    for intent in buckets:
        random.shuffle(buckets[intent])

    selected = []

    for intent, quota in QUOTAS.items():
        candidates = buckets[intent]

        if len(candidates) < quota:
            raise RuntimeError(
                f"Not enough candidates for {intent}: "
                f"need {quota}, found {len(candidates)}"
            )

        # Prefer cases with useful support context and reasonable text length.
        candidates = sorted(
            candidates,
            key=lambda x: (
                x["case"].get("customer_message_count", 1),
                len(customer_text(x["case"])),
            ),
        )

        # Take from across the bucket rather than only the first few.
        if len(candidates) > quota:
            positions = [
                round(i * (len(candidates) - 1) / (quota - 1))
                if quota > 1
                else 0
                for i in range(quota)
            ]
            chosen = [candidates[p] for p in positions]
        else:
            chosen = candidates

        selected.extend(chosen)

    # Stable ordering for review.
    selected.sort(key=lambda x: x["case"]["case_id"])

    output = []

    for number, item in enumerate(selected, start=140):
        case = item["case"]
        lbl = item["label"]

        text = customer_text(case)
        intent = lbl["intent"]
        confidence = float(lbl.get("labeling_confidence") or 0.0)

        output.append(
            {
                "id": number,
                "case_id": case["case_id"],
                "tweet": text,
                "historical_response": (
                    case.get("final_uber_response", {}).get("text", "")
                    if isinstance(case.get("final_uber_response"), dict)
                    else ""
                ),
                "resolution_type": case.get("resolution_type"),
                "heuristic_intent": intent,
                "heuristic_confidence": confidence,
                "difficulty_suggested": difficulty(
                    case, intent, confidence
                ),
                "safe_to_auto_handle": None,
                "reason": "",
                "intent": intent,
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Generated: {OUTPUT_PATH}")
    print(f"Candidates: {len(output)}")
    print()
    print("Intent distribution:")
    counts = Counter(x["intent"] for x in output)
    for intent, count in counts.most_common():
        print(f"  {intent}: {count}")

    print()
    print("Difficulty distribution:")
    difficulties = Counter(x["difficulty_suggested"] for x in output)
    for difficulty_name, count in difficulties.items():
        print(f"  {difficulty_name}: {count}")


if __name__ == "__main__":
    main()
