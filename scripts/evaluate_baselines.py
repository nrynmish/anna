"""Evaluate trivial and simple intent-classification baselines."""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.golden_set import evaluate_direct_intent


def main() -> None:
    golden_path = Path("data/golden/golden_200.json")

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    gold_labels = [str(item["intent"]) for item in golden]

    counts = Counter(gold_labels)
    majority_intent, majority_count = counts.most_common(1)[0]
    majority_accuracy = majority_count / len(gold_labels)

    heuristic = evaluate_direct_intent(golden_path)

    result = {
        "dataset": {
            "examples": len(gold_labels),
            "intents": len(counts),
        },
        "majority_class_baseline": {
            "intent": majority_intent,
            "count": majority_count,
            "accuracy": majority_accuracy,
        },
        "keyword_heuristic_baseline": {
            "accuracy": heuristic["accuracy"],
            "macro_f1": heuristic["macro_f1"],
            "weighted_f1": heuristic["weighted_f1"],
        },
        "lift": {
            "accuracy_absolute": heuristic["accuracy"] - majority_accuracy,
            "accuracy_relative": (
                (heuristic["accuracy"] / majority_accuracy) - 1
                if majority_accuracy
                else None
            ),
        },
    }

    output = Path("artifacts/baseline_evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
