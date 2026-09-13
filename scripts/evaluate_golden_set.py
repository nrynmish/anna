"""Run the primary direct golden-set intent evaluation."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.golden_set import (
    evaluate_direct_intent,
    write_direct_intent_report,
)


def main() -> None:
    golden_path = Path("data/golden/tweets.json")
    out_jsonl = Path("artifacts/golden_set.jsonl")
    out_md = Path("artifacts/golden_evaluation.md")

    result = evaluate_direct_intent(golden_path)

    write_direct_intent_report(
        result=result,
        out_jsonl=out_jsonl,
        out_md=out_md,
    )

    summary = {
        "golden_examples": result["total"],
        "evaluated": result["evaluated"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "jsonl": str(out_jsonl),
        "report": str(out_md),
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
