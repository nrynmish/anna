"""Evaluate a TF-IDF + Logistic Regression intent-classification baseline.

Training labels come from the existing weakly-labelled Uber support-case corpus.
Evaluation is performed directly on the human-labelled golden set.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.golden_set import load_golden
from src.evaluation.golden_set import normalize_text


CASES_PATH = Path("artifacts/uber_support_cases.jsonl")
LABELS_PATH = Path("artifacts/uber_case_intent_labels.jsonl")
GOLDEN_PATH = Path("data/golden/tweets.json")
OUTPUT_PATH = Path("artifacts/tfidf_baseline_evaluation.json")


def load_training_data():
    labels_by_case = {}

    with LABELS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            labels_by_case[record["case_id"]] = record["intent"]

    texts = []
    labels = []

    with CASES_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue

            case = json.loads(line)
            case_id = case.get("case_id")
            intent = labels_by_case.get(case_id)

            if not intent:
                continue

            messages = case.get("customer_messages") or []
            text = " ".join(
                str(message.get("text", "")).strip()
                for message in messages
                if str(message.get("text", "")).strip()
            )

            text = normalize_text(text)

            if not text:
                continue

            texts.append(text)
            labels.append(intent)

    return texts, labels


def main():
    train_texts, train_labels = load_training_data()
    golden = load_golden(GOLDEN_PATH)

    test_texts = [
        normalize_text(str(item.get("tweet", "")))
        for item in golden
    ]
    gold_labels = [
        str(item.get("intent"))
        for item in golden
    ]

    print("===== TRAINING DATA =====")
    print(f"Training examples: {len(train_texts)}")
    print(f"Training intents: {len(set(train_labels))}")

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    max_features=50000,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(test_texts)

    labels = sorted(set(gold_labels) | set(predictions))

    accuracy = accuracy_score(gold_labels, predictions)
    macro_f1 = f1_score(
        gold_labels,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        gold_labels,
        predictions,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels,
        predictions,
        labels=labels,
        zero_division=0,
    )

    per_intent = {}

    for i, intent in enumerate(labels):
        per_intent[intent] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    result = {
        "method": "tfidf_logistic_regression",
        "training": {
            "examples": len(train_texts),
            "intents": len(set(train_labels)),
            "excluded_intents": [],
            "label_source": "weak_labels_from_reconstructed_uber_cases",
        },
        "evaluation": {
            "examples": len(golden),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        },
        "per_intent": per_intent,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("===== TF-IDF + LOGISTIC REGRESSION =====")
    print(f"Accuracy:    {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Macro-F1:    {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
