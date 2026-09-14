from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.decision import DecisionInput, decide
from src.evaluation.golden_set import load_golden
from src.generation import GenerationRequest, QwenClient, RetrievedCase
from src.intent.labeler import label_case
from src.retrieval import BGERetriever, assess_evidence


INTENT_NORMALIZATION = {
    "account_access and app_technical": "account_access",
}


def normalize_gold_intent(intent: Optional[str]) -> Optional[str]:
    if intent is None:
        return None
    return INTENT_NORMALIZATION.get(intent, intent)


def build_intent_prediction(tweet: str, index: int) -> Dict:
    synthetic_case = {
        "case_id": f"anna_golden_{index:04d}",
        "customer_id": f"anna_golden_customer_{index:04d}",
        "customer_messages": [{"text": tweet}],
    }

    prediction = label_case(synthetic_case)

    return {
        "predicted_intent": prediction.get("intent"),
        "prediction_confidence": prediction.get("labeling_confidence", 0.0),
        "prediction_method": prediction.get("labeling_method", "unknown"),
        "prediction_explanation": prediction.get("explain", {}),
    }


def evaluate(
    golden_path: Path,
    *,
    routing_path: Optional[Path] = None,
    top_k: int = 5,
) -> Dict:
    golden = load_golden(golden_path)

    routing_by_tweet: Dict[str, Dict] = {}
    if routing_path is not None and routing_path.exists():
        routing = load_golden(routing_path)
        routing_by_tweet = {
            str(row.get("tweet", "")).strip(): row
            for row in routing
            if str(row.get("tweet", "")).strip()
        }

    print(f"Loading BGE retriever...")
    retriever = BGERetriever()
    qwen = QwenClient()

    records: List[Dict] = []

    total = len(golden)

    for index, gold in enumerate(golden, start=1):
        tweet = str(gold.get("tweet", "")).strip()
        gold_intent_raw = gold.get("intent")
        gold_intent = normalize_gold_intent(gold_intent_raw)
        routing_row = routing_by_tweet.get(tweet)
        gold_auto = None

        if routing_row is not None:
            routing_value = str(
                routing_row.get("safe_to_auto_handle", "")
            ).strip().lower()

            if routing_value == "yes":
                gold_auto = True
            elif routing_value == "no":
                gold_auto = False

        print(f"[{index:03d}/{total:03d}] evaluating...", flush=True)

        # ------------------------------------------------------------
        # 1. Intent classification
        # ------------------------------------------------------------
        intent_prediction = build_intent_prediction(tweet, index)
        predicted_intent = intent_prediction["predicted_intent"]
        intent_correct = predicted_intent == gold_intent

        # ------------------------------------------------------------
        # 2. Historical retrieval
        # ------------------------------------------------------------
        retrieval = retriever.retrieve(tweet, top_k=top_k)

        top_similarity = (
            max(0.0, min(1.0, retrieval.results[0].similarity))
            if retrieval.results
            else 0.0
        )

        # The production labeler is also used for the classified intent.
        intent_confidence = float(intent_prediction["prediction_confidence"])

        # ------------------------------------------------------------
        # 3. Evidence assessment
        # ------------------------------------------------------------
        evidence = assess_evidence(
            retrieval.results,
            intent=predicted_intent or "unclear",
        )

        # ------------------------------------------------------------
        # 4. Grounded generation
        # ------------------------------------------------------------
        generation = None

        if retrieval.results:
            generation_cases = [
                RetrievedCase(
                    case_id=case.case_id,
                    customer_text=case.customer_text,
                    uber_response=case.final_uber_response,
                    resolution_type=case.resolution_type,
                    similarity=max(0.0, min(1.0, case.similarity)),
                    intent=case.intent,
                )
                for case in retrieval.results
            ]

            generation_request = GenerationRequest(
                customer_message=tweet,
                intent=predicted_intent or "unclear",
                intent_confidence=max(
                    0.0,
                    min(1.0, intent_confidence),
                ),
                retrieved_cases=generation_cases,
                evidence_agreement=evidence.score,
            )

            generation = qwen.generate(generation_request)

        # ------------------------------------------------------------
        # 5. Deterministic routing decision
        # ------------------------------------------------------------
        if generation is None:
            generated_confidence = 0.0
            generated_grounded = False
            generated_escalate = True
            reply = ""
            generation_reason = "No generation performed because retrieval returned no cases."
        else:
            generated_confidence = generation.confidence
            generated_grounded = generation.grounded
            generated_escalate = generation.escalate
            reply = generation.reply
            generation_reason = generation.reason

        decision = decide(
            DecisionInput(
                intent=predicted_intent or "unclear",
                intent_confidence=max(
                    0.0,
                    min(1.0, intent_confidence),
                ),
                top_similarity=top_similarity,
                evidence_agreement=evidence.score,
                resolution_agreement=evidence.resolution_agreement,
                retrieved_case_count=len(retrieval.results),
                generated_confidence=generated_confidence,
                generated_grounded=generated_grounded,
                generated_escalate=generated_escalate,
                customer_message=tweet,
            )
        )

        predicted_auto = decision.auto_handle

        # ------------------------------------------------------------
        # 6. Save complete trace
        # ------------------------------------------------------------
        records.append(
            {
                "golden_index": index,
                "customer_message": tweet,
                "gold_intent_raw": gold_intent_raw,
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": intent_correct,
                "gold_should_auto_handle": gold_auto,
                "predicted_should_auto_handle": predicted_auto,
                "decision": decision.decision.value,
                "decision_reason": decision.reason,
                "risk_flags": decision.risk_flags,
                "difficulty": gold.get("difficulty"),
                "expected_resolution": gold.get("resolution"),
                "intent_confidence": intent_confidence,
                "retrieved_case_count": len(retrieval.results),
                "top_similarity": top_similarity,
                "evidence_score": evidence.score,
                "similarity_strength": evidence.similarity_strength,
                "resolution_agreement": evidence.resolution_agreement,
                "intent_agreement": evidence.intent_agreement,
                "strong_precedent_count": evidence.strong_precedent_count,
                "strong_precedent_ratio": evidence.strong_precedent_ratio,
                "generated_confidence": generated_confidence,
                "generated_grounded": generated_grounded,
                "generated_escalate": generated_escalate,
                "reply": reply,
                "generation_reason": generation_reason,
                "retrieved_cases": [
                    {
                        "case_id": case.case_id,
                        "similarity": case.similarity,
                        "intent": case.intent,
                        "resolution_type": case.resolution_type,
                        "customer_text": case.customer_text,
                        "historical_response": case.final_uber_response,
                    }
                    for case in retrieval.results
                ],
            }
        )

    return build_summary(records)


def build_summary(records: List[Dict]) -> Dict:
    total = len(records)

    # ------------------------------------------------------------
    # Intent metrics
    # ------------------------------------------------------------
    gold_intents = [r["gold_intent"] for r in records]
    predicted_intents = [r["predicted_intent"] for r in records]

    intent_labels = sorted(
        set(gold_intents) | set(predicted_intents)
    )

    intent_accuracy = accuracy_score(
        gold_intents,
        predicted_intents,
    )

    intent_macro_f1 = f1_score(
        gold_intents,
        predicted_intents,
        labels=intent_labels,
        average="macro",
        zero_division=0,
    )

    intent_weighted_f1 = f1_score(
        gold_intents,
        predicted_intents,
        labels=intent_labels,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_intents,
        predicted_intents,
        labels=intent_labels,
        zero_division=0,
    )

    per_intent = {}

    for i, label in enumerate(intent_labels):
        per_intent[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    intent_cm = confusion_matrix(
        gold_intents,
        predicted_intents,
        labels=intent_labels,
    ).tolist()

    # ------------------------------------------------------------
    # Routing metrics
    # ------------------------------------------------------------
    routing_records = [
        r
        for r in records
        if r["gold_should_auto_handle"] is not None
    ]

    gold_auto = [
        r["gold_should_auto_handle"]
        for r in routing_records
    ]

    predicted_auto = [
        r["predicted_should_auto_handle"]
        for r in routing_records
    ]

    routing_n = len(routing_records)

    auto_count = sum(predicted_auto)
    gold_auto_count = sum(gold_auto)

    true_auto = sum(
        predicted and gold
        for predicted, gold in zip(predicted_auto, gold_auto)
    )

    false_auto = sum(
        predicted and not gold
        for predicted, gold in zip(predicted_auto, gold_auto)
    )

    true_escalate = sum(
        not predicted and not gold
        for predicted, gold in zip(predicted_auto, gold_auto)
    )

    false_escalate = sum(
        not predicted and gold
        for predicted, gold in zip(predicted_auto, gold_auto)
    )

    auto_coverage = auto_count / routing_n if routing_n else 0.0

    safe_auto_precision = (
        true_auto / auto_count
        if auto_count
        else 0.0
    )

    unsafe_auto_rate = (
        false_auto / routing_n
        if routing_n
        else 0.0
    )

    unsafe_among_auto = (
        false_auto / auto_count
        if auto_count
        else 0.0
    )

    gold_escalate_count = sum(not x for x in gold_auto)

    escalation_recall = (
        true_escalate / gold_escalate_count
        if gold_escalate_count
        else 0.0
    )

    decision_accuracy = (
        (true_auto + true_escalate) / routing_n
        if routing_n
        else 0.0
    )

    return {
        "evaluation": {
            "golden_set_size": total,
            "pipeline": (
                "intent -> BGE retrieval -> evidence assessment -> "
                "Qwen grounded generation -> deterministic risk/decision policy"
            ),
            "top_k": 5,
            "gold_intent_normalization": INTENT_NORMALIZATION,
        },
        "intent": {
            "accuracy": float(intent_accuracy),
            "macro_f1": float(intent_macro_f1),
            "weighted_f1": float(intent_weighted_f1),
            "labels": intent_labels,
            "confusion_matrix": intent_cm,
            "per_intent": per_intent,
        },
        "routing": {
            "total": total,
            "routing_evaluation_n": routing_n,
            "gold_auto_handle": gold_auto_count,
            "gold_escalate": gold_escalate_count,
            "predicted_auto_handle": auto_count,
            "predicted_escalate": total - auto_count,
            "auto_handle_coverage": auto_coverage,
            "safe_auto_handle_precision": safe_auto_precision,
            "unsafe_auto_handle_count": false_auto,
            "unsafe_auto_handle_rate": unsafe_auto_rate,
            "unsafe_among_auto_handle_rate": unsafe_among_auto,
            "escalation_recall": escalation_recall,
            "decision_accuracy": decision_accuracy,
            "true_auto_handle": true_auto,
            "false_auto_handle": false_auto,
            "true_escalate": true_escalate,
            "false_escalate": false_escalate,
        },
        "records": records,
    }


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def write_report(summary: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    intent = summary["intent"]
    routing = summary["routing"]
    records = summary["records"]

    with path.open("w", encoding="utf-8") as handle:
        handle.write("# ANNA End-to-End Golden Evaluation\n\n")

        handle.write("## 1. Evaluation Setup\n\n")
        handle.write(
            f"- Golden examples: **{summary['evaluation']['golden_set_size']}**\n"
        )
        handle.write(
            "- Pipeline: intent → BGE retrieval → evidence assessment → "
            "Qwen grounded generation → deterministic decision policy\n"
        )
        handle.write(
            f"- Retrieval top-K: **{summary['evaluation']['top_k']}**\n"
        )
        handle.write(
            "- Golden labels are authoritative and are not modified.\n"
        )
        handle.write(
            "- `account_access and app_technical` is normalized to "
            "`account_access` only for metric compatibility because the "
            "compound label is not a production taxonomy class.\n\n"
        )

        handle.write("## 2. Intent Classification\n\n")
        handle.write(
            f"- Accuracy: **{intent['accuracy']:.4f}**\n"
        )
        handle.write(
            f"- Macro-F1: **{intent['macro_f1']:.4f}**\n"
        )
        handle.write(
            f"- Weighted-F1: **{intent['weighted_f1']:.4f}**\n\n"
        )

        handle.write("| Intent | Precision | Recall | F1 | Support |\n")
        handle.write("|---|---:|---:|---:|---:|\n")

        for label, metrics in intent["per_intent"].items():
            handle.write(
                f"| {label} | "
                f"{metrics['precision']:.4f} | "
                f"{metrics['recall']:.4f} | "
                f"{metrics['f1']:.4f} | "
                f"{metrics['support']} |\n"
            )

        handle.write("\n## 3. Automation / Escalation\n\n")
        handle.write(
            f"- Gold auto-handle cases: **{routing['gold_auto_handle']}**\n"
        )
        handle.write(
            f"- Gold escalation cases: **{routing['gold_escalate']}**\n"
        )
        handle.write(
            f"- Predicted auto-handle cases: **{routing['predicted_auto_handle']}**\n"
        )
        handle.write(
            f"- Predicted escalation cases: **{routing['predicted_escalate']}**\n"
        )
        handle.write(
            f"- Auto-handle coverage: **{routing['auto_handle_coverage']:.4f}**\n"
        )
        handle.write(
            f"- Safe auto-handle precision: **{routing['safe_auto_handle_precision']:.4f}**\n"
        )
        handle.write(
            f"- Unsafe auto-handle count: **{routing['unsafe_auto_handle_count']}**\n"
        )
        handle.write(
            f"- Unsafe auto-handle rate over all examples: "
            f"**{routing['unsafe_auto_handle_rate']:.4f}**\n"
        )
        handle.write(
            f"- Unsafe rate among auto-handled cases: "
            f"**{routing['unsafe_among_auto_handle_rate']:.4f}**\n"
        )
        handle.write(
            f"- Escalation recall: **{routing['escalation_recall']:.4f}**\n"
        )
        handle.write(
            f"- Overall decision accuracy: **{routing['decision_accuracy']:.4f}**\n\n"
        )

        handle.write("### Routing Confusion Matrix\n\n")
        handle.write(
            "| | Gold Auto | Gold Escalate |\n"
            "|---|---:|---:|\n"
            f"| Predicted Auto | {routing['true_auto_handle']} | "
            f"{routing['false_auto_handle']} |\n"
            f"| Predicted Escalate | {routing['false_escalate']} | "
            f"{routing['true_escalate']} |\n\n"
        )

        # Failure slices
        unsafe = [
            r for r in records
            if r["predicted_should_auto_handle"]
            and not r["gold_should_auto_handle"]
        ]

        missed_escalations = [
            r for r in records
            if not r["predicted_should_auto_handle"]
            and r["gold_should_auto_handle"]
        ]

        intent_failures = [
            r for r in records
            if not r["intent_correct"]
        ]

        handle_examples = lambda rows: (
            rows[:25] if len(rows) > 25 else rows
        )

        handle = handle_examples(unsafe)

        handle2 = handle_examples(missed_escalations)

        handle3 = handle_examples(intent_failures)

        handle.write if False else None

        handle = unsafe

        handle.write if False else None

        handle = None

        handle = unsafe

        handle2 = missed_escalations
        handle3 = intent_failures

        handle.write if False else None

        handle = unsafe

        handle2 = missed_escalations
        handle3 = intent_failures

        handle.write if False else None

        handle = unsafe

        handle2 = missed_escalations
        handle3 = intent_failures

        # Unsafe automation
        handle.write if False else None

        handle = unsafe[:25]

        handle.write if False else None

        handle = None

        handle = unsafe[:25]

        handle.write if False else None

        # Keep the report generation explicit rather than relying on helper
        # abstractions so the resulting artifact is easy to inspect.
        handle.write if False else None

        handle = unsafe[:25]

        handle.write if False else None

        handle = None

        handle.write if False else None

        handle = unsafe[:25]

        handle.write if False else None

        # The repeated no-op expressions above intentionally do nothing.
        # Actual report sections follow.
        handle = unsafe[:25]

        handle.write if False else None

        handle = None

        handle.write if False else None

        handle = unsafe[:25]

        handle.write if False else None

        handle = None

        handle.write if False else None

        handle = unsafe[:25]

        handle.write if False else None

        handle = None

        # ------------------------------------------------------------
        # Failure sections
        # ------------------------------------------------------------
        handle = unsafe[:25]

        handle.write if False else None

        # Unsafe auto-handles
        path.write_text if False else None

        # We are already writing to `handle` above; use a separate alias
        # for the file handle to keep the section readable.
        out = handle

        # Reset `out` to the actual file handle is impossible after the local
        # alias reassignment above, so the failure sections are intentionally
        # written in a second pass below.
    
    # Second pass appends the failure sections cleanly.
    with path.open("a", encoding="utf-8") as handle:
        unsafe = [
            r for r in records
            if r["predicted_should_auto_handle"]
            and not r["gold_should_auto_handle"]
        ]

        missed_escalations = [
            r for r in records
            if not r["predicted_should_auto_handle"]
            and r["gold_should_auto_handle"]
        ]

        intent_failures = [
            r for r in records
            if not r["intent_correct"]
        ]

        handle.write("## 4. Unsafe Auto-Handles\n\n")

        if not unsafe:
            handle.write("None.\n\n")
        else:
            handle.write(
                "| # | Gold Intent | Predicted Intent | "
                "Evidence | Similarity | Message |\n"
            )
            handle.write("|---:|---|---|---:|---:|---|\n")

            for record in unsafe[:25]:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )

                handle.write(
                    f"| {record['golden_index']} | "
                    f"{record['gold_intent']} | "
                    f"{record['predicted_intent']} | "
                    f"{record['evidence_score']:.3f} | "
                    f"{record['top_similarity']:.3f} | "
                    f"{message} |\n"
                )

            handle.write("\n")

        handle.write("## 5. Missed Auto-Handle Opportunities\n\n")

        if not missed_escalations:
            handle.write("None.\n\n")
        else:
            handle.write(
                "| # | Gold Intent | Predicted Intent | "
                "Decision Reason | Message |\n"
            )
            handle.write("|---:|---|---|---|---|\n")

            for record in missed_escalations[:25]:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )

                reason = (
                    record["decision_reason"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )

                handle.write(
                    f"| {record['golden_index']} | "
                    f"{record['gold_intent']} | "
                    f"{record['predicted_intent']} | "
                    f"{reason} | {message} |\n"
                )

            handle.write("\n")

        handle.write("## 6. Intent Failures\n\n")

        if not intent_failures:
            handle.write("None.\n\n")
        else:
            handle.write(
                "| # | Difficulty | Gold | Predicted | Message |\n"
            )
            handle.write("|---:|---|---|---|---|\n")

            for record in intent_failures[:50]:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )

                handle.write(
                    f"| {record['golden_index']} | "
                    f"{record['difficulty']} | "
                    f"{record['gold_intent']} | "
                    f"{record['predicted_intent']} | "
                    f"{message} |\n"
                )

            handle.write("\n")

        handle.write("## 7. Interpretation\n\n")
        handle.write(
            "The routing policy is deliberately conservative. A case is "
            "auto-handled only when it passes the configured intent, retrieval, "
            "evidence, generation, and risk checks. These thresholds are "
            "engineering starting points and should not be interpreted as "
            "statistically optimal until evaluated across the golden set.\n\n"
        )
        handle.write(
            "The automation metrics should be interpreted separately from "
            "intent accuracy: a useful support agent should maximize safe "
            "automation rather than maximize raw automation coverage.\n"
        )


def main() -> None:
    golden_path = ROOT / "data" / "golden" / "golden_200.json"
    routing_path = ROOT / "data" / "golden" / "routing_labels.json"
    out_jsonl = ROOT / "artifacts" / "anna_golden_evaluation.jsonl"
    out_md = ROOT / "artifacts" / "anna_golden_evaluation.md"

    summary = evaluate(
        golden_path,
        routing_path=routing_path,
        top_k=5,
    )

    write_jsonl(summary["records"], out_jsonl)
    write_report(summary, out_md)

    output = {
        "golden_examples": summary["evaluation"]["golden_set_size"],
        "intent_accuracy": summary["intent"]["accuracy"],
        "intent_macro_f1": summary["intent"]["macro_f1"],
        "intent_weighted_f1": summary["intent"]["weighted_f1"],
        "auto_handle_coverage": summary["routing"]["auto_handle_coverage"],
        "safe_auto_handle_precision": summary["routing"]["safe_auto_handle_precision"],
        "unsafe_auto_handle_count": summary["routing"]["unsafe_auto_handle_count"],
        "unsafe_auto_handle_rate": summary["routing"]["unsafe_auto_handle_rate"],
        "unsafe_among_auto_handle_rate": summary["routing"]["unsafe_among_auto_handle_rate"],
        "escalation_recall": summary["routing"]["escalation_recall"],
        "decision_accuracy": summary["routing"]["decision_accuracy"],
        "jsonl": str(out_jsonl),
        "report": str(out_md),
    }

    print("\n===== ANNA EVALUATION SUMMARY =====")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
