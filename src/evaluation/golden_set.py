"""Evaluate the 140-example human-labelled golden set.

Evaluation flow:
    golden tweet
        -> exact match against reconstructed case customer messages
        -> case_id
        -> heuristic prediction for that case
        -> intent metrics

The golden annotations are authoritative and are never modified.
Routing is retained as gold data, but the current heuristic baseline does
not produce a routing prediction, so routing metrics are intentionally
reported as unavailable for this baseline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.intent.labeler import label_case

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def stream_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError("Golden set must contain a JSON array.")

    return data


def normalize_text(text: str) -> str:
    """Normalize text for deterministic golden-to-case matching."""
    import re

    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_case_index(
    cases_path: Path,
) -> tuple[Dict[str, List[str]], Dict[str, str], int]:
    """Build normalized message/full-case indexes for golden mapping."""
    message_index: Dict[str, List[str]] = {}
    case_texts: Dict[str, str] = {}
    case_count = 0

    for case in stream_jsonl(cases_path):
        case_id = case.get("case_id")
        if not case_id:
            continue

        case_count += 1
        customer_texts = [
            str(message.get("text", ""))
            for message in case.get("customer_messages", [])
            if message.get("text")
        ]

        normalized_messages = [
            normalize_text(text)
            for text in customer_texts
            if normalize_text(text)
        ]

        for text in normalized_messages:
            message_index.setdefault(text, []).append(case_id)

        case_texts[case_id] = " ".join(normalized_messages)

    return message_index, case_texts, case_count


def build_labels_by_case(labels_path: Path) -> Dict[str, Dict]:
    """Map reconstructed case ID to its heuristic prediction."""
    labels: Dict[str, Dict] = {}

    for label in stream_jsonl(labels_path):
        case_id = label.get("case_id")
        if case_id:
            labels[case_id] = label

    return labels


def routable_to_bool(value: Optional[str]) -> Optional[bool]:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def map_golden_examples(
    golden: List[Dict],
    cases_index: Dict[str, List[str]],
    labels_by_case: Dict[str, Dict],
    case_texts: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Map human-written golden examples to reconstructed cases.

    Matching stages:
      1. unique normalized exact message match
      2. unique normalized containment match
      3. TF-IDF semantic similarity over reconstructed customer-side cases

    Semantic matching is deliberately conservative. A candidate is accepted
    only when its similarity clears the minimum threshold and is sufficiently
    ahead of the runner-up. Otherwise the example remains unmatched rather
    than being assigned arbitrarily.
    """
    records: List[Dict] = []

    case_ids = []
    case_documents = []

    if case_texts:
        for case_id, customer_text in case_texts.items():
            normalized = normalize_text(customer_text)
            if normalized:
                case_ids.append(case_id)
                case_documents.append(normalized)

    # Word n-grams capture support concepts; character n-grams help with
    # paraphrases, spelling variation, and short noisy Twitter text.
    word_vectorizer = None
    char_vectorizer = None
    case_word_matrix = None
    case_char_matrix = None

    if case_documents:
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            sublinear_tf=True,
        )
        case_word_matrix = word_vectorizer.fit_transform(case_documents)
        case_char_matrix = char_vectorizer.fit_transform(case_documents)

    # Conservative semantic thresholds. These are evaluation-mapping
    # thresholds, not model-performance thresholds.
    SEMANTIC_MIN_SCORE = 0.30
    SEMANTIC_MIN_MARGIN = 0.08
    SEMANTIC_MIN_CASE_WORDS = 5

    for gold in golden:
        tweet = gold.get("tweet")
        normalized_tweet = normalize_text(tweet or "")

        unique_candidates = list(
            dict.fromkeys(cases_index.get(normalized_tweet, []))
        )
        mapping_method = "normalized_exact"

        # Stage 2: conservative token-aware containment.
        #
        # Never use unrestricted character substring matching here. A tiny
        # historical message such as "no" would otherwise match unrelated
        # golden text containing words such as "another" or phrases with
        # "no", creating false provenance mappings.
        #
        # Containment is accepted only when the shorter side contains at
        # least MIN_CONTAINMENT_TOKENS tokens and its complete token sequence
        # occurs contiguously in the longer side.
        MIN_CONTAINMENT_TOKENS = 4

        def token_sequence_contains(shorter: str, longer: str) -> bool:
            short_tokens = shorter.split()
            long_tokens = longer.split()

            if len(short_tokens) < MIN_CONTAINMENT_TOKENS:
                return False

            if len(short_tokens) > len(long_tokens):
                return False

            width = len(short_tokens)
            return any(
                long_tokens[i:i + width] == short_tokens
                for i in range(len(long_tokens) - width + 1)
            )

        if not unique_candidates and case_texts and normalized_tweet:
            containment_candidates = []

            for candidate_case_id, customer_text in case_texts.items():
                if not customer_text:
                    continue

                if (
                    token_sequence_contains(normalized_tweet, customer_text)
                    or token_sequence_contains(customer_text, normalized_tweet)
                ):
                    containment_candidates.append(candidate_case_id)

            unique_candidates = list(dict.fromkeys(containment_candidates))

            if len(unique_candidates) == 1:
                mapping_method = "normalized_containment"
            elif len(unique_candidates) > 1:
                # Containment does not establish unique provenance.
                # Fall through to conservative semantic ranking.
                unique_candidates = []

        # Stage 3: semantic ranking.
        semantic_candidates = []
        semantic_scores = []

        if (
            not unique_candidates
            and normalized_tweet
            and case_documents
            and word_vectorizer is not None
            and char_vectorizer is not None
        ):
            query_word = word_vectorizer.transform([normalized_tweet])
            query_char = char_vectorizer.transform([normalized_tweet])

            word_scores = cosine_similarity(
                query_word, case_word_matrix
            )[0]
            char_scores = cosine_similarity(
                query_char, case_char_matrix
            )[0]

            scores = 0.65 * word_scores + 0.35 * char_scores
            ranked = np.argsort(scores)[::-1]

            best_idx = int(ranked[0])
            best_score = float(scores[best_idx])
            second_score = (
                float(scores[ranked[1]])
                if len(ranked) > 1
                else 0.0
            )
            margin = best_score - second_score

            # Keep a small diagnostic shortlist in the output even when
            # semantic matching is rejected.
            shortlist = ranked[:5]
            semantic_candidates = [case_ids[int(i)] for i in shortlist]
            semantic_scores = [float(scores[int(i)]) for i in shortlist]

            case_word_count = len(
                case_documents[best_idx].split()
            )

            if (
                best_score >= SEMANTIC_MIN_SCORE
                and margin >= SEMANTIC_MIN_MARGIN
                and case_word_count >= SEMANTIC_MIN_CASE_WORDS
            ):
                unique_candidates = [case_ids[best_idx]]
                mapping_method = "tfidf_semantic"
            else:
                unique_candidates = []

        if len(unique_candidates) == 1:
            case_id = unique_candidates[0]
            label = labels_by_case.get(case_id)

            if label is not None:
                predicted_intent = label.get("intent")
                prediction_status = "available"
            else:
                predicted_intent = None
                prediction_status = "unavailable"

            mapping_status = "unique"

        elif len(unique_candidates) == 0:
            case_id = None
            predicted_intent = None
            prediction_status = "unavailable"
            mapping_status = "unmatched"

            if mapping_method not in {
                "tfidf_semantic",
                "normalized_containment",
            }:
                mapping_method = "none"

        else:
            case_id = None
            predicted_intent = None
            prediction_status = "unavailable"
            mapping_status = "ambiguous"
            mapping_method = "ambiguous"

        gold_intent = gold.get("intent")

        records.append(
            {
                "case_id": case_id,
                "customer_message": tweet,
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": (
                    predicted_intent == gold_intent
                    if predicted_intent is not None
                    else None
                ),
                "gold_should_auto_handle": routable_to_bool(
                    gold.get("routable")
                ),
                "predicted_should_auto_handle": None,
                "difficulty": gold.get("difficulty"),
                "expected_resolution": gold.get("resolution"),
                "mapping_status": mapping_status,
                "mapping_method": mapping_method,
                "mapping_candidates": (
                    semantic_candidates
                    if mapping_status == "unmatched"
                    and semantic_candidates
                    else unique_candidates
                ),
                "semantic_scores": semantic_scores,
                "prediction_status": prediction_status,
            }
        )

    return records


def compute_intent_metrics(records: List[Dict]) -> Dict:
    evaluated = [
        r
        for r in records
        if r["mapping_status"] == "unique"
        and r["prediction_status"] == "available"
        and r["predicted_intent"] is not None
    ]

    total = len(records)
    uniquely_mapped = sum(
        r["mapping_status"] == "unique" for r in records
    )
    unmatched = sum(
        r["mapping_status"] == "unmatched" for r in records
    )
    ambiguous = sum(
        r["mapping_status"] == "ambiguous" for r in records
    )
    predictions_available = len(evaluated)
    predictions_unavailable = uniquely_mapped - predictions_available

    counts = {
        "total": total,
        "uniquely_mapped": uniquely_mapped,
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "predictions_available": predictions_available,
        "predictions_unavailable": predictions_unavailable,
    }

    if not evaluated:
        return {
            "counts": counts,
            "note": "No mapped heuristic predictions available.",
        }

    gold = [r["gold_intent"] for r in evaluated]
    pred = [r["predicted_intent"] for r in evaluated]

    labels = sorted(set(gold) | set(pred))

    accuracy = accuracy_score(gold, pred)
    macro_f1 = f1_score(
        gold,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        gold,
        pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold,
        pred,
        labels=labels,
        zero_division=0,
    )

    per_intent = {}

    for i, label in enumerate(labels):
        per_intent[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(gold, pred, labels=labels)

    return {
        "counts": counts,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "per_intent": per_intent,
    }


def compute_routing_metrics(records: List[Dict]) -> Dict:
    """Routing is intentionally unavailable for the heuristic baseline."""
    return {
        "available": False,
        "note": (
            "Routing metrics are unavailable for the heuristic baseline "
            "because the existing heuristic labeler produces no "
            "should_auto_handle prediction."
        ),
    }


def build_summary(records: List[Dict], case_count: int) -> Dict:
    intent = compute_intent_metrics(records)
    routing = compute_routing_metrics(records)

    disagreements = [
        r
        for r in records
        if r["intent_correct"] is False
    ]

    unmatched = [
        r
        for r in records
        if r["mapping_status"] == "unmatched"
    ]

    ambiguous = [
        r
        for r in records
        if r["mapping_status"] == "ambiguous"
    ]

    return {
        "evaluation": {
            "golden_set_size": len(records),
            "reconstructed_corpus_size": case_count,
            "baseline": "deterministic heuristic intent labeler",
        },
        "mapping": {
            "total": len(records),
            "unique": sum(
                r["mapping_status"] == "unique" for r in records
            ),
            "unmatched": len(unmatched),
            "ambiguous": len(ambiguous),
        },
        "intent": intent,
        "routing": routing,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
        "unmatched_examples": unmatched,
        "ambiguous_examples": ambiguous,
    }


def write_jsonl(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(
                json.dumps(record, ensure_ascii=False)
                + "\n"
            )


def write_markdown_report(
    summary: Dict,
    records: List[Dict],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fh:
        evaluation = summary["evaluation"]
        mapping = summary["mapping"]
        intent = summary["intent"]

        fh.write("# Golden Set Evaluation\n\n")

        fh.write("## 1. Evaluation Setup\n\n")
        fh.write(
            f"- Golden set: **{evaluation['golden_set_size']}** "
            "manually labelled examples\n"
        )
        fh.write(
            f"- Reconstructed historical corpus: "
            f"**{evaluation['reconstructed_corpus_size']}** cases\n"
        )
        fh.write(
            f"- Baseline: **{evaluation['baseline']}**\n"
        )
        fh.write(
            "- The golden labels are authoritative ground truth.\n"
        )
        fh.write(
            "- No routing prediction is available from this baseline.\n\n"
        )

        fh.write("## 2. Mapping Audit\n\n")
        fh.write(f"- Total gold examples: **{mapping['total']}**\n")
        fh.write(f"- Unique mappings: **{mapping['unique']}**\n")
        fh.write(f"- Unmatched: **{mapping['unmatched']}**\n")
        fh.write(f"- Ambiguous: **{mapping['ambiguous']}**\n")

        counts = intent.get("counts", {})
        fh.write(
            f"- Heuristic predictions available: "
            f"**{counts.get('predictions_available', 0)}**\n"
        )
        fh.write(
            f"- Heuristic predictions unavailable: "
            f"**{counts.get('predictions_unavailable', 0)}**\n\n"
        )

        fh.write("## 3. Intent Metrics\n\n")

        if "note" in intent:
            fh.write(intent["note"] + "\n\n")
        else:
            fh.write(
                f"- Accuracy: **{intent['accuracy']:.4f}**\n"
            )
            fh.write(
                f"- Macro-F1: **{intent['macro_f1']:.4f}**\n"
            )
            fh.write(
                f"- Weighted-F1: **{intent['weighted_f1']:.4f}**\n\n"
            )

            fh.write("### Per-Intent Metrics\n\n")
            fh.write(
                "| Intent | Precision | Recall | F1 | Support |\n"
            )
            fh.write(
                "|---|---:|---:|---:|---:|\n"
            )

            for label, metrics in intent["per_intent"].items():
                fh.write(
                    f"| {label} | "
                    f"{metrics['precision']:.4f} | "
                    f"{metrics['recall']:.4f} | "
                    f"{metrics['f1']:.4f} | "
                    f"{metrics['support']} |\n"
                )

            fh.write("\n### Confusion Matrix\n\n")

            labels = intent["labels"]
            matrix = intent["confusion_matrix"]

            header = "| Gold \\ Predicted | " + " | ".join(labels) + " |\n"
            separator = "|---|" + "|".join(["---"] * len(labels)) + "|\n"

            fh.write(header)
            fh.write(separator)

            for label, row in zip(labels, matrix):
                fh.write(
                    "| "
                    + label
                    + " | "
                    + " | ".join(str(x) for x in row)
                    + " |\n"
                )

            fh.write("\n")

        fh.write("## 4. Routing\n\n")
        fh.write(
            "- **Not available for this baseline.** "
            "The deterministic heuristic labeler produces intent labels "
            "but does not produce a `should_auto_handle` prediction. "
            "The gold routing labels are preserved for evaluating ANNA's "
            "future decision layer.\n\n"
        )

        disagreements = summary["disagreements"]

        fh.write(
            f"## 5. Intent Disagreements "
            f"({len(disagreements)})\n\n"
        )

        if not disagreements:
            fh.write("No intent disagreements.\n\n")
        else:
            fh.write(
                "| Case | Difficulty | Gold | Predicted | Customer message |\n"
            )
            fh.write(
                "|---|---|---|---|---|\n"
            )

            for record in disagreements:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )

                fh.write(
                    f"| {record['case_id']} | "
                    f"{record['difficulty']} | "
                    f"{record['gold_intent']} | "
                    f"{record['predicted_intent']} | "
                    f"{message} |\n"
                )

            fh.write("\n")

        fh.write("## 6. Unmatched Examples\n\n")

        unmatched = summary["unmatched_examples"]

        if not unmatched:
            fh.write("None.\n\n")
        else:
            fh.write(
                "| Golden tweet | Difficulty |\n"
            )
            fh.write("|---|---|\n")

            for record in unmatched:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )
                fh.write(
                    f"| {message} | {record['difficulty']} |\n"
                )

            fh.write("\n")

        fh.write("## 7. Ambiguous Examples\n\n")

        ambiguous = summary["ambiguous_examples"]

        if not ambiguous:
            fh.write("None.\n")
        else:
            fh.write(
                "| Golden tweet | Candidate case IDs |\n"
            )
            fh.write("|---|---|\n")

            for record in ambiguous:
                message = (
                    record["customer_message"]
                    .replace("\n", " ")
                    .replace("|", "\\|")
                )
                candidates = ", ".join(record["mapping_candidates"])

                fh.write(
                    f"| {message} | {candidates} |\n"
                )



def evaluate_direct_intent(golden_path: Path) -> Dict:
    """Evaluate the production heuristic labeler directly on the human gold set.

    This is the primary intent-classification evaluation.

    IMPORTANT:
    The golden examples are manually authored annotations and are not required
    to be verbatim historical tweets. Therefore this evaluation deliberately
    does NOT reverse-map golden examples to reconstructed historical cases.
    Every golden example is passed directly through the same production
    `label_case` function used by the corpus labeling pipeline.
    """
    golden = load_golden(golden_path)

    records: List[Dict] = []

    for index, gold in enumerate(golden):
        tweet = str(gold.get("tweet", "")).strip()

        # Construct the smallest valid support-case object accepted by the
        # production labeler. This is an in-memory evaluation adapter only;
        # it is never written into the support-case corpus.
        synthetic_case = {
            "case_id": f"golden_eval_{index + 1:04d}",
            "customer_id": f"golden_eval_customer_{index + 1:04d}",
            "customer_messages": [{"text": tweet}],
        }

        prediction = label_case(synthetic_case)

        predicted_intent = prediction.get("intent")
        gold_intent = gold.get("intent")

        records.append(
            {
                "golden_index": index + 1,
                "customer_message": tweet,
                "gold_intent": gold_intent,
                "predicted_intent": predicted_intent,
                "intent_correct": predicted_intent == gold_intent,
                "gold_should_auto_handle": routable_to_bool(
                    gold.get("routable")
                ),
                "predicted_should_auto_handle": None,
                "difficulty": gold.get("difficulty"),
                "expected_resolution": gold.get("resolution"),
                "prediction_confidence": prediction.get(
                    "labeling_confidence", 0.0
                ),
                "prediction_method": prediction.get(
                    "labeling_method", "unknown"
                ),
                "prediction_explanation": prediction.get("explain", {}),
            }
        )

    gold_labels = [r["gold_intent"] for r in records]
    predicted_labels = [r["predicted_intent"] for r in records]

    labels = sorted(set(gold_labels) | set(predicted_labels))

    accuracy = accuracy_score(gold_labels, predicted_labels)

    macro_f1 = f1_score(
        gold_labels,
        predicted_labels,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        gold_labels,
        predicted_labels,
        labels=labels,
        average="weighted",
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        gold_labels,
        predicted_labels,
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

    matrix = confusion_matrix(
        gold_labels,
        predicted_labels,
        labels=labels,
    )

    confusion = {
        "labels": labels,
        "matrix": matrix.tolist(),
    }

    return {
        "total": len(records),
        "evaluated": len(records),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_intent": per_intent,
        "confusion_matrix": confusion,
        "records": records,
    }


def write_direct_intent_report(
    result: Dict,
    out_jsonl: Path,
    out_md: Path,
) -> None:
    """Write the primary direct golden-set intent evaluation."""
    records = result["records"]

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    lines = [
        "# Golden Set Intent Evaluation",
        "",
        "## Evaluation design",
        "",
        (
            "Each manually labelled golden example is passed directly through "
            "the same deterministic production `label_case` function used "
            "for the support-case corpus."
        ),
        "",
        (
            "Historical-case reverse mapping is NOT required for intent "
            "classification accuracy. This avoids selection bias caused by "
            "trying to infer historical provenance for manually authored or "
            "paraphrased golden examples."
        ),
        "",
        "## Dataset",
        "",
        f"- Golden examples: **{result['total']}**",
        f"- Evaluated: **{result['evaluated']} / {result['total']}**",
        "",
        "## Intent metrics",
        "",
        f"- Accuracy: **{result['accuracy']:.4f}**",
        f"- Macro-F1: **{result['macro_f1']:.4f}**",
        f"- Weighted-F1: **{result['weighted_f1']:.4f}**",
        "",
        "## Per-intent performance",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]

    for intent, metrics in result["per_intent"].items():
        lines.append(
            f"| {intent} | "
            f"{metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['support']} |"
        )

    lines.extend(
        [
            "",
            "## Confusion matrix",
            "",
            "Label order:",
            "",
            "```text",
            json.dumps(result["confusion_matrix"]["labels"]),
            "```",
            "",
            "```text",
            json.dumps(result["confusion_matrix"]["matrix"], indent=2),
            "```",
            "",
            "## Automation evaluation",
            "",
            (
                "The golden set contains human `routable` labels, but the "
                "current deterministic intent baseline does not produce an "
                "automation decision. Therefore automation metrics are "
                "intentionally unavailable rather than fabricated."
            ),
            "",
            "## Reply evaluation",
            "",
            (
                "Reply-quality, groundedness, and LLM-as-judge evaluation "
                "are pending implementation of the ANNA retrieval and "
                "generation pipeline."
            ),
            "",
            "## Limitations",
            "",
            "- The current golden set contains 139 examples; the target is approximately 200.",
            "- Rare intents have low support and therefore high metric variance.",
            "- Some support requests are inherently ambiguous and may receive different reasonable labels.",
            "- The deterministic keyword/heuristic classifier is a baseline, not the final ANNA system.",
            "- Historical support data can contain inconsistent or unresolved outcomes.",
            "- Intent classification performance does not establish reply quality.",
            "- Intent classification performance does not establish safe automation.",
            "- Historical provenance is evaluated separately because manually authored golden messages are not guaranteed to be verbatim historical tweets.",
            "",
        ]
    )

    out_md.write_text("\n".join(lines), encoding="utf-8")

def evaluate(
    golden_path: Path,
    cases_path: Path,
    labels_path: Path,
    out_jsonl: Path,
    out_md: Path,
) -> Dict:
    golden = load_golden(golden_path)

    cases_index, case_texts, case_count = build_case_index(cases_path)
    labels_by_case = build_labels_by_case(labels_path)

    records = map_golden_examples(
        golden,
        cases_index,
        labels_by_case,
        case_texts,
    )

    summary = build_summary(records, case_count)

    write_jsonl(records, out_jsonl)
    write_markdown_report(summary, records, out_md)

    return {
        "records_path": str(out_jsonl),
        "report_path": str(out_md),
        "summary": summary,
    }
