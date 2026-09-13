"""Evaluate TF-IDF case retrieval against the human-labelled golden set."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.golden_set import load_golden, normalize_text


CORPUS_PATH = Path("artifacts/uber_retrieval_corpus.jsonl")
GOLDEN_PATH = Path("data/golden/tweets.json")
OUTPUT_PATH = Path("artifacts/tfidf_retrieval_evaluation.json")


def normalize_gold_intent(intent: str) -> str:
    """Map the one compound annotation to a deterministic evaluation label."""
    if intent == "account_access and app_technical":
        return "account_access"
    return intent


def load_corpus():
    records = []

    with CORPUS_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                records.append(json.loads(line))

    return records


def reciprocal_rank(relevant_ranks):
    if not relevant_ranks:
        return 0.0
    return 1.0 / relevant_ranks[0]


def main():
    corpus = load_corpus()
    golden = load_golden(GOLDEN_PATH)

    corpus_texts = [
        normalize_text(record["customer_text"])
        for record in corpus
    ]

    queries = [
        normalize_text(str(item.get("tweet", "")))
        for item in golden
    ]

    gold_intents = [
        normalize_gold_intent(str(item.get("intent")))
        for item in golden
    ]

    print("===== TF-IDF RETRIEVAL =====")
    print(f"Corpus cases: {len(corpus)}")
    print(f"Golden queries: {len(queries)}")

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        max_features=100000,
    )

    matrix = vectorizer.fit_transform(corpus_texts)
    query_matrix = vectorizer.transform(queries)

    similarities = cosine_similarity(query_matrix, matrix)

    ks = [1, 3, 5, 10]
    recall_at_k = {k: [] for k in ks}
    reciprocal_ranks = []
    examples = []

    for index, (gold_intent, scores) in enumerate(
        zip(gold_intents, similarities)
    ):
        ranked_indices = np.argsort(-scores)

        relevant_ranks = [
            rank + 1
            for rank, corpus_index in enumerate(ranked_indices)
            if corpus[corpus_index]["intent"] == gold_intent
        ]

        reciprocal_ranks.append(reciprocal_rank(relevant_ranks))

        for k in ks:
            recall_at_k[k].append(
                1.0
                if any(
                    corpus[corpus_index]["intent"] == gold_intent
                    for corpus_index in ranked_indices[:k]
                )
                else 0.0
            )

        top = ranked_indices[:5]

        examples.append(
            {
                "golden_index": index + 1,
                "query": golden[index].get("tweet"),
                "gold_intent": gold_intent,
                "top_cases": [
                    {
                        "case_id": corpus[i]["case_id"],
                        "intent": corpus[i]["intent"],
                        "resolution_type": corpus[i]["resolution_type"],
                        "similarity": float(scores[i]),
                    }
                    for i in top
                ],
            }
        )

    result = {
        "method": "tfidf_customer_issue_retrieval",
        "corpus_cases": len(corpus),
        "golden_queries": len(golden),
        "intent_normalization": {
            "account_access and app_technical": "account_access",
        },
        "metrics": {
            f"recall_at_{k}": float(np.mean(recall_at_k[k]))
            for k in ks
        },
        "mrr": float(np.mean(reciprocal_ranks)),
        "examples": examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("===== RESULTS =====")

    for k in ks:
        print(
            f"Recall@{k}: "
            f"{result['metrics'][f'recall_at_{k}']:.4f}"
        )

    print(f"MRR:       {result['mrr']:.4f}")
    print()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
