import json
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


CORPUS_PATH = Path("artifacts/uber_retrieval_corpus.jsonl")
GOLDEN_PATH = Path("data/golden/tweets.json")
BGE_EMBEDDINGS_PATH = Path("artifacts/uber_bge_embeddings.npy")
OUTPUT_PATH = Path("artifacts/hybrid_retrieval_evaluation.json")

K_VALUES = [1, 3, 5, 10]
BGE_WEIGHTS = [0.25, 0.50, 0.75]
CANDIDATE_K = 20


def normalize_text(text: str) -> str:
    import re

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_gold_intent(intent: str) -> str:
    mapping = {
        "account_access and app_technical": "account_access",
        "driver_behaviour": "driver_behavior",
        "lost_found": "lost_and_found",
        "payout_earnings": "payouts_and_earnings",
        "rides_trips": "rides_and_trips",
    }
    return mapping.get(intent, intent)


def minmax_normalize(scores):
    scores = np.asarray(scores, dtype=np.float32)

    minimum = scores.min()
    maximum = scores.max()

    if maximum - minimum < 1e-8:
        return np.zeros_like(scores)

    return (scores - minimum) / (maximum - minimum)


def evaluate_configuration(
    corpus,
    golden,
    tfidf_matrix,
    tfidf_vectorizer,
    bge_embeddings,
    bge_weight,
):
    tfidf_weight = 1.0 - bge_weight

    corpus_intents = [
        normalize_gold_intent(r["intent"])
        for r in corpus
    ]

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []
    details = []

    for index, gold in enumerate(golden, start=1):
        query = normalize_text(gold["tweet"])
        gold_intent = normalize_gold_intent(gold["intent"])

        # Lexical scores.
        query_tfidf = tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(
            query_tfidf,
            tfidf_matrix,
        )[0]

        # Dense scores.
        query_bge = bge_embeddings[index - 1]

        bge_scores = bge_embeddings @ query_bge

        # Candidate pool = union of top-20 from each retriever.
        tfidf_candidates = np.argpartition(
            -tfidf_scores,
            min(CANDIDATE_K, len(tfidf_scores)) - 1,
        )[:CANDIDATE_K]

        bge_candidates = np.argpartition(
            -bge_scores,
            min(CANDIDATE_K, len(bge_scores)) - 1,
        )[:CANDIDATE_K]

        candidate_indices = np.unique(
            np.concatenate(
                [tfidf_candidates, bge_candidates]
            )
        )

        # Normalize scores only within the candidate pool.
        candidate_tfidf = minmax_normalize(
            tfidf_scores[candidate_indices]
        )
        candidate_bge = minmax_normalize(
            bge_scores[candidate_indices]
        )

        hybrid_scores = (
            tfidf_weight * candidate_tfidf
            + bge_weight * candidate_bge
        )

        order = np.argsort(-hybrid_scores)
        ranked_indices = candidate_indices[order]

        ranked_intents = [
            corpus_intents[i]
            for i in ranked_indices
        ]

        first_relevant_rank = None

        for rank, intent in enumerate(
            ranked_intents,
            start=1,
        ):
            if intent == gold_intent:
                first_relevant_rank = rank
                break

        for k in K_VALUES:
            if gold_intent in ranked_intents[:k]:
                hits[k] += 1

        reciprocal_ranks.append(
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        details.append(
            {
                "golden_index": index,
                "gold_intent": gold_intent,
                "query": gold["tweet"],
                "first_relevant_rank": first_relevant_rank,
                "top_cases": [
                    {
                        "rank": rank,
                        "case_id": corpus[i]["case_id"],
                        "intent": corpus_intents[i],
                        "tfidf_score": float(
                            tfidf_scores[i]
                        ),
                        "bge_score": float(
                            bge_scores[i]
                        ),
                        "hybrid_score": float(
                            hybrid_scores[
                                np.where(
                                    candidate_indices == i
                                )[0][0]
                            ]
                        ),
                    }
                    for rank, i in enumerate(
                        ranked_indices[:10],
                        start=1,
                    )
                ],
            }
        )

    total = len(golden)

    metrics = {
        f"recall@{k}": hits[k] / total
        for k in K_VALUES
    }

    metrics["mrr"] = float(np.mean(reciprocal_ranks))

    return metrics, details


def main():
    print("===== ANNA HYBRID RETRIEVAL =====")

    start = time.perf_counter()

    with CORPUS_PATH.open(encoding="utf-8") as f:
        corpus = [
            json.loads(line)
            for line in f
        ]

    with GOLDEN_PATH.open(encoding="utf-8") as f:
        golden = json.load(f)

    bge_embeddings = np.load(BGE_EMBEDDINGS_PATH)

    corpus_texts = [
        normalize_text(r["customer_text"])
        for r in corpus
    ]

    golden_texts = [
        normalize_text(r["tweet"])
        for r in golden
    ]

    print(f"Corpus: {len(corpus):,}")
    print(f"Golden: {len(golden):,}")
    print(
        "BGE embeddings:",
        bge_embeddings.shape,
    )

    print("\nBuilding TF-IDF matrix...")

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
    )

    tfidf_matrix = vectorizer.fit_transform(
        corpus_texts
    )

    print(
        "TF-IDF matrix:",
        tfidf_matrix.shape,
    )

    print("\nEncoding golden queries with TF-IDF...")
    query_tfidf = vectorizer.transform(
        golden_texts
    )

    all_results = {}

    for bge_weight in BGE_WEIGHTS:
        print(
            f"\n===== BGE WEIGHT {bge_weight:.2f} "
            f"/ TF-IDF WEIGHT {1-bge_weight:.2f} ====="
        )

        metrics, details = evaluate_configuration(
            corpus=corpus,
            golden=golden,
            tfidf_matrix=tfidf_matrix,
            tfidf_vectorizer=vectorizer,
            bge_embeddings=bge_embeddings,
            bge_weight=bge_weight,
        )

        key = f"bge_{bge_weight:.2f}"

        all_results[key] = {
            "bge_weight": bge_weight,
            "tfidf_weight": 1.0 - bge_weight,
            "metrics": metrics,
            "details": details,
        }

        for k in K_VALUES:
            print(
                f"Recall@{k}: "
                f"{metrics[f'recall@{k}']:.4f} "
                f"({metrics[f'recall@{k}'] * 100:.2f}%)"
            )

        print(
            f"MRR: {metrics['mrr']:.4f} "
            f"({metrics['mrr'] * 100:.2f}%)"
        )

    elapsed = time.perf_counter() - start

    output = {
        "method": "hybrid_tfidf_bge",
        "corpus_cases": len(corpus),
        "golden_queries": len(golden),
        "candidate_k_per_retriever": CANDIDATE_K,
        "normalization": (
            "per-query min-max normalization "
            "within the union candidate pool"
        ),
        "results": all_results,
        "runtime_seconds": elapsed,
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== SAVED =====")
    print(OUTPUT_PATH)
    print(f"Runtime: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
