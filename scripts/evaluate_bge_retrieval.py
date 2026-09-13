import json
import time
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
CORPUS_PATH = Path("artifacts/uber_retrieval_corpus.jsonl")
GOLDEN_PATH = Path("data/golden/tweets.json")
OUTPUT_PATH = Path("artifacts/bge_retrieval_evaluation.json")
EMBEDDINGS_PATH = Path("artifacts/uber_bge_embeddings.npy")

K_VALUES = [1, 3, 5, 10]


def normalize_gold_intent(intent: str) -> str:
    """Normalize legacy/manual aliases used in the golden set."""
    mapping = {
        "account_access and app_technical": "account_access",
        "driver_behaviour": "driver_behavior",
        "lost_found": "lost_and_found",
        "payout_earnings": "payouts_and_earnings",
        "rides_trips": "rides_and_trips",
    }
    return mapping.get(intent, intent)


def load_corpus():
    records = []

    with CORPUS_PATH.open(encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("customer_text", "").strip()

            if not text:
                continue

            records.append(
                {
                    "case_id": record["case_id"],
                    "customer_text": text,
                    "intent": record.get("intent", "unclear"),
                }
            )

    return records


def load_golden():
    with GOLDEN_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def encode_corpus(model, records):
    texts = [r["customer_text"] for r in records]

    print(f"Encoding {len(texts):,} historical cases...")

    start = time.perf_counter()

    embeddings = model.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    elapsed = time.perf_counter() - start

    print(
        f"Corpus encoding: {elapsed:.2f}s "
        f"({len(texts) / elapsed:.1f} texts/sec)"
    )

    return embeddings.astype(np.float32), elapsed


def encode_queries(model, golden):
    queries = [item["tweet"].strip() for item in golden]

    start = time.perf_counter()

    embeddings = model.encode(
        queries,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    elapsed = time.perf_counter() - start

    print(
        f"Golden query encoding: {elapsed:.3f}s "
        f"({len(queries) / elapsed:.1f} queries/sec)"
    )

    return embeddings.astype(np.float32), elapsed


def evaluate(corpus, corpus_embeddings, golden, query_embeddings):
    corpus_intents = [
        normalize_gold_intent(r["intent"])
        for r in corpus
    ]

    hits = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    details = []

    for index, (gold, query_embedding) in enumerate(
        zip(golden, query_embeddings),
        start=1,
    ):
        gold_intent = normalize_gold_intent(gold["intent"])

        # Because both matrices are normalized, dot product == cosine similarity.
        scores = corpus_embeddings @ query_embedding

        # Only top-10 is required for all metrics.
        top_indices = np.argpartition(
            -scores,
            min(max(K_VALUES), len(scores)) - 1,
        )[: max(K_VALUES)]

        top_indices = top_indices[
            np.argsort(-scores[top_indices])
        ]

        ranked_intents = [corpus_intents[i] for i in top_indices]

        first_relevant_rank = None

        for rank, intent in enumerate(ranked_intents, start=1):
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
                        "similarity": float(scores[i]),
                        "intent": corpus_intents[i],
                    }
                    for rank, i in enumerate(top_indices, start=1)
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
    print("===== ANNA BGE RETRIEVAL EVALUATION =====")
    print("Model:", MODEL_NAME)
    print("CUDA:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    corpus = load_corpus()
    golden = load_golden()

    print(f"Corpus cases: {len(corpus):,}")
    print(f"Golden queries: {len(golden):,}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\nLoading model...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    if EMBEDDINGS_PATH.exists():
        print(f"\nLoading cached embeddings: {EMBEDDINGS_PATH}")
        corpus_embeddings = np.load(EMBEDDINGS_PATH)
        corpus_encode_time = 0.0
    else:
        corpus_embeddings, corpus_encode_time = encode_corpus(
            model,
            corpus,
        )

        np.save(EMBEDDINGS_PATH, corpus_embeddings)

        print(f"Saved embeddings: {EMBEDDINGS_PATH}")

    query_embeddings, query_encode_time = encode_queries(
        model,
        golden,
    )

    print("\nEvaluating retrieval...")
    metrics, details = evaluate(
        corpus,
        corpus_embeddings,
        golden,
        query_embeddings,
    )

    result = {
        "model": MODEL_NAME,
        "device": device,
        "corpus_size": len(corpus),
        "golden_queries": len(golden),
        "embedding_dimension": int(corpus_embeddings.shape[1]),
        "corpus_encoding_seconds": corpus_encode_time,
        "query_encoding_seconds": query_encode_time,
        "metrics": metrics,
        "details": details,
    }

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n===== RESULTS =====")

    for k in K_VALUES:
        print(
            f"Recall@{k}: "
            f"{metrics[f'recall@{k}']:.4f} "
            f"({metrics[f'recall@{k}'] * 100:.2f}%)"
        )

    print(f"MRR: {metrics['mrr']:.4f} ({metrics['mrr'] * 100:.2f}%)")

    print("\n===== ARTIFACTS =====")
    print("Embeddings:", EMBEDDINGS_PATH)
    print("Evaluation:", OUTPUT_PATH)

    free_gb = (
        __import__("shutil")
        .disk_usage(Path.home())
        .free
        / (1024 ** 3)
    )

    print(f"Free disk: {free_gb:.2f} GB")


if __name__ == "__main__":
    main()
