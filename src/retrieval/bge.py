from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import RetrievedCase, RetrievalResult


class BGERetriever:
    """BGE-small semantic retriever over historical Uber support cases."""

    def __init__(
        self,
        corpus_path: str | Path = "artifacts/uber_retrieval_corpus.jsonl",
        embeddings_path: str | Path = "artifacts/uber_bge_embeddings.npy",
        model_name: str = "BAAI/bge-small-en-v1.5",
        device: str | None = None,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.embeddings_path = Path(embeddings_path)

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Retrieval corpus not found: {self.corpus_path}")

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"BGE embeddings not found: {self.embeddings_path}"
            )

        self.model = SentenceTransformer(model_name, device=device)

        with self.corpus_path.open("r", encoding="utf-8") as handle:
            self.corpus = [json.loads(line) for line in handle if line.strip()]

        self.embeddings = np.load(self.embeddings_path)

        if len(self.corpus) != len(self.embeddings):
            raise ValueError(
                f"Corpus/embedding mismatch: "
                f"{len(self.corpus)} cases vs {len(self.embeddings)} embeddings."
            )

        if self.embeddings.ndim != 2:
            raise ValueError(
                f"Expected a 2D embedding matrix, got shape {self.embeddings.shape}."
            )

    def retrieve(self, query: str, top_k: int = 5) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(query=query, results=[])

        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        top_k = min(top_k, len(self.corpus))

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # Corpus embeddings were generated normalized. Dot product therefore
        # equals cosine similarity.
        scores = self.embeddings @ query_embedding

        top_indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = [
            RetrievedCase(
                case_id=self.corpus[index]["case_id"],
                customer_text=self.corpus[index]["customer_text"],
                uber_response=self.corpus[index]["uber_response"],
                final_uber_response=self.corpus[index].get(
                    "final_uber_response",
                    self.corpus[index]["uber_response"],
                ),
                resolution_type=self.corpus[index]["resolution_type"],
                intent=self.corpus[index]["intent"],
                intent_confidence=float(
                    self.corpus[index]["intent_confidence"]
                ),
                similarity=max(-1.0, min(1.0, float(scores[index]))),
                created_at=self.corpus[index].get("created_at"),
            )
            for index in top_indices
        ]

        return RetrievalResult(query=query, results=results)
