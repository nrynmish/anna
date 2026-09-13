import json

import numpy as np
import pytest

from src.retrieval.bge import BGERetriever


def test_retrieval_requires_existing_corpus(tmp_path):
    embeddings = tmp_path / "embeddings.npy"
    np.save(embeddings, np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(FileNotFoundError):
        BGERetriever(
            corpus_path=tmp_path / "missing.jsonl",
            embeddings_path=embeddings,
        )


def test_retrieval_detects_corpus_embedding_mismatch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps(
            {
                "case_id": "case_1",
                "customer_text": "test",
                "uber_response": "response",
                "final_uber_response": "response",
                "resolution_type": "unclear",
                "intent": "other",
                "intent_confidence": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    embeddings = tmp_path / "embeddings.npy"
    np.save(
        embeddings,
        np.zeros((2, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="Corpus/embedding mismatch"):
        BGERetriever(
            corpus_path=corpus,
            embeddings_path=embeddings,
        )


def test_retrieval_empty_query_returns_no_results():
    retriever = object.__new__(BGERetriever)
    retriever.corpus = [{"case_id": "x"}]

    result = retriever.retrieve("   ")

    assert result.results == []


def test_top_k_must_be_positive():
    retriever = object.__new__(BGERetriever)
    retriever.corpus = [{"case_id": "x"}]

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve("test", top_k=0)
