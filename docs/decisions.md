# ANNA Architectural Decision Log

This document records the major engineering decisions made while implementing ANNA. Decisions are based on dataset characteristics, retrieval benchmarks, evaluation results, system constraints, and safety requirements.

## Decision 01 — Focus on one brand

**Status:** accepted

ANNA currently focuses on `Uber_Support` rather than attempting to build a multi-brand system initially.

**Reason:** A single-brand knowledge base keeps the taxonomy, retrieval corpus, historical resolutions, tone, and evaluation benchmark coherent.

## Decision 02 — Use support cases as the retrieval unit

**Status:** accepted

Individual tweets are not treated as independent knowledge documents. Conversations are reconstructed into support cases containing the customer issue, relevant support turns, intent, and resolution metadata.

**Reason:** The useful support knowledge is the relationship between an issue and its support response, not an isolated tweet.

## Decision 03 — Conservative case reconstruction

**Status:** accepted

Case reconstruction prevents cross-customer contamination and preserves legitimate same-customer multi-turn interactions.

**Reason:** Incorrect customer-response associations would create invalid retrieval evidence. Precision of case boundaries is prioritized over maximizing multi-turn coverage.

## Decision 04 — Use a small explicit intent taxonomy

**Status:** accepted

The taxonomy contains 11 primary support intents plus `other` and `unclear`.

**Reason:** A bounded taxonomy is easier to evaluate, audit, and connect to routing policy than an unconstrained intent space.

## Decision 05 — Keep `other` and `unclear` separate

**Status:** accepted

`other` represents an understood request outside the primary taxonomy. `unclear` represents insufficient evidence to determine the intent confidently.

**Reason:** These states have different operational implications.

## Decision 06 — Use weak labels for corpus construction, not as gold truth

**Status:** accepted

The deterministic labeler is used to provide scalable labels for the historical corpus.

**Reason:** The historical corpus is too large for complete manual annotation. A fixed manually reviewed golden set provides the authoritative evaluation signal.

## Decision 07 — Benchmark semantic retrieval against TF-IDF

**Status:** accepted

TF-IDF is retained as a lexical baseline while BGE is used as the primary semantic retriever.

**Evidence:** BGE improved Recall@1, Recall@3, Recall@5, Recall@10, and MRR over TF-IDF on the retrieval benchmark. A naive weighted hybrid did not improve the benchmark.

## Decision 08 — Use BGE-small for embeddings

**Status:** accepted

ANNA uses `BAAI/bge-small-en-v1.5` for semantic retrieval.

**Reason:** It provides a measurable semantic retrieval improvement while remaining practical for local inference.

## Decision 09 — Retrieve cases before generation

**Status:** accepted

The generator receives retrieved historical cases rather than answering from model knowledge alone.

**Reason:** The objective is historical-resolution grounding, and retrieval makes the evidence used for a response explicit.

## Decision 10 — Separate evidence assessment from retrieval similarity

**Status:** accepted

Semantic similarity alone is not considered sufficient evidence for automation.

**Reason:** Similar cases can have different outcomes. Evidence assessment therefore considers similarity, intent agreement, resolution agreement, and precedent strength.

## Decision 11 — Use a local Qwen model

**Status:** accepted

Response generation and LLM judging use a local Qwen2.5-7B-Instruct GGUF model.

**Reason:** Local inference avoids dependence on an external generation API and provides a reproducible evaluation environment within the available hardware constraints.

## Decision 12 — Make routing deterministic

**Status:** accepted

The final `AUTO_HANDLE` / `ESCALATE` decision is made by an explicit policy layer rather than allowing the generation model to make the final decision.

**Reason:** Safety-sensitive routing should be inspectable and reproducible. The model can provide a recommendation, but deterministic gates enforce capability and risk boundaries.

## Decision 13 — Treat high-risk signals as escalation gates

**Status:** accepted

Signals such as account compromise, fraud, legal threats, and safety incidents force escalation.

**Reason:** Historical similarity cannot provide sufficient evidence for high-risk current incidents, especially when ANNA lacks live operational access.

## Decision 14 — Use conservative automation thresholds

**Status:** accepted

The current policy requires minimum intent confidence, retrieval similarity, evidence score, and generation confidence, together with risk and financial-resolution checks.

**Reason:** The routing objective is safe automation rather than maximizing the percentage of messages handled automatically.

## Decision 15 — Keep evaluation independent, reproducible, and safety-oriented

**Status:** accepted

Intent classification, routing safety, response quality, and human agreement are evaluated as separate dimensions. The manually reviewed golden labels remain authoritative, generated artifacts remain outside Git, and routing coverage is reported together with safe auto-handle precision and unsafe auto-handle counts. Failure examples are retained in the evaluation artifacts and final documentation.

**Reason:** A single aggregate metric can hide important trade-offs. Correct intent classification does not imply safe automation, high safety can be achieved by escalating nearly everything, and LLM-as-judge scores should not be treated as a substitute for human review. Keeping evaluation artifacts reproducible and labels independent of model output makes the results auditable.
