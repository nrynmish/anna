# ANNA Architecture

This document describes the implemented high-level architecture of ANNA: a brand-grounded customer-support agent combining intent classification, historical support-case retrieval, evidence assessment, grounded generation, and deterministic risk-aware routing.

## High-level pipeline

customer message
→ preprocessing
→ intent classification
→ historical resolution retrieval
→ grounded response generation
→ risk/automation policy
→ auto-handle or human escalation
→ evaluation

The pipeline is designed around a simple operational pattern:

1. Receive a customer support message.
2. Preprocess and normalize the input for downstream analysis.
3. Predict the support intent from a labeled intent taxonomy.
4. Retrieve historically similar conversations and resolutions from the brand-specific knowledge base.
5. Generate a response grounded in those historical resolutions.
6. Apply a risk and automation policy to determine whether the case is safe to auto-handle or should be escalated.
7. Return the final action with explainability support and evidence.
8. Evaluate the complete behavior using offline and online checks.

## Separation of responsibilities

### Offline data and index construction

This stage is responsible for preparing the dataset and building persistent artifacts used during inference:

- dataset ingestion and validation
- brand selection and filtering
- conversation cleaning and normalization
- intent labeling or taxonomy preparation
- historically similar case retrieval index construction
- embedding generation and vector store preparation
- evaluation set creation and annotation support

This is the domain of the data and retrieval pipelines, and it should happen before live request handling.

### Online inference

This stage handles a single incoming customer message in production-like conditions:

- message intake and preprocessing
- intent classification
- retrieval of matching historical conversations
- grounded response generation
- automation/risk decisioning
- evidence and reason collection
- final recommendation or escalation

The online path should be deterministic, explainable, and easy to trace with metadata for each decision.

### Evaluation

This stage provides quality assurance and operational confidence:

- intent classification evaluation
- retrieval quality measurement
- response quality assessment
- automation safety and escalation calibration
- LLM-as-judge validation against human ratings
- reproducible evaluation runs with a fixed golden set

Evaluation is intentionally separated so the project can validate offline generation quality and online automation behavior without coupling those concerns to the runtime service.

## Planned module boundaries

The repository is organized around the following conceptual layers:

- src/ingestion: raw dataset loading, validation, and preprocessing utilities
- src/intent: intent taxonomy and classification logic
- src/retrieval: historical case search and similarity retrieval
- src/generation: grounded response generation
- src/decision: risk and automation policy
- src/evaluation: evaluation, scoring, and LLM-as-judge workflows

## Design principles

- Ground responses and decisions in historical support resolutions rather than generic model output.
- Prefer explainable evidence chains for every automated action.
- Maintain a clean separation between offline artifact construction and the online inference path.
- Keep evaluation reproducible and inspectable, using a fixed golden benchmark and human comparison sample.
- Avoid hard-coded or fabricated performance claims in the repository.
