# ANNA

ANNA is a brand-grounded AI customer support agent designed for the Hiver SDE Intern take-home assignment. The project is intended to demonstrate how historical support data can be analyzed, organized, and used to classify customer intent, retrieve relevant past resolutions, generate grounded responses, and decide whether an issue should be auto-handled or escalated to a human operator.

## Current project status

This repository is in the foundational setup phase. The project structure, documentation, and configuration have been established, but no AI pipeline, dataset processing, model logic, or frontend implementation has been created yet. The focus for this stage is a clean and interview-friendly foundation that will later support the full end-to-end workflow.

## Planned system components

- Data ingestion and preprocessing
- Brand selection and support-case filtering
- Intent discovery and classification
- Historical retrieval and similarity search
- Grounded response generation
- Risk-aware automation and escalation policy
- Evaluation pipeline with human and LLM-as-judge checks
- Frontend demonstration interface

## Repository structure

```text
anna/
├── README.md
├── .gitignore
├── .env.example
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   └── decisions.md
├── src/
│   ├── ingestion/
│   ├── intent/
│   ├── retrieval/
│   ├── generation/
│   ├── decision/
│   └── evaluation/
├── scripts/
├── tests/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── golden/
│   └── samples/
├── artifacts/
├── app/
│   ├── backend/
│   └── frontend/
└── .gitkeep
```

## Local setup instructions

1. Clone the repository.
2. Create a Python 3.11+ virtual environment.
3. Activate the environment.
4. Install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Environment variable setup

Copy the example environment file and adjust values as needed:

```bash
cp .env.example .env
```

The repository is intentionally minimal at this stage and does not require production secrets yet. Future environment variables will be added when the AI pipeline and runtime service are implemented.

## Development commands

```bash
# Run tests
pytest

# Run linting (when configured in a later stage)
ruff check .

# Type checking (when configured in a later stage)
mypy src
```

## Notes

- The evaluation pipeline will eventually be reproducible in under 15 minutes.
- Actual model and evaluation results will never be hard-coded or fabricated in this repository.

## Dataset reconnaissance

To profile the raw Kaggle dataset without modifying or committing the source data, run:

```bash
python scripts/profile_dataset.py
```

This script inspects the actual CSV schema, measures quality and conversation structure, and writes profiling artifacts locally under the artifacts directory. The raw Kaggle files remain untouched and uncommitted. Profiling is evidence-based and uses no LLM or fabricated labels.

## Project philosophy

This project is designed to be easy to explain in a live technical interview: a clean Python backend foundation, explicit separation between offline and online work, and a clear path to a grounded support-agent pipeline without premature optimization or unnecessary infrastructure.
