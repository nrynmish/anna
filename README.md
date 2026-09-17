# ANNA

ANNA is a brand-grounded AI customer support agent built around historical customer-support cases. The project demonstrates how historical support data can be analyzed, organized, and used to classify customer intent, retrieve relevant past resolutions, generate grounded responses, and decide whether an issue should be auto-handled or escalated to a human operator.

## Current project status

ANNA now includes the end-to-end support-agent pipeline: dataset profiling, conservative support-case reconstruction, intent classification, semantic retrieval, evidence assessment, grounded response generation, deterministic risk-aware routing, and offline evaluation.

## System components

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

The repository does not contain production secrets. Runtime configuration is provided through environment variables where required.

## Development commands

```bash
# Run tests
pytest

# Run linting
ruff check .

# Type checking
mypy src
```

## Notes

- Reported model and evaluation results are generated from saved evaluation artifacts and are not hard-coded.

## Dataset reconnaissance

To profile the raw Kaggle dataset without modifying or committing the source data, run:

```bash
python scripts/profile_dataset.py
```

This script inspects the actual CSV schema, measures quality and conversation structure, and writes profiling artifacts locally under the artifacts directory. The raw Kaggle files remain untouched and uncommitted. Profiling is evidence-based and uses no LLM or fabricated labels.

## Project philosophy

This project is designed to be easy to explain in a live technical interview: a clean Python backend foundation, explicit separation between offline and online work, and a clear path to a grounded support-agent pipeline without premature optimization or unnecessary infrastructure.
