"""Deterministic sample labeling runner for Uber support cases.

Usage (module functions are testable):
- sample_and_label(input_jsonl, sample_size=200, seed=42, out_jsonl, out_summary)

This script reads `artifacts/uber_support_cases.jsonl`, samples a deterministic
subset, applies the heuristic labeler from `src.intent.labeler`, and writes
the sample labels and a small summary. Sampling seed is fixed and documented.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from typing import List, Dict, Any

# Ensure repository root is on sys.path so `src` imports resolve when
# executing `python scripts/label_sample.py` from the repo root (or elsewhere).
import sys
import os
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.intent import labeler, taxonomy

# Documented deterministic seed
SAMPLE_SEED = 42
DEFAULT_SAMPLE_SIZE = 200


def load_cases(input_path: str) -> List[Dict[str, Any]]:
    cases = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def sample_case_indices(num_cases: int, sample_size: int, seed: int) -> List[int]:
    import random

    rnd = random.Random(seed)
    # deterministic sample of indices without replacement
    if sample_size >= num_cases:
        return list(range(num_cases))
    return rnd.sample(list(range(num_cases)), sample_size)


def build_label_record(label: Dict[str, Any]) -> Dict[str, Any]:
    # ensure minimal fields
    return {
        "case_id": label.get("case_id"),
        "customer_id": label.get("customer_id"),
        "intent": label.get("intent"),
        "labeling_method": label.get("labeling_method"),
        "labeling_confidence": label.get("labeling_confidence"),
        "taxonomy_version": label.get("taxonomy_version"),
        "explain": label.get("explain"),
    }


def sample_and_label(input_jsonl: str, out_jsonl: str, out_summary: str, sample_size: int = DEFAULT_SAMPLE_SIZE, seed: int = SAMPLE_SEED) -> List[Dict[str, Any]]:
    cases = load_cases(input_jsonl)
    num = len(cases)
    indices = sample_case_indices(num, sample_size, seed)
    indices = sorted(indices)

    labels = []
    for i in indices:
        case = cases[i]
        # prepare minimal expected shape for labeler
        # support both nested structures and flat
        customer_messages = case.get("customer_messages") or []
        # preserve case_id/customer_id if present, else fall back
        input_case = {
            "case_id": case.get("case_id") or case.get("uber_case_id") or f"idx_{i}",
            "customer_id": case.get("customer") or case.get("customer_id") or case.get("customer_id_str"),
            "customer_messages": customer_messages,
        }
        lbl = labeler.label_case(input_case)
        record = build_label_record(lbl)
        labels.append(record)

    # write JSONL
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in labels:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # build summary
    intents = [r["intent"] for r in labels]
    counts = Counter(intents)
    avg_conf = statistics.mean([r["labeling_confidence"] for r in labels]) if labels else 0.0

    with open(out_summary, "w", encoding="utf-8") as f:
        f.write(f"Sample seed: {seed}\n")
        f.write(f"Sample size: {len(labels)}\n")
        f.write("\nIntent counts:\n")
        for intent, cnt in counts.most_common():
            pct = cnt / len(labels) * 100
            f.write(f"- {intent}: {cnt} ({pct:.1f}%)\n")
        f.write(f"\nNumber labeled as other: {counts.get('other', 0)}\n")
        f.write(f"Number labeled as unclear: {counts.get('unclear', 0)}\n")
        f.write(f"Average confidence: {avg_conf:.3f}\n")
        f.write("\nLabeling methods:\n")
        methods = Counter(r.get("labeling_method") for r in labels)
        for m, c in methods.items():
            f.write(f"- {m}: {c}\n")

        # representative examples (first 10)
        f.write("\nRepresentative examples:\n")
        f.write("case_id | customer_text | predicted_intent | confidence | explain\n")
        f.write("---|---|---|---|---\n")
        for r in labels[:10]:
            cid = r.get("case_id")
            # find original text from the loaded cases by matching case_id
            # fallback to empty
            orig = ""
            # search in sampled cases list
            # Note: this is O(n) but sample is small
            for i in indices:
                c = cases[i]
                if (c.get("case_id") or c.get("uber_case_id") or f"idx_{i}") == r.get("case_id"):
                    orig_msgs = c.get("customer_messages") or []
                    if isinstance(orig_msgs, list):
                        if orig_msgs and isinstance(orig_msgs[0], dict):
                            orig = " ".join(m.get("text", "") for m in orig_msgs)
                        else:
                            orig = " ".join(orig_msgs)
                    else:
                        orig = str(orig_msgs)
                    break
            explain = r.get("explain") or {}
            f.write(f"{cid} | {orig} | {r.get('intent')} | {r.get('labeling_confidence'):.2f} | {explain}\n")

    return labels


if __name__ == "__main__":
    import os

    INPUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "uber_support_cases.jsonl")
    INPUT = os.path.normpath(INPUT)
    OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sample_labels.jsonl")
    OUT_SUM = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sample_label_summary.md")
    sample_and_label(INPUT, OUT, OUT_SUM, sample_size=DEFAULT_SAMPLE_SIZE, seed=SAMPLE_SEED)
