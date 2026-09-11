#!/usr/bin/env python3
"""Stream-label all support cases using the deterministic heuristic labeler.

Reads `artifacts/uber_support_cases.jsonl`, labels each case with
`src.intent.labeler.label_case`, and writes labels to
`artifacts/uber_case_intent_labels.jsonl` and a summary markdown.

This script streams input and output line-by-line and is memory-efficient.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from typing import Dict, Any

from src.intent import labeler, taxonomy


def process(input_path: str, output_path: str, summary_path: str, progress_interval: int = 1000) -> Dict[str, Any]:
    start = time.time()
    total = 0
    intents = Counter()
    methods = Counter()
    total_conf = 0.0

    with open(input_path, "r", encoding="utf-8") as inf, open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            line = line.strip()
            if not line:
                continue
            try:
                case = json.loads(line)
            except Exception as e:
                print(f"Skipping invalid JSON line {total+1}: {e}", file=sys.stderr)
                continue

            lbl = labeler.label_case(case)

            # preserve original identifiers
            if "case_id" not in lbl:
                lbl["case_id"] = case.get("case_id")
            if "customer_id" not in lbl:
                lbl["customer_id"] = case.get("customer_id")

            outf.write(json.dumps(lbl, ensure_ascii=False) + "\n")

            total += 1
            intents[lbl.get("intent", "unclear")] += 1
            methods[lbl.get("labeling_method", "heuristic")] += 1
            total_conf += float(lbl.get("labeling_confidence", 0.0) or 0.0)

            if total % progress_interval == 0:
                elapsed = time.time() - start
                print(f"Processed {total} cases in {elapsed:.1f}s...", file=sys.stderr)

    avg_conf = total_conf / total if total else 0.0

    other = intents.get("other", 0)
    unclear = intents.get("unclear", 0)

    # Build summary
    with open(summary_path, "w", encoding="utf-8") as sf:
        sf.write("# Intent labeling summary\n\n")
        sf.write(f"- total_cases: {total}\n")
        sf.write(f"- taxonomy_version: {taxonomy.TAXONOMY_VERSION}\n\n")
        sf.write("- intent counts:\n")
        for intent, count in intents.most_common():
            pct = (count / total * 100) if total else 0.0
            sf.write(f"  - {intent}: {count} ({pct:.2f}%)\n")
        sf.write(f"\n- other: {other} ({(other/total*100) if total else 0.0:.2f}%)\n")
        sf.write(f"- unclear: {unclear} ({(unclear/total*100) if total else 0.0:.2f}%)\n")
        sf.write(f"- avg_labeling_confidence: {avg_conf:.4f}\n\n")
        sf.write("- labeling method counts:\n")
        for method, count in methods.items():
            sf.write(f"  - {method}: {count}\n")

    elapsed = time.time() - start
    print(f"Completed {total} labels in {elapsed:.1f}s", file=sys.stderr)

    return {"total": total, "intents": intents, "avg_conf": avg_conf}


def _main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label all Uber support cases with heuristic intent labeler")
    p.add_argument("--input", default="artifacts/uber_support_cases.jsonl")
    p.add_argument("--output", default="artifacts/uber_case_intent_labels.jsonl")
    p.add_argument("--summary", default="artifacts/uber_case_intent_summary.md")
    p.add_argument("--progress-interval", type=int, default=1000)
    args = p.parse_args(argv)

    process(args.input, args.output, args.summary, progress_interval=args.progress_interval)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
