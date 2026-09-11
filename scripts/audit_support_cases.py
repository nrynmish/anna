#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = PROJECT_ROOT / "artifacts" / "uber_support_cases.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "uber_support_case_audit.md"

ISSUE_KEYWORDS = {
    "billing": ["charge", "refund", "invoice", "payment", "billing", "charged", "refunds"],
    "account": ["login", "password", "account", "email", "profile", "security"],
    "trip": ["trip", "ride", "pickup", "driver", "fare", "route", "destination"],
    "driver": ["driver", "driver rating", "driver behavior", "driver cancellation"],
    "safety": ["unsafe", "harassment", "safety", "report", "incident"],
    "support": ["support", "help", "issue", "problem", "bug", "question"],
}

OBVIOUS_UNRELATED_TERMS = {
    "lottery",
    "casino",
    "politics",
    "sports",
    "weather",
    "dating",
    "job",
    "resume",
    "celebrity",
    "stock",
    "crypto",
}


def normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value)
    text = text.strip().lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_datetime(value: Any) -> datetime | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = pd.to_datetime(text, format="%a %b %d %H:%M:%S %z %Y", errors="raise")
        if pd.isna(parsed):
            return None
        dt = parsed.to_pydatetime()
        return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        try:
            parsed = pd.to_datetime(text, errors="raise")
            if pd.isna(parsed):
                return None
            dt = parsed.to_pydatetime()
            return dt.astimezone(timezone.utc) if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            return None


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return cases


def get_case_messages(case: dict[str, Any]) -> list[dict[str, Any]]:
    full_thread = case.get("full_thread", [])
    if isinstance(full_thread, list):
        return full_thread
    return []


def message_texts(messages: list[dict[str, Any]]) -> list[str]:
    return [normalize_text(msg.get("text", "")) for msg in messages if isinstance(msg, dict)]


def message_topics(text: str) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return set()
    hits: set[str] = set()
    for topic, keywords in ISSUE_KEYWORDS.items():
        if any(keyword in norm for keyword in keywords):
            hits.add(topic)
    return hits


def compute_case_quality(case: dict[str, Any]) -> dict[str, Any]:
    messages = get_case_messages(case)
    customer_messages = case.get("customer_messages", [])
    uber_messages = case.get("uber_messages", [])
    customer_texts = message_texts(customer_messages)
    uber_texts = message_texts(uber_messages)
    all_texts = message_texts(messages)

    flagged_reasons: list[str] = []
    topic_sets = [message_topics(text) for text in all_texts if text]
    unique_topics = set().union(*topic_sets) if topic_sets else set()
    if len(unique_topics) > 2:
        flagged_reasons.append("multiple distinct issue topics")

    timestamps = []
    for msg in messages:
        if isinstance(msg, dict):
            ts = parse_datetime(msg.get("created_at"))
            if ts is not None:
                timestamps.append(ts)
    if len(timestamps) >= 2:
        sorted_ts = sorted(timestamps)
        gaps = [
            (sorted_ts[i + 1] - sorted_ts[i]).total_seconds() / 3600.0
            for i in range(len(sorted_ts) - 1)
        ]
        if any(gap > 72 for gap in gaps):
            flagged_reasons.append("long gap between messages")

    customer_starts = 0
    for text in customer_texts:
        if not text:
            continue
        if text.startswith("hi") or text.startswith("hello") or text.startswith("hey"):
            customer_starts += 1
    if len(customer_messages) >= 3 and customer_starts >= 2:
        flagged_reasons.append("repeated customer starts")

    unrelated_terms = set()
    for text in all_texts:
        if not text:
            continue
        chunks = text.split()
        for word in chunks:
            if word in OBVIOUS_UNRELATED_TERMS:
                unrelated_terms.add(word)
    if unrelated_terms:
        flagged_reasons.append("obvious unrelated issue terms")

    if not uber_messages:
        flagged_reasons.append("no Uber response")
    if customer_messages and not uber_messages:
        flagged_reasons.append("customer messages but no support interaction")

    return {
        "case_id": case.get("case_id"),
        "customer_id": case.get("customer_id"),
        "customer_message_count": len(customer_messages),
        "uber_message_count": len(uber_messages),
        "reasons": flagged_reasons,
        "topics": sorted(unique_topics),
    }


def summarize_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {
            "total_cases": 0,
            "unique_case_ids": 0,
            "unique_customer_ids": 0,
            "duplicate_tweet_ids_across_cases": [],
            "cases_with_multiple_customer_ids": 0,
            "cases_with_misaddressed_uber_messages": 0,
            "customer_message_count_distribution": {},
            "uber_message_count_distribution": {},
            "single_turn_cases": 0,
            "multi_turn_cases": 0,
            "resolution_type_distribution": {},
            "cases_with_no_uber_response": 0,
            "cases_with_customer_messages_and_no_apparent_support_interaction": 0,
            "suspicious_cases": [],
        }

    case_ids = [case.get("case_id") for case in cases if case.get("case_id") is not None]
    customer_ids = [case.get("customer_id") for case in cases if case.get("customer_id") is not None]

    tweet_ids_seen: defaultdict[str, list[str]] = defaultdict(list)
    for case in cases:
        for msg in get_case_messages(case):
            tweet_id = msg.get("tweet_id")
            if tweet_id:
                tweet_ids_seen[str(tweet_id)].append(case.get("case_id", "unknown"))

    duplicate_tweet_ids = sorted(
        [tweet_id for tweet_id, case_ids in tweet_ids_seen.items() if len(case_ids) > 1]
    )

    cases_with_multiple_customer_ids = 0
    for case in cases:
        thread_customer_ids = {msg.get("author_id") for msg in get_case_messages(case) if msg.get("author_id")}
        if len(thread_customer_ids - {case.get("customer_id"), "Uber_Support"}) > 0:
            cases_with_multiple_customer_ids += 1

    misaddressed_cases = 0
    for case in cases:
        full_thread = get_case_messages(case)
        customer_id = case.get("customer_id")
        if not customer_id:
            continue
        thread_authors = {msg.get("tweet_id"): msg.get("author_id") for msg in full_thread if msg.get("tweet_id") and msg.get("author_id")}
        customer_tweet_ids = {msg.get("tweet_id") for msg in case.get("customer_messages", []) if msg.get("tweet_id")}
        for msg in case.get("uber_messages", []):
            parent_id = msg.get("parent_tweet_id")
            text = normalize_text(msg.get("text", ""))
            if parent_id is not None:
                parent_author = thread_authors.get(parent_id)
                if parent_id in thread_authors and parent_author != customer_id:
                    misaddressed_cases += 1
                    break
                if parent_id not in customer_tweet_ids and parent_id not in thread_authors and "@" + customer_id.lower() not in text:
                    misaddressed_cases += 1
                    break
            elif "@" + customer_id.lower() not in text:
                misaddressed_cases += 1
                break

    customer_message_counts = Counter(len(case.get("customer_messages", [])) for case in cases)
    uber_message_counts = Counter(len(case.get("uber_messages", [])) for case in cases)
    resolution_counts = Counter(case.get("resolution_type", "unclear") for case in cases)

    single_turn_cases = sum(1 for case in cases if len(case.get("customer_messages", [])) == 1 and len(case.get("uber_messages", [])) == 1)
    multi_turn_cases = len(cases) - single_turn_cases

    no_uber_response = sum(1 for case in cases if not case.get("uber_messages"))
    no_apparent_support = sum(
        1
        for case in cases
        if case.get("customer_messages") and not case.get("uber_messages")
    )

    suspicious = []
    for case in cases:
        quality = compute_case_quality(case)
        if quality["reasons"]:
            suspicious.append(
                {
                    "case_id": quality["case_id"],
                    "customer_id": quality["customer_id"],
                    "customer_messages": case.get("customer_messages", []),
                    "uber_responses": case.get("uber_messages", []),
                    "reasons": quality["reasons"],
                }
            )

    suspicious = sorted(suspicious, key=lambda item: (item["case_id"] or ""))[:20]

    return {
        "total_cases": len(cases),
        "unique_case_ids": len(set(case_ids)),
        "unique_customer_ids": len(set(customer_ids)),
        "duplicate_tweet_ids_across_cases": duplicate_tweet_ids,
        "cases_with_multiple_customer_ids": cases_with_multiple_customer_ids,
        "cases_with_misaddressed_uber_messages": misaddressed_cases,
        "customer_message_count_distribution": dict(sorted(customer_message_counts.items())),
        "uber_message_count_distribution": dict(sorted(uber_message_counts.items())),
        "single_turn_cases": single_turn_cases,
        "multi_turn_cases": multi_turn_cases,
        "resolution_type_distribution": dict(sorted(resolution_counts.items())),
        "cases_with_no_uber_response": no_uber_response,
        "cases_with_customer_messages_and_no_apparent_support_interaction": no_apparent_support,
        "suspicious_cases": suspicious,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Uber support case audit")
    lines.append("")
    lines.append(f"- Total cases: {summary['total_cases']}")
    lines.append(f"- Unique case IDs: {summary['unique_case_ids']}")
    lines.append(f"- Unique customer IDs: {summary['unique_customer_ids']}")
    lines.append(f"- Duplicate tweet IDs across cases: {len(summary['duplicate_tweet_ids_across_cases'])}")
    lines.append(f"- Cases containing multiple customer IDs: {summary['cases_with_multiple_customer_ids']}")
    lines.append(f"- Cases containing Uber messages that appear misaddressed: {summary['cases_with_misaddressed_uber_messages']}")
    lines.append(f"- Single-turn cases: {summary['single_turn_cases']}")
    lines.append(f"- Multi-turn cases: {summary['multi_turn_cases']}")
    lines.append(f"- Cases with no Uber response: {summary['cases_with_no_uber_response']}")
    lines.append(f"- Cases with customer messages but no apparent support interaction: {summary['cases_with_customer_messages_and_no_apparent_support_interaction']}")
    lines.append("")
    lines.append("## Customer-message count distribution")
    lines.append("")
    for count, total in summary["customer_message_count_distribution"].items():
        lines.append(f"- {count}: {total}")
    lines.append("")
    lines.append("## Uber-message count distribution")
    lines.append("")
    for count, total in summary["uber_message_count_distribution"].items():
        lines.append(f"- {count}: {total}")
    lines.append("")
    lines.append("## Resolution type distribution")
    lines.append("")
    for key, total in summary["resolution_type_distribution"].items():
        lines.append(f"- {key}: {total}")
    lines.append("")
    lines.append("## Potentially suspicious cases")
    lines.append("")
    if not summary["suspicious_cases"]:
        lines.append("No suspicious cases were flagged by the conservative audit heuristics.")
    else:
        for item in summary["suspicious_cases"]:
            lines.append(f"### {item['case_id']} | customer={item['customer_id']}")
            lines.append(f"- customer messages: {len(item['customer_messages'])}")
            lines.append(f"- Uber responses: {len(item['uber_responses'])}")
            lines.append(f"- heuristic flags: {', '.join(item['reasons'])}")
            for msg in item["customer_messages"][:3]:
                lines.append(f"  - customer: {msg.get('tweet_id')} | {msg.get('text', '')[:180]}")
            for msg in item["uber_responses"][:3]:
                lines.append(f"  - Uber: {msg.get('tweet_id')} | {msg.get('text', '')[:180]}")
            lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    cases = load_cases(CASE_PATH)
    summary = summarize_cases(cases)
    report = render_markdown(summary)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
