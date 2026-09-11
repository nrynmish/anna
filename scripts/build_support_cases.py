#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_BRAND = "Uber_Support"
DEFAULT_CHUNK_SIZE = 250_000


@dataclass
class SupportCase:
    case_id: str
    customer_id: str
    customer_messages: list[dict[str, Any]] = field(default_factory=list)
    uber_messages: list[dict[str, Any]] = field(default_factory=list)
    full_thread: list[dict[str, Any]] = field(default_factory=list)
    created_at: str | None = None
    customer_message_count: int = 0
    uber_message_count: int = 0
    final_uber_response: dict[str, Any] | None = None
    resolution_type: str = "unclear"


def normalize_text(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value)
    text = text.strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
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


def infer_resolution_type(final_uber_text: str) -> str:
    text = normalize_text(final_uber_text)
    if not text:
        return "unclear"
    if any(token in text for token in ["dm", "direct message", "private message", "message us", "send us a dm"]):
        return "redirected_to_dm"
    if any(token in text for token in ["please provide", "please send", "can you share", "need more info", "need additional info", "send us", "provide your", "tell us more"]):
        return "requested_more_information"
    if any(token in text for token in ["here is", "we have", "your trip", "your account", "tracking", "status", "information", "available here"]):
        return "provided_information"
    if any(token in text for token in ["resolved", "fixed", "apologize", "sorry", "happy to help", "thanks for reaching out", "issue resolved", "we have resolved"]):
        return "apparent_resolution"
    return "unclear"


def _thread_sort_key(tweet_id: str, tweet_rows: dict[str, dict[str, Any]]) -> tuple[datetime, str]:
    row = tweet_rows.get(tweet_id, {})
    created_at = row.get("created_at")
    timestamp = parse_datetime(created_at) or datetime.fromtimestamp(0, tz=timezone.utc)
    return timestamp, tweet_id


def split_thread_issue_components(thread_ids: list[str], tweet_rows: dict[str, dict[str, Any]]) -> list[list[str]]:
    if not thread_ids:
        return []
    thread_set = set(thread_ids)
    adjacency: defaultdict[str, set[str]] = defaultdict(set)
    for tweet_id in thread_ids:
        row = tweet_rows.get(tweet_id)
        if row is None:
            continue
        parent_id = row.get("in_response_to_tweet_id")
        if parent_id in thread_set:
            adjacency[tweet_id].add(parent_id)
            adjacency[parent_id].add(tweet_id)

    ordered_ids = sorted(thread_ids, key=lambda tweet_id: _thread_sort_key(tweet_id, tweet_rows))
    components: list[list[str]] = []
    seen: set[str] = set()
    for tweet_id in ordered_ids:
        if tweet_id in seen:
            continue
        stack = [tweet_id]
        component: list[str] = []
        while stack:
            current_id = stack.pop()
            if current_id in seen:
                continue
            seen.add(current_id)
            component.append(current_id)
            for neighbor_id in sorted(adjacency.get(current_id, set()), key=lambda item: _thread_sort_key(item, tweet_rows), reverse=True):
                if neighbor_id in thread_set and neighbor_id not in seen:
                    stack.append(neighbor_id)
        components.append(sorted(component, key=lambda item: _thread_sort_key(item, tweet_rows)))
    return components


def validate_case_invariants(cases: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    seen_tweet_ids: set[str] = set()

    for index, case in enumerate(cases, start=1):
        case_id = case.get("case_id", f"case_{index:04d}")
        customer_id = case.get("customer_id")
        full_thread = case.get("full_thread", [])
        customer_messages = case.get("customer_messages", [])
        uber_messages = case.get("uber_messages", [])

        if not customer_messages or not uber_messages or not full_thread:
            issues.append(f"{case_id}:empty_case")
        if not customer_id:
            issues.append(f"{case_id}:missing_customer_id")

        thread_ids = [msg["tweet_id"] for msg in full_thread]
        if len(thread_ids) != len(set(thread_ids)):
            issues.append(f"{case_id}:duplicate_tweet_ids_in_case")

        for tweet_id in thread_ids:
            if tweet_id in seen_tweet_ids:
                issues.append(f"{case_id}:duplicate_tweet_id_across_cases:{tweet_id}")
            seen_tweet_ids.add(tweet_id)

        thread_author_ids = {msg["author_id"] for msg in full_thread}
        if customer_id and thread_author_ids - {customer_id, TARGET_BRAND}:
            issues.append(f"{case_id}:mixed_customer_ids")

        for msg in full_thread:
            parent_id = msg.get("parent_tweet_id")
            if parent_id is None:
                continue
            if parent_id in thread_ids:
                parent_row = next((item for item in full_thread if item.get("tweet_id") == parent_id), None)
                if parent_row is not None and parent_row.get("author_id") not in {customer_id, TARGET_BRAND}:
                    issues.append(f"{case_id}:invalid_internal_parent_child:{msg['tweet_id']}->{parent_id}")

        if customer_id:
            customer_tweet_ids = {msg["tweet_id"] for msg in customer_messages}
            thread_authors = {msg["tweet_id"]: msg["author_id"] for msg in full_thread if "tweet_id" in msg and "author_id" in msg}
            for msg in uber_messages:
                parent_id = msg.get("parent_tweet_id")
                text = str(msg.get("text", ""))
                directly_addresses_customer = bool(f"@{customer_id.lower()}" in text.lower())
                if parent_id is not None:
                    parent_author = thread_authors.get(parent_id)
                    if parent_id in thread_authors and parent_author != customer_id:
                        issues.append(f"{case_id}:misaddressed_uber_response:{msg['tweet_id']}")
                    elif parent_id not in customer_tweet_ids and parent_id not in thread_ids and not directly_addresses_customer:
                        issues.append(f"{case_id}:misaddressed_uber_response:{msg['tweet_id']}")
                elif not directly_addresses_customer:
                    issues.append(f"{case_id}:misaddressed_uber_response:{msg['tweet_id']}")

    return sorted(set(issues))


def build_tweet_index(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict[str, dict[str, Any]]:
    tweet_rows: dict[str, dict[str, Any]] = {}
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        required = {"tweet_id", "author_id", "inbound", "in_response_to_tweet_id", "response_tweet_id", "created_at", "text"}
        if not required.issubset(chunk.columns):
            continue
        subset = chunk.copy()
        subset = subset.loc[
            subset["author_id"].astype(str).str.lower().eq(TARGET_BRAND.lower())
            | subset["text"].fillna("").astype(str).str.lower().str.contains(f"@{TARGET_BRAND.lower()}")
            | subset["inbound"].astype(str).str.lower().eq("true")
        ].copy()
        if subset.empty:
            continue

        subset["tweet_id"] = subset["tweet_id"].astype(str)
        subset["author_id"] = subset["author_id"].astype(str)
        subset["inbound"] = subset["inbound"].astype(str).str.lower().eq("true")
        subset["in_response_to_tweet_id"] = pd.to_numeric(subset["in_response_to_tweet_id"], errors="coerce")
        subset["response_tweet_id"] = pd.to_numeric(subset["response_tweet_id"], errors="coerce")
        subset["text"] = subset["text"].fillna("").astype(str)
        subset["created_at"] = subset["created_at"].fillna("").astype(str)

        for row in subset.itertuples(index=False):
            tweet_id = str(row.tweet_id)
            author_id = str(row.author_id)
            payload = {
                "tweet_id": tweet_id,
                "author_id": author_id,
                "inbound": bool(row.inbound),
                "created_at": row.created_at,
                "text": str(row.text),
                "in_response_to_tweet_id": None if pd.isna(row.in_response_to_tweet_id) else str(int(row.in_response_to_tweet_id)),
                "response_tweet_id": None if pd.isna(row.response_tweet_id) else str(int(row.response_tweet_id)),
            }
            tweet_rows[tweet_id] = payload
    return tweet_rows


def build_child_map(tweet_rows: dict[str, dict[str, Any]]) -> dict[str, list[str]]:
    child_map: defaultdict[str, list[str]] = defaultdict(list)
    for row in tweet_rows.values():
        parent_id = row["in_response_to_tweet_id"]
        if parent_id and parent_id in tweet_rows:
            child_map[parent_id].append(row["tweet_id"])
    for parent_id in child_map:
        child_map[parent_id] = sorted(child_map[parent_id], key=lambda tweet_id: (parse_datetime(tweet_rows[tweet_id]["created_at"]) or datetime.fromtimestamp(0, tz=timezone.utc), tweet_id))
    return child_map


def expand_customer_chain(root_id: str, tweet_rows: dict[str, dict[str, Any]], child_map: dict[str, list[str]]) -> list[str]:
    root_row = tweet_rows.get(root_id)
    if root_row is None:
        return []
    customer_id = root_row["author_id"]
    allowed_authors = {customer_id, TARGET_BRAND}
    visited: set[str] = set()
    queue: deque[str] = deque([root_id])
    while queue:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        visited.add(current_id)
        current_row = tweet_rows.get(current_id)
        if current_row is None:
            continue

        for child_id in child_map.get(current_id, []):
            child_row = tweet_rows.get(child_id)
            if child_row is None:
                continue
            child_author = child_row["author_id"]
            if child_author not in allowed_authors:
                continue
            if current_row["author_id"] == customer_id and child_author == TARGET_BRAND:
                queue.append(child_id)
            elif current_row["author_id"] == TARGET_BRAND and child_author == customer_id:
                queue.append(child_id)
            elif current_row["author_id"] == customer_id and child_author == customer_id:
                queue.append(child_id)

        if current_id != root_id:
            parent_id = current_row["in_response_to_tweet_id"]
            if parent_id and parent_id in tweet_rows:
                parent_row = tweet_rows[parent_id]
                parent_author = parent_row["author_id"]
                if parent_author in allowed_authors and parent_id not in visited:
                    if current_row["author_id"] == customer_id and parent_author == TARGET_BRAND:
                        queue.append(parent_id)
                    elif current_row["author_id"] == TARGET_BRAND and parent_author == customer_id:
                        queue.append(parent_id)

    return sorted(visited, key=lambda tweet_id: (parse_datetime(tweet_rows[tweet_id]["created_at"]) or datetime.fromtimestamp(0, tz=timezone.utc), tweet_id))


def build_cases(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict[str, Any]]:
    tweet_rows = build_tweet_index(path, chunk_size=chunk_size)
    child_map = build_child_map(tweet_rows)
    root_candidates: list[str] = []
    for tweet_id, row in tweet_rows.items():
        author_id = row["author_id"]
        text = row["text"]
        if row["inbound"] and author_id != TARGET_BRAND and f"@{TARGET_BRAND.lower()}" in text.lower():
            root_candidates.append(tweet_id)

    candidates: list[dict[str, Any]] = []
    seen_signatures: set[tuple[str, ...]] = set()
    for root_id in sorted(root_candidates, key=lambda tweet_id: (_thread_sort_key(tweet_id, tweet_rows)[0], tweet_id)):
        root_row = tweet_rows.get(root_id)
        if root_row is None:
            continue
        customer_id = root_row["author_id"]
        thread_ids = expand_customer_chain(root_id, tweet_rows, child_map)
        if not thread_ids:
            continue

        ordered_messages: list[dict[str, Any]] = []
        for tweet_id in thread_ids:
            row = tweet_rows[tweet_id]
            if row["author_id"] not in {customer_id, TARGET_BRAND}:
                continue
            ordered_messages.append({
                "tweet_id": row["tweet_id"],
                "author_id": row["author_id"],
                "inbound": row["inbound"],
                "created_at": row["created_at"],
                "text": row["text"],
                "parent_tweet_id": row["in_response_to_tweet_id"],
            })

        if not ordered_messages:
            continue
        ordered_messages = sorted(ordered_messages, key=lambda item: (_thread_sort_key(item["tweet_id"], tweet_rows)[0], item["tweet_id"]))

        customer_messages = [msg for msg in ordered_messages if msg["author_id"] == customer_id]
        uber_messages = [msg for msg in ordered_messages if msg["author_id"] == TARGET_BRAND]
        if not customer_messages or not uber_messages:
            continue

        issue_components = split_thread_issue_components([msg["tweet_id"] for msg in ordered_messages], tweet_rows)
        if not issue_components:
            continue

        for component_ids in issue_components:
            component_set = set(component_ids)
            component_messages = [msg for msg in ordered_messages if msg["tweet_id"] in component_set]
            component_customer_messages = [msg for msg in component_messages if msg["author_id"] == customer_id]
            component_uber_messages = [msg for msg in component_messages if msg["author_id"] == TARGET_BRAND]
            if not component_customer_messages or not component_uber_messages:
                continue

            valid_uber_messages: list[dict[str, Any]] = []
            customer_tweet_ids = {msg["tweet_id"] for msg in component_customer_messages}
            for msg in component_uber_messages:
                parent_tweet_id = msg["parent_tweet_id"]
                if parent_tweet_id in customer_tweet_ids:
                    valid_uber_messages.append(msg)
                elif f"@{customer_id.lower()}" in msg["text"].lower():
                    valid_uber_messages.append(msg)
            if not valid_uber_messages:
                continue

            final_uber = max(valid_uber_messages, key=lambda msg: _thread_sort_key(msg["tweet_id"], tweet_rows)[0])
            component_ordered = sorted(component_messages, key=lambda item: (_thread_sort_key(item["tweet_id"], tweet_rows)[0], item["tweet_id"]))
            component_signature = tuple(msg["tweet_id"] for msg in component_ordered)
            if component_signature in seen_signatures:
                continue
            seen_signatures.add(component_signature)

            case = {
                "customer_id": customer_id,
                "customer_messages": component_customer_messages,
                "uber_messages": valid_uber_messages,
                "full_thread": component_ordered,
                "created_at": component_ordered[0]["created_at"],
                "customer_message_count": len(component_customer_messages),
                "uber_message_count": len(valid_uber_messages),
                "final_uber_response": {
                    "tweet_id": final_uber["tweet_id"],
                    "author_id": final_uber["author_id"],
                    "created_at": final_uber["created_at"],
                    "text": final_uber["text"],
                    "parent_tweet_id": final_uber["parent_tweet_id"],
                },
                "resolution_type": infer_resolution_type(final_uber["text"]),
                "tweet_ids": component_signature,
            }
            candidates.append(case)

    deduped: list[dict[str, Any]] = []
    by_customer: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        by_customer[case["customer_id"]].append(case)

    for customer_id in sorted(by_customer):
        customer_cases = sorted(by_customer[customer_id], key=lambda item: (-len(item["tweet_ids"]), item["created_at"] or "", item["customer_id"]))
        kept: list[dict[str, Any]] = []
        for case in customer_cases:
            matched = False
            for index, existing in enumerate(kept):
                existing_ids = set(existing["tweet_ids"])
                case_ids = set(case["tweet_ids"])
                if case_ids.issubset(existing_ids):
                    matched = True
                    break
                if existing_ids.issubset(case_ids):
                    kept[index] = case
                    matched = True
                    break
            if not matched:
                kept.append(case)
        deduped.extend(kept)

    unique_by_signature: dict[tuple[str, ...], dict[str, Any]] = {}
    for case in deduped:
        unique_by_signature.setdefault(case["tweet_ids"], case)
    final_cases = list(unique_by_signature.values())
    final_cases.sort(key=lambda item: (_thread_sort_key(item["tweet_ids"][0] if item["tweet_ids"] else "", tweet_rows)[0], item["customer_id"]))

    invariant_issues = validate_case_invariants(final_cases)
    if invariant_issues:
        raise ValueError("Support-case invariant validation failed: " + "; ".join(invariant_issues))

    case_records: list[dict[str, Any]] = []
    for index, case in enumerate(final_cases, start=1):
        record = {
            "case_id": f"uber_case_{index:04d}",
            "customer_id": case["customer_id"],
            "customer_messages": case["customer_messages"],
            "uber_messages": case["uber_messages"],
            "full_thread": case["full_thread"],
            "created_at": case["created_at"],
            "customer_message_count": case["customer_message_count"],
            "uber_message_count": case["uber_message_count"],
            "final_uber_response": case["final_uber_response"],
            "resolution_type": case["resolution_type"],
        }
        case_records.append(record)
    return case_records


def write_jsonl(cases: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")


def write_markdown(cases: list[dict[str, Any]], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Uber_Support reconstructed support cases")
    lines.append("")
    lines.append(f"- Total cases: {len(cases)}")
    lines.append(f"- Unique customers: {len({case['customer_id'] for case in cases})}")
    if cases:
        avg_customer = sum(case["customer_message_count"] for case in cases) / len(cases)
        avg_uber = sum(case["uber_message_count"] for case in cases) / len(cases)
        lines.append(f"- Average customer messages per case: {avg_customer:.2f}")
        lines.append(f"- Average Uber messages per case: {avg_uber:.2f}")
        one_turn = sum(1 for case in cases if case["customer_message_count"] == 1 and case["uber_message_count"] == 1)
        multi_turn = sum(1 for case in cases if case["customer_message_count"] > 1 or case["uber_message_count"] > 1)
        lines.append(f"- 1-turn cases: {one_turn}")
        lines.append(f"- Multi-turn cases: {multi_turn}")
        resolution_counts = Counter(case["resolution_type"] for case in cases)
        lines.append("- Resolution type distribution: " + ", ".join(f"{key}={value}" for key, value in sorted(resolution_counts.items())))
    else:
        lines.append("- Average customer messages per case: 0.00")
        lines.append("- Average Uber messages per case: 0.00")
        lines.append("- 1-turn cases: 0")
        lines.append("- Multi-turn cases: 0")
        lines.append("- Resolution type distribution: unclear=0")

    lines.append("")
    lines.append("## Representative cases")
    lines.append("")
    for index, case in enumerate(cases[:20], start=1):
        lines.append(f"### {index}. {case['case_id']} | customer={case['customer_id']} | created_at={case['created_at']}")
        lines.append(f"- customer_message_count: {case['customer_message_count']}")
        lines.append(f"- uber_message_count: {case['uber_message_count']}")
        lines.append(f"- resolution_type: {case['resolution_type']}")
        lines.append("- thread:")
        for message in case["full_thread"]:
            direction = "customer" if message["author_id"] == case["customer_id"] else "uber"
            lines.append(f"  - [{direction}] {message['author_id']} | {message['tweet_id']} | parent={message['parent_tweet_id']} | {message['text']}")
        lines.append(f"- final_uber_response: {case['final_uber_response']['text']}")
        lines.append("")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = build_cases(RAW_PATH)
    write_jsonl(cases, ARTIFACTS_DIR / "uber_support_cases.jsonl")
    write_markdown(cases, ARTIFACTS_DIR / "uber_support_cases.md")


if __name__ == "__main__":
    main()
