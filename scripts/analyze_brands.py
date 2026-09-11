#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_CHUNK_SIZE = 250_000


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
    return values.map(mapping).fillna(False)


def resolve_root(tweet_id: str, parent_map: dict[str, str], cache: dict[str, str]) -> str:
    if tweet_id in cache:
        return cache[tweet_id]
    current = tweet_id
    seen: set[str] = set()
    while current in parent_map and current not in seen:
        seen.add(current)
        current = parent_map[current]
    root = current
    for node in seen:
        cache[node] = root
    return root


def pick_parent_reference(columns: list[str]) -> str | None:
    preferred = [
        "in_response_to_tweet_id",
        "in_reply_to_status_id",
        "in_response_to",
        "in_reply_to",
        "parent_tweet_id",
        "parent_id",
    ]
    lower_map = {str(column).lower(): str(column) for column in columns}
    for name in preferred:
        if name in lower_map:
            return lower_map[name]
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in ["in_response_to", "in_reply_to", "parent", "reply_to", "thread", "conversation"]) and "response_tweet_id" not in lower:
            return str(column)
    return None


def analyze_brands(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> tuple[list[dict[str, Any]], dict[str, list[str]], dict[str, int]]:
    parent_map: dict[str, str] = {}
    cache: dict[str, str] = {}
    root_size: dict[str, int] = {}
    root_customer_authors: defaultdict[str, set[str]] = defaultdict(set)
    root_support_authors: defaultdict[str, set[str]] = defaultdict(set)
    root_customer_count: defaultdict[str, int] = defaultdict(int)
    root_support_count: defaultdict[str, int] = defaultdict(int)
    root_examples: dict[str, list[str]] = defaultdict(list)

    parent_reference: str | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        columns = list(chunk.columns)
        if parent_reference is None:
            parent_reference = pick_parent_reference(columns)
        if parent_reference and parent_reference in chunk.columns:
            relevant = chunk[["tweet_id", parent_reference]].copy()
            relevant = relevant.dropna(subset=["tweet_id", parent_reference])
            if not relevant.empty:
                relevant["tweet_id"] = relevant["tweet_id"].astype(str)
                relevant[parent_reference] = relevant[parent_reference].astype(str)
                for tweet_id, parent_id in relevant.itertuples(index=False, name=None):
                    parent_map[str(tweet_id)] = str(parent_id)

        if {"tweet_id", "author_id", "inbound"}.issubset(set(columns)):
            relevant = chunk[["tweet_id", "author_id", "inbound"]].copy()
            relevant["tweet_id"] = relevant["tweet_id"].astype(str)
            relevant["author_id"] = relevant["author_id"].astype(str)
            relevant["inbound"] = as_bool_series(relevant["inbound"])
            for tweet_id, author_id, inbound in relevant.itertuples(index=False, name=None):
                tweet_id = str(tweet_id)
                author_id = str(author_id)
                root = resolve_root(tweet_id, parent_map, cache)
                root_size[root] = root_size.get(root, 0) + 1
                if inbound:
                    root_customer_authors[root].add(author_id)
                    root_customer_count[root] += 1
                else:
                    root_support_authors[root].add(author_id)
                    root_support_count[root] += 1

    brand_inbound_messages: defaultdict[str, int] = defaultdict(int)
    brand_outbound_messages: defaultdict[str, int] = defaultdict(int)
    brand_customer_authors: defaultdict[str, set[str]] = defaultdict(set)
    brand_both_roots: defaultdict[str, set[str]] = defaultdict(set)
    brand_roots: defaultdict[str, set[str]] = defaultdict(set)
    brand_mixed_lengths: defaultdict[str, list[int]] = defaultdict(list)
    brand_multi_turn: defaultdict[str, int] = defaultdict(int)

    for root, support_authors in root_support_authors.items():
        customer_authors = root_customer_authors.get(root, set())
        inbound_messages = root_customer_count.get(root, 0)
        outbound_messages = root_support_count.get(root, 0)
        root_length = root_size.get(root, 0)
        for author in support_authors:
            brand_roots[author].add(root)
            brand_inbound_messages[author] += inbound_messages
            brand_outbound_messages[author] += outbound_messages
            brand_customer_authors[author].update(customer_authors)
            if customer_authors and inbound_messages > 0:
                brand_both_roots[author].add(root)
                brand_mixed_lengths[author].append(root_length)
            if root_length >= 2:
                brand_multi_turn[author] += 1

    brand_rows: list[dict[str, Any]] = []
    for brand in sorted(brand_roots):
        mixed_roots = brand_both_roots.get(brand, set())
        mixed_sizes = brand_mixed_lengths.get(brand, [])
        brand_rows.append(
            {
                "brand": brand,
                "inbound_customer_messages": int(brand_inbound_messages.get(brand, 0)),
                "outbound_brand_messages": int(brand_outbound_messages.get(brand, 0)),
                "conversations_with_customer_and_brand": len(mixed_roots),
                "unique_customer_accounts": len(brand_customer_authors.get(brand, set())),
                "multi_turn_conversations": int(brand_multi_turn.get(brand, 0)),
                "average_conversation_length_for_mixed_customer_and_brand_conversations": (sum(mixed_sizes) / len(mixed_sizes)) if mixed_sizes else 0.0,
            }
        )

    brand_rows.sort(key=lambda row: (-row["conversations_with_customer_and_brand"], -row["inbound_customer_messages"], str(row["brand"])))

    top_brands = {row["brand"]: sorted((brand_roots.get(row["brand"], set())), key=lambda root: (-root_size.get(root, 0), str(root)))[:5] for row in brand_rows[:20]}
    return brand_rows, top_brands, root_size


def gather_thread_examples(path: Path, root_ids_by_brand: dict[str, list[str]], parent_map: dict[str, str], root_size: dict[str, int]) -> dict[str, list[dict[str, Any]]]:
    selected_roots = {root for roots in root_ids_by_brand.values() for root in roots}
    examples_by_root: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    cache: dict[str, str] = {}

    for chunk in pd.read_csv(path, chunksize=DEFAULT_CHUNK_SIZE, low_memory=False):
        if not {"tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"}.issubset(set(chunk.columns)):
            continue
        relevant = chunk[["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]].copy()
        relevant["tweet_id"] = relevant["tweet_id"].astype(str)
        relevant["author_id"] = relevant["author_id"].astype(str)
        relevant["inbound"] = as_bool_series(relevant["inbound"])
        relevant["text"] = relevant["text"].fillna("").astype(str)
        relevant["in_response_to_tweet_id"] = pd.to_numeric(relevant["in_response_to_tweet_id"], errors="coerce")
        for row in relevant.itertuples(index=False):
            tweet_id = str(row.tweet_id)
            root = resolve_root(tweet_id, parent_map, cache)
            if root not in selected_roots:
                continue
            parent_id = row.in_response_to_tweet_id
            examples_by_root[root].append(
                {
                    "tweet_id": tweet_id,
                    "author": str(row.author_id),
                    "inbound": bool(row.inbound),
                    "text": str(row.text),
                    "parent_tweet_id": "" if pd.isna(parent_id) else str(int(parent_id)) if float(parent_id).is_integer() else str(parent_id),
                    "created_at": None,
                }
            )

    thread_examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for brand, roots in root_ids_by_brand.items():
        for root in roots:
            rows = examples_by_root.get(root, [])
            if not rows:
                continue
            rows = sorted(rows, key=lambda item: (item["tweet_id"], item["author"]))
            thread_examples[brand].append({"root": root, "messages": rows})
    return thread_examples


def write_brand_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "brand",
        "inbound_customer_messages",
        "outbound_brand_messages",
        "conversations_with_customer_and_brand",
        "unique_customer_accounts",
        "multi_turn_conversations",
        "average_conversation_length_for_mixed_customer_and_brand_conversations",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_brand_markdown(rows: list[dict[str, Any]], examples_by_brand: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# ANNA brand analysis")
    lines.append("")
    lines.append("This analysis is focused on selecting one support brand for ANNA by prioritizing the accounts with the strongest customer-support conversation signal.")
    lines.append("")
    lines.append("## Top 20 brands ranked by conversations with customer and brand messages, then inbound customer messages")
    lines.append("")
    lines.append("| Rank | Brand | Inbound customer messages | Outbound brand messages | Conversations with customer and brand | Unique customer accounts | Multi-turn conversations | Avg. conversation length (mixed) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for index, row in enumerate(rows[:20], start=1):
        lines.append(
            f"| {index} | {row['brand']} | {row['inbound_customer_messages']} | {row['outbound_brand_messages']} | {row['conversations_with_customer_and_brand']} | {row['unique_customer_accounts']} | {row['multi_turn_conversations']} | {row['average_conversation_length_for_mixed_customer_and_brand_conversations']:.2f} |"
        )

    lines.append("")
    lines.append("## Example conversation threads for the top 20 brands")
    lines.append("")
    for index, row in enumerate(rows[:20], start=1):
        brand = str(row["brand"])
        lines.append(f"### {index}. {brand}")
        brand_examples = examples_by_brand.get(brand, [])
        if not brand_examples:
            lines.append("No example threads available.")
            lines.append("")
            continue
        for thread_index, thread in enumerate(brand_examples[:5], start=1):
            lines.append(f"#### Example thread {thread_index} (root tweet {thread['root']})")
            lines.append("")
            lines.append("| tweet_id | author | direction | parent_tweet_id | text |")
            lines.append("| --- | --- | --- | --- | --- |")
            for message in thread["messages"]:
                direction = "inbound" if message["inbound"] else "outbound"
                text = message["text"].replace("|", "\\|").replace("\n", " ")
                lines.append(f"| {message['tweet_id']} | {message['author']} | {direction} | {message['parent_tweet_id']} | {text} |")
            lines.append("")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    brand_rows, root_ids_by_brand, root_size = analyze_brands(RAW_PATH)
    parent_map: dict[str, str] = {}
    cache: dict[str, str] = {}
    for chunk in pd.read_csv(RAW_PATH, chunksize=DEFAULT_CHUNK_SIZE, low_memory=False):
        if "in_response_to_tweet_id" in chunk.columns:
            relevant = chunk[["tweet_id", "in_response_to_tweet_id"]].copy().dropna(subset=["tweet_id", "in_response_to_tweet_id"])
            relevant["tweet_id"] = relevant["tweet_id"].astype(str)
            relevant["in_response_to_tweet_id"] = relevant["in_response_to_tweet_id"].astype(str)
            for tweet_id, parent_id in relevant.itertuples(index=False, name=None):
                parent_map[str(tweet_id)] = str(parent_id)

    examples_by_brand = gather_thread_examples(RAW_PATH, root_ids_by_brand, parent_map, root_size)
    write_brand_csv(brand_rows, ARTIFACTS_DIR / "brand_analysis.csv")
    write_brand_markdown(brand_rows, examples_by_brand, ARTIFACTS_DIR / "brand_analysis.md")


if __name__ == "__main__":
    main()
