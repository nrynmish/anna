#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_CHUNK_SIZE = 250_000


def infer_role(column_name: str, series: pd.Series) -> str:
    name = str(column_name).lower()
    if any(token in name for token in ["inbound", "outbound", "direction"]):
        return "direction"
    if any(token in name for token in ["created", "timestamp", "time", "date"]) or name.endswith("_at"):
        return "timestamp"
    if any(token in name for token in ["author", "user", "handle", "screen_name", "account"]):
        return "user_account"
    if any(token in name for token in ["text", "message", "content", "body", "tweet", "status"]) and "id" not in name:
        return "text"
    if any(token in name for token in ["in_response_to", "in_reply_to", "parent", "reply_to", "thread", "conversation"]):
        return "reply_reference"
    if any(token in name for token in ["response", "reply"]) and any(token in name for token in ["tweet_id", "id"]) and "in_response" not in name and "in_reply" not in name:
        return "response_reference"
    if any(token in name for token in ["id", "uid"]):
        return "identifier"
    if series.dtype == bool or series.dropna().map(lambda value: str(value).lower() in {"true", "false", "1", "0"}).all():
        return "boolean"
    return "other"


def inspect_schema(path: Path, sample_rows: int = 5000) -> dict[str, Any]:
    sample = pd.read_csv(path, nrows=sample_rows, low_memory=False)
    fields: list[dict[str, Any]] = []
    for column in sample.columns:
        series = sample[column]
        fields.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "inferred_role": infer_role(column, series),
                "sample_values": [str(value) for value in series.dropna().head(3).tolist()],
                "null_count": int(series.isna().sum()),
            }
        )
    return {
        "file": str(path),
        "column_count": len(sample.columns),
        "columns": list(sample.columns),
        "fields": fields,
    }


def text_like_column_name(name: str) -> bool:
    lower = name.lower()
    return any(token in lower for token in ["text", "message", "content", "body"]) and "id" not in lower


def normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def as_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    values = series.fillna("").astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False, "y": True, "n": False}
    return values.map(mapping).fillna(False)


def identify_relevant_id_columns(columns: list[str]) -> list[str]:
    selected: list[str] = []
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in ["tweet_id", "author_id", "user_id", "response_tweet_id", "in_response_to_tweet_id", "in_reply_to_status_id"]) and "text" not in lower:
            selected.append(column)
    return selected


def parse_timestamp_column(series: pd.Series, sample_size: int = 1000) -> pd.Series:
    values = series.astype("string").str.strip()
    if values.empty:
        return pd.Series(pd.NaT, index=series.index)

    sample = values.dropna().head(sample_size)
    if sample.empty:
        return pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")

    # Actual Twitter-created timestamps use: Tue Oct 31 22:10:47 +0000 2017
    # Parse them in one vectorized pass with the exact format to avoid dateutil
    # inference and the associated warning/performance overhead.
    return pd.to_datetime(series, format="%a %b %d %H:%M:%S %z %Y", errors="coerce")


def stream_dataset_quality(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict[str, Any]:
    total_rows = 0
    null_counts: dict[str, int] = {}
    text_quality: dict[str, Counter[str]] = {}
    timestamp_coverage: dict[str, Any] = {}
    exact_cardinality: dict[str, int] = {}
    exact_duplicates: dict[str, int] = {}
    columns: list[str] | None = None
    id_counters: dict[str, Counter[str]] = {}

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
            null_counts = {col: 0 for col in columns}
            text_quality = {
                col: Counter({"empty": 0, "whitespace_only": 0, "url_only": 0, "very_short": 0, "very_long": 0})
                for col in columns
            }
            id_counters = {col: Counter() for col in identify_relevant_id_columns(columns)}

        total_rows += len(chunk)
        for col in columns:
            null_counts[col] += int(chunk[col].isna().sum())
            if text_like_column_name(str(col)):
                values = normalize_text(chunk[col])
                text_quality[col]["empty"] += int((values == "").sum())
                text_quality[col]["whitespace_only"] += int(values.str.len().eq(0).sum())
                text_quality[col]["url_only"] += int(values.str.fullmatch(r"https?://\S+|www\.\S+", na=False).sum())
                text_quality[col]["very_short"] += int(values.str.len().lt(20).sum())
                text_quality[col]["very_long"] += int(values.str.len().gt(500).sum())
            name = str(col).lower()
            if any(token in name for token in ["time", "date", "created", "timestamp"]) or name.endswith("_at"):
                ts = parse_timestamp_column(chunk[col])
                if col not in timestamp_coverage:
                    timestamp_coverage[col] = {"min": None, "max": None, "invalid_count": 0, "monthly_counts": Counter()}
                timestamp_coverage[col]["invalid_count"] += int(ts.isna().sum())
                valid = ts.dropna()
                if not valid.empty:
                    normalized = valid
                    if getattr(valid.dt, "tz", None) is not None:
                        normalized = valid.dt.tz_convert("UTC").dt.tz_localize(None)
                    min_value = normalized.min()
                    max_value = normalized.max()
                    timestamp_coverage[col]["min"] = min_value if timestamp_coverage[col]["min"] is None else min(timestamp_coverage[col]["min"], min_value)
                    timestamp_coverage[col]["max"] = max_value if timestamp_coverage[col]["max"] is None else max(timestamp_coverage[col]["max"], max_value)
                    timestamp_coverage[col]["monthly_counts"].update(normalized.dt.to_period("M").astype(str).tolist())

        for col in id_counters:
            id_counters[col].update(chunk[col].dropna().astype(str).tolist())

    for col, counter in id_counters.items():
        exact_cardinality[col] = len(counter)
        exact_duplicates[col] = sum(value - 1 for value in counter.values() if value > 1)

    return {
        "total_rows": total_rows,
        "null_counts": null_counts,
        "null_percentages": {col: (count / max(total_rows, 1)) * 100.0 for col, count in null_counts.items()},
        "text_quality": {col: dict(counter) for col, counter in text_quality.items()},
        "timestamp_coverage": timestamp_coverage,
        "exact_id_cardinality": exact_cardinality,
        "exact_id_duplicates": exact_duplicates,
        "exact_id_fields": sorted(id_counters.keys()),
    }


def detect_parent_reference_column(columns: list[str]) -> str | None:
    lower_columns = [str(column).lower() for column in columns]
    preferred = [
        "in_response_to_tweet_id",
        "in_reply_to_status_id",
        "in_response_to",
        "in_reply_to",
        "parent_tweet_id",
        "parent_id",
        "reply_to_tweet_id",
        "reply_to_id",
    ]
    for expected in preferred:
        for column in columns:
            if str(column).lower() == expected:
                return str(column)
    for column in columns:
        lower = str(column).lower()
        if any(token in lower for token in ["in_response_to", "in_reply_to", "parent", "reply_to", "thread", "conversation"]) and "response_tweet_id" not in lower:
            return str(column)
    return None


def find_root(tweet_id: str, parent_map: dict[str, str], cache: dict[str, str]) -> str:
    if tweet_id in cache:
        return cache[tweet_id]
    current = tweet_id
    seen: set[str] = set()
    while current in parent_map and current not in seen:
        seen.add(current)
        current = parent_map[current]
    final_root = current
    for node in seen:
        cache[node] = final_root
    return final_root


def compute_conversation_summary_from_roots(root_map: dict[str, str]) -> dict[str, Any]:
    if not root_map:
        return {
            "conversation_count": 0,
            "average_messages_per_conversation": 0.0,
            "median_messages_per_conversation": 0.0,
            "p90_messages_per_conversation": 0.0,
            "max_messages_per_conversation": 0,
            "pct_ge_2": 0.0,
            "pct_ge_3": 0.0,
            "pct_ge_5": 0.0,
            "pct_ge_10": 0.0,
        }

    conversation_sizes = sorted(Counter(root_map.values()).values())
    conversation_count = len(conversation_sizes)
    if conversation_count == 0:
        return {
            "conversation_count": 0,
            "average_messages_per_conversation": 0.0,
            "median_messages_per_conversation": 0.0,
            "p90_messages_per_conversation": 0.0,
            "max_messages_per_conversation": 0,
            "pct_ge_2": 0.0,
            "pct_ge_3": 0.0,
            "pct_ge_5": 0.0,
            "pct_ge_10": 0.0,
        }

    total_messages = sum(conversation_sizes)
    median_index = (conversation_count - 1) // 2
    median = float(conversation_sizes[median_index]) if conversation_count % 2 == 1 else float((conversation_sizes[median_index] + conversation_sizes[median_index + 1]) / 2.0)

    p90_index = max(0, min(conversation_count - 1, math.ceil(0.9 * conversation_count) - 1))
    p90 = float(conversation_sizes[p90_index])
    max_size = max(conversation_sizes)

    def pct_at_least(threshold: int) -> float:
        return sum(1 for size in conversation_sizes if size >= threshold) / conversation_count

    return {
        "conversation_count": conversation_count,
        "average_messages_per_conversation": float(total_messages / conversation_count),
        "median_messages_per_conversation": median,
        "p90_messages_per_conversation": p90,
        "max_messages_per_conversation": max_size,
        "pct_ge_2": pct_at_least(2),
        "pct_ge_3": pct_at_least(3),
        "pct_ge_5": pct_at_least(5),
        "pct_ge_10": pct_at_least(10),
    }


def compute_conversation_roots(df: pd.DataFrame) -> tuple[dict[str, Any], dict[str, str]]:
    if "tweet_id" not in df.columns:
        return {
            "conversation_count": 0,
            "average_messages_per_conversation": 0.0,
            "median_messages_per_conversation": 0.0,
            "p90_messages_per_conversation": 0.0,
            "max_messages_per_conversation": 0,
            "pct_ge_2": 0.0,
            "pct_ge_3": 0.0,
            "pct_ge_5": 0.0,
            "pct_ge_10": 0.0,
        }, {}

    parent_col = detect_parent_reference_column(df.columns)
    if parent_col is None:
        root_map = {str(tweet_id): str(tweet_id) for tweet_id in df["tweet_id"].dropna().astype(str).unique().tolist()}
        return compute_conversation_summary_from_roots(root_map), root_map

    subset = df[["tweet_id", parent_col]].dropna(subset=["tweet_id", parent_col]).copy()
    subset["tweet_id"] = subset["tweet_id"].astype(str)
    subset[parent_col] = subset[parent_col].astype(str)
    parent_map = subset.set_index("tweet_id")[parent_col].to_dict()
    cache: dict[str, str] = {}
    root_map = {tweet_id: find_root(tweet_id, parent_map, cache) for tweet_id in parent_map}
    summary = compute_conversation_summary_from_roots(root_map)
    summary["method"] = f"Reconstructed from {parent_col} using parent references."
    return summary, root_map


def compute_customer_brand_signal(df: pd.DataFrame) -> dict[str, Any]:
    if "inbound" not in df.columns:
        return {
            "status": "inbound/outbound field unavailable; no direct customer-brand signal found.",
            "inbound_messages": None,
            "outbound_messages": None,
            "inbound_share": None,
            "outbound_share": None,
            "inbound_outbound_ratio": None,
        }

    inbound_series = as_bool_series(df["inbound"])
    inbound_count = int(inbound_series.sum())
    outbound_count = int((~inbound_series).sum())
    total = len(df)
    return {
        "status": "Actual inbound field semantics are based on the dataset value rather than a guessed label.",
        "inbound_messages": inbound_count,
        "outbound_messages": outbound_count,
        "inbound_share": inbound_count / max(total, 1),
        "outbound_share": outbound_count / max(total, 1),
        "inbound_outbound_ratio": (inbound_count / outbound_count) if outbound_count else None,
    }


def build_brand_profiles(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> list[dict[str, Any]]:
    sample_columns: list[str] | None = None
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        sample_columns = list(chunk.columns)
        break
    if not sample_columns:
        return []

    required = {"author_id", "inbound", "tweet_id"}
    if not required.issubset(set(sample_columns)):
        return []

    parent_reference = detect_parent_reference_column(sample_columns)
    if parent_reference is None:
        return []

    parent_map: dict[str, str] = {}
    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if parent_reference not in chunk.columns or "tweet_id" not in chunk.columns:
            continue
        relevant = chunk[["tweet_id", parent_reference]].copy()
        relevant["tweet_id"] = relevant["tweet_id"].astype(str)
        relevant[parent_reference] = relevant[parent_reference].astype(str)
        for tweet_id, parent_id in relevant.dropna().itertuples(index=False, name=None):
            tweet_id = str(tweet_id)
            parent_id = str(parent_id)
            if tweet_id and parent_id:
                parent_map[tweet_id] = parent_id

    def resolve_root(tweet_id: str, cache: dict[str, str]) -> str:
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

    root_cache: dict[str, str] = {}
    root_map = {tweet_id: resolve_root(tweet_id, root_cache) for tweet_id in parent_map}

    root_size: Counter[str] = Counter()
    root_to_customer_authors: defaultdict[str, set[str]] = defaultdict(set)
    root_to_support_authors: defaultdict[str, set[str]] = defaultdict(set)
    root_support_outbound_count: defaultdict[str, int] = defaultdict(int)

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if not required.issubset(set(chunk.columns)):
            continue
        relevant = chunk[["tweet_id", "author_id", "inbound"]].copy()
        relevant["tweet_id"] = relevant["tweet_id"].astype(str)
        relevant["author_id"] = relevant["author_id"].astype(str)
        relevant["inbound"] = as_bool_series(relevant["inbound"])
        for tweet_id, author_id, inbound in relevant.itertuples(index=False, name=None):
            tweet_id = str(tweet_id)
            author_id = str(author_id)
            inbound = bool(inbound)
            root = root_map.get(tweet_id, tweet_id)
            root_size[root] += 1
            if inbound:
                root_to_customer_authors[root].add(author_id)
            else:
                root_to_support_authors[root].add(author_id)
                root_support_outbound_count[root] += 1

    brand_outbound_count: defaultdict[str, int] = defaultdict(int)
    brand_roots: defaultdict[str, set[str]] = defaultdict(set)
    brand_lengths: defaultdict[str, list[int]] = defaultdict(list)
    brand_customer_authors: defaultdict[str, set[str]] = defaultdict(set)
    brand_both_roots: defaultdict[str, set[str]] = defaultdict(set)

    for root, support_authors in root_to_support_authors.items():
        conversation_length = root_size.get(root, 0)
        customer_authors = root_to_customer_authors.get(root, set())
        for author in support_authors:
            brand_roots[author].add(root)
            brand_outbound_count[author] += root_support_outbound_count.get(root, 0)
            brand_lengths[author].append(conversation_length)
            brand_customer_authors[author].update(customer_authors)
            if root in root_to_customer_authors:
                brand_both_roots[author].add(root)

    results: list[dict[str, Any]] = []
    for author, outbound_count in sorted(brand_outbound_count.items(), key=lambda item: item[1], reverse=True):
        lengths = brand_lengths.get(author, [])
        roots = brand_roots.get(author, set())
        results.append(
            {
                "brand": author,
                "total_outbound_messages": outbound_count,
                "total_conversations": len(roots),
                "conversations_with_customer_and_brand": len(brand_both_roots.get(author, set())),
                "unique_customer_accounts": len(brand_customer_authors.get(author, set())),
                "average_conversation_length": (sum(lengths) / len(lengths)) if lengths else 0.0,
                "median_conversation_length": float(pd.Series(lengths).median()) if lengths else 0.0,
                "multi_turn_conversation_percentage": (sum(1 for value in lengths if value >= 2) / len(lengths)) if lengths else 0.0,
            }
        )
    return results


def merge_top_counter(target: Counter[str], new_items: Counter[str], top_n: int) -> Counter[str]:
    for key, value in new_items.items():
        target[key] += value
    if len(target) <= top_n:
        return target
    return Counter(dict(target.most_common(top_n)))


def aggregate_text_signals_from_chunks(path: Path, chunk_size: int = DEFAULT_CHUNK_SIZE, top_n: int = 50) -> dict[str, Any]:
    unigram_counter: Counter[str] = Counter()
    bigram_counter: Counter[tuple[str, str]] = Counter()
    length_histogram: Counter[int] = Counter()
    total_length = 0
    total_messages = 0

    for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
        if "text" not in chunk.columns:
            continue
        chunk_unigrams: Counter[str] = Counter()
        chunk_bigrams: Counter[tuple[str, str]] = Counter()
        text_series = normalize_text(chunk["text"])
        for value in text_series.dropna().astype(str).tolist():
            if not value:
                continue
            total_messages += 1
            total_length += len(value)
            length_histogram[len(value)] += 1
            tokens = [token for token in value.lower().replace("https://", " ").replace("http://", " ").replace("www.", " ").replace("?", " ").replace("!", " ").split() if token and token.isalnum()]
            for token in tokens:
                chunk_unigrams[token] += 1
            for index in range(len(tokens) - 1):
                chunk_bigrams[(tokens[index], tokens[index + 1])] += 1

        unigram_counter = merge_top_counter(unigram_counter, chunk_unigrams, top_n)
        bigram_counter = merge_top_counter(bigram_counter, chunk_bigrams, top_n)

    if total_messages == 0:
        return {"status": "No text field available for text reconnaissance.", "top_unigrams": [], "top_bigrams": [], "average_length": 0.0, "median_length_estimate": 0.0}

    median_target = total_messages / 2
    cumulative = 0
    median_estimate = 0.0
    for length, count in sorted(length_histogram.items()):
        cumulative += count
        if cumulative >= median_target:
            median_estimate = float(length)
            break

    return {
        "status": "Top-N bounded counters only; no unbounded token retention.",
        "top_unigrams": unigram_counter.most_common(top_n),
        "top_bigrams": bigram_counter.most_common(top_n),
        "average_length": total_length / total_messages,
        "median_length_estimate": median_estimate,
        "approximate_median": True,
    }


def write_profile_artifacts(profile: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "dataset_profile.json").write_text(json.dumps(profile, default=str, indent=2), encoding="utf-8")

    brand_rows = profile.get("brand_profile_rows", [])
    if brand_rows:
        pd.DataFrame(brand_rows).to_csv(ARTIFACTS_DIR / "brand_profile.csv", index=False)

    markdown_lines = [
        "# ANNA dataset reconnaissance",
        "",
        f"- Raw file: {profile['raw_file']}",
        f"- Total rows: {profile['total_rows']}",
        f"- Total columns: {profile['total_columns']}",
        "",
        "## Schema",
        "",
    ]
    for field in profile.get("schema", {}).get("fields", []):
        markdown_lines.append(f"- {field['column']}: dtype={field['dtype']}, role={field['inferred_role']}")
    markdown_lines.extend(["", "## Data quality", ""])
    for col, value in profile.get("quality", {}).get("null_counts", {}).items():
        markdown_lines.append(f"- {col}: {value} nulls")
    markdown_lines.extend(["", "## Conversation structure", ""])
    for key, value in profile.get("conversation_summary", {}).items():
        markdown_lines.append(f"- {key}: {value}")
    markdown_lines.extend(["", "## Customer / brand signal", ""])
    for key, value in profile.get("customer_brand_signal", {}).items():
        markdown_lines.append(f"- {key}: {value}")
    (ARTIFACTS_DIR / "dataset_profile.md").write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")


def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset file not found: {RAW_PATH}")

    schema = inspect_schema(RAW_PATH, sample_rows=5000)
    quality = stream_dataset_quality(RAW_PATH, chunk_size=DEFAULT_CHUNK_SIZE)
    conversation_summary, root_map = compute_conversation_roots(pd.read_csv(RAW_PATH, nrows=5000, low_memory=False))
    customer_brand_signal = compute_customer_brand_signal(pd.read_csv(RAW_PATH, nrows=5000, low_memory=False))
    brand_profiles = build_brand_profiles(RAW_PATH, chunk_size=DEFAULT_CHUNK_SIZE)
    text_signals = aggregate_text_signals_from_chunks(RAW_PATH, chunk_size=DEFAULT_CHUNK_SIZE)

    profile = {
        "raw_file": str(RAW_PATH),
        "total_rows": quality["total_rows"],
        "total_columns": len(schema["columns"]),
        "schema": schema,
        "quality": quality,
        "conversation_summary": conversation_summary,
        "customer_brand_signal": customer_brand_signal,
        "brand_profile_rows": brand_profiles,
        "text_signals": text_signals,
    }
    write_profile_artifacts(profile)
    print(json.dumps({
        "status": "profile artifacts prepared for review",
        "dataset_file": str(RAW_PATH),
        "rows": quality["total_rows"],
        "columns": len(schema["columns"]),
        "brand_profiles": len(brand_profiles),
    }, indent=2))


if __name__ == "__main__":
    main()
