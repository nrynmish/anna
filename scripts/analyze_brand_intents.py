#!/usr/bin/env python3
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "twcs" / "twcs.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TARGET_BRANDS = ["AmazonHelp", "AppleSupport", "Uber_Support", "TMobileHelp", "Delta"]


def normalize_message(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"@\w+", " ", value)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def is_meaningful_token(token: str) -> bool:
    return len(token) > 2 and token not in {"the", "and", "for", "with", "this", "that", "have", "from", "your", "into", "just", "been", "will", "more", "over", "than", "they", "them", "there", "their", "what", "when", "where", "who", "why", "how", "about", "after", "before", "could", "would", "should", "still", "need", "please", "help", "thanks", "thank", "pls", "can", "also"}


def extract_brand_messages(df: pd.DataFrame, brand: str) -> pd.DataFrame:
    if "author_id" not in df.columns or "text" not in df.columns or "inbound" not in df.columns:
        return pd.DataFrame(columns=["tweet_id", "author_id", "inbound", "text", "response_tweet_id", "in_response_to_tweet_id"])

    brand_df = df.loc[
        (df["author_id"].astype(str).str.lower() == brand.lower())
        | (df["text"].astype(str).str.lower().str.contains(f"@{brand.lower()}"))
    ].copy()
    if brand_df.empty:
        return brand_df

    # Keep direct customer inbound conversations addressed to the brand.
    # This favors support requests and removes unrelated content when possible.
    brand_df = brand_df.loc[
        brand_df["inbound"].astype(str).str.lower().eq("true")
        | brand_df["text"].astype(str).str.lower().str.contains(f"@{brand.lower()}")
    ].copy()

    if "text" in brand_df.columns:
        brand_df["normalized_text"] = brand_df["text"].map(normalize_message)
    else:
        brand_df["normalized_text"] = ""

    brand_df = brand_df.loc[brand_df["normalized_text"].str.len() > 0].copy()
    return brand_df


def summarize_text_metrics(messages: list[str]) -> dict[str, Any]:
    lengths = [len(message) for message in messages]
    if not lengths:
        return {"count": 0, "avg_length": 0.0, "median_length": 0.0, "min_length": 0, "max_length": 0}
    ordered = sorted(lengths)
    median = ordered[len(ordered) // 2] if len(ordered) % 2 == 1 else (ordered[len(ordered) // 2 - 1] + ordered[len(ordered) // 2]) / 2
    return {
        "count": len(lengths),
        "avg_length": sum(lengths) / len(lengths),
        "median_length": median,
        "min_length": min(lengths),
        "max_length": max(lengths),
    }


def summarize_messages(messages: list[str]) -> tuple[Counter[str], Counter[tuple[str, str]], list[str]]:
    unigrams: Counter[str] = Counter()
    bigrams: Counter[tuple[str, str]] = Counter()
    for message in messages:
        tokens = [token for token in message.split() if is_meaningful_token(token)]
        for token in tokens:
            unigrams[token] += 1
        for index in range(len(tokens) - 1):
            bigrams[(tokens[index], tokens[index + 1])] += 1

    representatives = [message for message in messages[:10]]
    return unigrams, bigrams, representatives


def extract_brand_intent_summary(brand: str, df: pd.DataFrame) -> dict[str, Any]:
    messages = extract_brand_messages(df, brand)
    text_values = [value for value in messages["normalized_text"].dropna().astype(str).tolist() if value]

    unigrams, bigrams, reps = summarize_messages(text_values)
    length_stats = summarize_text_metrics(text_values)

    return {
        "brand": brand,
        "message_count": len(text_values),
        "top_unigrams": unigrams,
        "top_bigrams": bigrams,
        "length_stats": length_stats,
        "representative_messages": reps,
    }


def markdown_for_brand(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"## {summary['brand']}")
    lines.append("")
    lines.append(f"- Inbound customer messages analyzed: {summary['message_count']}")
    lines.append(f"- Mean length: {summary['length_stats']['avg_length']:.1f}")
    lines.append(f"- Median length: {summary['length_stats']['median_length']:.1f}")
    lines.append(f"- Min length: {summary['length_stats']['min_length']}")
    lines.append(f"- Max length: {summary['length_stats']['max_length']}")
    lines.append("")
    lines.append("### Top 30 meaningful unigrams")
    top_unigrams = summary["top_unigrams"].most_common(30)
    if top_unigrams:
        for token, count in top_unigrams:
            lines.append(f"- {token}: {count}")
    else:
        lines.append("- No meaningful unigrams found.")
    lines.append("")
    lines.append("### Top 30 meaningful bigrams")
    top_bigrams = summary["top_bigrams"].most_common(30)
    if top_bigrams:
        for (left, right), count in top_bigrams:
            lines.append(f"- {left} {right}: {count}")
    else:
        lines.append("- No meaningful bigrams found.")
    lines.append("")
    lines.append("### 10 representative inbound messages")
    representative_messages = summary["representative_messages"][:10]
    if representative_messages:
        for index, message in enumerate(representative_messages, start=1):
            lines.append(f"{index}. {message}")
    else:
        lines.append("- No representative messages found.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_iter = pd.read_csv(RAW_PATH, chunksize=250_000, low_memory=False)
    brand_summaries: list[dict[str, Any]] = []
    for chunk in chunk_iter:
        if {"tweet_id", "author_id", "inbound", "text"}.issubset(set(chunk.columns)):
            for brand in TARGET_BRANDS:
                summary = extract_brand_intent_summary(brand, chunk)
                if not brand_summaries or not any(item["brand"] == brand for item in brand_summaries):
                    brand_summaries.append(summary)
                else:
                    idx = next(i for i, item in enumerate(brand_summaries) if item["brand"] == brand)
                    brand_summaries[idx]["top_unigrams"] = brand_summaries[idx]["top_unigrams"] + summary["top_unigrams"]
                    brand_summaries[idx]["top_bigrams"] = brand_summaries[idx]["top_bigrams"] + summary["top_bigrams"]
                    brand_summaries[idx]["representative_messages"] = (brand_summaries[idx]["representative_messages"] + summary["representative_messages"])[:10]
                    brand_summaries[idx]["message_count"] += summary["message_count"]

    # Ensure a full pass over the dataset for consistent totals while keeping the script limited to the target brands.
    if not brand_summaries:
        brand_summaries = [extract_brand_intent_summary(brand, pd.read_csv(RAW_PATH, nrows=0)) for brand in TARGET_BRANDS]

    brand_summaries = [extract_brand_intent_summary(brand, pd.read_csv(RAW_PATH, low_memory=False)) for brand in TARGET_BRANDS]
    markdown_sections = [markdown_for_brand(summary) for summary in brand_summaries]
    output = "# Brand intent analysis for ANNA candidate brands\n\n"
    output += "\n\n".join(markdown_sections)
    (ARTIFACTS_DIR / "brand_intent_analysis.md").write_text(output + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
