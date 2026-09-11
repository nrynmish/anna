from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.profile_dataset import (
    as_bool_series,
    compute_conversation_summary_from_roots,
    detect_parent_reference_column,
    normalize_text,
    text_like_column_name,
)


def test_detect_parent_reference_column_prefers_in_response_to():
    columns = ["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"]
    assert detect_parent_reference_column(columns) == "in_response_to_tweet_id"


def test_compute_conversation_summary_from_roots_counts_conversation_size():
    root_map = {
        "1": "root-1",
        "2": "root-1",
        "3": "root-1",
        "4": "root-2",
        "5": "root-2",
        "6": "root-3",
    }
    summary = compute_conversation_summary_from_roots(root_map)
    assert summary["conversation_count"] == 3
    assert summary["max_messages_per_conversation"] == 3
    assert summary["pct_ge_2"] == 2.0 / 3.0


def test_normalize_text_handles_missing_values():
    series = pd.Series([" hello ", None, "world"])
    result = normalize_text(series)
    assert result.tolist() == ["hello", "", "world"]


def test_text_like_column_name_detection():
    assert text_like_column_name("text") is True
    assert text_like_column_name("tweet_id") is False
    assert text_like_column_name("response_tweet_id") is False


def test_as_bool_series_coerces_true_false_values():
    series = pd.Series([True, False, "true", "false", "1", "0", None])
    result = as_bool_series(series)
    assert result.tolist() == [True, False, True, False, True, False, False]
