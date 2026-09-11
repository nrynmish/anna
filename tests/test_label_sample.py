import hashlib
import json
import os

from scripts import label_sample
from src.intent import taxonomy


TEST_INPUT = os.path.join(os.path.dirname(__file__), "..", "artifacts", "uber_support_cases.jsonl")
OUT_JSONL = os.path.join(os.path.dirname(__file__), "..", "artifacts", "sample_labels.jsonl")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def test_sample_size_and_determinism(tmp_path):
    # compute original checksum to ensure input not modified
    orig_hash = file_sha256(TEST_INPUT)

    labels1 = label_sample.sample_and_label(TEST_INPUT, OUT_JSONL, OUT_JSONL + ".md", sample_size=200, seed=label_sample.SAMPLE_SEED)
    labels2 = label_sample.sample_and_label(TEST_INPUT, OUT_JSONL, OUT_JSONL + ".md", sample_size=200, seed=label_sample.SAMPLE_SEED)

    # exactly 200 records
    assert len(labels1) == 200
    assert len(labels2) == 200

    # same case_ids in same order (deterministic)
    ids1 = [l["case_id"] for l in labels1]
    ids2 = [l["case_id"] for l in labels2]
    assert ids1 == ids2

    # input file unchanged
    assert file_sha256(TEST_INPUT) == orig_hash


def test_labels_valid_and_confidence():
    labels = label_sample.sample_and_label(TEST_INPUT, OUT_JSONL, OUT_JSONL + ".md", sample_size=200, seed=label_sample.SAMPLE_SEED)
    valid_ids = {i["id"] for i in taxonomy.TAXONOMY} | {"other", "unclear"}
    for l in labels:
        assert l["intent"] in valid_ids
        assert 0.0 <= float(l["labeling_confidence"]) <= 1.0
        assert l.get("taxonomy_version") == taxonomy.TAXONOMY_VERSION
