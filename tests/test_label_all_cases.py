import json
import importlib.util
from pathlib import Path

from src.intent import taxonomy


def load_module_from_path(path: Path):
    spec = importlib.util.spec_from_file_location("label_all_cases", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_label_all_cases_small(tmp_path):
    script_path = Path("scripts") / "label_all_cases.py"
    mod = load_module_from_path(script_path)

    inp = tmp_path / "cases.jsonl"
    out = tmp_path / "labels.jsonl"
    summary = tmp_path / "summary.md"

    cases = [
        {"case_id": "c1", "customer_id": "u1", "customer_messages": ["Charged me $140 for a $90 ride. This was after quoting me $65"]},
        {"case_id": "c2", "customer_id": "u2", "customer_messages": ["I need someone to call my mom back she left her bag in an Uber and nobody is responding."]},
        {"case_id": "c3", "customer_id": "u3", "customer_messages": ["I was charged twice, please refund me the extra charge"]},
    ]

    with inp.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    res = mod.process(str(inp), str(out), str(summary), progress_interval=1)

    assert res["total"] == 3

    intents = [json.loads(l)["intent"] for l in out.read_text(encoding="utf-8").splitlines()]
    assert "payments_charges" in intents
    assert "lost_and_found" in intents
    assert "refunds_adjustments" in intents

    summary_text = summary.read_text(encoding="utf-8")
    assert f"taxonomy_version: {taxonomy.TAXONOMY_VERSION}" in summary_text
