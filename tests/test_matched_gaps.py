"""Free citation fixes must preserve semantic content and independent agreement."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import matched_gaps as study

ROOT = Path(__file__).resolve().parents[1]


def saved_case():
    proof = json.loads((study.base(ROOT) / "free-repairs-v1.json").read_text())
    identity = proof["repairs"][0]["assignment_id"]
    p = (
        ROOT
        / "runs/reviewer-observation/matched-syntax-v1/requests"
        / identity
        / "judge"
    )
    return json.loads((p / "raw.json").read_text())["completion"], json.loads(
        (p / "packet.json").read_text()
    )


def test_all_free_repairs_reproduce_without_semantic_changes():
    assert len(study.free_repairs(ROOT)["repairs"]) == 62
    text, packet = saved_case()
    fixed = study.normalize(text, packet)
    old = json.loads(text)
    for key in ("status", "flags", "target_forms", "rationale"):
        assert fixed["corrected"][key] == old[key]
    assert fixed["label"] == fixed["audit_label"]


def test_missing_rationale_or_wrong_forms_not_repaired():
    text, packet = saved_case()
    value = json.loads(text)
    value["rationale"] = ""
    with pytest.raises(ValueError):
        study.normalize(json.dumps(value), packet)
    value["rationale"] = "Cited source supports this form."
    value["target_forms"] = ["snake_case"]
    with pytest.raises(ValueError):
        study.normalize(json.dumps(value), packet)


def test_naming_ranges_cannot_be_repaired_by_quote_rule():
    text, packet = saved_case()
    packet["family"] = "naming"
    with pytest.raises(ValueError):
        study.normalize(text, packet)


def test_complete_scope_excludes_held_and_previously_recovered_attempts():
    rows = study.recovery_records(ROOT)
    assert len(rows) == 311
    assert all((r["study"], r["profile"]) != ("naming", "glm") for r in rows)
    assert all("matched-syntax-v1/requests/" in r["source_path"] for r in rows)
    assert sum(r["recovery_reason"] == "incomplete" for r in rows) == 165


def test_empty_rationale_requires_all_remaining_checks_and_source_agreement():
    b = study.base(ROOT)
    audit = json.loads((b / "operations/empty-rationale-review.json").read_text())
    case = next(r for r in audit["eligible"] if r["source"].endswith("matched-gaps-v1"))
    path = ROOT / case["source"] / "requests" / case["assignment_id"] / "judge"
    packet = json.loads((path / "packet.json").read_text())
    text = json.loads((path / "raw.json").read_text())["completion"]
    fixed = study.qualify_empty_rationale(text, packet)
    assert (
        fixed["judgment"]["rationale"] == "" and fixed["rationale_fabricated"] is False
    )
    assert fixed["label"] == case["label"] and fixed["strict_failed"]
    bad = json.loads(text)
    bad["evidence"][0]["end_line"] = 999999
    with pytest.raises(ValueError):
        study.qualify_empty_rationale(json.dumps(bad), packet)
    bad = json.loads(text)
    bad["target_forms"] = ["camelCase"]
    with pytest.raises(ValueError):
        study.qualify_empty_rationale(json.dumps(bad), packet)
    with pytest.raises(ValueError):
        study.qualify_empty_rationale(text + text, packet)
    with pytest.raises(ValueError):
        study.qualify_empty_rationale(
            text.replace('"rationale":""', '"rationale":"", "rationale":""'), packet
        )
