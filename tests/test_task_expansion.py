"""Expansion coverage, lineage and admission checks without executing task code."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "audit_task_expansion", ROOT / "scripts/audit_task_expansion.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_membership_rejects_old_missing_and_duplicate_tasks() -> None:
    reviews = [{"task_id": "new", "decision": "validate_pair"}]
    refs = [{"task_id": "new"}]
    audit.check_membership({"old", "new"}, {"old"}, reviews, refs)
    for invalid in (
        [],
        reviews + reviews,
        [{"task_id": "old", "decision": "validate_pair"}],
    ):
        with pytest.raises(ValueError):
            audit.check_membership({"old", "new"}, {"old"}, invalid, refs)
    with pytest.raises(ValueError):
        audit.check_membership({"old", "new"}, {"old"}, reviews, [{"task_id": "old"}])


def test_style_requires_target_construction_and_rejects_mixed() -> None:
    code = "def task(xs):\n    out = [x * 2 for x in xs]\n    return sorted(out)\n"
    assert audit.reference_style(code, "task", "comprehension")["passed"]
    mixed = (
        "def task(xs):\n    out = [x for x in xs]\n    out.append(3)\n    return out\n"
    )
    assert not audit.reference_style(mixed, "task", "comprehension")["passed"]
    helper_only = "def helper(xs):\n    return [x for x in xs]\ndef task(xs):\n    return helper(xs)\n"
    assert not audit.reference_style(helper_only, "task", "comprehension")["passed"]


def test_sandbox_gate_rejects_skipped_or_partial_checks(tmp_path: Path) -> None:
    report = tmp_path / "gate.xml"
    report.write_text("<testsuite>" + "<testcase/>" * 9 + "</testsuite>")
    audit.verify_gate(report)
    for cases in (
        "<testcase/>" * 8,
        "<testcase/>" * 8 + "<testcase><skipped/></testcase>",
    ):
        report.write_text("<testsuite>" + cases + "</testsuite>")
        with pytest.raises(ValueError):
            audit.verify_gate(report)


def test_exclusive_artifact_writes_preserve_old_evidence(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    audit.save_new(path, {"old": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        audit.save_new(path, {"old": False})
    assert path.read_bytes() == original


def test_real_bank_excludes_every_old_accepted_task_and_records_holds() -> None:
    pairs, _, records, reviews = audit.load(ROOT)
    assert len(reviews) == 491
    assert len(pairs) == len(records) == 25
    assert all(row["human_review"] == "not_requested_yet" for row in records)
    assert sum(row["hold"] is None for row in records) == 14
    assert all(
        all(style["passed"] for style in row["explicit_target_style"].values())
        for row in records
    )
