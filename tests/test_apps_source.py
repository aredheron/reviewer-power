"""Source parsing, strict output decoding and audit evidence checks without execution."""

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from grader_power.datasets import apps_source as apps


def audit_module() -> ModuleType:
    """Load the CLI like other script tests without changing package search paths."""
    path = Path(__file__).resolve().parents[1] / "scripts/audit_apps.py"
    spec = importlib.util.spec_from_file_location("audit_apps", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source(io: dict) -> dict:
    """Build inert source data for parser checks."""
    return {
        "id": 1,
        "question": "Return a list.",
        "solutions": "[]",
        "input_output": json.dumps(io),
        "difficulty": "introductory",
        "url": "https://example.test/task",
        "starter_code": "def f(x):\n    ",
    }


def test_stream_preserves_strings_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    row = source(
        {"fn_name": "f", "inputs": [["__import__('os')"]], "outputs": [[["literal"]]]}
    )
    path = tmp_path / "source.jsonl"
    path.write_text(json.dumps(row) + "\n")
    assert list(apps.records(path)) == [row]
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        list(apps.records(path))


def test_wrapper_is_removed_exactly_once() -> None:
    row = source({"fn_name": "f", "inputs": [[1], [2]], "outputs": [[[[1, 2]]], [[]]]})
    name, cases = apps.callable_cases(row)
    assert name == "f"
    assert cases == [{"args": [1], "expected": [[1, 2]]}, {"args": [2], "expected": []}]


@pytest.mark.parametrize(
    "io",
    [
        {"fn_name": "f", "inputs": [[1]], "outputs": [[1, 2]]},
        {"fn_name": "f", "inputs": [1], "outputs": [[[1]]]},
        {"fn_name": "f", "inputs": [[1]], "outputs": []},
        {"fn_name": "f", "inputs": [], "outputs": []},
        {"inputs": [[1]], "outputs": [[[1]]]},
    ],
)
def test_no_heuristic_coercion_or_empty_suite(io: dict) -> None:
    with pytest.raises(ValueError):
        apps.callable_cases(source(io))


def test_unsupported_nested_json_is_retained_explicitly() -> None:
    row = source({})
    row["input_output"] = '{"inputs":[' + "1" * 10000 + "]}"
    result = apps.inspect_record(row, "train")
    assert result["task_id"] == "APPS/train/1"
    assert result["screening"] == "nested_json_unsupported"
    assert result["source_record_sha256"]


@pytest.mark.parametrize(
    "report",
    [
        {},
        {"results": []},
        {"results": [{"passed": 1, "error_type": None}]},
        {"results": [{"passed": True, "error_type": "RuntimeError"}]},
        {"results": [{"passed": False, "error_type": None}]},
    ],
)
def test_incomplete_or_invalid_evidence_is_rejected(report: dict) -> None:
    with pytest.raises(ValueError):
        apps.validate_details(report, 1)


def test_skipped_sandbox_verification_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.xml"
    path.write_text(
        '<testsuite><testcase classname="tests.integration.test_apps_docker" name="x"><skipped/></testcase></testsuite>'
    )
    with pytest.raises(ValueError, match="complete passing"):
        audit_module().verify_sandbox_report(path)


def test_frozen_bank_interfaces_sources_and_history() -> None:
    root = Path(__file__).resolve().parents[1]
    sources, refs = audit_module().load_bank(root)
    assert len(sources) == len(refs) == 34
    assert all(r["split"] == "train" for r in refs)
    assert len({r["source_id"] for r in refs}) == 34
