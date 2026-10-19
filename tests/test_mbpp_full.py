"""Original MBPP import and validation contracts; no benchmark code runs on host."""

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from grader_power.datasets import mbpp_full as mbpp


def record() -> dict[str, Any]:
    """Provide an independent assertion task, including a setup dependency."""
    return {
        "task_id": 1,
        "text": "Double numbers",
        "code": "def f(x): return x * 2",
        "test_setup_code": "x = 3",
        "test_list": ["assert f(x) == 6", "assert f(0) == 0", "assert f(-2) == -4"],
        "challenge_test_list": ["assert f(10) == 20"],
    }


def test_import_rejects_duplicates_incomplete_tests_and_nontext() -> None:
    row = record()
    raw = json.dumps(row).encode()
    assert mbpp.parse_records(raw)[1] == row
    with pytest.raises(ValueError, match="duplicate"):
        mbpp.parse_records(raw + b"\n" + raw)
    with pytest.raises(ValueError, match="three"):
        mbpp.parse_records(json.dumps(row | {"test_list": ["assert True"]}).encode())
    with pytest.raises(TypeError, match="code"):
        mbpp.parse_records(json.dumps(row | {"code": None}).encode())


def test_pinned_load_and_write_refuse_changes(tmp_path: Path) -> None:
    path = tmp_path / mbpp.CACHE
    mbpp.write_once(path, b"old")
    mbpp.write_once(path, b"old")
    with pytest.raises(ValueError, match="replace"):
        mbpp.write_once(path, b"new")
    mbpp.write_once(
        tmp_path / mbpp.MANIFEST,
        json.dumps({"revision": mbpp.REVISION, "sha256": mbpp.SOURCE_SHA256}).encode(),
    )
    with pytest.raises(ValueError, match="binding"):
        mbpp.load(tmp_path)


def test_signature_does_not_evaluate_defaults() -> None:
    code = "def f(x=raise_if_executed()): return x"
    assert mbpp.signature(code, "f")
    with pytest.raises(ValueError, match="one"):
        mbpp.signature(code, "absent")


@pytest.mark.parametrize(
    "details",
    [[], [{"passed": True}] * 4, [{"passed": True, "error_type": "ValueError"}] * 4],
)
def test_incomplete_reports_fail_closed_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, details: list[dict]
) -> None:
    instances = []

    class FakeWorkspace:
        """Avoid Docker while checking the host report acceptance boundary."""

        cleanup_verified = False

        def __init__(self, *args: Any) -> None:
            instances.append(self)

        def start(self) -> None:
            pass

        def command(self, _: str) -> dict:
            return {"exit_code": 0, "output": json.dumps({"results": details})}

        def close(self) -> None:
            self.cleanup_verified = True

    monkeypatch.setattr(mbpp, "DockerWorkspace", FakeWorkspace)
    with pytest.raises(ValueError, match="evidence"):
        mbpp.evaluate_assertions(record(), record()["code"], "synthetic")
    assert instances and instances[0].cleanup_verified


def test_no_assertion_cannot_count_as_a_pass() -> None:
    source: dict[str, Any] = record() | {"test_list": ["f(3)"]}
    with pytest.raises(ValueError, match="no assertion"):
        mbpp.evaluate_assertions(source, source["code"], "synthetic")


@pytest.fixture
def audit_module() -> Any:
    """Load the standalone audit CLI without running its entry point."""
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "audit_mbpp_full", root / "scripts/audit_mbpp_full.py"
    )
    assert spec is not None and spec.loader is not None
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    return audit


def test_sandbox_admission_requires_complete_passing_named_checks(
    tmp_path: Path, audit_module: Any
) -> None:
    audit = audit_module
    report = tmp_path / "checks.xml"
    cases = []
    for name, count in audit.SANDBOX_CHECKS.items():
        cases.extend(
            f'<testcase classname="{name}" name="check{i}"/>' for i in range(count)
        )
    report.write_text("<testsuite>" + "".join(cases) + "</testsuite>")
    audit.verify_sandbox_report(report)
    for invalid in (
        cases[:-1],
        cases[:-1] + [cases[-1].replace("/>", "><skipped/></testcase>")],
        [cases[0]] * 10,
    ):
        report.write_text("<testsuite>" + "".join(invalid) + "</testsuite>")
        with pytest.raises(ValueError, match="sandbox"):
            audit.verify_sandbox_report(report)


def test_original_mbpp_namespace_works_with_legacy_diagnostics(
    audit_module: Any,
) -> None:
    reference = {
        "source_id": 27,
        "task_id": "MbppFull/27",
        "entry_point": "f",
        "family": "numeric_mapping",
        "notes": "Synthetic reference",
        "implementations": {
            "comprehension": "def f(xs): return [x * 2 for x in xs]",
            "loop": "def f(xs):\n    out = []\n    for x in xs:\n        out.append(x * 2)\n    return out\n",
        },
    }
    result = audit_module.static_audit(reference)
    assert result["static_status"] == "pass"
    assert reference["task_id"] == "MbppFull/27"
    assert result["scoring_region"]["task_id"] == "Mbpp/27"
