"""Check frozen selection and fail-closed reference gates without executing code."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from grader_power import pilot_tasks as pilot
from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.oracle import OracleResult
from grader_power.datasets.reference_solutions import ReferencePair

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:" + "a" * 64


@pytest.fixture
def prepared_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Use inert source/reference strings and mocked audit; no cache or Docker needed."""
    records = [
        json.loads(line)
        for line in (ROOT / "manifests/reference_validation.jsonl")
        .read_text()
        .splitlines()
    ]
    pairs = [
        ReferencePair.model_validate_json(
            (ROOT / row["reference_bundle"]["path"]).read_text()
        )
        for row in records
    ]
    tasks = {}
    for row in records:
        task_id = row["task_id"]
        tasks[task_id] = {
            "task_id": task_id,
            "entry_point": row["entry_point"],
            "prompt": (
                f"def {row['original_signature']}:\n"
                if task_id.startswith("HumanEval/")
                else '"""Original task statement and example."""\n'
            ),
            "base_input": [None] * row["suite_counts"]["base"],
            "plus_input": [None] * row["suite_counts"]["plus"],
        }
    dataset = LoadedDataset(
        tasks=tasks, provenance={"variant": "full", "name": "synthetic"}
    )
    monkeypatch.setattr(
        pilot, "audit_references", lambda root: (pairs, [dataset], records)
    )
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/task_suitability_review.jsonl").write_text(
        "review provenance\n"
    )
    for name in (*pilot.ORACLE_FILES, "src/grader_power/score_style.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("trusted source bytes\n")
    gate = {
        "passed": True,
        "image": IMAGE,
        "integration_cases": 9,
        "oracle_files": {
            name: pilot.digest((tmp_path / name).read_bytes())
            for name in pilot.ORACLE_FILES
        },
    }
    pilot.save(tmp_path / "data/sandbox/mode-a-example-verification.json", gate)
    report = tmp_path / "runs/pilot-verification/docker-20261010.xml"
    report.parent.mkdir(parents=True)
    report.write_text("<testsuite>" + '<testcase name="passed"/>' * 9 + "</testsuite>")
    return tmp_path


def passing_oracle(
    dataset: LoadedDataset, name: str, task_id: str, solution: str, **kwargs: Any
) -> OracleResult:
    """Produce realistic full-suite parity evidence without executing a code string."""
    assert kwargs == {"mode": "full", "image": IMAGE, "verify_upstream": True}
    suites = {}
    for suite in ("base", "plus"):
        count = len(dataset.tasks[task_id][f"{suite}_input"])
        suites[suite] = {
            "status": "pass",
            "total": count,
            "details": [True] * count,
            "upstream_parity": {
                "status": "matched",
                "official_status": "pass",
                "official_details": [True] * count,
            },
        }
    return OracleResult(
        "pass", task_id, IMAGE, pilot.digest(solution), "Full suites passed", suites
    )


def test_selection_ignores_order_and_model_outcomes() -> None:
    """Family/source balancing and development containment are deterministic."""
    records = [
        json.loads(line)
        for line in (ROOT / "manifests/reference_validation.jsonl")
        .read_text()
        .splitlines()
    ]
    selected = pilot.select_tasks(records)
    for record in records:
        record["grader_effect"] = 999 if record["task_id"] in selected else -999
    assert pilot.select_tasks(list(reversed(records))) == selected
    assert selected[0] == "HumanEval/7"
    assert len(selected) == len(set(selected)) == 11
    assert sum(task.startswith("HumanEval/") for task in selected) == 5
    assert "Mbpp/749" not in selected and "HumanEval/58" not in selected
    families = {row["family"] for row in records if row["task_id"] in selected}
    assert len(families) == 11


def test_freeze_approval_and_visible_contract(prepared_root: Path) -> None:
    """Keep 40 source IDs outside the pilot and expose no implementations or suites."""
    pilot.freeze_pilot_tasks(prepared_root)
    tasks = pilot.load_pilot_tasks(prepared_root)
    selection = json.loads((prepared_root / pilot.SELECTION).read_text())
    approval = json.loads((prepared_root / pilot.APPROVAL).read_text())
    assert len(selection["remaining_task_ids"]) == 40
    assert len(approval["task_ids"]) == 51
    assert "does not assert" in approval["scope"]
    controls = [task for task in tasks if task.provenance["control_subset"]]
    assert {task.dataset_name for task in controls} == {"humaneval", "mbpp"}
    assert all(task.reference.task_id != "HumanEval/7" for task in controls)
    for task in tasks:
        source_prompt = task.dataset.tasks[task.reference.task_id]["prompt"]
        assert task.visible.stub.startswith(source_prompt)
        assert "NotImplementedError" in task.visible.stub
        assert task.reference.implementations.loop not in task.visible.stub
        assert task.reference.implementations.comprehension not in task.visible.stub
        assert task.region.task_id == task.visible.derived_task_id
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        pilot.freeze_pilot_tasks(prepared_root)


def test_changed_fixture_or_membership_fails_closed(prepared_root: Path) -> None:
    """Do not silently use edited stubs even when the original source cache matches."""
    pilot.freeze_pilot_tasks(prepared_root)
    task = pilot.load_pilot_tasks(prepared_root)[0]
    path = prepared_root / task.provenance["fixture_path"] / task.visible.stub_path
    path.write_text(task.visible.stub + "# unreviewed edit\n")
    with pytest.raises(ValueError, match="fixture changed"):
        pilot.load_pilot_tasks(prepared_root)


def test_all_references_require_complete_official_parity(
    prepared_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every chosen style is separately checked; missing parity cannot pass the gate."""
    pilot.freeze_pilot_tasks(prepared_root)
    evaluator = Mock(side_effect=passing_oracle)
    monkeypatch.setattr(pilot, "evaluate_solution", evaluator)
    evidence = pilot.verify_pilot_tasks(
        prepared_root,
        IMAGE,
        prepared_root / "runs/pilot-verification/docker-20261010.xml",
    )
    assert evaluator.call_count == 22
    assert pilot.require_pilot_verification(prepared_root, IMAGE) == evidence
    del evidence["checks"][0]["suites"]["plus"]["upstream_parity"]
    (prepared_root / pilot.VERIFICATION).write_text(json.dumps(evidence))
    with pytest.raises(ValueError, match="official parity"):
        pilot.require_pilot_verification(prepared_root, IMAGE)


def test_failed_reference_stops_without_retries(
    prepared_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep failed evidence and do not mark the remaining references verified."""
    pilot.freeze_pilot_tasks(prepared_root)
    evaluator = Mock(
        return_value=OracleResult(
            "infrastructure_error", "HumanEval/7", IMAGE, "f" * 64, "Docker unavailable"
        )
    )
    monkeypatch.setattr(pilot, "evaluate_solution", evaluator)
    with pytest.raises(RuntimeError, match="evidence retained"):
        pilot.verify_pilot_tasks(
            prepared_root,
            IMAGE,
            prepared_root / "runs/pilot-verification/docker-20261010.xml",
        )
    assert evaluator.call_count == 1
    assert not (prepared_root / pilot.VERIFICATION).exists()
    assert (
        len(
            list(
                (prepared_root / "runs/pilot-reference-validation").glob("*-loop.json")
            )
        )
        == 1
    )


def test_failed_integration_report_prevents_execution(
    prepared_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Skipped or failed integration cases must block all reference execution."""
    pilot.freeze_pilot_tasks(prepared_root)
    report = prepared_root / "runs/pilot-verification/docker-20261010.xml"
    report.write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
    evaluator = Mock()
    monkeypatch.setattr(pilot, "evaluate_solution", evaluator)
    with pytest.raises(ValueError, match="nine passing cases"):
        pilot.verify_pilot_tasks(prepared_root, IMAGE, report)
    evaluator.assert_not_called()
