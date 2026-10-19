"""Test reference audit/transport without importing or executing reference code."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from grader_power.datasets import reference_solutions as references
from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.oracle import OracleResult

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "sha256:" + "a" * 64


def pair_data() -> dict[str, Any]:
    """Use synthetic code strings that must never be evaluated by these tests."""
    return {
        "schema_version": 1,
        "task_id": "HumanEval/999",
        "entry_point": "transform",
        "transformation": "Double each value in input order",
        "implementations": {
            "comprehension": "def transform(items):\n    return [2 * value for value in items]\n",
            "loop": "def transform(items):\n    result = []\n    for value in items:\n        result.append(2 * value)\n    return result\n",
        },
        "review_notes": ["Synthetic contract; no human approval or execution."],
    }


def synthetic_dataset() -> LoadedDataset:
    """Build inert trusted input metadata for a mocked oracle transport."""
    return LoadedDataset(
        tasks={
            "HumanEval/999": {
                "task_id": "HumanEval/999",
                "entry_point": "transform",
                "prompt": "def transform(items):\n",
                "canonical_solution": "    return [2 * value for value in items]\n",
                "base_input": [[[1]]],
                "plus_input": [[[2]], [[]]],
                "atol": 0,
            }
        },
        provenance={"name": "HumanEval+", "variant": "full"},
    )


def test_audit_does_not_execute_module_or_function() -> None:
    """A top-level trap and undefined helper can compile/score but must never run."""
    data = pair_data()
    data["implementations"]["comprehension"] = (
        "raise AssertionError('reference code must stay inert')\n"
        "def transform(items):\n    return [undefined(value) for value in items]\n"
    )
    result = references.audit_pair(
        references.ReferencePair.model_validate(data), "transform(items)"
    )
    assert result["static_status"] == "pass"
    assert "full_correct" not in result


@pytest.mark.parametrize(
    "code,compile_status,signature_status",
    [
        (
            "def transform(items, extra=0):\n    return [x for x in items]\n",
            "pass",
            "mismatch",
        ),
        (
            "def transform(items: list):\n    return [x for x in items]\n",
            "pass",
            "mismatch",
        ),
        (
            "def transform(items, /):\n    return [x for x in items]\n",
            "pass",
            "mismatch",
        ),
        (
            "def transform(items) -> list:\n    return [x for x in items]\n",
            "pass",
            "mismatch",
        ),
        (
            "def different(items):\n    return [x for x in items]\n",
            "pass",
            "unavailable",
        ),
        ("def transform(items)\n    return []\n", "invalid", "unavailable"),
    ],
)
def test_contract_and_syntax_failures_remain_explicit(
    code: str, compile_status: str, signature_status: str
) -> None:
    """Reject changed call contracts without treating valid syntax as a parse error."""
    data = pair_data()
    data["implementations"]["comprehension"] = code
    result = references.audit_pair(
        references.ReferencePair.model_validate(data), "transform(items)"
    )
    item = result["implementations"]["comprehension"]
    assert item["compile_status"] == compile_status
    assert item["signature_status"] == signature_status
    assert result["static_status"] == "needs_resolution"


def test_wrong_style_and_postsort_are_not_repaired() -> None:
    """A named style is not evidence; unsupported natural constructions stay visible."""
    data = pair_data()
    data["implementations"]["comprehension"] = data["implementations"]["loop"]
    data["implementations"]["loop"] = data["implementations"]["loop"].replace(
        "    return result", "    result.sort()\n    return result"
    )
    result = references.audit_pair(
        references.ReferencePair.model_validate(data), "transform(items)"
    )
    assert result["static_status"] == "needs_resolution"
    assert result["implementations"]["comprehension"]["style"]["label"] == "loop_only"
    assert result["implementations"]["loop"]["style"]["status"] == "region_unsupported"


def test_bundle_schema_rejects_payloads_approvals_and_invalid_ids() -> None:
    """Reject hidden oracle fields and status assertions at the bundle boundary."""
    for key, value in (
        ("base_input", []),
        ("human_review", "approved"),
        ("task_id", "../../file"),
    ):
        data = pair_data()
        data[key] = value
        with pytest.raises(ValidationError):
            references.ReferencePair.model_validate(data)
    with pytest.raises(ValueError):
        references.reference_path("../../file")


def test_report_does_not_claim_failed_signatures_are_preserved() -> None:
    """A failing audit must not become a positive signature claim in the report."""
    pair = references.ReferencePair.model_validate(pair_data())
    row = {
        "task_id": pair.task_id,
        "entry_point": pair.entry_point,
        "family": "synthetic_mapping",
        "reference_bundle": {"path": references.reference_path(pair.task_id)},
        "suite_counts": {"base": 1, "plus": 2},
        "review_notes": pair.review_notes,
        **references.audit_pair(pair, "transform(items, extra)"),
    }
    report = references.render_reference_review([row])
    assert (
        "0 pairs pass signature, compilation, and intended AST style checks" in report
    )
    assert "signature `mismatch`" in report
    assert "preserve the original signature" not in report


def test_every_suitable_task_has_both_actual_reference_styles() -> None:
    """Audit all 102 delivered implementations with no dependency on source caches."""
    reviews = [
        json.loads(line)
        for line in (ROOT / references.REVIEW_PATH).read_text().splitlines()
    ]
    selected = {
        row["task_id"]: row
        for row in reviews
        if row["recommendation"] == "suitable_for_reference_validation"
    }
    paths = sorted((ROOT / references.REFERENCE_DIR).glob("*.json"))
    pairs = [
        references.ReferencePair.model_validate_json(path.read_text()) for path in paths
    ]
    assert len(pairs) == len(selected) == 51
    assert {pair.task_id for pair in pairs} == set(selected)
    for path, pair in zip(paths, pairs, strict=True):
        assert path == ROOT / references.reference_path(pair.task_id)
        result = references.audit_pair(
            pair, selected[pair.task_id]["original_signature"]
        )
        for item in result["implementations"].values():
            assert item["compile_status"] == item["signature_status"] == "pass"
        if pair.task_id == "HumanEval/58":
            assert result["static_status"] == "needs_resolution"
            assert all(
                item["style"]["status"] == "region_unsupported"
                for item in result["implementations"].values()
            )
        else:
            assert result["static_status"] == "pass"


def test_tracked_evidence_is_reproducible_and_never_approval() -> None:
    """Keep code hashes, actual AST evidence, report and provisional statuses in sync."""
    records = [
        json.loads(line)
        for line in (ROOT / references.MANIFEST_PATH).read_text().splitlines()
    ]
    assert len(records) == len({row["task_id"] for row in records}) == 51
    assert (
        ROOT / references.REPORT_PATH
    ).read_text() == references.render_reference_review(records)
    for row in records:
        pair_path = ROOT / row["reference_bundle"]["path"]
        pair = references.ReferencePair.model_validate_json(pair_path.read_text())
        assert row["reference_bundle"]["sha256"] == references.digest(
            pair_path.read_bytes()
        )
        for path, expected in row["provenance"].items():
            assert references.digest((ROOT / path).read_bytes()) == expected
        audit = references.audit_pair(pair, row["original_signature"])
        # JSON normalizes the scorer's dataclass tuples to arrays.
        assert {key: row[key] for key in audit} == json.loads(json.dumps(audit))
        assert row["statuses"] == {
            "author": "agent",
            "dual_style_correctness": "not_verified",
            "oracle_execution": "not_run",
            "human_review": "not_reviewed",
            "final_acceptance": "not_accepted",
            "pilot_membership": "not_assigned",
        }
        assert (
            not {
                "prompt",
                "canonical_solution",
                "base_input",
                "plus_input",
                "implementations_code",
            }
            & row.keys()
        )


@pytest.fixture
def synthetic_repository(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Supply one reviewed source and one pair for fail-closed provenance tests."""
    dataset = synthetic_dataset()
    monkeypatch.setattr(references, "load_screening_pool", lambda *_: [dataset])
    for path in (
        "manifests/evalplus_source.json",
        "src/grader_power/score_style.py",
        "src/grader_power/datasets/reference_solutions.py",
    ):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic non-executable artifact\n")
    task = dataset.tasks["HumanEval/999"]
    review = {
        "task_id": task["task_id"],
        "entry_point": task["entry_point"],
        "recommendation": "suitable_for_reference_validation",
        "source": {"name": "HumanEval+"},
        "evalplus_version": "0.3.1",
        "transformation_family": "synthetic_mapping",
        "original_signature": "transform(items)",
        "provenance": {
            "prompt_sha256": references.digest(task["prompt"]),
            "canonical_solution_sha256": references.digest(task["canonical_solution"]),
            "auxiliary_contract_sha256": references.digest(""),
            **{
                key: {
                    "path": path,
                    "sha256": references.digest((tmp_path / path).read_bytes()),
                }
                for key, path in (
                    ("source_manifest", "manifests/evalplus_source.json"),
                    ("scorer", "src/grader_power/score_style.py"),
                )
            },
        },
    }
    (tmp_path / references.REVIEW_PATH).write_text(json.dumps(review) + "\n")
    bundle = tmp_path / references.reference_path(task["task_id"])
    bundle.parent.mkdir(parents=True)
    bundle.write_text(json.dumps(pair_data()) + "\n")
    return tmp_path


def test_full_static_audit_is_stable(synthetic_repository: Path) -> None:
    """Repeated source-linked audits produce identical metadata without execution."""
    _, _, first = references.audit_references(synthetic_repository)
    _, _, second = references.audit_references(synthetic_repository)
    assert first == second
    assert first[0]["suite_counts"] == {"base": 1, "plus": 2}


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "extra",
        "wrong_id",
        "duplicate_review",
        "stale_source",
        "stale_contract",
        "stale_signature",
    ],
)
def test_population_and_stale_provenance_fail_closed(
    synthetic_repository: Path, change: str
) -> None:
    """No silent missing tasks, replacement IDs, duplicate reviews or stale sources."""
    path = synthetic_repository / references.reference_path("HumanEval/999")
    review_path = synthetic_repository / references.REVIEW_PATH
    review = json.loads(review_path.read_text())
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (path.parent / "extra.json").write_text(path.read_text())
    elif change == "wrong_id":
        data = pair_data()
        data["task_id"] = "HumanEval/998"
        path.write_text(json.dumps(data))
    elif change == "duplicate_review":
        review_path.write_text(review_path.read_text() * 2)
    else:
        if change == "stale_signature":
            review["original_signature"] = "transform(items, extra)"
        else:
            key = (
                "auxiliary_contract_sha256"
                if change == "stale_contract"
                else "canonical_solution_sha256"
            )
            review["provenance"][key] = "0" * 64
        review_path.write_text(json.dumps(review) + "\n")
    with pytest.raises(ValueError):
        references.audit_references(synthetic_repository)


@pytest.mark.parametrize("verify_upstream", [False, True])
def test_oracle_transport_preserves_both_styles_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, verify_upstream: bool
) -> None:
    """Mock full execution: unchanged task/code goes to the existing Docker adapter."""
    pair = references.ReferencePair.model_validate(pair_data())
    dataset = synthetic_dataset()
    original = deepcopy(dataset.tasks)
    suites = {
        name: {"status": "pass", "details": [True] * count, "total": count}
        for name, count in (("base", 1), ("plus", 2))
    }
    fake = Mock(
        side_effect=[
            OracleResult(
                "pass",
                pair.task_id,
                IMAGE,
                references.digest(pair.implementations.comprehension),
                "synthetic",
                suites,
            ),
            OracleResult(
                "wrong_answer",
                pair.task_id,
                IMAGE,
                references.digest(pair.implementations.loop),
                "synthetic",
                {
                    "base": suites["base"],
                    "plus": {
                        "status": "wrong_answer",
                        "details": [True, False],
                        "total": 2,
                    },
                },
            ),
        ]
    )
    monkeypatch.setattr(references, "docker_available", lambda: True)
    monkeypatch.setattr(references, "evaluate_solution", fake)
    output = tmp_path / "run"
    results = references.run_reference_oracles(
        [pair],
        [dataset],
        [],
        image=IMAGE,
        output_dir=output,
        verify_upstream=verify_upstream,
    )
    assert [row["full_correct"] for row in results] == [True, False]
    assert [row["style"] for row in results] == ["comprehension", "loop"]
    assert dataset.tasks == original
    for call, style in zip(fake.call_args_list, references.STYLES, strict=True):
        assert call.args == (
            dataset,
            "humaneval",
            pair.task_id,
            getattr(pair.implementations, style),
        )
        assert call.kwargs == {
            "mode": "full",
            "image": IMAGE,
            "verify_upstream": verify_upstream,
        }
    with pytest.raises(FileExistsError):
        references.run_reference_oracles(
            [pair], [dataset], [], image=IMAGE, output_dir=output
        )
    assert fake.call_count == 2


def test_missing_sandbox_has_no_host_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absent daemon must not execute or manufacture an attempt."""
    fake = Mock()
    monkeypatch.setattr(references, "docker_available", lambda: False)
    monkeypatch.setattr(references, "evaluate_solution", fake)
    with pytest.raises(RuntimeError, match="prerequisite"):
        references.run_reference_oracles(
            [], [], [], image=IMAGE, output_dir=tmp_path / "run"
        )
    fake.assert_not_called()
    assert not (tmp_path / "run").exists()
    with pytest.raises(ValueError, match="immutable"):
        references.run_reference_oracles(
            [], [], [], image="mutable:latest", output_dir=tmp_path / "run"
        )


def test_infrastructure_failure_retains_attempt_and_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not hide failed cleanup or continue creating containers after infra errors."""
    pair = references.ReferencePair.model_validate(pair_data())
    fake = Mock(
        return_value=OracleResult(
            "infrastructure_error",
            pair.task_id,
            IMAGE,
            "a" * 64,
            "synthetic cleanup failed",
        )
    )
    monkeypatch.setattr(references, "docker_available", lambda: True)
    monkeypatch.setattr(references, "evaluate_solution", fake)
    with pytest.raises(RuntimeError, match="remaining references unattempted"):
        references.run_reference_oracles(
            [pair], [synthetic_dataset()], [], image=IMAGE, output_dir=tmp_path / "run"
        )
    assert fake.call_count == 1
    attempt = json.loads(
        (tmp_path / "run/humaneval-999-comprehension.json").read_text()
    )
    assert attempt["status"] == "infrastructure_error"
    assert attempt["full_correct"] is False
