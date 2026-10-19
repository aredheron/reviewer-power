"""Freeze human-reviewed pilot membership and verify references in Docker only."""

import hashlib
import json
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from grader_power.datasets.evalplus_loader import DatasetName, LoadedDataset
from grader_power.datasets.oracle import evaluate_solution
from grader_power.datasets.reference_solutions import ReferencePair, audit_references
from grader_power.schemas import RenderTask, ScoringRegion

MANIFEST = "manifests/tasks_pilot.jsonl"
SELECTION = "manifests/pilot_selection.json"
APPROVAL = "manifests/task_human_review.json"
VERIFICATION = "data/sandbox/pilot-task-verification.json"
SEED = "grader-power-pilot-11-20261010"
DEVELOPMENT_TASK = "HumanEval/7"
SCORER_COVERAGE_EXCLUSIONS = {
    "Mbpp/749": (
        "Natural convert-then-sort implementations are outside returned_list_v1; "
        "the proposed pair instead repeats integer conversion to move sorting "
        "outside the scored region. Excluded before inference for scorer coverage, "
        "not as a reversal of human task suitability approval."
    )
}
ORACLE_FILES = (
    "configs/sandbox/Dockerfile",
    "configs/sandbox/runner.py",
    "src/grader_power/datasets/oracle.py",
    "tests/integration/test_oracle_docker.py",
)


def digest(content: str | bytes) -> str:
    """Hash exact persisted bytes, including newlines, for frozen provenance."""
    return hashlib.sha256(
        content.encode() if isinstance(content, str) else content
    ).hexdigest()


def save(path: Path, value: Any) -> None:
    """Create an exclusive artifact; never overwrite earlier selection or evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class PilotTask:
    """Separate the renderer's public contract from private references and tests."""

    dataset: LoadedDataset
    reference: ReferencePair
    visible: RenderTask
    region: ScoringRegion
    provenance: dict[str, Any]

    @property
    def dataset_name(self) -> DatasetName:
        """Return the unchanged source's adapter name."""
        return (
            "humaneval" if self.reference.task_id.startswith("HumanEval/") else "mbpp"
        )


def select_tasks(records: list[dict[str, Any]]) -> list[str]:
    """Choose 11 distinct families without using model or runtime outcomes.

    Include the already exposed development task to leave 40 unexposed source
    tasks. Then hash-rank statically passing candidates, with five HumanEval and
    six MBPP tasks. The seed is a fixed date label, not tuned against outcomes.
    """
    by_id = {row["task_id"]: row for row in records}
    if len(by_id) != len(records) or DEVELOPMENT_TASK not in by_id:
        raise ValueError(
            "Unique candidates including the development task are required"
        )
    first = by_id[DEVELOPMENT_TASK]
    if first["static_status"] != "pass":
        raise ValueError("Development task must pass static validation")
    chosen = [DEVELOPMENT_TASK]
    families = {first["family"]}
    counts = {"HumanEval": 1, "Mbpp": 0}
    quotas = {"HumanEval": 5, "Mbpp": 6}
    for row in sorted(records, key=lambda item: digest(f"{SEED}:{item['task_id']}")):
        source = row["task_id"].split("/")[0]
        if (
            row["static_status"] == "pass"
            and row["task_id"] not in SCORER_COVERAGE_EXCLUSIONS
            and counts[source] < quotas[source]
            and row["family"] not in families
        ):
            chosen.append(row["task_id"])
            families.add(row["family"])
            counts[source] += 1
    if len(chosen) != 11 or counts != quotas:
        raise ValueError("Insufficient distinct statically passing families for quotas")
    return chosen


def _task_record(
    dataset: LoadedDataset, pair: ReferencePair, audit: dict[str, Any], control: bool
) -> tuple[dict[str, Any], RenderTask, ScoringRegion]:
    """Append an empty stub without modifying the original task statement."""
    source = dataset.tasks[pair.task_id]
    derived_id = pair.task_id.lower().replace("/", "-") + "-pilot"
    stub = source["prompt"]
    adaptation = "Original prompt unchanged; append TODO and NotImplementedError only."
    if pair.task_id.startswith("Mbpp/"):
        stub += f"\n\ndef {audit['original_signature']}:\n"
        adaptation = (
            "Original prompt unchanged; append original canonical signature, TODO, "
            "and NotImplementedError. No canonical implementation is visible."
        )
    stub += "    # TODO\n    raise NotImplementedError\n"
    visible = RenderTask(
        source_task_id=pair.task_id,
        derived_task_id=derived_id,
        family=audit["family"],
        stub_path="src/task.py",
        stub=stub,
        readme=("# Sequence utilities\n\nUtilities for working with sequences.\n"),
    )
    region = ScoringRegion(
        task_id=derived_id,
        entry_point=pair.entry_point,
        transformation=pair.transformation,
        rule="returned_list_v1",
    )
    record = {
        "schema_version": 1,
        "source_task_id": pair.task_id,
        "derived_task_id": derived_id,
        "family": audit["family"],
        "split": "pilot",
        "control_subset": control,
        "development_exposed": pair.task_id == DEVELOPMENT_TASK,
        "source": dataset.provenance,
        "evalplus_version": audit["evalplus_version"],
        "original_prompt_sha256": digest(source["prompt"]),
        "original_signature": audit["original_signature"],
        "visible_stub_sha256": digest(stub),
        "readme_sha256": digest(visible.readme),
        "stub_path": visible.stub_path,
        "fixture_path": f"fixtures/repos/{derived_id}",
        "scoring_region": asdict(region),
        "reference_bundle": audit["reference_bundle"],
        "reference_sha256": {
            style: digest(getattr(pair.implementations, style))
            for style in ("loop", "comprehension")
        },
        "suite_counts": audit["suite_counts"],
        "human_review": {
            "status": "approved",
            "date": "2026-10-10",
            "record": APPROVAL,
        },
        "adaptation": adaptation,
        "oracle_evidence": VERIFICATION,
    }
    return record, visible, region


def freeze_pilot_tasks(root: Path) -> list[str]:
    """Freeze membership, user approval, and minimal fixtures before oracle outcomes."""
    for name in (MANIFEST, SELECTION, APPROVAL):
        if (root / name).exists():
            raise FileExistsError(f"Refusing to replace frozen artifact: {name}")
    pairs, datasets, records = audit_references(root)
    if len(pairs) != 51:
        raise ValueError("The human approval applies to exactly 51 reviewed candidates")
    selected = select_tasks(records)
    controls = [
        next(
            task
            for task in selected
            if task.startswith(prefix) and task != DEVELOPMENT_TASK
        )
        for prefix in ("HumanEval/", "Mbpp/")
    ]
    selection = {
        "schema_version": 1,
        "frozen_at": datetime.now(UTC).isoformat(),
        "seed": SEED,
        "procedure": (
            "Include HumanEval/7 because it is development-exposed; SHA256-rank "
            "seed:task_id, select static passes with unique families, quotas "
            "HumanEval=5 and Mbpp=6. No model treatment outcomes consulted."
        ),
        "reference_failure_policy": (
            "Stop before inference on any failure. Functional/scorer failures may "
            "justify a separately recorded replacement, taking the next hash-ranked "
            "static pass in the same source quota with an unused family. "
            "Infrastructure failures require repair, not task replacement."
        ),
        "selected_task_ids": selected,
        "control_task_ids": controls,
        "control_selection": "First nondevelopment selected task from each source.",
        "remaining_task_ids": sorted(
            row["task_id"] for row in records if row["task_id"] not in selected
        ),
        "static_exclusions": [
            row["task_id"] for row in records if row["static_status"] != "pass"
        ],
        "scorer_coverage_exclusions": SCORER_COVERAGE_EXCLUSIONS,
        "remaining_caveat": (
            "40 human-approved candidates remain, not 40 oracle-validated tasks. "
            "HumanEval/58 still requires static scoring resolution; Mbpp/749 "
            "requires scorer coverage of natural convert-then-sort solutions."
        ),
        "review_manifest_sha256": digest(
            (root / "manifests/task_suitability_review.jsonl").read_bytes()
        ),
        "reference_bundle_sha256": {
            row["task_id"]: row["reference_bundle"]["sha256"] for row in records
        },
    }
    save(root / SELECTION, selection)
    save(
        root / APPROVAL,
        {
            "schema_version": 1,
            "review_date": "2026-10-10",
            "reviewer": "human_project_owner",
            "evidence": "User: i've already checked all 51, they look good.",
            "scope": (
                "Human approval of the 51 recommended task candidates. This record "
                "does not assert full-oracle validity or independent review of later fixture adaptations."
            ),
            "task_ids": sorted(pair.task_id for pair in pairs),
            "review_manifest_sha256": selection["review_manifest_sha256"],
            "pilot_size_override": {"previous": 12, "approved": 11, "remaining": 40},
        },
    )
    by_pair = {pair.task_id: pair for pair in pairs}
    by_record = {row["task_id"]: row for row in records}
    by_dataset = {task_id: dataset for dataset in datasets for task_id in dataset.tasks}
    frozen = []
    for task_id in selected:
        record, visible, _ = _task_record(
            by_dataset[task_id],
            by_pair[task_id],
            by_record[task_id],
            task_id in controls,
        )
        fixture = root / record["fixture_path"]
        (fixture / "src").mkdir(parents=True, exist_ok=False)
        (fixture / visible.stub_path).write_text(visible.stub)
        (fixture / "README.md").write_text(visible.readme)
        frozen.append(record)
    with (root / MANIFEST).open("x") as stream:
        stream.write("".join(json.dumps(row, sort_keys=True) + "\n" for row in frozen))
    return selected


def load_pilot_tasks(root: Path) -> tuple[PilotTask, ...]:
    """Reject source, reference, membership, or fixture drift before live inference."""
    pairs, datasets, audits = audit_references(root)
    rows = [json.loads(line) for line in (root / MANIFEST).read_text().splitlines()]
    selection = json.loads((root / SELECTION).read_text())
    approval = json.loads((root / APPROVAL).read_text())
    if (
        len(rows) != 11
        or [row["source_task_id"] for row in rows] != selection["selected_task_ids"]
        or selection["selected_task_ids"] != select_tasks(audits)
        or set(approval["task_ids"]) != {pair.task_id for pair in pairs}
        or approval["review_manifest_sha256"] != selection["review_manifest_sha256"]
        or selection["review_manifest_sha256"]
        != digest((root / "manifests/task_suitability_review.jsonl").read_bytes())
    ):
        raise ValueError("Frozen pilot selection or human-review provenance changed")
    by_pair = {pair.task_id: pair for pair in pairs}
    by_audit = {row["task_id"]: row for row in audits}
    by_dataset = {task_id: dataset for dataset in datasets for task_id in dataset.tasks}
    result = []
    for row in rows:
        task_id = row["source_task_id"]
        dataset, pair = by_dataset[task_id], by_pair[task_id]
        expected, visible, region = _task_record(
            dataset, pair, by_audit[task_id], task_id in selection["control_task_ids"]
        )
        if row != expected:
            raise ValueError(f"Frozen pilot contract changed: {task_id}")
        fixture = root / row["fixture_path"]
        if (fixture / visible.stub_path).read_text() != visible.stub or (
            fixture / "README.md"
        ).read_text() != visible.readme:
            raise ValueError(f"Frozen pilot fixture changed: {task_id}")
        result.append(PilotTask(dataset, pair, visible, region, row))
    return tuple(result)


def verification_bindings(root: Path, image: str) -> dict[str, Any]:
    """Bind evidence to immutable contracts and current sandbox/scorer code."""
    return {
        "image": image,
        "files": {
            name: digest((root / name).read_bytes())
            for name in (
                *ORACLE_FILES,
                MANIFEST,
                SELECTION,
                APPROVAL,
                "src/grader_power/score_style.py",
            )
        },
    }


def verify_pilot_tasks(
    root: Path, image: str, integration_report: Path
) -> dict[str, Any]:
    """Run every selected reference through full original suites and official parity.

    Requires the nine-case Docker integration report from the current adapter and
    image. Each result is persisted exclusively before proceeding. Failures stop
    collection; reference code never executes on the host. An earlier model
    demonstration is not a prerequisite. The report is test-run evidence, not a
    cryptographic certificate of sandbox isolation.
    """
    tasks = load_pilot_tasks(root)
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Use the immutable local image ID from the verified sandbox")
    directory = root / "runs/pilot-reference-validation"
    cases = ET.parse(integration_report).getroot().findall(".//testcase")
    if len(cases) != 9 or any(list(case) for case in cases):
        raise ValueError(
            "The fresh pilot integration report must contain nine passing cases"
        )
    directory.mkdir(parents=True, exist_ok=False)
    bindings = verification_bindings(root, image)
    report_hash = digest(integration_report.read_bytes())
    save(directory / "definition.json", bindings)
    attempts = []
    for task in tasks:
        for style in ("loop", "comprehension"):
            outcome = evaluate_solution(
                task.dataset,
                task.dataset_name,
                task.reference.task_id,
                getattr(task.reference.implementations, style),
                mode="full",
                image=image,
                verify_upstream=True,
            )
            record = {
                "style": style,
                **asdict(outcome),
                "full_correct": outcome.full_correct,
            }
            name = f"{task.visible.derived_task_id}-{style}.json"
            save(directory / name, record)
            attempts.append(record)
            print(f"{task.reference.task_id} {style}: {outcome.status}", flush=True)
            if not outcome.full_correct:
                raise RuntimeError(
                    f"Pilot reference failed: {task.reference.task_id} {style}; evidence retained"
                )
    if bindings != verification_bindings(root, image) or report_hash != digest(
        integration_report.read_bytes()
    ):
        raise RuntimeError(
            "Validation inputs changed during execution; evidence retained"
        )
    evidence = {
        "schema_version": 1,
        "verified_at": datetime.now(UTC).isoformat(),
        **bindings,
        "integration_report": {
            "sha256": report_hash,
            "cases": len(cases),
        },
        "checks": attempts,
        "passed": True,
    }
    save(root / VERIFICATION, evidence)
    return evidence


def require_pilot_verification(root: Path, image: str) -> dict[str, Any]:
    """Require all 22 exact reference solutions to pass complete parity-checked suites."""
    tasks = load_pilot_tasks(root)
    evidence = json.loads((root / VERIFICATION).read_text())
    if evidence.get("passed") is not True or any(
        evidence.get(key) != value
        for key, value in verification_bindings(root, image).items()
    ):
        raise ValueError("Pilot reference verification is missing or stale")
    checks = evidence.get("checks", [])
    expected = {
        (task.reference.task_id, style): task
        for task in tasks
        for style in ("loop", "comprehension")
    }
    if len(checks) != len(expected) or {
        (row["task_id"], row["style"]) for row in checks
    } != set(expected):
        raise ValueError("Both reference styles for all 11 tasks must be verified")
    for row in checks:
        task = expected[row["task_id"], row["style"]]
        if (
            row.get("full_correct") is not True
            or row.get("status") != "pass"
            or row.get("solution_sha256")
            != task.provenance["reference_sha256"][row["style"]]
        ):
            raise ValueError(
                "Reference verification failed or belongs to different code"
            )
        for suite, total in task.provenance["suite_counts"].items():
            result = row["suites"][suite]
            if (
                result["status"] != "pass"
                or result["total"] != total
                or len(result["details"]) != total
                or not all(result["details"])
                or result.get("upstream_parity", {}).get("status") != "matched"
            ):
                raise ValueError("Missing complete official parity evidence")
    return evidence
