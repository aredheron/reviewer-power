"""Audit provisional reference code as data; execution uses only the Docker oracle.

JSON bundles keep complete reference modules out of import discovery and model
fixtures. Static checks establish syntax/signature/style evidence, not behavior.
"""

import ast
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator

from grader_power.code_extract import extract_code, find_target
from grader_power.config import Contract, SchemaVersion, Text
from grader_power.datasets.evalplus_loader import SOURCES, DatasetName, LoadedDataset
from grader_power.datasets.oracle import docker_available, evaluate_solution
from grader_power.datasets.task_filter import load_screening_pool, write_text_atomic
from grader_power.schemas import ScoringRegion, validate_identifier
from grader_power.score_style import score_style

STYLES = {"comprehension": "comprehension_only", "loop": "loop_only"}
REFERENCE_DIR = "fixtures/reference_solutions"
REVIEW_PATH = "manifests/task_suitability_review.jsonl"
MANIFEST_PATH = "manifests/reference_validation.jsonl"
REPORT_PATH = "planning/reference_validation_review.md"


class Implementations(Contract):
    """Complete, inert Python modules; no completion concatenation or repairs."""

    comprehension: Text
    loop: Text


class ReferencePair(Contract):
    """Agent-authored pair with no eligibility or approval fields."""

    schema_version: SchemaVersion
    task_id: Text
    entry_point: Text
    transformation: Text
    implementations: Implementations
    review_notes: list[Text] = Field(min_length=1)

    @field_validator("task_id")
    @classmethod
    def source_id(cls, value: str) -> str:
        """Preserve an original numeric source ID and reject path-like values."""
        if not re.fullmatch(r"(?:HumanEval|Mbpp)/[0-9]+", value):
            raise ValueError("Reference task_id must be an original EvalPlus ID")
        return value

    @field_validator("entry_point")
    @classmethod
    def entry(cls, value: str) -> str:
        """Require a valid Python function name without evaluating anything."""
        validate_identifier(value)
        return value


def digest(content: str | bytes) -> str:
    """Hash exact UTF-8 code or artifact bytes, including their final newline."""
    return hashlib.sha256(
        content.encode() if isinstance(content, str) else content
    ).hexdigest()


def reference_path(task_id: str) -> str:
    """Map an original ID to a stable artifact name, not an accepted fixture ID."""
    if not re.fullmatch(r"(?:HumanEval|Mbpp)/[0-9]+", task_id):
        raise ValueError("Invalid reference task ID")
    return f"{REFERENCE_DIR}/{task_id.lower().replace('/', '-')}.json"


def signature(source: str, entry_point: str) -> str:
    """Compare all argument kinds/defaults/annotations and return annotation by AST."""
    function, _ = find_target(ast.parse(source), entry_point)
    if not isinstance(function, ast.FunctionDef):
        raise TypeError("Reference requires one synchronous top-level target")
    return (
        ast.dump(function.args)
        + ":"
        + (ast.dump(function.returns) if function.returns is not None else "None")
    )


def audit_pair(pair: ReferencePair, original_signature: str) -> dict[str, Any]:
    """Compile without executing and locate each actual reference's target region.

    Code failures remain explicit records. Malformed metadata raises an error;
    compilation discards the code object without importing or invoking the module.
    """
    expected = signature(f"def {original_signature}:\n    pass\n", pair.entry_point)
    region = ScoringRegion(
        task_id=pair.task_id,
        entry_point=pair.entry_point,
        transformation=pair.transformation,
        rule="returned_list_v1",
    )
    results = {}
    for style, expected_label in STYLES.items():
        code = getattr(pair.implementations, style)
        compile_status = "pass"
        signature_status = "mismatch"
        try:
            compile(code, "<provisional-reference>", "exec", dont_inherit=True)
        except (SyntaxError, ValueError, TypeError):
            compile_status = "invalid"
        try:
            if signature(code, pair.entry_point) == expected:
                signature_status = "pass"
        except (SyntaxError, ValueError, TypeError):
            signature_status = "unavailable"
        score = score_style(extract_code(code, pair.entry_point), region)
        results[style] = {
            "sha256": digest(code),
            "compile_status": compile_status,
            "signature_status": signature_status,
            "expected_label": expected_label,
            "style": asdict(score),
            "static_checks_passed": compile_status == signature_status == "pass"
            and score.status == "scored"
            and score.label == expected_label,
        }
    return {
        "scoring_region": asdict(region),
        "implementations": results,
        "static_status": "pass"
        if all(item["static_checks_passed"] for item in results.values())
        else "needs_resolution",
    }


def audit_references(
    root: Path,
) -> tuple[list[ReferencePair], list[LoadedDataset], list[dict[str, Any]]]:
    """Check exact review population, pinned sources, and all code without execution.

    Reject missing/extra bundles or stale reviewed provenance. Source cache loading
    uses the existing hash-checked, no-download path. No source records are changed.
    """
    datasets = load_screening_pool(
        root / "data/evalplus", root / "manifests/evalplus_source.json"
    )
    reviews = [
        json.loads(line) for line in (root / REVIEW_PATH).read_text().splitlines()
    ]
    by_id = {row["task_id"]: row for row in reviews}
    if len(by_id) != len(reviews):
        raise ValueError("Duplicate suitability review IDs")
    selected = {
        task_id: row
        for task_id, row in by_id.items()
        if row["recommendation"] == "suitable_for_reference_validation"
    }
    if not selected:
        raise ValueError("No tasks recommended for reference validation")
    expected_paths = {reference_path(task_id) for task_id in selected}
    actual_paths = {
        str(path.relative_to(root)) for path in (root / REFERENCE_DIR).glob("*.json")
    }
    if actual_paths != expected_paths:
        raise ValueError("Reference bundles must match every suitable task exactly")
    source_tasks = {
        task_id: (dataset, task)
        for dataset in datasets
        for task_id, task in dataset.tasks.items()
    }
    fingerprints = {
        path: digest((root / path).read_bytes())
        for path in (
            REVIEW_PATH,
            "manifests/evalplus_source.json",
            "src/grader_power/score_style.py",
            "src/grader_power/datasets/reference_solutions.py",
        )
    }
    pairs = []
    records = []
    for task_id, review in sorted(selected.items()):
        path = reference_path(task_id)
        pair = ReferencePair.model_validate_json((root / path).read_text())
        if pair.task_id != task_id or pair.entry_point != review["entry_point"]:
            raise ValueError(f"Reference identity mismatch for {task_id}")
        dataset, task = source_tasks[task_id]
        for key, value in review["source"].items():
            if dataset.provenance[key] != value:
                raise ValueError(f"Reviewed source provenance is stale for {task_id}")
        for field in ("prompt", "canonical_solution"):
            if digest(task[field]) != review["provenance"][f"{field}_sha256"]:
                raise ValueError(f"Reviewed source text is stale for {task_id}")
        if (
            digest(task.get("contract", ""))
            != review["provenance"]["auxiliary_contract_sha256"]
        ):
            raise ValueError(f"Reviewed auxiliary contract is stale for {task_id}")
        for key in ("source_manifest", "scorer"):
            link = review["provenance"][key]
            if fingerprints.get(link["path"]) != link["sha256"]:
                raise ValueError(f"Reviewed {key} is stale for {task_id}")
        source_code = (
            task["prompt"] + task["canonical_solution"]
            if task_id.startswith("HumanEval/")
            else task["canonical_solution"]
        )
        original_signature = review["original_signature"]
        if signature(source_code, pair.entry_point) != signature(
            f"def {original_signature}:\n    pass\n", pair.entry_point
        ):
            raise ValueError(f"Reviewed signature is stale for {task_id}")
        records.append(
            {
                "schema_version": 1,
                "task_id": task_id,
                "entry_point": pair.entry_point,
                "source": review["source"],
                "evalplus_version": review["evalplus_version"],
                "family": review["transformation_family"],
                "reference_bundle": {
                    "path": path,
                    "sha256": digest((root / path).read_bytes()),
                },
                "provenance": fingerprints,
                "original_signature": original_signature,
                "source_text_hashes": {
                    field: review["provenance"][field]
                    for field in (
                        "prompt_sha256",
                        "canonical_solution_sha256",
                        "auxiliary_contract_sha256",
                    )
                },
                "suite_counts": {
                    name: len(task[f"{name}_input"]) for name in ("base", "plus")
                },
                **audit_pair(pair, original_signature),
                "review_notes": pair.review_notes,
                "statuses": {
                    "author": "agent",
                    "dual_style_correctness": "not_verified",
                    "oracle_execution": "not_run",
                    "human_review": "not_reviewed",
                    "final_acceptance": "not_accepted",
                    "pilot_membership": "not_assigned",
                },
            }
        )
        pairs.append(pair)
    return pairs, datasets, records


def render_reference_review(records: list[dict[str, Any]]) -> str:
    """Render reproducible static evidence and unresolved per-task notes."""
    passing = sum(row["static_status"] == "pass" for row in records)
    lines = [
        "# Provisional dual-style references",
        "",
        (
            "Generated by `uv run --locked python scripts/validate_references.py`. "
            "Reference code is agent-authored data under `fixtures/reference_solutions/`."
        ),
        "",
        (
            f"{len(records)} original tasks; {len(records) * 2} complete implementations. "
            f"{passing} pairs pass signature, compilation, and intended AST style checks; "
            f"{len(records) - passing} pairs need static/scoring resolution."
        ),
        "",
        (
            "These are proposed references, not accepted tasks. No reference or canonical "
            "solution executes during this audit. Full base/plus correctness, equivalent "
            "types/order/side effects, idiomaticity, and human QA remain unverified. "
            "No prompt edits, pilot assignments, or changes to the prior suitability "
            "recommendations are made. No grader effects are consulted."
        ),
        "",
        "## Verification boundary",
        "",
        (
            "The source pool is loaded only after its existing cache hashes match pinned "
            "provenance. Every task previously recommended for reference validation has "
            "one bundle; each module is compared with the original signature. Code "
            "objects are compiled and discarded, never executed. The unchanged "
            "`returned_list_v1` scorer records labels, features, and source spans for "
            "these actual references. A syntax/style pass says nothing about runtime correctness."
        ),
        "",
        (
            "Sandbox execution requires the build, isolation/cleanup integration gate, "
            "and source parity checks in `configs/sandbox/README.md`. The optional "
            "`--oracle-image` command uses only that existing isolated Docker adapter; "
            "it never builds/pulls an image or falls back to host execution. Execution "
            "attempts are stored separately in ignored `runs/reference-validation/`. "
            "Even two full EvalPlus passes do not prove unrestricted semantic equivalence, "
            "object identity, input nonmutation, or researcher approval."
        ),
        "",
        (
            "The known Mbpp/793 malformed plus-input record remains unchanged outside "
            "this 51-task reference population; its original oracle-review flag stands."
        ),
        "",
        "## Per-task evidence and remaining review",
        "",
    ]
    for row in records:
        styles = row["implementations"]
        lines.extend(
            [
                f"### {row['task_id']} — `{row['entry_point']}`",
                "",
                f"Bundle: `{row['reference_bundle']['path']}`. Family: `{row['family']}`.",
                "",
                f"Target: {row['scoring_region']['transformation']}",
                "",
                f"Static result: `{row['static_status']}`. "
                + "; ".join(
                    f"{style}: compile `{item['compile_status']}`, signature "
                    f"`{item['signature_status']}`, scorer `{item['style']['status']}` / "
                    f"`{item['style']['label']}`"
                    for style, item in styles.items()
                )
                + ".",
                "",
                (
                    f"Unexecuted source cases: base {row['suite_counts']['base']}, "
                    f"plus {row['suite_counts']['plus']}."
                ),
                "",
                *[f"- {note}" for note in row["review_notes"]],
                "",
            ]
        )
    return "\n".join(lines)


def run_reference_oracles(
    pairs: list[ReferencePair],
    datasets: list[LoadedDataset],
    records: list[dict[str, Any]],
    *,
    image: str,
    output_dir: Path,
    verify_upstream: bool = False,
) -> list[dict[str, Any]]:
    """Run both unchanged full suites only in a previously verified Docker image.

    Each explicit run uses a fresh output directory. Persist each result before
    continuing; no retries or successful-result substitution. Infrastructure
    failure stops the run with earlier attempts retained and remaining unattempted.
    The caller must complete the documented sandbox integration/parity gate first.
    """
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Use the immutable local image ID from the verified sandbox")
    if not docker_available():
        raise RuntimeError("Unmet prerequisite: usable, verified Docker sandbox")
    output_dir.mkdir(parents=True, exist_ok=False)
    write_text_atomic(
        output_dir / "static_manifest.jsonl",
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
    )
    source_by_task = {
        task_id: dataset for dataset in datasets for task_id in dataset.tasks
    }
    attempts = []
    for pair in pairs:
        dataset = source_by_task[pair.task_id]
        name: DatasetName = next(
            name
            for name, spec in SOURCES.items()
            if spec.name == dataset.provenance["name"]
        )
        for style in STYLES:
            result = evaluate_solution(
                dataset,
                name,
                pair.task_id,
                getattr(pair.implementations, style),
                mode="full",
                image=image,
                verify_upstream=verify_upstream,
            )
            attempt = {
                "style": style,
                "full_correct": result.full_correct,
                **asdict(result),
            }
            attempts.append(attempt)
            filename = f"{pair.task_id.lower().replace('/', '-')}-{style}.json"
            write_text_atomic(
                output_dir / filename,
                json.dumps(attempt, indent=2, sort_keys=True) + "\n",
            )
            if result.status == "infrastructure_error":
                raise RuntimeError(
                    "Oracle infrastructure failed; prior attempts retained, remaining references unattempted"
                )
    return attempts
