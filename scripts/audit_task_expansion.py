"""Verify a separate task-bank expansion without changing historical task banks."""

import argparse
import ast
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from grader_power.datasets.oracle import OracleResult
from grader_power.datasets.reference_solutions import (
    ReferencePair,
    audit_pair,
    run_reference_oracles,
)
from grader_power.datasets.task_filter import load_screening_pool

BANK = Path("fixtures/task_expansion_v1")


def sha(path: Path) -> str:
    """Hash exact saved bytes, including source and verification artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_new(path: Path, value: Any) -> None:
    """Write a new artifact exclusively; never replace previous evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def check_membership(
    all_ids: set[str], old_ids: set[str], reviews: list[dict], references: list[dict]
) -> None:
    """Require complete remaining-pool coverage and exact reference membership."""
    ids = [row["task_id"] for row in reviews]
    ref_ids = [row["task_id"] for row in references]
    if len(ids) != len(set(ids)) or set(ids) != all_ids - old_ids:
        raise ValueError(
            "Review must cover each nonaccepted original task exactly once"
        )
    expected = {row["task_id"] for row in reviews if row["decision"] == "validate_pair"}
    if len(ref_ids) != len(set(ref_ids)) or set(ref_ids) != expected:
        raise ValueError("References must exactly match the validation shortlist")
    if set(ref_ids) & old_ids:
        raise ValueError("Previously accepted tasks are not new tasks")


def reference_style(code: str, entry: str, style: str) -> dict:
    """Check explicit target-function constructions, retaining legacy scorer output.

    This structural check does not certify arbitrary model outputs. References
    receive separate semantic review, including helper and postprocessing scope.
    """
    module = ast.parse(code)
    target = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == entry
    )
    nodes = list(ast.walk(target))
    comprehensions = sum(isinstance(node, ast.ListComp) for node in nodes)
    appends = sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "append"
        for node in nodes
    )
    passed = (
        comprehensions > 0 and appends == 0
        if style == "comprehension"
        else comprehensions == 0 and appends > 0
    )
    return {
        "list_comprehensions": comprehensions,
        "append_calls": appends,
        "passed": passed,
    }


def load(root: Path) -> tuple[list[ReferencePair], list, list[dict], list[dict]]:
    """Load only hash-verified cached sources and preserve all historical bindings."""
    baseline = json.loads((root / BANK / "preserved-bindings.json").read_text())
    for filename, expected in baseline.items():
        if sha(root / filename) != expected:
            raise ValueError(f"Historical binding changed: {filename}")
    datasets = load_screening_pool(
        root / "data/evalplus", root / "manifests/evalplus_source.json"
    )
    tasks = {key: task for dataset in datasets for key, task in dataset.tasks.items()}
    old = {
        row["task_id"]
        for row in map(
            json.loads,
            (root / "manifests/task_suitability_review.jsonl").read_text().splitlines(),
        )
        if row["recommendation"] == "suitable_for_reference_validation"
    }
    references = json.loads((root / BANK / "references.json").read_text())
    reviews = json.loads((root / BANK / "review.json").read_text())
    check_membership(set(tasks), old, reviews, references)
    pairs = []
    records = []
    for row in references:
        task = tasks[row["task_id"]]
        code = task["canonical_solution"]
        if row["task_id"].startswith("HumanEval/"):
            code = task["prompt"] + code
        original = next(
            node
            for node in ast.parse(code).body
            if isinstance(node, ast.FunctionDef) and node.name == row["entry_point"]
        )
        signature = f"{original.name}({ast.unparse(original.args)})"
        if original.returns is not None:
            signature += " -> " + ast.unparse(original.returns)
        pair = ReferencePair.model_validate(
            {
                key: row[key]
                for key in (
                    "schema_version",
                    "task_id",
                    "entry_point",
                    "transformation",
                    "implementations",
                )
            }
            | {
                "review_notes": row["review_notes"]
                or ["Agent semantic review; human review remains separate."]
            }
        )
        audit = audit_pair(pair, signature)
        structural = {
            style: reference_style(value, row["entry_point"], style)
            for style, value in row["implementations"].items()
        }
        if not all(item["passed"] for item in structural.values()):
            raise ValueError(f"Reference construction check failed: {row['task_id']}")
        if not all(
            item["compile_status"] == item["signature_status"] == "pass"
            for item in audit["implementations"].values()
        ):
            raise ValueError(f"Reference syntax/signature failed: {row['task_id']}")
        record = {
            "task_id": row["task_id"],
            "family": row["family"],
            "hold": row["hold"],
            "human_review": row["human_review"],
            "source_sha256": {
                field: hashlib.sha256(task.get(field, "").encode()).hexdigest()
                for field in ("prompt", "canonical_solution", "contract")
            },
            "suite_counts": {
                suite: len(task[f"{suite}_input"]) for suite in ("base", "plus")
            },
            "explicit_target_style": structural,
            "legacy_static_audit": audit,
            "review_notes": row["review_notes"],
            "redundancy": row["redundancy"],
        }
        records.append(record)
        pairs.append(pair)
    return pairs, datasets, records, reviews


def verify_gate(path: Path) -> None:
    """Require an actual complete nine-test sandbox report, with no skipped tests."""
    suites = ET.parse(path).getroot()
    cases = suites.findall(".//testcase")
    if len(cases) != 9 or any(
        case.find(tag) is not None
        for case in cases
        for tag in ("failure", "error", "skipped")
    ):
        raise ValueError("All nine sandbox checks must pass before reference execution")


def summarize(root: Path, run_dir: Path, records: list[dict]) -> dict:
    """Check code/source bindings and full-suite outcomes before counting readiness."""
    run = json.loads((run_dir / "run.json").read_text())
    for filename, expected in run["bindings"].items():
        if sha(root / filename) != expected:
            raise ValueError(f"Run binding changed: {filename}")
    output = []
    for record in records:
        task_id = record["task_id"]
        results = {}
        for style, audit in record["legacy_static_audit"]["implementations"].items():
            path = (
                run_dir
                / "outcomes"
                / f"{task_id.lower().replace('/', '-')}-{style}.json"
            )
            saved = json.loads(path.read_text())
            result = OracleResult(
                **{
                    key: value
                    for key, value in saved.items()
                    if key not in {"style", "full_correct"}
                }
            )
            if (
                result.solution_sha256 != audit["sha256"]
                or result.image != run["image"]
            ):
                raise ValueError(f"Outcome binding changed: {task_id}/{style}")
            if result.task_id != task_id or saved["style"] != style:
                raise ValueError("Outcome identity mismatch")
            for suite in ("base", "plus"):
                if (
                    result.full_correct
                    and result.suites[suite]["total"] != record["suite_counts"][suite]
                ):
                    raise ValueError("Incomplete source suite")
            results[style] = {
                "full_correct": result.full_correct,
                "status": result.status,
                "path": str(path.relative_to(root)),
                "sha256": sha(path),
            }
        full = all(item["full_correct"] for item in results.values())
        output.append(
            record
            | {
                "outcomes": results,
                "both_references_pass": full,
                "recommended_without_changes": full and record["hold"] is None,
            }
        )
    return {
        "target_new_tasks": 64,
        "reviewed_remaining_pool": 491,
        "reference_pairs": len(records),
        "both_references_pass": sum(row["both_references_pass"] for row in output),
        "recommended_without_changes": sum(
            row["recommended_without_changes"] for row in output
        ),
        "records": output,
        "human_approval": "not_claimed",
        "model_inference_calls": 0,
        "interpretation": "Agent suitability review plus full finite-suite verification; no experimental collection or task split assigned.",
    }


def main() -> None:
    """Audit statically, execute in the verified sandbox, or summarize saved checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "execute", "summarize"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sandbox-report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    run_dir.relative_to(root / "runs/task-bank-expansion")
    pairs, datasets, records, reviews = load(root)
    if args.action == "audit":
        save_new(
            run_dir / "static-audit.json",
            {"records": records, "reviewed": len(reviews)},
        )
    elif args.action == "execute":
        if args.sandbox_report is None:
            parser.error("--sandbox-report is required for execution")
        verify_gate(args.sandbox_report)
        image = (root / "data/sandbox/oracle-image.id").read_text().strip()
        bound = [
            BANK / name
            for name in ("references.json", "review.json", "preserved-bindings.json")
        ]
        bound += [
            Path("scripts/audit_task_expansion.py"),
            Path("configs/sandbox/runner.py"),
            Path("src/grader_power/datasets/oracle.py"),
            Path("manifests/evalplus_source.json"),
        ]
        save_new(
            run_dir / "run.json",
            {
                "image": image,
                "verify_upstream": True,
                "bindings": {str(path): sha(root / path) for path in bound},
                "sandbox_report_sha256": sha(args.sandbox_report),
            },
        )
        run_reference_oracles(
            pairs,
            datasets,
            records,
            image=image,
            output_dir=run_dir / "outcomes",
            verify_upstream=True,
        )
        save_new(run_dir / "summary.json", summarize(root, run_dir, records))
    else:
        print(json.dumps(summarize(root, run_dir, records), indent=2))
    print(f"Audited {len(reviews)} remaining tasks; {len(pairs)} reference pairs.")


if __name__ == "__main__":
    main()
