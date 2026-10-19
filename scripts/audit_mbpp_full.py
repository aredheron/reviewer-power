"""Validate original MBPP candidates without changing previous task banks."""

import argparse
import ast
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from grader_power.datasets import mbpp_full as mbpp
from grader_power.datasets.reference_solutions import ReferencePair, audit_pair

BANK = Path("fixtures/mbpp_full_v1")
SANDBOX_CHECKS = {
    "tests.integration.test_mode_b_docker": 5,
    "tests.integration.test_mode_b_isolation_independent": 1,
    "tests.integration.test_mbpp_full_docker": 4,
}


def verify_sandbox_report(path: Path) -> None:
    """Require the workspace, independent boundary and new assertion checks."""
    report = ET.parse(path).getroot()
    cases = list(report.iter("testcase"))
    counts = Counter(case.get("classname") for case in cases)
    identities = {(case.get("classname"), case.get("name")) for case in cases}
    if (
        len(cases) != 10
        or len(identities) != 10
        or counts != SANDBOX_CHECKS
        or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        )
    ):
        raise ValueError("A complete passing ten-check sandbox report is required")


def load_bank(root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Verify frozen sources, complete review coverage, interfaces and style pairs."""
    preservation = json.loads((root / BANK / "preserved-bindings.json").read_text())
    for name, expected in preservation.items():
        if mbpp.digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Previously saved artifact changed: {name}")
    sources = mbpp.load(root)
    refs = json.loads((root / BANK / "references.json").read_text())
    reviews = json.loads((root / BANK / "review.json").read_text())
    if len(reviews) != 974 or {r["source_id"] for r in reviews} != set(sources):
        raise ValueError("Review must account for all original source records")
    shortlisted = {r["source_id"] for r in reviews if r["decision"] == "validate_pair"}
    if len(refs) != len(shortlisted) or {r["source_id"] for r in refs} != shortlisted:
        raise ValueError("Reference and shortlist memberships differ")
    for ref in refs:
        source = sources[ref["source_id"]]
        if (
            ref["task_id"] != f"MbppFull/{source['task_id']}"
            or ref["source_record_sha256"]
            != mbpp.digest(json.dumps(source, sort_keys=True).encode())
            or ref["original_prompt"] != source["text"]
            or set(ref["implementations"]) != {"loop", "comprehension"}
        ):
            raise ValueError("Candidate source, prompt or identity changed")
        expected = mbpp.signature(source["code"], ref["entry_point"])
        for style, code in ref["implementations"].items():
            if mbpp.signature(code, ref["entry_point"]) != expected:
                raise ValueError("Reference changed the original function signature")
            target = next(
                node
                for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef) and node.name == ref["entry_point"]
            )
            nodes = list(ast.walk(target))
            comps = sum(isinstance(node, ast.ListComp) for node in nodes)
            appends = sum(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                for node in nodes
            )
            if not (
                comps > 0 and appends == 0
                if style == "comprehension"
                else comps == 0 and appends > 0
            ):
                raise ValueError("Reference does not contain the intended construction")
    return sources, refs


def static_audit(ref: dict[str, Any]) -> dict[str, Any]:
    """Retain legacy region diagnostics; do not change historical style scoring."""
    pair = ReferencePair(
        schema_version=1,
        # The legacy diagnostic schema accepts the upstream Mbpp/N spelling.
        # This adapter does not relabel the source bank or claim MBPP+ coverage.
        task_id=f"Mbpp/{ref['source_id']}",
        entry_point=ref["entry_point"],
        transformation=ref["family"],
        implementations=ref["implementations"],
        review_notes=[ref["notes"] or "Original MBPP supplied-test candidate audit"],
    )
    target = next(
        node
        for node in ast.parse(ref["implementations"]["comprehension"]).body
        if isinstance(node, ast.FunctionDef) and node.name == ref["entry_point"]
    )
    original_signature = f"{target.name}({ast.unparse(target.args)})"
    if target.returns is not None:
        original_signature += " -> " + ast.unparse(target.returns)
    return audit_pair(pair, original_signature)


def summarize(root: Path, run_dir: Path, refs: list[dict[str, Any]]) -> dict[str, Any]:
    """Count only complete, source-bound canonical and dual-reference outcomes."""
    run = json.loads((run_dir / "run.json").read_text())
    for filename, expected in run["bindings"].items():
        if mbpp.digest((root / filename).read_bytes()) != expected:
            raise ValueError(f"Validation binding changed: {filename}")
    sources = mbpp.load(root)
    rows = []
    for ref in refs:
        source = sources[ref["source_id"]]
        outcomes = {}
        for style, code in {
            "canonical": source["code"],
            **ref["implementations"],
        }.items():
            path = run_dir / "outcomes" / f"{ref['source_id']}-{style}.json"
            saved = json.loads(path.read_text())
            if (
                saved["source_id"] != ref["source_id"]
                or saved["code_sha256"] != mbpp.digest(code.encode())
                or saved["source_record_sha256"] != ref["source_record_sha256"]
                or saved["image"] != run["image"]
                or not saved["cleanup_verified"]
                or saved["supplied_tests"] != len(source["test_list"])
                or saved["challenge_tests"] != len(source["challenge_test_list"])
                or len(saved["details"])
                != len(source["test_list"] + source["challenge_test_list"])
                or saved["passed"] != all(x["passed"] for x in saved["details"])
            ):
                raise ValueError("Incomplete or changed validation outcome")
            outcomes[style] = {
                "passed": saved["passed"],
                "sha256": mbpp.digest(path.read_bytes()),
                "path": str(path.relative_to(root)),
            }
        passed = all(x["passed"] for x in outcomes.values())
        rows.append(
            {
                "task_id": ref["task_id"],
                "family": ref["family"],
                "hold": ref["hold"],
                "all_supplied_tests_pass": passed,
                "recommended_candidate": passed and ref["hold"] is None,
                "legacy_style_audit": static_audit(ref),
                "outcomes": outcomes,
            }
        )
    return {
        "source_tasks": 974,
        "outside_existing_mbpp_plus": 596,
        "reference_pairs": len(refs),
        "all_three_implementations_pass": sum(
            r["all_supplied_tests_pass"] for r in rows
        ),
        "recommended_candidates": sum(r["recommended_candidate"] for r in rows),
        "records": rows,
        "model_api_calls": 0,
        "coverage": "Supplied original MBPP assertions only, accepted by user; not MBPP+ coverage or a universal correctness proof",
        "human_review_and_experiment_admission": "not_claimed",
    }


def main() -> None:
    """Run the fixed candidate bank or verify saved outcomes offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "summarize"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sandbox-report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    run_dir.relative_to(root / "runs/mbpp-full-audit")
    sources, refs = load_bank(root)
    for ref in refs:
        static_audit(ref)
    if args.action == "validate":
        if args.sandbox_report is None:
            parser.error("--sandbox-report is required for validation")
        verify_sandbox_report(args.sandbox_report)
        if (run_dir / "run.json").exists():
            raise ValueError("Use a new validation directory; no automatic replay")
        image = (root / "data/sandbox/oracle-image.id").read_text().strip()
        paths = [
            BANK / f
            for f in ("references.json", "review.json", "preserved-bindings.json")
        ]
        paths += [
            mbpp.MANIFEST,
            mbpp.CACHE,
            Path("scripts/audit_mbpp_full.py"),
            Path("src/grader_power/datasets/mbpp_full.py"),
            Path("src/grader_power/agent_sandbox.py"),
            Path("tests/integration/test_mode_b_docker.py"),
            Path("tests/integration/test_mode_b_isolation_independent.py"),
            Path("tests/integration/test_mbpp_full_docker.py"),
            args.sandbox_report.resolve().relative_to(root),
        ]
        run = {
            "image": image,
            "bindings": {str(p): mbpp.digest((root / p).read_bytes()) for p in paths},
        }
        mbpp.write_once(
            run_dir / "run.json",
            (json.dumps(run, indent=2, sort_keys=True) + "\n").encode(),
        )
        for ref in refs:
            source = sources[ref["source_id"]]
            for style, code in {
                "canonical": source["code"],
                **ref["implementations"],
            }.items():
                result = mbpp.evaluate_assertions(source, code, image)
                mbpp.write_once(
                    run_dir / "outcomes" / f"{ref['source_id']}-{style}.json",
                    (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
                )
            print(f"Validated MbppFull/{ref['source_id']}", flush=True)
        result = summarize(root, run_dir, refs)
        mbpp.write_once(
            run_dir / "summary.json",
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        )
    else:
        result = summarize(root, run_dir, refs)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
