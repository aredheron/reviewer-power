"""Validate the frozen APPS candidate pairs in Docker or verify saved results offline."""

import argparse
import ast
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from grader_power.datasets import apps_source as apps
from grader_power.datasets.mbpp_full import digest, signature, write_once

BANK = Path("fixtures/apps_v1")
SANDBOX_CHECKS = {
    "tests.integration.test_mode_b_docker": 5,
    "tests.integration.test_mode_b_isolation_independent": 1,
    "tests.integration.test_apps_docker": 5,
}


def verify_sandbox_report(path: Path) -> None:
    """Require real passing workspace, isolation and APPS callable-runner checks."""
    cases = list(ET.parse(path).getroot().iter("testcase"))
    if (
        len(cases) != 11
        or len({(c.get("classname"), c.get("name")) for c in cases}) != 11
        or Counter(c.get("classname") for c in cases) != SANDBOX_CHECKS
        or any(
            c.find(tag) is not None
            for c in cases
            for tag in ("failure", "error", "skipped")
        )
    ):
        raise ValueError("A complete passing eleven-check sandbox report is required")


def load_bank(root: Path) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Fail closed on changed historical files, sources, prompts, interfaces or styles."""
    for name, expected in json.loads(
        (root / BANK / "preserved-bindings.json").read_text()
    ).items():
        if apps.file_digest(root / name) != expected:
            raise ValueError(f"Historical artifact changed: {name}")
    refs = json.loads((root / BANK / "references.json").read_text())
    reviews = json.loads((root / BANK / "review.json").read_text())
    identities = {f"APPS/{split}/{i}" for split in apps.SPLITS for i in range(5000)}
    if len(reviews) != 10000 or {r["task_id"] for r in reviews} != identities:
        raise ValueError("Review must account for all 10000 source records")
    selected = {r["task_id"] for r in reviews if r["decision"] == "validate_pair"}
    if len(refs) != len(selected) or {r["task_id"] for r in refs} != selected:
        raise ValueError("Reference and review populations differ")
    sources = apps.load_selected(root, "train", {r["source_id"] for r in refs})
    for ref in refs:
        source = sources[ref["source_id"]]
        entry, cases = apps.callable_cases(source)
        canonical = json.loads(source["solutions"])[ref["canonical_solution_index"]]
        if (
            ref["task_id"] != f"APPS/train/{source['id']}"
            or ref["split"] != "train"
            or ref["entry_point"] != entry
            or ref["original_prompt"] != source["question"]
            or ref["original_starter_code"] != source["starter_code"]
            or ref["source_record_sha256"]
            != digest(json.dumps(source, sort_keys=True).encode())
            or ref["canonical_sha256"] != digest(canonical.encode())
            or ref["decoder"] != "singleton_wrapped_list"
            or set(ref["implementations"]) != {"loop", "comprehension"}
        ):
            raise ValueError("APPS source or reference binding changed")
        ast.parse(canonical)
        expected = signature(source["starter_code"] + "pass\n", entry)
        for style, code in ref["implementations"].items():
            if signature(code, entry) != expected:
                raise ValueError("APPS reference changed the interface")
            target = next(
                n
                for n in ast.parse(code).body
                if isinstance(n, ast.FunctionDef) and n.name == entry
            )
            nodes = list(ast.walk(target))
            comps = sum(isinstance(n, ast.ListComp) for n in nodes)
            appends = sum(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "append"
                for n in nodes
            )
            if not (
                comps > 0 and appends == 0
                if style == "comprehension"
                else comps == 0 and appends > 0
            ):
                raise ValueError("Missing intended list construction")
        if not cases:
            raise ValueError("No supplied APPS cases")
    return sources, refs


def summarize(
    root: Path,
    run_dir: Path,
    sources: dict[int, dict[str, Any]],
    refs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Count complete, source-bound canonical and paired-reference evidence only."""
    run = json.loads((run_dir / "run.json").read_text())
    for name, expected in run["bindings"].items():
        if apps.file_digest(root / name) != expected:
            raise ValueError(f"APPS verification binding changed: {name}")
    rows = []
    for ref in refs:
        source = sources[ref["source_id"]]
        _, cases = apps.callable_cases(source)
        outcomes = {}
        for style, code in {
            "canonical": json.loads(source["solutions"])[
                ref["canonical_solution_index"]
            ],
            **ref["implementations"],
        }.items():
            path = run_dir / "outcomes" / f"{ref['source_id']}-{style}.json"
            saved = json.loads(path.read_text())
            details = apps.validate_details({"results": saved["details"]}, len(cases))
            if (
                saved["source_id"] != ref["source_id"]
                or saved["code_sha256"] != digest(code.encode())
                or saved["source_record_sha256"] != ref["source_record_sha256"]
                or saved["image"] != run["image"]
                or saved["decoder"] != ref["decoder"]
                or saved["supplied_cases"] != len(cases)
                or saved["cleanup_verified"] is not True
                or saved["passed"] != all(r["passed"] for r in details)
            ):
                raise ValueError("Changed or incomplete APPS outcome")
            outcomes[style] = {
                "passed": saved["passed"],
                "path": str(path.relative_to(root)),
                "sha256": apps.file_digest(path),
            }
        passed = all(o["passed"] for o in outcomes.values())
        rows.append(
            {
                "task_id": ref["task_id"],
                "entry_point": ref["entry_point"],
                "family": ref["family"],
                "supplied_cases": len(cases),
                "recommended_candidate": passed and ref["hold"] is None,
                "outcomes": outcomes,
            }
        )
    return {
        "source_records": 10000,
        "reference_pairs": len(refs),
        "recommended_candidates": sum(r["recommended_candidate"] for r in rows),
        "implementation_case_checks": sum(r["supplied_cases"] * 3 for r in rows),
        "records": rows,
        "model_api_calls": 0,
        "coverage": "All supplied cases for this subset only; no augmented tests or universal equivalence claim",
        "human_review_and_experiment_admission": "not_claimed",
        "style_evidence": "Assistant code review plus static interface/list-construction checks; not arbitrary-output judge calibration",
    }


def main() -> None:
    """Validate a fresh run or summarize existing outcomes without replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "summarize"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--sandbox-report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.resolve()
    run_dir.relative_to(root / "runs/apps-audit")
    sources, refs = load_bank(root)
    if args.action == "validate":
        if args.sandbox_report is None:
            parser.error("--sandbox-report is required")
        verify_sandbox_report(args.sandbox_report)
        if (run_dir / "run.json").exists():
            raise ValueError("Use a new directory; no automatic replay")
        image = (root / "data/sandbox/oracle-image.id").read_text().strip()
        paths = [
            BANK / name
            for name in ("references.json", "review.json", "preserved-bindings.json")
        ]
        paths += [
            apps.MANIFEST,
            Path("scripts/audit_apps.py"),
            Path("scripts/import_apps.py"),
            Path("src/grader_power/datasets/apps_source.py"),
            Path("src/grader_power/datasets/mbpp_full.py"),
            Path("src/grader_power/agent_sandbox.py"),
            Path("tests/test_apps_source.py"),
            Path("tests/integration/test_apps_docker.py"),
            Path("tests/integration/test_mode_b_docker.py"),
            Path("tests/integration/test_mode_b_isolation_independent.py"),
            args.sandbox_report.resolve().relative_to(root),
        ]
        run = {
            "image": image,
            "bindings": {str(p): apps.file_digest(root / p) for p in paths},
        }
        write_once(
            run_dir / "run.json",
            (json.dumps(run, indent=2, sort_keys=True) + "\n").encode(),
        )
        for ref in refs:
            source = sources[ref["source_id"]]
            for style, code in {
                "canonical": json.loads(source["solutions"])[
                    ref["canonical_solution_index"]
                ],
                **ref["implementations"],
            }.items():
                result = apps.evaluate(source, code, image)
                write_once(
                    run_dir / "outcomes" / f"{ref['source_id']}-{style}.json",
                    (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
                )
                print(
                    f"{ref['task_id']} {style}: {'PASS' if result['passed'] else 'FAIL'}",
                    flush=True,
                )
    result = summarize(root, run_dir, sources, refs)
    if args.action == "validate":
        write_once(
            run_dir / "summary.json",
            (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
        )
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
