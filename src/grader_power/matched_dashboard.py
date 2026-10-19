"""Read completed matched robustness and recovery views without invoking collectors."""

import hashlib
import json
from pathlib import Path
from typing import Any

SYNTAX_FAMILIES = {"MATCHED_NAMING": "naming", "MATCHED_QUOTES": "quotes"}


def read(path: Path) -> Any:
    """Read an immutable JSON export."""
    return json.loads(path.read_text())


def verify_exports(run: Path, *, syntax: bool) -> dict[str, Any]:
    """Verify completion bindings and reproduce the full planned paired analysis.

    Completion audits independently verify source/outcome provenance. This reader
    fails closed on altered exports, cohort identities, or statistical results.
    It performs no inference calls and does not import the experiment runners.
    """
    from grader_power import matched_review_analysis, matched_syntax_analysis

    audit = read(run / "operations/completion/dashboard-audit.json")
    size, families = (20736, [6, 12, 6]) if syntax else (32256, [12, 15, 3])
    required = {
        "manifest.json",
        "analysis-v1/analysis.json",
        "analysis-v1/measured-rows.json",
    }
    required |= (
        {"assignments.json"}
        if syntax
        else {"designated-records.json", "free-repairs-v1.json"}
    )
    if (
        audit.get("verified") is not True
        or audit.get("source_and_retained_artifact_bindings_verified") is not True
        or audit.get("assigned") != size
        or audit.get("families") != families
        or audit.get("bootstrap_draws") != 10000
        or not required <= set(audit.get("bindings", {}))
    ):
        raise ValueError("Incomplete matched completion audit")
    for name, digest in audit["bindings"].items():
        path = run / name
        if (
            not path.resolve().is_relative_to(run.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != digest
        ):
            raise ValueError("Matched completion binding changed")
    rows = read(run / "analysis-v1/measured-rows.json")
    result = read(run / "analysis-v1/analysis.json")
    if len(rows) != size or len({r["assignment_id"] for r in rows}) != size:
        raise ValueError("Matched assignments missing or duplicated")
    if syntax:
        assignments = {
            r["assignment_id"]: r
            for r in read(run / "assignments.json")
            if r["phase"] == "main"
        }
        if set(assignments) != {r["assignment_id"] for r in rows}:
            raise ValueError("Syntax assignment coverage changed")
        for row in rows:
            if any(
                row[k] != assignments[row["assignment_id"]][k]
                for k in (
                    "phase",
                    "profile",
                    "provider",
                    "model",
                    "study",
                    "condition",
                    "source_task_id",
                    "repetition_index",
                )
            ):
                raise ValueError("Syntax assignment factors changed")
        expected = matched_syntax_analysis.analyze(rows)
    else:
        parent = (
            run.parents[1]
            / "reviewer-observation/matched-review-v1/analysis-v1/measured-rows.json"
        )
        if (
            hashlib.sha256(parent.read_bytes()).hexdigest()
            != audit["parent_rows_sha256"]
        ):
            raise ValueError("Original matched strict view changed")
        original = {r["assignment_id"]: r for r in read(parent)}
        designated = read(run / "designated-records.json")
        if isinstance(designated, dict):
            designated = designated["records"]
        repairs = read(run / "free-repairs-v1.json")["repairs"]
        targets = {r["replaces"]: r["assignment_id"] for r in designated}
        free = {r["assignment_id"] for r in repairs}
        if len(targets) != 147 or len(free) != 86 or targets.keys() & free:
            raise ValueError("Recovery designation changed")
        if set(original) != {r["assignment_id"] for r in rows}:
            raise ValueError("Recovery parent coverage changed")
        for row in rows:
            identity = row["assignment_id"]
            old = original[identity]
            if any(
                row[k] != old[k]
                for k in (
                    "condition",
                    "profile",
                    "provider",
                    "model",
                    "source_task_id",
                    "repetition_index",
                    "historical_assignment_id",
                )
            ):
                raise ValueError("Recovery factors changed")
            if identity not in targets and identity not in free and row != old:
                raise ValueError("Untreated parent outcome changed")
            if (
                identity in targets
                and row.get("recovery_assignment_id") != targets[identity]
            ):
                raise ValueError("Designated recovery linkage changed")
        expected = matched_review_analysis.analyze(rows)
        if result.get("pending_recoveries") != []:
            raise ValueError("Recovery has unfinished stages")
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Matched saved inference differs from planned analysis")
    if [f["holm_family_size"] for f in result["families"]] != families:
        raise ValueError("Matched inference family was reduced")
    return result
