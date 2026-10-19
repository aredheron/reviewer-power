"""Additive recovery design; original matched-review outcomes and ledger are immutable."""

import json
from pathlib import Path
from typing import Any

from inspect_ai.model import ModelOutput

from evals import inspect_outcome_recovery as precedent
from evals.inspect_mode_a import digest, json_text
from grader_power import matched_review as original

RUN = "matched-review-recovery-v1"
PROFILES = original.PROFILES
JUDGE_PROFILE = original.JUDGE_PROFILE
SEED = 2026101901
CAP = 15
MEDIAN_LIMIT = 15
AUTHORIZATION = "User asked to resolve last night's unresolved generations/judgments, relying on past rulings: one fresh attempt at unchanged limits, fresh blinded judgments where needed, free mechanical format repairs, completed TODO stub retained. All outcomes preserved; additive designated recovery only."


def base(root: Path) -> Path:
    return root / "runs/recovery" / RUN


def normalize(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Reuse the approved exact-quote/metadata rule, never alter semantic fields."""
    return precedent.normalize_judgment(text, packet, "mapping")


def free_repairs(root: Path) -> dict[str, Any]:
    """Revalidate every proposed repair and hash-bound source before separate publication."""
    b = base(root)
    audit_path = b / "operations/initial-audit.json"
    audit = json.loads(audit_path.read_text())
    parent = original.base(root)
    if (
        digest((parent / "manifest.json").read_bytes())
        != audit["parent_manifest_sha256"]
        or digest((parent / "analysis-v1/measured-rows.json").read_bytes())
        != audit["parent_analysis_sha256"]
    ):
        raise ValueError("Parent identity or strict analysis changed")
    repairs = []
    for item in audit["free_repairs"]:
        for name, sha in item["bindings"].items():
            if digest((root / name).read_bytes()) != sha:
                raise ValueError("Original repair evidence changed")
        p = parent / "requests" / item["assignment_id"] / "judge"
        raw = ModelOutput.model_validate(json.loads((p / "raw.json").read_text()))
        result = json.loads((p / "result.json").read_text())
        packet = json.loads((p / "packet.json").read_text())
        if (
            result["label"] is not None
            or not raw.choices
            or raw.choices[0].stop_reason != "stop"
        ):
            raise ValueError("Ineligible mechanical repair")
        fixed = normalize(raw.completion, packet)
        if fixed != item["repair"]:
            raise ValueError("Mechanical repair changed")
        repairs.append(item | {"label_source": "separate_precedent_mechanical_repair"})
    if len(repairs) != 86 or len({r["assignment_id"] for r in repairs}) != 86:
        raise ValueError("Audited repair scope changed")
    return {
        "authorization": AUTHORIZATION,
        "source_audit_sha256": digest(audit_path.read_bytes()),
        "originals_unchanged": True,
        "paid_calls": 0,
        "repairs": repairs,
    }


def recovery_records(root: Path) -> list[dict[str, Any]]:
    """Designate each fresh attempt in advance; never fall back to a better old outcome."""
    audit = json.loads((base(root) / "operations/initial-audit.json").read_text())
    old = {
        r["assignment_id"]: r
        for r in json.loads((original.base(root) / "assignments.json").read_text())
    }
    result = []
    for stage, key in (
        ("generation", "fresh_generation_candidates"),
        ("judge", "fresh_judge_candidates"),
    ):
        for item in audit[key]:
            source = old[item["assignment_id"]]
            row = dict(
                source,
                replaces=source["assignment_id"],
                original_workflow=source["workflow"],
                workflow=RUN,
                recovery_stage=stage,
            )
            row["assignment_id"] = digest(
                json_text(
                    {
                        "recovery": RUN,
                        "source": source["assignment_id"],
                        "stage": stage,
                        "attempt": 1,
                    }
                )
            )
            row["source_path"] = str(
                (
                    original.base(root) / "requests" / source["assignment_id"]
                ).relative_to(root)
            )
            result.append(row)
    if len(result) != 147 or len({r["replaces"] for r in result}) != 147:
        raise ValueError("Paid recovery scope changed")
    return result
