"""Frozen additive syntax gap recovery; prior requests and analyses remain immutable."""

import json
from pathlib import Path
from typing import Any

from evals.inspect_all_recovery import repair_reversed_ranges
from evals.inspect_mode_a import digest, json_text
from grader_power import matched_syntax as original

RUN = "matched-gaps-v1"
PROFILES = original.PROFILES
JUDGE_PROFILE = original.JUDGE_PROFILE
CAP = 15
MEDIAN_LIMIT = 15
AUTHORIZATION = "User requested all remaining generation/judgment gaps resolved using precedent. One designated same-limit replacement of each missing/incomplete original syntax generation; one fresh blind judgment of each unrepaired saved response; exact reversed-range repairs separate. No retries of replacements or best-of fallback. Original outcomes and held GLM naming unchanged."


def base(root: Path) -> Path:
    """Locate the isolated recovery, never a parent's request directory."""
    return root / "runs/recovery" / RUN


def normalize(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Apply approved free rules separately, never combine defects or change semantics."""
    if packet["family"] == "quotes":
        try:
            result = repair_reversed_ranges(text, packet)
        except (ValueError, TypeError, KeyError):
            pass
        else:
            return result | {"status": result["judgment"]["status"]}
    return qualify_empty_rationale(text, packet)


def qualify_empty_rationale(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Accept only absent narrative with otherwise valid wire and independent agreement.

    A temporary nonempty sentinel exercises the unchanged parser's other checks.
    It is removed from the returned judgment; no explanation is fabricated.
    Duplicate keys, multiple JSON objects, wrong forms, bad ranges and abstentions
    are rejected. This is a separate authorized qualification, not strict parsing.
    """
    from evals.inspect_style_preferences import parse_packet
    from grader_power.preference_judge import JudgeInput, audit_style

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    wire = json.loads(text, object_pairs_hook=unique)
    if not isinstance(wire, dict) or wire.get("rationale") != "":
        raise ValueError("Empty rationale must be the sole defect")
    candidate = dict(wire, rationale="Validation sentinel; not supplied by model.")
    parsed = parse_packet(packet["family"], json_text(candidate), packet)
    audit = audit_style(JudgeInput.model_validate(packet))
    if parsed.status != "classified" or parsed.label != audit.label:
        raise ValueError("Independent source audit disagrees or judgment abstains")
    judgment = parsed.model_dump(mode="json")
    judgment["rationale"] = ""
    return {
        "status": "classified",
        "label": parsed.label,
        "judgment": judgment,
        "original_wire": wire,
        "audit": audit.model_dump(mode="json"),
        "qualification": "approved_empty_rationale_only",
        "strict_failed": True,
        "changes": [],
        "rationale_fabricated": False,
    }


def free_repairs(root: Path) -> dict[str, Any]:
    """Revalidate fixed saved repairs against every original source artifact."""
    b = base(root)
    audit = json.loads((b / "operations/initial-audit.json").read_text())
    for name, sha in audit["source_bindings"].items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Original recovery evidence changed: " + name)
    proof = json.loads((b / "free-repairs-v1.json").read_text())
    for item in proof["repairs"]:
        p = original.base(root) / "requests" / item["assignment_id"] / "judge"
        raw = json.loads((p / "raw.json").read_text())
        if raw["choices"][0]["stop_reason"] != "stop":
            raise ValueError("Incomplete original judgment")
        if (
            repair_reversed_ranges(
                raw["completion"], json.loads((p / "packet.json").read_text())
            )
            != item["repair"]
        ):
            raise ValueError("Free repair changed")
    if len(proof["repairs"]) != 62:
        raise ValueError("Frozen free repair scope changed")
    return proof


def recovery_records(root: Path) -> list[dict[str, Any]]:
    """Verify all designated replacements against the original complete assignment grid."""
    records = json.loads((base(root) / "designated-records.json").read_text())
    old = {
        r["assignment_id"]: r
        for r in json.loads((original.base(root) / "assignments.json").read_text())
    }
    if len(records) != 311 or len({r["replaces"] for r in records}) != 311:
        raise ValueError("Frozen paid recovery scope changed")
    if sum(r["recovery_stage"] == "generation" for r in records) != 197:
        raise ValueError("Generation scope changed")
    for row in records:
        source = old[row["replaces"]]
        if source["phase"] != "main" or (source["study"], source["profile"]) == (
            "naming",
            "glm",
        ):
            raise ValueError("Held or pilot assignment cannot be recovered")
        for key in (
            "messages",
            "condition",
            "profile",
            "model",
            "provider",
            "study",
            "repetition_index",
            "source_task_id",
            "prompt_sha256",
        ):
            if row[key] != source[key]:
                raise ValueError("Recovery changed scientific factors")
        expected = digest(
            json_text(
                {
                    "recovery": RUN,
                    "source": source["assignment_id"],
                    "stage": row["recovery_stage"],
                    "attempt": 1,
                }
            )
        )
        if row["assignment_id"] != expected or row["source_path"] != str(
            (original.base(root) / "requests" / source["assignment_id"]).relative_to(
                root
            )
        ):
            raise ValueError("Recovery identity/source changed")
    return records
