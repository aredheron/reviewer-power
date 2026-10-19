"""Inert, single-attempt recovery for an explicitly frozen generation cohort.

These helpers neither authorize requests nor read or write experiment artifacts.
The collector must bind the human approval, original outcomes and returned fresh
records before dispatch. Recovered analysis replaces each selected cell with its
one fresh attempt, including a missing or failed attempt, without selecting the
better result. Original analysis and all source artifacts remain separate.
"""

import copy
from collections import Counter
from itertools import product
from typing import Any

from grader_power import style_preference_experiment as study

RECOVERY = "generation-rerun-v1"
PHASE = "generation_rerun_v1"
TARGET_COUNT = 20
MAIN_COUNT = 10080
REPETITIONS = 3

_CELL_FIELDS = (
    "profile",
    "study",
    "kind",
    "source_task_id",
    "condition",
    "mode",
    "repetition_index",
)
OUTCOME_FIELDS = (
    "reused_generation",
    "judge_raw_saved",
    "started",
    "attempt_complete",
    "raw_saved",
    "scored",
    "full_correct",
    "judge_started",
    "judge_result",
    "label",
    "audit_label",
    "audit_agrees",
    "probe_correct",
    "probe_correct_strict",
    "generation_status",
    "judge_status",
)


def _indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {row["assignment_id"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate recovery assignment IDs")
    return indexed


def _main_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Validate every selected main cell without imposing outcome requirements."""
    indexed = _indexed(rows)
    if len(rows) != MAIN_COUNT or any(
        r["phase"] != "main" or r["kind"] != "core" or r["mode"] != "A" for r in rows
    ):
        raise ValueError("Recovery requires exactly the 10,080 selected main rows")
    for family in study.FAMILIES:
        group = [r for r in rows if r["study"] == family]
        tasks = {r["source_task_id"] for r in group}
        expected = set(
            product(
                study.PROFILES,
                tasks,
                study.STYLES[family],
                study.ARMS,
                range(REPETITIONS),
            )
        )
        cells = Counter(
            (
                r["profile"],
                r["source_task_id"],
                r["condition"]["user_preference"],
                r["condition"]["arm"],
                r["repetition_index"],
            )
            for r in group
        )
        if len(tasks) != 24 or set(cells) != expected or set(cells.values()) != {1}:
            raise ValueError("Recovery main cells, repetitions or models changed")
        for row in group:
            if type(row["repetition_index"]) is not int or row[
                "condition"
            ] != study.condition(
                family,
                row["condition"]["user_preference"],
                row["condition"]["arm"],
            ):
                raise ValueError("Recovery main condition changed")
    return indexed


def _failed_generation(row: dict[str, Any]) -> bool:
    status = row["generation_status"]
    if status not in ("request_error", "incomplete"):
        return False
    if row["started"] is not True or row["attempt_complete"] is not True:
        raise ValueError("Recovery cannot select unfinished original generations")
    if row["raw_saved"] is not (status == "incomplete"):
        raise ValueError(
            "Recovery requires missing errors or saved incomplete responses"
        )
    return True


def fresh_assignment_id(source_assignment_id: str) -> str:
    """Return the only fresh identity permitted for a frozen original assignment."""
    if len(source_assignment_id) != 64 or any(
        c not in "0123456789abcdef" for c in source_assignment_id
    ):
        raise ValueError("Recovery source assignment must be a SHA-256 identity")
    return study.digest(
        study.json_text(
            {"recovery": RECOVERY, "source_assignment_id": source_assignment_id}
        )
    )


def build_records(
    main_records: list[dict[str, Any]], measured_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Copy the frozen 20 terminal generation failures, changing only identities.

    Coding correctness and judging results never determine eligibility. Prompts,
    models, task metadata and any per-record settings are deep-copied unchanged.
    The caller verifies transport evidence and approval separately; this function
    does not treat a terminal request error as proof that dispatch was rejected.
    """
    original = _main_rows(main_records)
    measured = _main_rows(measured_rows)
    if set(original) != set(measured):
        raise ValueError("Measured recovery cohort identities changed")
    for identity, row in original.items():
        if any(row[k] != measured[identity][k] for k in _CELL_FIELDS):
            raise ValueError("Measured recovery cohort cells changed")
    targets = [
        identity for identity, row in measured.items() if _failed_generation(row)
    ]
    if len(targets) != TARGET_COUNT:
        raise ValueError("Recovery is limited to exactly 20 frozen generation failures")
    additions = []
    for identity in sorted(targets):
        row = copy.deepcopy(original[identity])
        if (
            row.get("source_assignment_id") is not None
            or row.get("generation_source_assignment_id") is not None
            or row.get("judge_only")
            or row.get("recovery_generation")
            or row["request_id"] != identity
            or row["prompt_sha256"] != study.digest(study.json_text(row["messages"]))
            or row["provider"] != study.PROFILES[row["profile"]]["provider"]
            or row["model"] != study.PROFILES[row["profile"]]["model"]
        ):
            raise ValueError(
                "Recovery source prompt, model or original lineage changed"
            )
        fresh_id = fresh_assignment_id(identity)
        if fresh_id in original:
            raise ValueError("Recovery identity collides with an original assignment")
        row.update(
            assignment_id=fresh_id,
            request_id=fresh_id,
            phase=PHASE,
            source_assignment_id=identity,
            recovery_generation=True,
            generation_recovery=RECOVERY,
        )
        additions.append(row)
    return additions


def recovered_main_rows(
    original_measured: list[dict[str, Any]], fresh_measured: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace 20 main-cell outcomes with fresh evidence, never with the best outcome.

    All original row IDs, cells and provider provenance are retained. Every
    outcome field is replaced even when the fresh attempt is unstarted or failed;
    the complete original measured row is preserved in
    ``original_generation_outcome``. Later
    failures in unrelated main cells remain unchanged and are never auto-retried.
    """
    original = _main_rows(original_measured)
    fresh = _indexed(fresh_measured)
    if len(fresh) != TARGET_COUNT:
        raise ValueError("Recovered analysis requires exactly 20 frozen fresh rows")
    replacements = {}
    for row in fresh.values():
        source = row["source_assignment_id"]
        if (
            source not in original
            or source in replacements
            or row["assignment_id"] != fresh_assignment_id(source)
            or row["phase"] != PHASE
            or row.get("recovery_generation") is not True
            or row.get("generation_recovery") != RECOVERY
            or any(row[k] != original[source][k] for k in _CELL_FIELDS)
            or any(
                row[k] != original[source][k] for k in ("provider", "generation_model")
            )
            or not _failed_generation(original[source])
        ):
            raise ValueError(
                "Recovered analysis source, identity, cell or lineage changed"
            )
        if any(k not in row or k not in original[source] for k in OUTCOME_FIELDS):
            raise ValueError(
                "Recovered analysis requires complete measured outcome fields"
            )
        replacements[source] = row
    output = []
    for original_row in original_measured:
        row = copy.deepcopy(original_row)
        source = row["assignment_id"]
        replacement = replacements.get(source)
        if replacement is not None:
            prior = copy.deepcopy(row)
            row.update({k: copy.deepcopy(replacement[k]) for k in OUTCOME_FIELDS})
            row.update(
                recovery_assignment_id=replacement["assignment_id"],
                recovery_source_assignment_id=source,
                recovery_generation=True,
                generation_recovery=RECOVERY,
                original_generation_outcome=prior,
            )
        output.append(row)
    return output
