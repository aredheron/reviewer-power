"""One newly authorized attempt after the frozen first recovery was rejected.

This inert layer derives only the same twenty original scientific cells from the
unchanged v1 builder. It neither dispatches requests nor authorizes retries. The
collector separately verifies explicit human approval and the transport proof
that all first recovery requests were rejected or never sent.
"""

import copy
from typing import Any

from grader_power import style_preference_experiment as study
from grader_power import style_preference_generation_recovery as first

RECOVERY = "generation-rerun-v2"
PHASE = "generation_rerun_v2"
TARGET_COUNT = first.TARGET_COUNT

_CELL_FIELDS = (
    "profile",
    "study",
    "kind",
    "source_task_id",
    "condition",
    "mode",
    "repetition_index",
)


def _indexed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed = {r["assignment_id"]: r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate second-recovery assignment IDs")
    return indexed


def _failed_first_rows(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Require terminal missing first attempts; transport acceptance is checked upstream."""
    indexed = _indexed(rows)
    if len(indexed) != TARGET_COUNT:
        raise ValueError("Second recovery requires exactly twenty first attempts")
    for row in rows:
        if (
            row["phase"] != first.PHASE
            or row.get("generation_recovery") != first.RECOVERY
            or row.get("recovery_generation") is not True
            or row["assignment_id"]
            != first.fresh_assignment_id(row["source_assignment_id"])
            or row["started"] is not True
            or row["attempt_complete"] is not True
            or row["generation_status"] != "request_error"
            or row["raw_saved"] is not False
            or row["scored"] is not False
            or row["judge_started"] is not False
            or row["judge_result"] is not False
            or row["judge_raw_saved"] is not False
            or row["label"] is not None
            or row["reused_generation"] is not False
        ):
            raise ValueError(
                "First recovery must retain twenty terminal missing generations"
            )
    return indexed


def fresh_assignment_id(source_assignment_id: str) -> str:
    """Bind the single v2 identity to both the original cell and its v1 attempt."""
    prior = first.fresh_assignment_id(source_assignment_id)
    return study.digest(
        study.json_text(
            {
                "recovery": RECOVERY,
                "source_assignment_id": source_assignment_id,
                "prior_recovery_assignment_id": prior,
            }
        )
    )


def build_records(
    main_records: list[dict[str, Any]],
    original_baseline: list[dict[str, Any]],
    first_records: list[dict[str, Any]],
    first_measured: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy exactly the original twenty v1 records with new identity and lineage.

    No failed main request arriving later can enter this cohort. Every scientific
    field, including prompt bytes, provider, model and settings, must still equal
    the unchanged v1 builder applied to its frozen original baseline.
    """
    expected = _indexed(first.build_records(main_records, original_baseline))
    records = _indexed(first_records)
    measured = _failed_first_rows(first_measured)
    if records != expected or set(measured) != set(expected):
        raise ValueError(
            "Second recovery cannot change the frozen first recovery cohort"
        )
    for identity, row in measured.items():
        prior = records[identity]
        if (
            row["source_assignment_id"] != prior["source_assignment_id"]
            or any(row[k] != prior[k] for k in _CELL_FIELDS)
            or row["provider"] != prior["provider"]
            or row["generation_model"] != prior["model"]
        ):
            raise ValueError(
                "First recovery measured cells or provider provenance changed"
            )
    additions = []
    occupied = set(records) | {r["assignment_id"] for r in main_records}
    for prior in sorted(records.values(), key=lambda r: r["source_assignment_id"]):
        row = copy.deepcopy(prior)
        identity = fresh_assignment_id(row["source_assignment_id"])
        if identity in occupied:
            raise ValueError(
                "Second recovery identity collides with an earlier attempt"
            )
        row.update(
            assignment_id=identity,
            request_id=identity,
            phase=PHASE,
            generation_recovery=RECOVERY,
            prior_recovery_assignment_id=prior["assignment_id"],
        )
        additions.append(row)
    return additions


def recovered_main_rows(
    original_rows: list[dict[str, Any]],
    first_measured: list[dict[str, Any]],
    second_measured: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Use the designated v2 attempt, preserving complete original and v1 outcomes.

    The original 10,080 assignment IDs and scientific cells remain unchanged.
    Missing or failed second attempts replace earlier outcomes too; this is never
    a success-selected merge. Other main failures remain in their original cells.
    """
    prior = _failed_first_rows(first_measured)
    # This also verifies every original cell and each first-attempt source link.
    rows = first.recovered_main_rows(original_rows, first_measured)
    prior_by_source = {r["source_assignment_id"]: r for r in prior.values()}
    second = _indexed(second_measured)
    if len(second) != TARGET_COUNT:
        raise ValueError("Second recovered analysis requires exactly twenty fresh rows")
    replacements = {}
    for row in second.values():
        source = row["source_assignment_id"]
        previous = prior_by_source.get(source)
        if (
            previous is None
            or source in replacements
            or row["assignment_id"] != fresh_assignment_id(source)
            or row["phase"] != PHASE
            or row.get("generation_recovery") != RECOVERY
            or row.get("recovery_generation") is not True
            or row.get("prior_recovery_assignment_id") != previous["assignment_id"]
            or any(row[k] != previous[k] for k in _CELL_FIELDS)
            or any(row[k] != previous[k] for k in ("provider", "generation_model"))
        ):
            raise ValueError(
                "Second recovery cell, identity or prior-attempt lineage changed"
            )
        if any(k not in row for k in first.OUTCOME_FIELDS):
            raise ValueError(
                "Second recovery requires complete measured outcome fields"
            )
        replacements[source] = row
    for row in rows:
        source = row["assignment_id"]
        replacement = replacements.get(source)
        if replacement is not None:
            row.update({k: copy.deepcopy(replacement[k]) for k in first.OUTCOME_FIELDS})
            row.update(
                generation_recovery=RECOVERY,
                recovery_assignment_id=replacement["assignment_id"],
                prior_recovery_assignment_id=prior_by_source[source]["assignment_id"],
                prior_recovery_outcome=copy.deepcopy(prior_by_source[source]),
            )
    return rows
