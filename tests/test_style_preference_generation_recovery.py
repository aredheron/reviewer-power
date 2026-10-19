"""Single authorized fresh attempts preserve every original scientific cell."""

import copy
from itertools import product
from typing import Any

import pytest

from grader_power import style_preference_experiment as study
from grader_power import style_preference_generation_recovery as recovery


@pytest.fixture(scope="module")
def cohort() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    for family in study.FAMILIES:
        for profile, task, user, arm, repetition in product(
            study.PROFILES,
            range(24),
            study.STYLES[family],
            study.ARMS,
            range(3),
        ):
            identity = study.digest(
                study.json_text((family, profile, task, user, arm, repetition))
            )
            messages = [{"role": "user", "content": "Exact frozen coding prompt"}]
            records.append(
                {
                    "assignment_id": identity,
                    "request_id": identity,
                    "phase": "main",
                    "profile": profile,
                    "study": family,
                    "kind": "core",
                    "mode": "A",
                    "source_task_id": f"{family}/{task}",
                    "condition": study.condition(family, user, arm),
                    "repetition_index": repetition,
                    "messages": messages,
                    "prompt_sha256": study.digest(study.json_text(messages)),
                    "provider": study.PROFILES[profile]["provider"],
                    "model": study.PROFILES[profile]["model"],
                    "settings": {"max_tokens": 4096, "nested": ["preserved"]},
                }
            )
    measured = []
    for i, record in enumerate(records):
        row = copy.deepcopy(record)
        row.pop("messages")
        row.update(
            generation_model=record["model"],
            reused_generation=False,
            judge_raw_saved=False,
            started=i < 21,
            attempt_complete=i < 21,
            raw_saved=18 <= i < 21,
            scored=i == 20,
            full_correct=False if i == 20 else None,
            judge_started=i == 20,
            judge_result=i == 20,
            label=None,
            audit_label=None,
            audit_agrees=None,
            probe_correct=None,
            probe_correct_strict=None,
            generation_status="request_error"
            if i < 18
            else "incomplete"
            if i < 20
            else "completed"
            if i == 20
            else None,
            judge_status="judge_error" if i == 20 else None,
        )
        measured.append(row)
    return records, measured


def fresh_rows(
    records: list[dict[str, Any]], measured: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Make unstarted fresh measured rows matching the collector's public shape."""
    additions = recovery.build_records(records, measured)
    result = []
    for record in additions:
        row = copy.deepcopy(record)
        row.update({key: None for key in recovery.OUTCOME_FIELDS})
        row.update(
            generation_model=record["model"],
            reused_generation=False,
            judge_raw_saved=False,
            started=False,
            attempt_complete=False,
            raw_saved=False,
            scored=False,
            judge_started=False,
            judge_result=False,
        )
        result.append(row)
    return result


def test_fresh_records_are_deterministic_exact_scientific_copies(cohort):
    records, measured = cohort
    before = copy.deepcopy((records, measured))
    additions = recovery.build_records(records, measured)
    assert additions == recovery.build_records(records[::-1], measured[::-1])
    assert len(additions) == len({r["assignment_id"] for r in additions}) == 20
    assert {r["source_assignment_id"] for r in additions} == {
        r["assignment_id"] for r in measured[:20]
    }
    indexed = {r["assignment_id"]: r for r in records}
    changed = {
        "assignment_id",
        "request_id",
        "phase",
        "source_assignment_id",
        "recovery_generation",
        "generation_recovery",
    }
    for row in additions:
        source = indexed[row["source_assignment_id"]]
        assert {k: v for k, v in row.items() if k not in changed} == {
            k: v for k, v in source.items() if k not in changed
        }
        assert row["assignment_id"] == row["request_id"]
        assert row["assignment_id"] not in indexed
    additions[0]["settings"]["nested"].append("changed only copy")
    assert (records, measured) == before


@pytest.mark.parametrize("count", [19, 21])
def test_scope_is_exactly_twenty_frozen_generation_failures(cohort, count):
    records, original = cohort
    measured = copy.deepcopy(original)
    changed = measured[19 if count == 19 else 20]
    changed.update(
        generation_status="completed" if count == 19 else "incomplete",
        raw_saved=True,
    )
    with pytest.raises(ValueError, match="exactly 20"):
        recovery.build_records(records, measured)


@pytest.mark.parametrize(
    "field,value",
    [("started", False), ("attempt_complete", False), ("raw_saved", True)],
)
def test_unfinished_or_ambiguous_saved_errors_cannot_enter_cohort(cohort, field, value):
    records, original = cohort
    measured = copy.deepcopy(original)
    measured[0][field] = value
    with pytest.raises(ValueError, match="unfinished|missing errors"):
        recovery.build_records(records, measured)


@pytest.mark.parametrize(
    "field,value",
    [
        ("phase", "pilot"),
        ("repetition_index", 3),
        ("mode", "B"),
        ("kind", "comprehension_probe"),
        ("profile", "other"),
    ],
)
def test_only_selected_main_matrix_is_eligible(cohort, field, value):
    records, measured = copy.deepcopy(cohort)
    records[0][field] = value
    with pytest.raises(ValueError, match="main|cells"):
        recovery.build_records(records, measured)


@pytest.mark.parametrize(
    "field,value",
    [
        ("messages", [{"role": "user", "content": "Altered"}]),
        ("model", "different-model"),
        ("provider", "different-provider"),
        ("source_assignment_id", "already-recovered"),
        ("judge_only", True),
        ("recovery_generation", True),
    ],
)
def test_changed_prompts_models_or_prior_retry_lineage_rejected(cohort, field, value):
    records, measured = copy.deepcopy(cohort)
    records[0][field] = value
    with pytest.raises(ValueError, match="source prompt"):
        recovery.build_records(records, measured)


def test_missing_and_duplicate_original_rows_fail(cohort):
    records, measured = cohort
    with pytest.raises(ValueError, match="10,080"):
        recovery.build_records(records[:-1], measured)
    with pytest.raises(ValueError, match="Duplicate"):
        recovery.build_records(records[:-1] + [records[0]], measured)


def test_overlay_keeps_cells_and_full_originals_even_when_fresh_attempt_is_missing(
    cohort,
):
    records, measured = cohort
    fresh = fresh_rows(records, measured)
    before = copy.deepcopy((measured, fresh))
    result = recovery.recovered_main_rows(measured, fresh)
    assert len(result) == 10080
    assert [r["assignment_id"] for r in result] == [
        r["assignment_id"] for r in measured
    ]
    for original, row in zip(measured[:20], result[:20], strict=True):
        assert row["original_generation_outcome"] == original
        assert row["raw_saved"] is False
        assert row["attempt_complete"] is False
        assert row["generation_status"] is None
        assert row["phase"] == "main"
        assert row["condition"] == original["condition"]
        assert row["recovery_source_assignment_id"] == row["assignment_id"]
    assert result[20:] == measured[20:]
    assert (measured, fresh) == before


def test_overlay_always_uses_fresh_failure_without_old_label_fallback(cohort):
    records, original = cohort
    measured = copy.deepcopy(original)
    measured[18].update(label="snake_case_only", judge_result=True, full_correct=True)
    fresh = fresh_rows(records, measured)
    source = measured[18]["assignment_id"]
    replacement = next(r for r in fresh if r["source_assignment_id"] == source)
    replacement.update(
        started=True, attempt_complete=True, generation_status="request_error"
    )
    rows = recovery.recovered_main_rows(measured, fresh)
    assert rows[18]["label"] is None
    assert rows[18]["full_correct"] is None
    assert rows[18]["generation_status"] == "request_error"
    assert rows[18]["original_generation_outcome"]["label"] == "snake_case_only"


def test_later_untargeted_failure_is_retained_without_another_recovery(cohort):
    records, original = cohort
    fresh = fresh_rows(records, original)
    measured = copy.deepcopy(original)
    measured[21].update(
        started=True, attempt_complete=True, generation_status="request_error"
    )
    result = recovery.recovered_main_rows(measured, fresh)
    assert result[21] == measured[21]
    assert "recovery_assignment_id" not in result[21]


@pytest.mark.parametrize(
    "field,value",
    [
        ("assignment_id", "f" * 64),
        ("source_assignment_id", "e" * 64),
        ("phase", "main"),
        ("generation_recovery", "generation-rerun-v2"),
        ("recovery_generation", False),
        ("provider", "wrong-provider"),
        ("generation_model", "wrong-model"),
        ("repetition_index", 2.5),
    ],
)
def test_overlay_rejects_wrong_fresh_lineage(cohort, field, value):
    records, measured = cohort
    fresh = fresh_rows(records, measured)
    fresh[0][field] = value
    with pytest.raises(ValueError, match="lineage"):
        recovery.recovered_main_rows(measured, fresh)


def test_overlay_rejects_missing_outcomes_duplicate_sources_and_partial_cohort(cohort):
    records, measured = cohort
    fresh = fresh_rows(records, measured)
    with pytest.raises(ValueError, match="exactly 20"):
        recovery.recovered_main_rows(measured, fresh[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        recovery.recovered_main_rows(measured, fresh[:-1] + [fresh[0]])
    del fresh[0]["label"]
    with pytest.raises(ValueError, match="complete measured"):
        recovery.recovered_main_rows(measured, fresh)


def test_fresh_id_rejects_invalid_original_identity():
    with pytest.raises(ValueError, match="SHA-256"):
        recovery.fresh_assignment_id("not-an-assignment")
