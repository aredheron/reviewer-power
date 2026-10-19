"""The second authorization is bound to the same twenty original failures."""

import copy
from itertools import product
from typing import Any

import pytest

from grader_power import style_preference_experiment as study
from grader_power import style_preference_generation_recovery as first
from grader_power import style_preference_generation_recovery_v2 as second


def _measured(record: dict[str, Any], *, failed: bool) -> dict[str, Any]:
    row = copy.deepcopy(record)
    row.update({key: None for key in first.OUTCOME_FIELDS})
    row.update(
        provider=record["provider"],
        generation_model=record["model"],
        reused_generation=False,
        judge_raw_saved=False,
        started=failed,
        attempt_complete=failed,
        raw_saved=False,
        scored=False,
        judge_started=False,
        judge_result=False,
        generation_status="request_error" if failed else None,
    )
    return row


@pytest.fixture(scope="module")
def cohort():
    records = []
    for family in study.FAMILIES:
        for profile, task, user, arm, repetition in product(
            study.PROFILES, range(24), study.STYLES[family], study.ARMS, range(3)
        ):
            identity = study.digest(
                study.json_text((family, profile, task, user, arm, repetition))
            )
            messages = [{"role": "user", "content": "Frozen prompt"}]
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
                    "settings": {"max_tokens": 4096, "extra": ["preserved"]},
                }
            )
    baseline = [_measured(r, failed=i < 20) for i, r in enumerate(records)]
    baseline[18].update(generation_status="incomplete", raw_saved=True, label="invalid")
    baseline[19].update(generation_status="incomplete", raw_saved=True, label="invalid")
    first_records = first.build_records(records, baseline)
    first_measured = [_measured(r, failed=True) for r in first_records]
    return records, baseline, first_records, first_measured


def test_second_records_are_deterministic_with_only_new_identity_and_prior_link(cohort):
    before = copy.deepcopy(cohort)
    result = second.build_records(*cohort)
    assert result == second.build_records(*(rows[::-1] for rows in cohort))
    previous = {r["assignment_id"]: r for r in cohort[2]}
    assert len(result) == len({r["assignment_id"] for r in result}) == 20
    changed = {
        "assignment_id",
        "request_id",
        "phase",
        "generation_recovery",
        "prior_recovery_assignment_id",
    }
    for row in result:
        prior = previous[row["prior_recovery_assignment_id"]]
        assert {k: v for k, v in row.items() if k not in changed} == {
            k: v for k, v in prior.items() if k not in changed
        }
        assert row["assignment_id"] not in previous
        assert row["request_id"] == row["assignment_id"]
        assert row["source_assignment_id"] == prior["source_assignment_id"]
    result[0]["settings"]["extra"].append("copy only")
    assert cohort == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("raw_saved", True),
        ("started", False),
        ("attempt_complete", False),
        ("generation_status", "incomplete"),
        ("generation_status", "completed"),
        ("scored", True),
        ("judge_started", True),
        ("label", "snake_case_only"),
        ("phase", "main"),
        ("generation_recovery", second.RECOVERY),
    ],
)
def test_only_terminal_missing_first_attempts_are_eligible(cohort, field, value):
    args = copy.deepcopy(cohort)
    args[3][0][field] = value
    with pytest.raises(ValueError, match="terminal missing"):
        second.build_records(*args)


@pytest.mark.parametrize("field,value", [("model", "other"), ("phase", "main")])
def test_first_records_must_equal_original_frozen_builder(cohort, field, value):
    args = copy.deepcopy(cohort)
    args[2][0][field] = value
    with pytest.raises(ValueError, match="frozen first"):
        second.build_records(*args)


def test_changed_first_prompt_is_not_rehashed_into_new_authorization(cohort):
    args = copy.deepcopy(cohort)
    args[2][0]["messages"][0]["content"] = "New prompt"
    args[2][0]["prompt_sha256"] = study.digest(study.json_text(args[2][0]["messages"]))
    with pytest.raises(ValueError, match="frozen first"):
        second.build_records(*args)


def test_first_measured_provider_cannot_differ_from_first_record(cohort):
    args = copy.deepcopy(cohort)
    args[3][0]["provider"] = "other"
    with pytest.raises(ValueError, match="provider provenance"):
        second.build_records(*args)


def test_partial_or_duplicate_first_cohort_rejected(cohort):
    records, baseline, first_records, first_measured = cohort
    with pytest.raises(ValueError, match="twenty first"):
        second.build_records(records, baseline, first_records, first_measured[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        second.build_records(
            records, baseline, first_records, first_measured[:-1] + first_measured[:1]
        )


def test_recovered_rows_preserve_entire_original_and_prior_outcomes(cohort):
    _, original, _, first_measured = cohort
    second_measured = [
        _measured(r, failed=False) for r in second.build_records(*cohort)
    ]
    before = copy.deepcopy((original, first_measured, second_measured))
    result = second.recovered_main_rows(original, first_measured, second_measured)
    prior = {r["source_assignment_id"]: r for r in first_measured}
    latest = {r["source_assignment_id"]: r for r in second_measured}
    assert len(result) == 10080
    assert [r["assignment_id"] for r in result] == [
        r["assignment_id"] for r in original
    ]
    for source, row in zip(original[:20], result[:20], strict=True):
        identity = source["assignment_id"]
        assert row["original_generation_outcome"] == source
        assert row["prior_recovery_outcome"] == prior[identity]
        assert row["prior_recovery_assignment_id"] == prior[identity]["assignment_id"]
        assert row["recovery_assignment_id"] == latest[identity]["assignment_id"]
        assert row["phase"] == "main"
        assert row["generation_recovery"] == second.RECOVERY
        assert row["raw_saved"] is False
        assert row["generation_status"] is None
        assert row["label"] is None
    assert result[20:] == original[20:]
    assert (original, first_measured, second_measured) == before


def test_new_failure_replaces_even_a_scored_original_incomplete_outcome(cohort):
    _, original, _, first_measured = cohort
    second_measured = [_measured(r, failed=True) for r in second.build_records(*cohort)]
    result = second.recovered_main_rows(original, first_measured, second_measured)
    assert result[18]["original_generation_outcome"]["label"] == "invalid"
    assert result[18]["generation_status"] == "request_error"
    assert result[18]["label"] is None
    assert result[18]["raw_saved"] is False


def test_new_success_replaces_all_outcome_fields_without_changing_cells(cohort):
    _, original, _, first_measured = cohort
    second_measured = [_measured(r, failed=True) for r in second.build_records(*cohort)]
    selected = second_measured[0]
    selected.update(
        raw_saved=True,
        generation_status="completed",
        scored=True,
        full_correct=True,
        judge_started=True,
        judge_result=True,
        judge_raw_saved=True,
        label="snake_case_only",
        judge_status="classified",
    )
    result = second.recovered_main_rows(original, first_measured, second_measured)
    row = next(
        r for r in result if r["assignment_id"] == selected["source_assignment_id"]
    )
    assert all(row[k] == selected[k] for k in first.OUTCOME_FIELDS)
    assert row["condition"] == selected["condition"]
    assert row["provider"] == selected["provider"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("prior_recovery_assignment_id", "f" * 64),
        ("assignment_id", "e" * 64),
        ("source_assignment_id", "d" * 64),
        ("generation_recovery", first.RECOVERY),
        ("phase", first.PHASE),
        ("provider", "different"),
        ("generation_model", "different"),
    ],
)
def test_second_overlay_rejects_changed_lineage(cohort, field, value):
    _, original, _, first_measured = cohort
    measured = [_measured(r, failed=False) for r in second.build_records(*cohort)]
    measured[0][field] = value
    with pytest.raises(ValueError, match="lineage"):
        second.recovered_main_rows(original, first_measured, measured)


def test_second_overlay_requires_complete_twenty_rows(cohort):
    _, original, _, first_measured = cohort
    measured = [_measured(r, failed=False) for r in second.build_records(*cohort)]
    with pytest.raises(ValueError, match="twenty fresh"):
        second.recovered_main_rows(original, first_measured, measured[:-1])
    del measured[0]["full_correct"]
    with pytest.raises(ValueError, match="complete measured"):
        second.recovered_main_rows(original, first_measured, measured)
