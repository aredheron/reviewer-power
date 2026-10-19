"""Approved fresh validation preserves old evidence and reduces only repetitions."""

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from grader_power import style_preference_experiment as study
from grader_power import style_preference_revision as revision

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records():
    return study.schedule(ROOT)[0]


@pytest.fixture(scope="module")
def measured(records):
    rows = [
        {
            **copy.deepcopy(r),
            "raw_saved": True,
            "scored": r["kind"] == "core",
            "full_correct": r["kind"] == "core",
            "label": study.STYLES[r["study"]][0] + "_only"
            if r["kind"] == "core"
            else None,
            "probe_correct": True,
            "probe_correct_strict": True,
            "judge_started": r["kind"] == "core",
            "judge_result": r["kind"] == "core",
            "judge_status": "completed" if r["kind"] == "core" else None,
        }
        for r in records
        if r["phase"] == "pilot"
    ]
    invalid = next(r for r in rows if r["study"] == "naming" and r["kind"] == "core")
    invalid.update(label="invalid", full_correct=False)
    return rows


@pytest.fixture(scope="module")
def fresh(records, measured):
    return revision.build_revision_records(
        records, measured, naming_judge_prompt_sha256="a" * 64
    )


def measure_fresh(fresh):
    return [
        {
            **copy.deepcopy(r),
            "started": not r["judge_only"],
            "attempt_complete": not r["judge_only"],
            "raw_saved": True,
            "scored": r["kind"] == "core",
            "full_correct": r["kind"] == "core",
            "label": study.STYLES[r["study"]][0] + "_only"
            if r["kind"] == "core"
            else None,
            "probe_correct": True,
            "probe_correct_strict": True,
            "judge_started": r["kind"] == "core",
            "judge_result": r["kind"] == "core",
            "judge_status": "completed" if r["kind"] == "core" else None,
        }
        for r in fresh
    ]


def test_revision_exact_scope_and_fresh_identities(records, measured, fresh):
    assert Counter(r["revision_kind"] for r in fresh) == {
        "facts": 224,
        "quote_core": 140,
        "naming_judge": 279,
    }
    assert sum(not r["judge_only"] for r in fresh) == 364
    assert len({r["assignment_id"] for r in fresh}) == 643
    assert not {r["assignment_id"] for r in fresh} & {
        r["assignment_id"] for r in records
    }
    assert len({r["source_assignment_id"] for r in fresh}) == 643
    assert all(r["phase"] == revision.PHASE for r in fresh)
    assert fresh == revision.build_revision_records(
        records, measured, naming_judge_prompt_sha256="a" * 64
    )
    changed_prompt = revision.build_revision_records(
        records, measured, naming_judge_prompt_sha256="b" * 64
    )
    assert {r["assignment_id"] for r in fresh if not r["judge_only"]} == {
        r["assignment_id"] for r in changed_prompt if not r["judge_only"]
    }
    assert not {r["assignment_id"] for r in fresh if r["judge_only"]} & {
        r["assignment_id"] for r in changed_prompt if r["judge_only"]
    }


def test_prompts_only_change_the_approved_pilot_scope(records, fresh):
    original = {r["assignment_id"]: r for r in records}
    proposal = json.loads(
        (study.base(ROOT) / "operations/proposed-validation-revision.json").read_text()
    )
    assert revision.PROBE_CLARIFICATION == proposal["proposed_probe_clarification"]
    assert (
        revision.TASK_CLARIFICATIONS
        == proposal["proposed_pilot_only_contract_clarifications"]
    )
    for row in fresh:
        old = original[row["source_assignment_id"]]
        before, after = old["messages"][0]["content"], row["messages"][0]["content"]
        assert row["condition"] == old["condition"]
        assert row["request_id"] == row["assignment_id"]
        assert row["prompt_sha256"] == study.digest(study.json_text(row["messages"]))
        if row["revision_kind"] == "facts":
            assert row["profile"] in revision.PROBE_PROFILES
            assert after == before + "\n\n" + revision.PROBE_CLARIFICATION
        elif row["revision_kind"] == "quote_core":
            assert row["study"] == "quotes"
            addition = (
                "\n\nPilot task clarification:\n"
                + revision.TASK_CLARIFICATIONS[row["source_task_id"]]
            )
            assert after.replace(addition, "", 1) == before
            assert after.endswith(study.OUTPUT_CONTRACT)
        else:
            assert row["study"] == "naming"
            assert before == after
            assert row["generation_source_assignment_id"] == old["assignment_id"]
            assert row["judge_prompt_sha256"] == "a" * 64
        if not row["judge_only"]:
            assert "generation_source_assignment_id" not in row


def test_original_inputs_are_never_mutated(records, measured):
    original_records_hash = study.digest(study.json_text(records))
    measured_hash = study.digest(study.json_text(measured))
    fresh = revision.build_revision_records(
        records, measured, naming_judge_prompt_sha256="a" * 64
    )
    effective = revision.qualified_pilot_rows(measured, measure_fresh(fresh))
    fresh[0]["condition"]["arm"] = "mutated"
    effective[0]["condition"]["arm"] = "mutated"
    assert study.digest(study.json_text(records)) == original_records_hash
    assert study.digest(study.json_text(measured)) == measured_hash


def test_overlay_reuses_correctness_but_never_falls_back_from_failed_fresh_judge(
    measured, fresh
):
    revised = measure_fresh(fresh)
    effective = revision.qualified_pilot_rows(measured, revised)
    assert len(effective) == 1120
    assert {r["assignment_id"] for r in effective} == {
        r["assignment_id"] for r in measured
    }
    assert study.pilot_gate(effective, naming_mixed_policy_approved=True)["passed"]
    judges = [r for r in revised if r["judge_only"]]
    for row in judges[:3]:
        row.update(label=None, judge_status="judge_error", full_correct=False)
    effective = revision.qualified_pilot_rows(measured, revised)
    by_id = {r["assignment_id"]: r for r in effective}
    for judge in judges[:3]:
        row = by_id[judge["source_assignment_id"]]
        assert row["label"] is None
        assert row["full_correct"] is True
        assert row["qualification_generation_assignment_id"] == row["assignment_id"]
        assert row["qualification_evidence_assignment_id"] == judge["assignment_id"]
    retained = [r for r in effective if r.get("label") == "invalid"]
    assert len(retained) == 1 and "qualification_revision" not in retained[0]


def test_failed_fresh_facts_replace_old_correct_answers(measured, fresh):
    revised = measure_fresh(fresh)
    for row in revised:
        if row["revision_kind"] == "facts":
            row.update(raw_saved=False, probe_correct=None, probe_correct_strict=None)
    effective = revision.qualified_pilot_rows(measured, revised)
    assert sum(r["raw_saved"] is False for r in effective) == 224
    assert not study.pilot_gate(effective)["passed"]


def test_overlay_rejects_missing_duplicate_or_mismatched_replacement(measured, fresh):
    revised = measure_fresh(fresh)
    with pytest.raises(ValueError, match="643"):
        revision.qualified_pilot_rows(measured, revised[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        revision.qualified_pilot_rows(measured, revised[:-1] + [revised[0]])
    revised[0]["source_task_id"] = "Mbpp/999999"
    with pytest.raises(ValueError, match="approved pilot cell"):
        revision.qualified_pilot_rows(measured, revised)


def test_builder_refuses_unreviewed_invalid_or_missing_naming_code(records, measured):
    changed = copy.deepcopy(measured)
    invalid = next(r for r in changed if r.get("label") == "invalid")
    invalid["label"] = "snake_case_only"
    with pytest.raises(ValueError, match="279"):
        revision.build_revision_records(
            records, changed, naming_judge_prompt_sha256="a" * 64
        )
    invalid["scored"] = False
    with pytest.raises(ValueError, match="saved, scored"):
        revision.build_revision_records(
            records, changed, naming_judge_prompt_sha256="a" * 64
        )


def test_main_reduction_preserves_ids_prompts_and_all_twenty_four_tasks(records):
    selected = revision.selected_main_records(records)
    excluded = revision.excluded_main_records(records)
    assert len(selected) == 10080 and len(excluded) == 3360
    assert {r["repetition_index"] for r in selected} == {0, 1, 2}
    assert {r["repetition_index"] for r in excluded} == {3}
    original = {r["assignment_id"]: r for r in records if r["phase"] == "main"}
    assert {r["assignment_id"] for r in selected + excluded} == set(original)
    assert all(r == original[r["assignment_id"]] for r in selected + excluded)
    assert Counter((r["profile"], r["study"]) for r in selected) == {
        (model, family): 1008 for model in study.PROFILES for family in study.FAMILIES
    }
    for family in study.FAMILIES:
        assert (
            len({r["source_task_id"] for r in selected if r["study"] == family}) == 24
        )
    with pytest.raises(ValueError, match="13,440"):
        revision.selected_main_records(selected)


def test_three_repetition_analysis_retains_scaling_and_missingness(records):
    rows = revision.selected_main_records(records)
    for row in rows:
        row.update(raw_saved=True, full_correct=True)
        row["label"] = (
            row["condition"]["grader_preference"] + "_only"
            if row["condition"]["grader_consequence"] == "rl_update"
            else study.STYLES[row["study"]][0] + "_only"
        )
    result = study.analyze(rows, draws=100, repetitions=3)
    assert result["assigned"] == 10080
    assert result["assigned_by_family"] == {"naming": 5040, "quotes": 5040}
    assert result["repetitions"] == 3 and result["holm_family_size"] == 20
    assert len(result["no_reviewer_descriptive"]) == 120
    assert len(result["by_condition"]) == 140
    assert all(r["assigned"] == 72 for r in result["by_condition"])
    for contrast in result["comparisons"]:
        expected = 1.0 if contrast["contrast"].startswith("rl_update") else 0.0
        assert contrast["estimate"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="13,440"):
        study.analyze(rows, draws=100)
    with pytest.raises(ValueError, match="four or approved three"):
        study.analyze(rows, draws=100, repetitions=2)
    rows[0]["label"] = None
    result = study.analyze(rows, draws=100, repetitions=3)
    assert result["missing_labels"] == 1
    assert all(r["holm_p_value"] is None for r in result["comparisons"])


def test_revision_does_not_relax_factual_parsing(fresh):
    row = next(r for r in fresh if r["revision_kind"] == "facts")
    answer = json.dumps(study.expected_facts(row))
    assert study.score_probe(answer, row)["strict"]["all_correct"]
    assert study.score_probe("```json\n" + answer + "\n```", row)["fence_only"][
        "all_correct"
    ]
    assert not study.score_probe(answer + answer, row)["fence_only"]["all_correct"]
