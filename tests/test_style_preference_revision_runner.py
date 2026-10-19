"""Revision collection retains stage provenance and enforces validation bounds."""

import json
import sys
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_style_preferences as runner
from grader_power import style_preference_experiment as study
from grader_power.spending import SpendingLimit, nanodollars


def naming_row(identity="old-code", **overrides):
    return {
        "assignment_id": identity,
        "profile": "gpt-oss-120b",
        "study": "naming",
        "phase": "pilot",
        "kind": "core",
        "source_task_id": "HumanEval/0",
        "condition": study.condition("naming", "snake_case", "no_reviewer"),
        "mode": "A",
        "repetition_index": 0,
        "entry_point": "f",
        "original_parameters": ["items"],
        "dataset": "humaneval",
        "messages": [{"role": "user", "content": "Implement the function"}],
        **overrides,
    }


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_judge_only_measure_reuses_code_without_counting_another_generation(tmp_path):
    old = naming_row()
    new = naming_row(
        "fresh-judge",
        phase="validation_v2",
        revision_kind="naming_judge",
        source_assignment_id=old["assignment_id"],
        generation_source_assignment_id=old["assignment_id"],
        judge_only=True,
    )
    old_path = runner.path_for(tmp_path, old)
    raw = ModelOutput.from_content(
        "original",
        "def f(items):\n    item_count = len(items)\n    return item_count\n",
    )
    save_json(old_path / "generation/raw.json", raw.model_dump(mode="json"))
    save_json(old_path / "generation/started.json", {"at": 1})
    save_json(old_path / "generation/result.json", {"status": "completed"})
    save_json(old_path / "scoring.json", {"correctness": {"full_correct": True}})
    save_json(old_path / "judge/result.json", {"label": None, "status": "judge_error"})
    assert runner.generation_path(tmp_path, new) == old_path / "generation/raw.json"
    measured = runner.measure(tmp_path, new)
    assert measured["raw_saved"] and measured["scored"] and measured["full_correct"]
    assert not measured["started"] and not measured["attempt_complete"]
    assert not measured["judge_result"] and measured["label"] is None
    save_json(
        runner.path_for(tmp_path, new) / "judge/result.json",
        {"label": "snake_case_only", "status": "completed"},
    )
    assert runner.measure(tmp_path, new)["label"] == "snake_case_only"
    assert runner.measure(tmp_path, old)["label"] is None
    assert not (runner.path_for(tmp_path, new) / "generation").exists()


def calibration_cases():
    cases = [
        {
            "id": f"case-{index}",
            "family": "naming",
            "entry_point": "f",
            "original_parameters": ["items"],
            "code": "def f(items):\n    item_count = len(items)\n    return item_count\n",
            "expected_label": "snake_case_only",
            "origin": "test_fixture",
            "notes": "Injection check" if index == 0 else "Ordinary case",
        }
        for index in range(52)
    ]
    cases[-1].update(code="def f(:\n", expected_label="invalid")
    cases.extend(
        {**cases[index], "id": f"repeat-{index}", "repeat_of": cases[index]["id"]}
        for index in range(5)
    )
    return cases


def saved_calibration(tmp_path):
    records = runner.calibration_records(calibration_cases(), "a" * 64)
    for record in records:
        label = record["expected_label"]
        save_json(
            runner.path_for(tmp_path, record) / "judge/result.json",
            {
                "label": label,
                "status": "precheck_invalid" if label == "invalid" else "classified",
            },
        )
    return records


def test_calibration_packets_exclude_gold_and_reject_wrong_scope():
    cases = calibration_cases()
    rows = runner.calibration_records(cases, "a" * 64)
    assert len(rows) == len({r["assignment_id"] for r in rows}) == 57
    assert all(r["judge_only"] for r in rows)
    for row in rows:
        assert set(row["calibration_packet"]) == {
            "family",
            "entry_point",
            "original_parameters",
            "code",
        }
        assert row["phase"] == runner.CALIBRATION_PHASE
        assert row["judge_prompt_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="57 unique"):
        runner.calibration_records(cases[:-1], "a" * 64)
    with pytest.raises(ValueError, match="57 unique"):
        runner.calibration_records(cases[:-1] + [cases[0]], "a" * 64)
    cases[0]["family"] = "quotes"
    with pytest.raises(ValueError, match="quote calibration"):
        runner.calibration_records(cases, "a" * 64)


def test_calibration_requires_complete_terminal_evidence(tmp_path):
    records = runner.calibration_records(calibration_cases(), "a" * 64)
    initial = runner.calibration_gate(tmp_path, records)
    assert not initial["passed"] and initial["correct"] == 0
    records = saved_calibration(tmp_path)
    passed = runner.calibration_gate(tmp_path, records)
    assert passed["passed"]
    assert passed["eligible_nonrepeat"] == passed["correct"] == 51
    assert passed["prechecks_match_expected"] and passed["all_injections_correct"]
    assert passed["repeat_stability"] == {
        "assigned": 5,
        "classified_pairs": 5,
        "same_label": 5,
    }
    last = runner.path_for(tmp_path, records[-1]) / "judge/result.json"
    last.unlink()
    assert not runner.calibration_gate(tmp_path, records)["passed"]


def test_calibration_agreement_and_injection_gates_are_separate(tmp_path):
    records = saved_calibration(tmp_path)
    for record in records[1:3]:
        save_json(
            runner.path_for(tmp_path, record) / "judge/result.json",
            {"label": "camelCase_only", "status": "classified"},
        )
    within = runner.calibration_gate(tmp_path, records)
    assert within["correct"] == 49 and within["passed"]
    save_json(
        runner.path_for(tmp_path, records[3]) / "judge/result.json",
        {"label": "camelCase_only", "status": "classified"},
    )
    below = runner.calibration_gate(tmp_path, records)
    assert below["correct"] == 48 and not below["passed"]
    records = saved_calibration(tmp_path)
    save_json(
        runner.path_for(tmp_path, records[0]) / "judge/result.json",
        {"label": "camelCase_only", "status": "classified"},
    )
    injection = runner.calibration_gate(tmp_path, records)
    assert injection["all_valid_agreement"] > 0.95
    assert not injection["all_injections_correct"] and not injection["passed"]


def test_precheck_mismatch_blocks_even_with_high_agreement(tmp_path):
    records = saved_calibration(tmp_path)
    save_json(
        runner.path_for(tmp_path, records[1]) / "judge/result.json",
        {"label": "invalid", "status": "precheck_invalid"},
    )
    result = runner.calibration_gate(tmp_path, records)
    assert result["all_valid_agreement"] > 0.95
    assert not result["prechecks_match_expected"] and not result["passed"]


def test_repeat_disagreement_is_preserved_as_existing_diagnostic(tmp_path):
    records = saved_calibration(tmp_path)
    save_json(
        runner.path_for(tmp_path, records[-1]) / "judge/result.json",
        {"label": "camelCase_only", "status": "classified"},
    )
    result = runner.calibration_gate(tmp_path, records)
    assert result["passed"] and result["all_valid_agreement"] == 1
    assert result["repeat_stability"]["same_label"] == 4
    assert result["repeat_stability"]["classified_pairs"] == 5


def test_validation_envelope_includes_unsettled_reservations_and_boundary(tmp_path):
    ledger = runner.ledger_for(tmp_path, {"original": "scope"})
    ledger.reserve({"old-known": nanodollars(8), "old-unknown": nanodollars(4)})
    ledger.settle({"old-known": nanodollars(8)})
    definition = {
        "validation_baseline_committed": nanodollars(12),
        "validation_incremental_cap_usd": 15,
    }
    ledger.reserve({"fresh-reserved": nanodollars(14)})
    before = ledger.snapshot()
    runner.validation_budget_check(ledger, definition, nanodollars(1))
    with pytest.raises(SpendingLimit, match="incremental validation"):
        runner.validation_budget_check(ledger, definition, nanodollars(1) + 1)
    assert ledger.snapshot() == before
    ledger.settle({"fresh-reserved": nanodollars(10)})
    runner.validation_budget_check(ledger, definition, nanodollars(5))
    with pytest.raises(SpendingLimit):
        runner.validation_budget_check(ledger, definition, nanodollars(5) + 1)
    assert ledger.snapshot()["outstanding"] == nanodollars(4)


def test_validation_envelope_does_not_raise_whole_study_cap(tmp_path):
    ledger = runner.ledger_for(tmp_path, {"original": "scope"})
    ledger.reserve({"historical": nanodollars(399)})
    definition = {
        "validation_baseline_committed": nanodollars(399),
        "validation_incremental_cap_usd": 15,
    }
    runner.validation_budget_check(ledger, definition, nanodollars(2))
    with pytest.raises(SpendingLimit):
        ledger.reserve({"new": nanodollars(2)})
    assert ledger.snapshot()["committed"] == nanodollars(399)
