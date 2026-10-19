"""Deterministic contract tests; snippets are inert text, never executed."""

import json
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from grader_power.schemas import (
    CorrectnessOutcome,
    CorrectnessResults,
    ErrorDetails,
    ExtractionOutcome,
    ExtractionResult,
    FixtureMetadata,
    ModelVisibleTask,
    OracleTask,
    ResponseResult,
    ScoringRegion,
    SourceTaskIdentity,
    StyleResult,
    StyleScore,
    TrialRecord,
)


@pytest.fixture
def source() -> SourceTaskIdentity:
    """Synthetic provenance shaped like the loader's HumanEval manifest entry."""
    return SourceTaskIdentity(
        source="humaneval",
        task_id="HumanEval/9999",
        source_revision="synthetic-v1",
        source_hash="a" * 64,
        evalplus_version="synthetic-package",
    )


@pytest.fixture
def fixture_data(source: SourceTaskIdentity) -> dict[str, Any]:
    """Provisional metadata with no human review or verified references."""
    return {
        "task": {
            "source": source,
            "derived_task_id": "synthetic-task",
            "family": "mapping",
            "split": "candidate",
        },
        "fixture_version": "synthetic-v1",
        "oracle_version": "synthetic-v1",
        "visible_contract_hash": "b" * 64,
        "oracle_hash": "c" * 64,
        "loop_ref_hash": None,
        "comp_ref_hash": None,
        "scoring_region": ScoringRegion(
            "synthetic-task", "transform", "Build the returned list", "returned_list_v1"
        ),
        "screening_reasons": (),
        "adaptation_notes": (),
        "human_review_status": "pending",
    }


@pytest.fixture
def unevaluated() -> dict[str, Any]:
    """Explicit absence of correctness evidence, without a synthetic pass/fail."""
    return {
        "status": "not_evaluated",
        "tests_passed": None,
        "tests_total": None,
        "reason": "No verified Docker evaluation has run",
        "error": None,
        "test_log_ref": None,
    }


@pytest.fixture
def trial_data(
    run_config_data: dict[str, Any],
    factor_data: dict[str, Any],
    fixture_data: dict[str, Any],
    unevaluated: dict[str, Any],
) -> dict[str, Any]:
    """An assigned attempt with missing model output; keep it in the logged population."""
    return {
        "schema_version": 1,
        "trial_id": "synthetic-trial",
        "attempt_id": "synthetic-attempt-1",
        "attempt_number": 1,
        "retry_of": None,
        "retry_error": None,
        "config": run_config_data,
        "fixture": fixture_data,
        "factors": factor_data,
        "started_at": datetime(2026, 1, 1, tzinfo=UTC),
        "finished_at": datetime(2026, 1, 1, tzinfo=UTC),
        "messages": (),
        "rendered_context_hash": "d" * 64,
        "prompt_template_hash": "e" * 64,
        "response": {
            "status": "missing",
            "raw_response": None,
            "provider_response_ref": None,
            "error": {
                "category": "response",
                "code": "response_missing",
                "message": "Synthetic missing output",
            },
        },
        "extraction": {
            "status": "not_attempted",
            "result": None,
            "reason": "No response",
        },
        "style": {"status": "not_evaluated", "result": None, "reason": "No extraction"},
        "correctness": {"base": dict(unevaluated), "plus": dict(unevaluated)},
        "compile_valid": None,
        "format_compliant": None,
        "attempted_policy_change": None,
        "token_usage": {
            "input_tokens": None,
            "output_tokens": None,
            "reasoning_tokens": None,
        },
        "wall_time_seconds": 0.0,
    }


def received_trial(data: dict[str, Any]) -> dict[str, Any]:
    """Replace only response/scoring stages with synthetic successful static results."""
    raw = "def transform(items):\n    return [item for item in items]\n"
    return {
        **data,
        "messages": (
            {"role": "user", "content": "Synthetic request", "tool_call_id": None},
        ),
        "response": {
            "status": "received",
            "raw_response": raw,
            "provider_response_ref": "runs/synthetic/raw.json",
            "error": None,
        },
        "extraction": {
            "status": "completed",
            "result": ExtractionResult(
                raw, "transform", raw, "raw_python", "ok", "Synthetic extraction"
            ),
            "reason": None,
        },
        "style": {
            "status": "completed",
            "result": StyleScore(
                "synthetic-task",
                "comprehension_only",
                "scored",
                "Synthetic static score",
            ),
            "reason": None,
        },
        "compile_valid": True,
        "format_compliant": True,
    }


def test_trial_round_trip_preserves_missing_states(trial_data: dict[str, Any]) -> None:
    """No absent response/evaluation is converted to a style label or correctness value."""
    record = TrialRecord.model_validate(trial_data)
    assert TrialRecord.model_validate_json(record.model_dump_json()) == record
    dumped = json.loads(record.model_dump_json())
    assert dumped["style"]["result"] is None
    assert dumped["correctness"]["base"]["status"] == "not_evaluated"
    assert dumped["correctness"]["plus"]["tests_passed"] is None
    assert dumped["token_usage"]["output_tokens"] is None


def test_received_mock_response_round_trip(trial_data: dict[str, Any]) -> None:
    """Static success is preserved independently of unevaluated functional correctness."""
    record = TrialRecord.model_validate(received_trial(trial_data))
    restored = TrialRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.extraction.result is not None
    assert restored.extraction.result.code == restored.response.raw_response
    assert restored.correctness.plus.status == "not_evaluated"


@pytest.mark.parametrize(
    "status", ["not_evaluated", "infrastructure_error", "passed", "failed"]
)
def test_correctness_states_are_distinct(
    unevaluated: dict[str, Any], status: str
) -> None:
    """Timeouts are failed solutions; unavailable sandboxes are infrastructure failures."""
    data = {**unevaluated, "status": status}
    if status == "infrastructure_error":
        data["error"] = {
            "category": "infrastructure",
            "code": "sandbox_unavailable",
            "message": "Synthetic missing sandbox",
        }
    elif status in ("passed", "failed"):
        data.update(
            tests_total=3,
            tests_passed=3 if status == "passed" else 1,
            reason=None,
            test_log_ref="runs/synthetic/tests.json",
        )
        if status == "failed":
            data["error"] = {
                "category": "solution",
                "code": "solution_timeout",
                "message": "Synthetic timeout",
            }
    outcome = CorrectnessOutcome.model_validate(data)
    assert CorrectnessOutcome.model_validate_json(outcome.model_dump_json()) == outcome
    assert outcome.status == status


@pytest.mark.parametrize(
    "change",
    [
        {"status": "passed"},
        {"status": "failed"},
        {"status": "infrastructure_error"},
        {"tests_passed": 0},
        {"reason": None},
        {"tests_total": 0},
        {
            "status": "passed",
            "tests_total": 3,
            "tests_passed": 2,
            "test_log_ref": "runs/synthetic/log",
        },
        {
            "status": "passed",
            "tests_total": 3,
            "tests_passed": 4,
            "test_log_ref": "runs/synthetic/log",
        },
        {
            "status": "failed",
            "tests_total": 3,
            "tests_passed": 3,
            "test_log_ref": "runs/synthetic/log",
        },
        {
            "status": "infrastructure_error",
            "error": {
                "category": "solution",
                "code": "assertion_failure",
                "message": "Wrong result",
            },
        },
    ],
)
def test_inconsistent_correctness_rejected(
    unevaluated: dict[str, Any], change: dict[str, Any]
) -> None:
    """Absent or contradictory evidence cannot create a success-shaped result."""
    with pytest.raises(ValidationError):
        CorrectnessOutcome.model_validate({**unevaluated, **change})


def test_both_suites_and_all_result_fields_required(
    unevaluated: dict[str, Any], trial_data: dict[str, Any]
) -> None:
    """Missing required data never receives a passed/failed or empty-result default."""
    with pytest.raises(ValidationError, match="plus"):
        CorrectnessResults.model_validate({"base": unevaluated})
    for field in trial_data:
        incomplete = dict(trial_data)
        del incomplete[field]
        with pytest.raises(ValidationError, match=field):
            TrialRecord.model_validate(incomplete)


def test_valid_retry_links_same_trial(trial_data: dict[str, Any]) -> None:
    """A linked eligible transport retry remains a distinct attempt row."""
    data = {
        **trial_data,
        "attempt_id": "synthetic-attempt-2",
        "attempt_number": 2,
        "retry_of": "synthetic-attempt-1",
        "retry_error": {
            "category": "infrastructure",
            "code": "transport_error",
            "message": "Synthetic transport failure",
        },
    }
    record = TrialRecord.model_validate(data)
    assert record.trial_id == trial_data["trial_id"]
    assert record.retry_of != record.attempt_id


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_number": 2},
        {"retry_of": "other"},
        {"attempt_number": 3},
        {"compile_valid": True},
        {"finished_at": datetime(2025, 1, 1, tzinfo=UTC)},
        {"started_at": datetime(2026, 1, 1, tzinfo=UTC).replace(tzinfo=None)},
    ],
)
def test_invalid_trial_states(
    trial_data: dict[str, Any], change: dict[str, Any]
) -> None:
    """Reject missing retry evidence, impossible timing and stages without code."""
    with pytest.raises(ValidationError):
        TrialRecord.model_validate({**trial_data, **change})


def test_solution_failure_cannot_authorize_retry(trial_data: dict[str, Any]) -> None:
    """A real failed solution cannot be promoted to an infrastructure retry."""
    trial_data.update(
        attempt_number=2,
        attempt_id="second",
        retry_of="first",
        retry_error={
            "category": "solution",
            "code": "solution_timeout",
            "message": "Timed out",
        },
    )
    with pytest.raises(ValidationError, match="eligible infrastructure"):
        TrialRecord.model_validate(trial_data)
    with pytest.raises(ValidationError, match="category"):
        ErrorDetails.model_validate(
            {
                "category": "infrastructure",
                "code": "solution_timeout",
                "message": "Timed out",
            }
        )


def test_cross_stage_mismatches_rejected(trial_data: dict[str, Any]) -> None:
    """Code, raw output, style and fixture identities must refer to the same attempt."""
    data = received_trial(trial_data)
    data["response"]["raw_response"] = "different response"
    with pytest.raises(ValidationError, match="same raw"):
        TrialRecord.model_validate(data)
    data = received_trial(trial_data)
    data["style"]["result"] = StyleScore(
        "other-task", "loop_only", "scored", "Synthetic"
    )
    with pytest.raises(ValidationError, match="task_id"):
        TrialRecord.model_validate(data)


def test_compile_failure_retains_static_style(trial_data: dict[str, Any]) -> None:
    """Persist the real scorer's AST result even when compilation independently fails."""
    from grader_power.scoring import score_response

    data = received_trial(trial_data)
    raw = "return 3\ndef transform(items):\n    return [item for item in items]\n"
    result = score_response(raw, data["fixture"]["scoring_region"])
    assert result.compile_status == "invalid"
    assert result.style.label == "comprehension_only"
    data["response"]["raw_response"] = raw
    data["extraction"]["result"] = result.extraction
    data["style"]["result"] = result.style
    data.update(compile_valid=False, format_compliant=result.format_compliant)

    record = TrialRecord.model_validate(data)
    restored = TrialRecord.model_validate_json(record.model_dump_json())
    assert restored == record
    assert restored.style.result == result.style
    assert restored.compile_valid is False
    assert restored.correctness.base.status == "not_evaluated"
    assert restored.correctness.plus.status == "not_evaluated"

    data["correctness"]["base"] = {
        "status": "passed",
        "tests_passed": 1,
        "tests_total": 1,
        "reason": None,
        "error": None,
        "test_log_ref": "runs/synthetic/base.json",
    }
    with pytest.raises(ValidationError, match="cannot pass correctness"):
        TrialRecord.model_validate(data)


def test_failed_extraction_cannot_claim_style(trial_data: dict[str, Any]) -> None:
    """Keep rejecting successful style labels when the extractor found no target."""
    from grader_power.code_extract import extract_code

    data = received_trial(trial_data)
    raw = "def other(items): return []\n"
    data["response"]["raw_response"] = raw
    data["extraction"]["result"] = extract_code(raw, "transform")
    with pytest.raises(ValidationError, match="valid style"):
        TrialRecord.model_validate(data)


def test_parse_failure_retains_source_without_correctness(
    trial_data: dict[str, Any],
) -> None:
    """Malformed source stays available for audit with an explicit invalid style label."""
    raw = "def transform(:\n"
    data = received_trial(trial_data)
    data["response"]["raw_response"] = raw
    data["extraction"]["result"] = ExtractionResult(
        raw, "transform", raw, "raw_python", "parse_error", "Synthetic syntax error"
    )
    data["style"]["result"] = StyleScore(
        "synthetic-task",
        "invalid_or_unparseable",
        "invalid_or_unparseable",
        "Synthetic syntax error",
    )
    data.update(compile_valid=False, format_compliant=False)
    record = TrialRecord.model_validate(data)
    assert record.correctness.base.status == "not_evaluated"
    assert record.extraction.result is not None and record.extraction.result.code == raw


def test_response_and_extraction_absence_are_not_success() -> None:
    """A received empty response is different from no response; extraction still can fail."""
    response = ResponseResult(
        status="received",
        raw_response="",
        provider_response_ref="runs/synthetic/raw",
        error=None,
    )
    assert response.raw_response == ""
    with pytest.raises(ValidationError, match="requires"):
        ResponseResult(
            status="received", raw_response=None, provider_response_ref=None, error=None
        )
    with pytest.raises(ValidationError, match="result"):
        ExtractionOutcome(status="completed", result=None, reason=None)
    with pytest.raises(ValidationError, match="result"):
        StyleResult(status="completed", result=None, reason=None)


def test_source_and_fixture_round_trip(
    source: SourceTaskIdentity, fixture_data: dict[str, Any]
) -> None:
    """Provenance, provisional status and native scoring-region contract are retained."""
    fixture = FixtureMetadata.model_validate(fixture_data)
    assert FixtureMetadata.model_validate_json(fixture.model_dump_json()) == fixture
    assert fixture.task.source == source
    assert fixture.human_review_status == "pending"
    with pytest.raises(ValidationError, match="task_id"):
        SourceTaskIdentity.model_validate({**source.model_dump(), "source": "mbpp"})
    with pytest.raises(ValidationError, match="source_hash"):
        SourceTaskIdentity.model_validate(
            {**source.model_dump(), "source_hash": "unresolved"}
        )
    with pytest.raises(ValidationError, match="human_review_status"):
        FixtureMetadata.model_validate(
            {**fixture_data, "human_review_status": "approved"}
        )


def test_visible_contract_cannot_accept_whole_record(
    source: SourceTaskIdentity,
) -> None:
    """Hidden fields cannot leak via generic model serialization or ignored extras."""
    visible = {
        "derived_task_id": "synthetic-task",
        "target_file": "src/task.py",
        "readme": "Synthetic fixture",
        "stub": "def transform(items):\n    raise NotImplementedError\n",
    }
    task = ModelVisibleTask.model_validate(visible)
    assert ModelVisibleTask.model_validate_json(task.model_dump_json()) == task
    for hidden in (
        "canonical_solution",
        "base_input",
        "plus_input",
        "expected_outputs",
    ):
        with pytest.raises(ValidationError, match=hidden):
            ModelVisibleTask.model_validate({**visible, hidden: "HIDDEN_CANARY"})
    inputs = [(1.5, 2j)]
    oracle = OracleTask(
        source=source,
        entry_point="transform",
        canonical_solution="HIDDEN_CANARY",
        contract="",
        base_input=inputs,
        plus_input=[(3.5, 4j)],
        atol=0.0,
    )
    assert oracle.base_input is inputs
    assert oracle.base_input[0][1] == 2j
    assert "HIDDEN_CANARY" not in repr(oracle)
    assert not hasattr(oracle, "model_dump")
    assert "HIDDEN_CANARY" not in task.model_dump_json()
    malformed: Any = {}
    with pytest.raises(TypeError, match="plus_input"):
        OracleTask(
            source=source,
            entry_point="transform",
            canonical_solution="HIDDEN_CANARY",
            contract="",
            base_input=inputs,
            plus_input=malformed,
            atol=0.0,
        )
    assert malformed == {}


@pytest.mark.parametrize(
    "path",
    [
        "../task.py",
        "/tmp/task.py",
        "src/../task.py",
        "src//task.py",
        "fixtures/oracles/task.py",
        "src/task.txt",
    ],
)
def test_visible_paths_are_constrained(path: str) -> None:
    """Hidden oracle roots and path traversal cannot become renderer source targets."""
    with pytest.raises(ValidationError, match="target_file"):
        ModelVisibleTask(
            derived_task_id="synthetic",
            target_file=path,
            readme="Synthetic",
            stub="Synthetic",
        )


def test_contract_pipeline(
    trial_data: dict[str, Any], fixture_data: dict[str, Any]
) -> None:
    """Render eight conditions, score mock text statically, and round-trip attempt logs.

    Correctness deliberately stays not_evaluated: this test makes no provider
    requests and does not compile, import, or execute the synthetic solution.
    """
    from grader_power.code_extract import extract_code
    from grader_power.render import render_pilot_block
    from grader_power.score_style import score_style

    fixture = FixtureMetadata.model_validate(fixture_data)
    visible = ModelVisibleTask(
        derived_task_id="synthetic-task",
        target_file="src/task.py",
        readme="Synthetic contract demonstration; not research data.",
        stub="def transform(items):\n    raise NotImplementedError\n",
    )
    rendered = render_pilot_block(
        visible.to_render_task(fixture.task), preflight_variant="status", seed=7
    )
    raw = "def transform(items):\n    return [item for item in items]\n"
    extraction = extract_code(raw, fixture.scoring_region.entry_point)
    style = score_style(extraction, fixture.scoring_region)
    records = []
    for index, row in enumerate(rendered):
        data = received_trial(trial_data)
        data.update(
            trial_id=f"synthetic-trial-{index}",
            attempt_id=f"synthetic-attempt-{index}",
            factors=row.to_factor_assignment(),
            messages=tuple(
                {"role": message.role, "content": message.content, "tool_call_id": None}
                for message in row.messages
            ),
            rendered_context_hash=row.prompt_sha256,
            prompt_template_hash=row.metadata.template_sha256,
            compile_valid=None,
        )
        data["extraction"]["result"] = extraction
        data["style"]["result"] = style
        record = TrialRecord.model_validate(data)
        assert TrialRecord.model_validate_json(record.model_dump_json()) == record
        assert (
            record.correctness.base.status
            == record.correctness.plus.status
            == "not_evaluated"
        )
        assert record.factors.grader_consequence == row.condition.grader_consequence
        assert record.style.result is not None
        assert record.style.result.label == "comprehension_only"
        records.append(record)
    assert len({record.trial_id for record in records}) == 8
    assert len({record.rendered_context_hash for record in records}) == 8
