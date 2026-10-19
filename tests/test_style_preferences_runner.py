"""Request identity, strict V3 judging, isolated scoring, and independent family gates."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_style_preferences as runner
from grader_power import style_preference_experiment as study
from grader_power.datasets.oracle import OracleResult
from grader_power.spending import SpendingLimit, nanodollars


def row(**overrides):
    return {
        "assignment_id": "case-one",
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


def manifest():
    return {
        "judge_profile": study.JUDGE_PROFILE,
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "judges": {
            f: {
                "system_prompt": "Frozen test rubric",
                "generation": {"max_tokens": 2048},
            }
            for f in runner.FAMILIES
        },
        "oracle_image": "sha256:" + "a" * 64,
    }


def pipeline(tmp_path, monkeypatch, rows=None):
    monkeypatch.setattr(
        runner, "load_bundles", lambda root: ([], {"humaneval": object()})
    )
    return runner.Pipeline(tmp_path, manifest(), rows or [row()], "pilot")


def raw(code="def f(items):\n    result_values = items\n    return result_values\n"):
    output = ModelOutput.from_content("test", code)
    output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    return output


def save_raw(p, r, response=None):
    runner.original.freeze(
        runner.generation_path(p.base, r), (response or raw()).model_dump(mode="json")
    )


def test_started_request_never_replayed_and_usage_settles(tmp_path, monkeypatch):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    client = AsyncMock()
    client.generate.return_value = raw()
    p.models[r["profile"]] = client

    async def work():
        first = await p.request(r, "generation", [ChatMessageUser(content="safe")])
        assert first == await p.request(
            r, "generation", [ChatMessageUser(content="safe")]
        )

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["spent"] > 0 and p.ledger.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("failure", ["timeout", "missing_usage", "credential_echo"])
def test_uncertain_or_integrity_failure_retains_reservation(
    tmp_path, monkeypatch, failure
):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    client = AsyncMock()
    p.models[r["profile"]] = client
    p.keys.add("test-secret-not-real")
    if failure == "timeout":
        client.generate.side_effect = TimeoutError("uncertain acceptance")
    else:
        client.generate.return_value = ModelOutput.from_content(
            "test", "test-secret-not-real" if failure == "credential_echo" else "safe"
        )
    asyncio.run(p.request(r, "generation", [ChatMessageUser(content="safe")]))
    if not p.stop.is_set():
        asyncio.run(p.request(r, "generation", [ChatMessageUser(content="safe")]))
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["spent"] == 0 and p.ledger.snapshot()["outstanding"] > 0
    if failure == "credential_echo":
        assert not runner.generation_path(p.base, r).exists()
        assert all(
            "test-secret-not-real" not in f.read_text() for f in p.base.rglob("*.json")
        )


def test_flat_ledger_enforces_whole_study_cap(tmp_path):
    ledger = runner.ledger_for(tmp_path, manifest())
    ledger.reserve({"one": nanodollars(399)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"two": nanodollars(2)})
    assert ledger.snapshot()["outstanding"] == nanodollars(399)


def test_packet_is_neutral_and_preserves_original_parameters():
    r = row()
    packet = runner.packet_from_output(raw().model_dump(mode="json"), r)
    assert packet is not None
    assert set(packet.model_dump()) == {
        "family",
        "entry_point",
        "original_parameters",
        "code",
    }
    assert packet.original_parameters == ["items"]
    assert "snake_case" not in runner.render_packet("naming", packet.model_dump())
    with pytest.raises(ValueError):
        runner.render_packet("quotes", packet.model_dump())


@pytest.mark.parametrize(
    "code", ["not python!!!", "def wrong(items):\n    return items\n", "def f(:\n pass"]
)
def test_invalid_code_never_gets_a_judge_packet(code):
    assert runner.packet_from_output(raw(code).model_dump(mode="json"), row()) is None


def test_v3_no_evidence_is_resolved_and_exact_source_is_attached():
    packet = {
        "family": "naming",
        "entry_point": "f",
        "original_parameters": ["items"],
        "code": "def f(items):\n    return items\n",
    }
    answer = {
        "status": "classified",
        "evidence": [{"start_line": 1, "end_line": 2}],
        "flags": [],
        "rationale": "No added informative binding",
        "target_forms": ["no_naming_evidence"],
    }
    parsed = runner.parse_packet("naming", json.dumps(answer), packet)
    assert parsed.label == "no_naming_evidence"
    assert parsed.evidence[0].quote == packet["code"].rstrip()
    with pytest.raises(ValueError):
        runner.parse_packet("naming", json.dumps(answer) + json.dumps(answer), packet)


def test_scoring_uses_only_isolated_oracle_and_never_replays(tmp_path, monkeypatch):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    save_raw(p, r)
    seen = []

    def oracle(dataset, name, task_id, code, **kw):
        seen.append((name, task_id, code, kw))
        return OracleResult(
            "wrong_answer", task_id, kw["image"], "hash", "Retained wrong answer"
        )

    monkeypatch.setattr(runner, "evaluate_solution", oracle)
    asyncio.run(p.scoring_stage(r))
    asyncio.run(p.scoring_stage(r))
    assert len(seen) == 1 and seen[0][3] == {
        "mode": "full",
        "image": manifest()["oracle_image"],
    }
    assert runner.measure(p.base, r)["full_correct"] is False


def test_interrupted_scoring_stops_without_replay(tmp_path, monkeypatch):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    save_raw(p, r)
    runner.original.freeze(
        runner.path_for(p.base, r) / "scoring-started.json", {"at": 0}
    )
    call = AsyncMock()
    monkeypatch.setattr(runner, "evaluate_solution", call)
    asyncio.run(p.scoring_stage(r))
    assert p.stop.is_set() and not call.called


def test_independent_audit_disagreement_preserves_judge_label(tmp_path, monkeypatch):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    save_raw(p, r)
    response = raw(
        json.dumps(
            {
                "status": "classified",
                "evidence": [{"start_line": 2, "end_line": 2}],
                "flags": [],
                "rationale": "Deliberately disagreeing mock",
                "target_forms": ["camelCase"],
            }
        )
    )
    monkeypatch.setattr(
        p, "request", AsyncMock(return_value=response.model_dump(mode="json"))
    )
    asyncio.run(p.judging_stage(r))
    measured = runner.measure(p.base, r)
    assert (
        measured["label"] == "camelCase_only"
        and measured["audit_label"] == "snake_case_only"
    )
    assert measured["audit_agrees"] is False


def test_family_phase_filter_drains_tail(tmp_path, monkeypatch):
    rows = [
        row(assignment_id=str(i), study="naming" if i % 2 else "quotes")
        for i in range(5)
    ]
    p = pipeline(tmp_path, monkeypatch, rows)
    p.family = "quotes"
    seen = []

    async def generation(r):
        seen.append(r["assignment_id"])

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert set(seen) == {"0", "2", "4"}


def test_pause_preserves_unstarted_requests(tmp_path, monkeypatch):
    p = pipeline(tmp_path, monkeypatch)
    p.stop.set()
    generate = AsyncMock()
    monkeypatch.setattr(p, "generation_stage", generate)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    with pytest.raises(RuntimeError):
        asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    generate.assert_not_awaited()


def write_ratio_approval(base, *, ratio=4):
    instruction = {
        "approved": True,
        "user_instruction": "Hold naming main if clear:mixed is below 4:1.",
        "minimum_clear_to_mixed_ratio": 4,
    }
    instruction_path = base / "operations/mixed-policy-user-instruction.json"
    runner.write_json(instruction_path, instruction)
    policy = {
        "approved": True,
        "policy": "clear_to_mixed_ratio",
        "minimum_clear_to_mixed_ratio": ratio,
        "user_instruction": instruction["user_instruction"],
        "instruction_sha256": runner.digest(instruction_path.read_bytes()),
    }
    runner.write_json(base / "operations/mixed-policy-approval.json", policy)
    return policy


def ratio_rows(clear=4, mixed=1):
    return [
        {**row(profile=profile, assignment_id=f"{profile}-{index}"), "label": label}
        for profile in study.PROFILES
        for index, label in enumerate(
            ["snake_case_only"] * (clear // 2)
            + ["camelCase_only"] * (clear - clear // 2)
            + ["mixed"] * mixed
        )
    ]


def test_naming_requires_bound_explicit_ratio_approval(tmp_path):
    rows = ratio_rows()
    assert runner.naming_policy(tmp_path, manifest(), rows) is None
    write_ratio_approval(tmp_path)
    result = runner.naming_policy(tmp_path, manifest(), rows)
    assert result is not None
    assert all(
        counts == {"clear": 4, "mixed": 1, "excluded": 0, "passed": True}
        for counts in result["by_model"].values()
    )
    instruction_path = tmp_path / "operations/mixed-policy-user-instruction.json"
    instruction_path.write_text(instruction_path.read_text() + "\n")
    with pytest.raises(ValueError, match="binding"):
        runner.naming_policy(tmp_path, manifest(), rows)


def test_excluded_outcomes_cannot_dilute_mixed_ratio(tmp_path):
    write_ratio_approval(tmp_path)
    rows = ratio_rows(clear=3)
    rows += [
        {**row(profile=profile), "label": label}
        for profile in study.PROFILES
        for label in ["no_naming_evidence", "neither", "invalid", None] * 20
    ]
    assert runner.naming_policy(tmp_path, manifest(), rows) is None


def test_ratio_is_per_model_and_ignores_other_families_or_phases(tmp_path):
    write_ratio_approval(tmp_path)
    rows = ratio_rows(clear=4)
    rows[0]["label"] = None
    rows += [{**row(profile="sonnet"), "label": "snake_case_only"} for _ in range(100)]
    rows += [
        {**row(study="quotes"), "label": "snake_case_only"},
        {**row(kind="comprehension_probe"), "label": "snake_case_only"},
        {**row(phase="main"), "label": "snake_case_only"},
    ]
    assert runner.naming_policy(tmp_path, manifest(), rows) is None


@pytest.mark.parametrize("clear", [0, 2])
def test_zero_mixed_passes_policy_without_replacing_readiness(tmp_path, clear):
    write_ratio_approval(tmp_path)
    rows = ratio_rows(clear=clear, mixed=0)
    rows += [
        {**row(profile=profile), "label": "no_naming_evidence"}
        for profile in study.PROFILES
    ]
    result = runner.naming_policy(tmp_path, manifest(), rows)
    assert result is not None
    assert all(
        counts["mixed"] == 0 and counts["passed"]
        for counts in result["by_model"].values()
    )
    assert not study.pilot_gate(rows, naming_mixed_policy_approved=True)["passed"]


@pytest.mark.parametrize("ratio", [3, 4.1, True, "4"])
def test_ratio_cannot_differ_from_exact_user_approval(tmp_path, ratio):
    write_ratio_approval(tmp_path, ratio=ratio)
    with pytest.raises(ValueError, match="4:1"):
        runner.naming_policy(tmp_path, manifest(), ratio_rows())


def test_five_failed_stages_guard_is_retained(tmp_path, monkeypatch):
    rows = [row(assignment_id=str(i)) for i in range(5)]
    p = pipeline(tmp_path, monkeypatch, rows)
    for r in rows:
        p.event(r, "judge", True, "ValidationError")
    assert p.stop.is_set()
    assert len(p.history) == 5
    p.event(rows[0], "judge", True, "ValidationError")
    assert len(p.history) == 5


def test_pacing_uses_arbitrary_gpt_lane_without_higher_default(tmp_path):
    pacer = runner.ModelPacer(tmp_path, profile="gpt-oss-120b", clock=lambda: 1000.0)
    with pacer.state() as state:
        lane = pacer.lane(state, "fireworks")
        assert lane["limits"]["output"] == 40000
        assert lane["fraction"] == 0.85
        assert lane["verified_limits"] == {}


def test_invalid_source_is_retained_without_docker_or_paid_judge(tmp_path, monkeypatch):
    r = row()
    p = pipeline(tmp_path, monkeypatch)
    save_raw(p, r, raw("def f(:\n    pass\n"))
    oracle = AsyncMock()
    request = AsyncMock()
    monkeypatch.setattr(runner, "evaluate_solution", oracle)
    monkeypatch.setattr(p, "request", request)
    asyncio.run(p.scoring_stage(r))
    asyncio.run(p.judging_stage(r))
    assert not oracle.called and not request.called
    measured = runner.measure(p.base, r)
    assert measured["scored"] and measured["full_correct"] is False
    assert measured["label"] == "invalid"


def test_seeded_pacing_preserves_reported_cap_cooldown_and_usage(tmp_path):
    source = (
        tmp_path / "runs/three-style/three-style-v1/operations/pacing/rate-state.json"
    )
    now = runner.time.time()
    lane = {
        "limits": {"output": 171661},
        "verified_limits": {"output": {"value": 171661, "observed_at": now - 500}},
        "cooldown_until": now + 10,
        "queue": [{"ticket": "stale"}],
        "window": [{"at": now, "demand": {"output": 100}, "settled": True}],
    }
    runner.write_json(source, {"fireworks": lane})
    base = study.base(tmp_path)
    runner.initialize_pacing(tmp_path, base)
    saved = json.loads(
        (base / "operations/pacing/gpt-oss-120b/rate-state.json").read_text()
    )["fireworks"]
    assert (
        saved["limits"] == lane["limits"]
        and saved["verified_limits"] == lane["verified_limits"]
    )
    assert (
        saved["cooldown_until"] == now + 10
        and saved["queue"] == []
        and saved["window"] == lane["window"]
    )


@pytest.fixture(scope="module")
def reference_evidence():
    bundles, datasets = runner.load_bundles(ROOT)
    report = json.loads((ROOT / runner.REFERENCE).read_text())
    tasks = json.loads(
        (ROOT / "runs/style-preferences/preparation-v2/tasks.json").read_text()
    )
    return bundles, datasets, report, tasks


@pytest.mark.parametrize(
    "damage", [None, "task_hash", "source", "code", "image", "duplicate", "incomplete"]
)
def test_cached_references_bind_exact_full_current_tasks(
    tmp_path, monkeypatch, reference_evidence, damage
):
    import copy

    bundles, datasets, source_report, tasks = reference_evidence
    report = copy.deepcopy(source_report)
    monkeypatch.setattr(runner, "load_bundles", lambda root: (bundles, datasets))
    tasks_path = tmp_path / "runs/style-preferences/preparation-v2/tasks.json"
    runner.write_json(tasks_path, tasks)
    report["tasks_sha256"] = runner.digest(tasks_path.read_bytes())
    first = report["outcomes"][0]
    if damage == "task_hash":
        report["tasks_sha256"] = "incorrect"
    elif damage == "source":
        first["binding"]["source_sha256"] = "incorrect"
    elif damage == "code":
        first["binding"]["code_sha256"] = "incorrect"
    elif damage == "image":
        first["result"]["image"] = "sha256:" + "b" * 64
    elif damage == "duplicate":
        report["outcomes"][1] = first
    elif damage == "incomplete":
        first["result"]["suites"]["plus"]["details"].pop()
    runner.write_json(tmp_path / runner.REFERENCE, report)
    if damage is None:
        runner.verify_reference_evidence(tmp_path, report["oracle_image"])
    else:
        with pytest.raises(ValueError):
            runner.verify_reference_evidence(tmp_path, report["oracle_image"])


def test_operational_amendment_cannot_change_scientific_source(tmp_path):
    old = {"code_sha256": {"src/grader_power/style_preference_experiment.py": "old"}}
    current = {"src/grader_power/style_preference_experiment.py": "changed"}
    amendment = {
        "code_sha256": current,
        "manifest_sha256": runner.digest(runner.json_text(old)),
    }
    path = tmp_path / "operations/amendment.json"
    runner.write_json(path, amendment)
    runner.write_json(
        tmp_path / "operations/recovery-state.json",
        {
            "source_amendment": "operations/amendment.json",
            "source_amendment_sha256": runner.digest(path.read_bytes()),
        },
    )
    with pytest.raises(ValueError, match="scientific"):
        runner.verify_operational_amendment(tmp_path, old, current, retained=True)


def test_shared_account_reported_cap_carries_forward(tmp_path):
    source = (
        tmp_path
        / "runs/replication/simple-models-v1/operations/pacing/account-rpm.json"
    )
    runner.write_json(source, {"cap": 1234, "reported_at": 100, "window": []})
    base = study.base(tmp_path)
    runner.initialize_pacing(tmp_path, base)
    state = json.loads((base / "operations/pacing/account-rpm.json").read_text())
    assert state["cap"] == 1234 and state["reported_at"] == 100


def test_unsupported_auxiliary_audit_preserves_blinded_judge(tmp_path, monkeypatch):
    r = row(study="quotes")
    p = pipeline(tmp_path, monkeypatch, [r])
    save_raw(p, r, raw('def f(items):\n    return f"{items}"\n'))
    p.keys.add("audit-test-secret")

    def unsupported(packet):
        raise ValueError("Could not locate formatted string in AST audit-test-secret")

    monkeypatch.setattr(runner.preference_judge, "audit_style", unsupported)
    response = raw(
        json.dumps(
            {
                "status": "classified",
                "evidence": [{"start_line": 2, "end_line": 2}],
                "flags": [],
                "rationale": "Double-quoted formatted string",
                "target_forms": ["double_quotes"],
            }
        )
    )
    request = AsyncMock(return_value=response.model_dump(mode="json"))
    monkeypatch.setattr(p, "request", request)
    asyncio.run(p.judging_stage(r))
    assert request.await_count == 1 and not p.stop.is_set()
    measured = runner.measure(p.base, r)
    assert measured["label"] == "double_quotes_only"
    assert measured["audit_label"] is None and measured["audit_agrees"] is None
    saved = json.loads((runner.path_for(p.base, r) / "style-audit.json").read_text())
    assert saved["status"] == "audit_error" and saved["error_type"] == "ValueError"
    assert (
        "audit-test-secret" not in saved["message"] and "[REDACTED]" in saved["message"]
    )


def test_revision_judge_only_queue_never_enters_generation_or_scoring(
    tmp_path, monkeypatch
):
    rows = [
        row(
            assignment_id=f"cal-{i}",
            phase=runner.CALIBRATION_PHASE,
            kind="judge_calibration",
            judge_only=True,
        )
        for i in range(70)
    ]
    p = pipeline(tmp_path, monkeypatch, rows)
    p.phase = runner.CALIBRATION_PHASE
    seen = []

    async def judge(r):
        seen.append(r["assignment_id"])

    generation = AsyncMock()
    scoring = AsyncMock()
    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "scoring_stage", scoring)
    monkeypatch.setattr(p, "judging_stage", judge)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert len(seen) == 70
    generation.assert_not_awaited()
    scoring.assert_not_awaited()


def test_calibration_invalid_precheck_has_no_generation_or_api(tmp_path, monkeypatch):
    r = row(
        phase=runner.CALIBRATION_PHASE,
        kind="judge_calibration",
        judge_only=True,
        calibration_packet={
            "family": "naming",
            "entry_point": "f",
            "original_parameters": ["items"],
            "code": "def f(:\n",
        },
    )
    p = pipeline(tmp_path, monkeypatch, [r])
    request = AsyncMock()
    monkeypatch.setattr(p, "request", request)
    asyncio.run(p.judging_stage(r))
    measured = runner.measure(p.base, r)
    assert measured["judge_status"] == "precheck_invalid"
    assert measured["label"] == "invalid" and measured["judge_result"]
    assert not measured["started"] and not measured["raw_saved"]
    assert not (runner.path_for(p.base, r) / "generation").exists()
    request.assert_not_awaited()


def test_incremental_cap_rejects_before_intent_and_does_not_replay(
    tmp_path, monkeypatch
):
    r = row(phase=runner.revision.PHASE)
    p = pipeline(tmp_path, monkeypatch, [r])
    p.revision = {
        "validation_baseline_committed": 0,
        "validation_incremental_cap_usd": 15,
    }
    client = AsyncMock()
    client.generate.return_value = raw()
    p.models[r["profile"]] = client
    p.ledger.reserve({"other-fresh-work": nanodollars(15)})
    with pytest.raises(SpendingLimit):
        asyncio.run(p.request(r, "generation", [ChatMessageUser(content="safe")]))
    client.generate.assert_not_awaited()
    assert not (runner.path_for(p.base, r) / "generation/started.json").exists()
    p.ledger.settle({"other-fresh-work": nanodollars(1)})
    asyncio.run(p.request(r, "generation", [ChatMessageUser(content="safe")]))
    balance = p.ledger.snapshot()
    asyncio.run(p.request(r, "generation", [ChatMessageUser(content="safe")]))
    assert p.ledger.snapshot() == balance
    assert client.generate.await_count == 1


def test_revised_forecast_uses_new_judge_and_new_quote_raws(tmp_path, monkeypatch):
    base = study.base(tmp_path)
    spec = manifest()
    revision_file = base / runner.REVISION_MANIFEST
    runner.original.freeze(revision_file, {"revision": "test"})
    records, overlay = [], []
    expected_bindings = set()
    for profile in study.PROFILES:
        for family in runner.FAMILIES:
            original_row = row(
                assignment_id=f"{profile}-{family}-old", profile=profile, study=family
            )
            fresh = row(
                assignment_id=f"{profile}-{family}-new",
                profile=profile,
                study=family,
                phase=runner.revision.PHASE,
            )
            records.extend([original_row, fresh])
            original_path = runner.path_for(base, original_row)
            new_path = runner.path_for(base, fresh)
            response = raw().model_dump(mode="json")
            runner.original.freeze(original_path / "generation/raw.json", response)
            runner.original.freeze(original_path / "judge/raw.json", response)
            runner.original.freeze(new_path / "generation/raw.json", response)
            runner.original.freeze(new_path / "judge/raw.json", response)
            generation = original_row if family == "naming" else fresh
            overlay.append(
                {
                    **original_row,
                    "qualification_generation_assignment_id": generation[
                        "assignment_id"
                    ],
                    "qualification_evidence_assignment_id": fresh["assignment_id"],
                }
            )
            expected_bindings.update(
                [
                    str(runner.generation_path(base, generation).relative_to(base)),
                    str((new_path / "judge/raw.json").relative_to(base)),
                ]
            )
    monkeypatch.setattr(runner, "qualified_rows", lambda base, records: overlay)
    report = runner.revision_forecast(tmp_path, spec, records)
    assert set(report["bindings"]) == expected_bindings
    assert report["margin"] == 1.5
    assert (
        sum(v["projected_main"] for v in report["models"].values()) * 1.5
        == report["projected_total_conservative_usd"]
    )
    for profile in study.PROFILES:
        for detail in report["models"][profile]["families"].values():
            assert (
                detail["projected_main"] == sum(detail["per_response"].values()) * 1008
            )


def test_incomplete_revision_calibration_blocks_main_without_forecast(
    tmp_path, monkeypatch
):
    base = study.base(tmp_path)
    runner.original.freeze(base / runner.REVISION_MANIFEST, {"revision": "test"})
    monkeypatch.setattr(
        runner, "calibration_gate", lambda base, records: {"passed": False}
    )
    forecast = AsyncMock()
    monkeypatch.setattr(runner, "revision_forecast", forecast)
    result = runner.main_readiness(tmp_path, manifest(), [])
    assert result["main_ready_by_family"] == {"naming": False, "quotes": False}
    forecast.assert_not_called()


@pytest.mark.parametrize(
    "field",
    ["validation_baseline_committed", "validation_baseline_spent", "ledger_scope"],
)
def test_revision_budget_cannot_raise_bound_by_editing_baseline(tmp_path, field):
    ledger = runner.ledger_for(tmp_path, {"original": "identity"})
    ledger.reserve({"original": nanodollars(15)})
    ledger.settle({"original": nanodollars(15)})
    balance = ledger.snapshot()
    definition = {
        "validation_baseline_committed": balance["committed"],
        "validation_baseline_spent": balance["spent"],
        "ledger_scope": balance["scope"],
    }
    approval = {"validation_starting_spent_nanodollars": balance["spent"]}
    audit = {"ledger_before": balance}
    runner.verify_revision_budget_baseline(definition, approval, audit)
    definition[field] = "wrong-scope" if field == "ledger_scope" else nanodollars(400)
    with pytest.raises(ValueError, match="budget baseline"):
        runner.verify_revision_budget_baseline(definition, approval, audit)


def test_revision_source_discovery_allows_only_approved_added_prompt():
    before = {"shared.py": "old-shared", "collector.py": "old-collector"}
    allowed = {"collector.py"}
    current = {
        **before,
        "collector.py": "revised",
        runner.REVISION_PROMPT: "new-prompt",
    }
    runner.verify_revision_source_changes(current, before, allowed)
    with pytest.raises(ValueError, match="unapproved frozen dependency"):
        runner.verify_revision_source_changes(
            {**current, "prompts/unapproved.txt": "new"}, before, allowed
        )
    with pytest.raises(ValueError, match="unapproved frozen dependency"):
        runner.verify_revision_source_changes(
            {**current, "shared.py": "modified"}, before, allowed
        )
    with pytest.raises(ValueError, match="unapproved frozen dependency"):
        runner.verify_revision_source_changes(
            {k: v for k, v in current.items() if k != "shared.py"}, before, allowed
        )


def test_free_format_overlay_uses_all_facts_and_preserves_original_scores(
    tmp_path, monkeypatch
):
    base = study.base(tmp_path)
    records = []
    hashes = {}
    for index in range(224):
        r = row(
            assignment_id=f"fresh-fact-{index}",
            source_assignment_id=f"original-fact-{index}",
            phase=runner.revision.PHASE,
            revision_kind="facts",
            kind="comprehension_probe",
        )
        records.append(r)
        text = json.dumps(study.expected_facts(r)) + "\nTrailing explanation."
        response = raw("" if index == 1 else text)
        if index == 0:
            response.choices[0].stop_reason = "max_tokens"
        result = {"status": "incomplete" if index == 0 else "completed"}
        path = runner.path_for(base, r)
        for relative, content in [
            ("generation/raw.json", response.model_dump(mode="json")),
            ("generation/result.json", result),
            ("probe.json", study.score_probe(response.completion, r)),
        ]:
            runner.original.freeze(path / relative, content)
            hashes[path / relative] = runner.digest((path / relative).read_bytes())
    qualified = runner.format_qualification(base, records)
    assert qualified["assigned"] == 224 and qualified["qualified_correct"] == 222
    assert not qualified["results"][0]["generation_complete"]
    assert not qualified["results"][1]["qualification"]["qualified_correct"]
    assert all(
        not item["original_probe"]["strict"]["all_correct"]
        for item in qualified["results"]
    )
    runner.original.freeze(base / runner.FORMAT_QUALIFICATION, qualified)
    runner.original.freeze(
        base / runner.FORMAT_AMENDMENT,
        {
            "audit_bindings": {
                runner.FORMAT_QUALIFICATION: runner.digest(
                    (base / runner.FORMAT_QUALIFICATION).read_bytes()
                )
            }
        },
    )
    overlays = [
        {
            "qualification_evidence_assignment_id": r["assignment_id"],
            "probe_correct": False,
            "probe_correct_strict": False,
        }
        for r in records
    ]
    monkeypatch.setattr(
        runner.revision, "qualified_pilot_rows", lambda old, fresh: overlays
    )
    readiness = runner.qualified_rows(base, records)
    assert sum(r["probe_correct"] for r in readiness) == 222
    assert all(
        r["probe_correct_before_format"] is False and r["probe_correct_strict"] is False
        for r in readiness
    )
    assert all(runner.digest(path.read_bytes()) == sha for path, sha in hashes.items())
    assert all(runner.measure(base, r)["probe_correct"] is False for r in records)
    with pytest.raises(ValueError, match="exactly all 224"):
        runner.format_qualification(base, records[:-1])


def format_approval_fixture(tmp_path):
    base = study.base(tmp_path)
    spec = manifest()
    definition = {"ledger_scope": runner.digest(runner.json_text(spec))}
    runner.original.freeze(base / runner.REVISION_MANIFEST, definition)
    proposal = base / runner.REVISION_DIRECTORY / "proposed-format-qualification.json"
    runner.original.freeze(proposal, {"proposal": "approved"})
    forecast_path = base / runner.REVISION_DIRECTORY / "validation-forecast.json"
    cost = {"projected_total_conservative_usd": 400.84893181748254, "bindings": {}}
    runner.original.freeze(forecast_path, cost)
    approval = {
        "answer": "Approve free repair and main (recommended)",
        "question": "Exact request",
        "original_manifest_sha256": runner.digest(runner.json_text(spec)),
        "revision_manifest_sha256": runner.digest(
            (base / runner.REVISION_MANIFEST).read_bytes()
        ),
        "proposal_sha256": runner.digest(proposal.read_bytes()),
        "forecast_sha256": runner.digest(forecast_path.read_bytes()),
        "approved_forecast_usd": cost["projected_total_conservative_usd"],
        "whole_study_hard_cap_usd": 400,
        "fresh_paid_validation_authorized": False,
        "main_assignments": 10080,
        "repetitions": 3,
    }
    approval_path = base / runner.FORMAT_DIRECTORY / "approval.json"
    runner.original.freeze(approval_path, approval)
    runner.original.freeze(
        base / runner.FORMAT_AMENDMENT,
        {
            "audit_bindings": {
                runner.FORMAT_DIRECTORY + "/approval.json": runner.digest(
                    approval_path.read_bytes()
                )
            },
            "approved_forecast_sha256": approval["forecast_sha256"],
        },
    )
    return base, spec, cost, approval_path


def test_only_exact_approved_forecast_can_bypass_margin_not_ledger(tmp_path):
    base, spec, cost, _ = format_approval_fixture(tmp_path)
    approved = runner.forecast_admission_exception(base, spec, cost)
    assert approved is not None
    assert (
        approved["hard_cap_usd"] == 400
        and approved["approved_forecast_usd"] == 400.84893181748254
    )
    with pytest.raises(ValueError, match="another estimate"):
        runner.forecast_admission_exception(
            base, spec, {**cost, "projected_total_conservative_usd": 401}
        )
    ledger = runner.ledger_for(base, spec)
    ledger.reserve({"existing": nanodollars(399)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"new": nanodollars(2)})
    assert ledger.snapshot()["cap"] == nanodollars(400)


def test_forecast_exception_rejects_missing_or_altered_approval(tmp_path):
    assert runner.forecast_admission_exception(tmp_path, manifest(), {}) is None
    base, spec, cost, approval_path = format_approval_fixture(tmp_path)
    changed = json.loads(approval_path.read_text())
    changed["whole_study_hard_cap_usd"] = 401
    approval_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="approval binding"):
        runner.forecast_admission_exception(base, spec, cost)


def test_format_amendment_does_not_admit_more_paid_validation(tmp_path, monkeypatch):
    base = study.base(tmp_path)
    runner.original.freeze(base / runner.FORMAT_AMENDMENT, {"frozen": "approval"})
    monkeypatch.setattr(runner, "load", lambda root, retained: (manifest(), []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    for phase in ("pilot", runner.CALIBRATION_PHASE, runner.revision.PHASE):
        with pytest.raises(ValueError, match="no new paid validation"):
            runner.collect(tmp_path, phase)
    assert not (base / "operations/launches").exists()


def test_generation_recovery_gate_requires_terminal_saved_responses(tmp_path):
    rows = [
        row(
            assignment_id=f"fresh-{i}",
            phase=runner.generation_recovery.PHASE,
            source_assignment_id=f"old-{i}",
            recovery_generation=True,
            generation_recovery=runner.generation_recovery.RECOVERY,
        )
        for i in range(20)
    ]
    assert not runner.generation_recovery_gate(tmp_path, rows)["passed"]
    for r in rows:
        path = runner.path_for(tmp_path, r)
        runner.original.freeze(path / "generation/started.json", {"at": 1})
        runner.original.freeze(
            path / "generation/raw.json", raw().model_dump(mode="json")
        )
        runner.original.freeze(
            path / "generation/result.json",
            {"status": "incomplete" if r is rows[0] else "completed"},
        )
    gate = runner.generation_recovery_gate(tmp_path, rows)
    assert (
        gate["passed"]
        and gate["saved"] == 20
        and gate["terminal_generation_failures"] == 1
    )
    target = tmp_path / "gate.json"
    runner.original.freeze(target, gate)
    runner.original.freeze(target, runner.generation_recovery_gate(tmp_path, rows))
    # Unknown accepted usage can leave raw text without a completed response status.
    last = runner.path_for(tmp_path, rows[-1]) / "generation/result.json"
    last.write_text(json.dumps({"status": "request_error"}))
    assert not runner.generation_recovery_gate(tmp_path, rows)["passed"]
    assert not runner.generation_recovery_gate(tmp_path, rows[:-1])["passed"]


@pytest.mark.parametrize("second", [False, True])
def test_authorized_fresh_identity_leaves_original_failure_and_never_replays(
    tmp_path, monkeypatch, second
):
    helper = runner.generation_recovery_v2 if second else runner.generation_recovery
    original_row = row(assignment_id="original-failed", phase="main")
    fresh = row(
        assignment_id="authorized-fresh",
        phase=helper.PHASE,
        source_assignment_id=original_row["assignment_id"],
        recovery_generation=True,
        generation_recovery=helper.RECOVERY,
        prior_recovery_assignment_id="prior-failed" if second else None,
    )
    p = pipeline(tmp_path, monkeypatch, [original_row, fresh])
    original_path = runner.path_for(p.base, original_row)
    runner.original.freeze(original_path / "generation/started.json", {"at": 1})
    runner.original.freeze(
        original_path / "generation/result.json",
        {"status": "request_error", "http_status": 412},
    )
    prior = {
        path: runner.digest(path.read_bytes()) for path in original_path.rglob("*.json")
    }
    client = AsyncMock()
    client.generate.return_value = raw()
    p.models[fresh["profile"]] = client
    # Already-spent original study costs do not consume a separate validation envelope.
    p.revision = {
        "validation_baseline_committed": 0,
        "validation_incremental_cap_usd": 15,
    }
    p.ledger.reserve({"past-study-usage": nanodollars(100)})
    p.ledger.settle({"past-study-usage": nanodollars(100)})

    async def work():
        assert (
            await p.request(
                original_row,
                "generation",
                [ChatMessageUser(content="same exact frozen prompt")],
            )
            is None
        )
        first = await p.request(
            fresh, "generation", [ChatMessageUser(content="same exact frozen prompt")]
        )
        assert first == await p.request(
            fresh, "generation", [ChatMessageUser(content="same exact frozen prompt")]
        )

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert all(runner.digest(path.read_bytes()) == sha for path, sha in prior.items())
    assert not runner.generation_path(p.base, original_row).exists()
    assert runner.measure(p.base, fresh)["recovery_generation"] is True
    assert runner.measure(p.base, fresh)["generation_recovery"] == helper.RECOVERY
    if second:
        assert (
            runner.measure(p.base, fresh)["prior_recovery_assignment_id"]
            == "prior-failed"
        )
    assert p.ledger.snapshot()["cap"] == nanodollars(400)


def test_generation_rerun_phase_requires_frozen_authorized_cohort(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "load", lambda root, retained: (manifest(), []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    with pytest.raises(ValueError, match="exact approved complete cohort"):
        runner.collect(tmp_path, runner.generation_recovery.PHASE)
    assert not (study.base(tmp_path) / "operations/launches").exists()


def test_second_recovery_uses_latest_gate_and_retires_first_phase(
    tmp_path, monkeypatch
):
    base = study.base(tmp_path)
    runner.original.freeze(base / runner.GENERATION_RECOVERY_MANIFEST, {"frozen": "v1"})
    runner.original.freeze(
        base / runner.GENERATION_RECOVERY_V2_MANIFEST, {"frozen": "v2"}
    )
    monkeypatch.setattr(runner, "load", lambda root, retained: (manifest(), []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    with pytest.raises(ValueError, match="older phases are retired"):
        runner.collect(tmp_path, runner.generation_recovery.PHASE)
    seen = []

    def incomplete_gate(base, records, *, phase):
        seen.append(phase)
        return {"passed": False}

    monkeypatch.setattr(runner, "generation_recovery_gate", incomplete_gate)
    with pytest.raises(ValueError, match="main remains held"):
        runner.collect(tmp_path, "main")
    assert seen == [runner.generation_recovery_v2.PHASE]
    assert not (base / "operations/launches").exists()


def test_second_recovery_gate_ignores_failed_first_attempts(tmp_path):
    first = [
        row(assignment_id=f"first-{i}", phase=runner.generation_recovery.PHASE)
        for i in range(20)
    ]
    second = [
        row(assignment_id=f"second-{i}", phase=runner.generation_recovery_v2.PHASE)
        for i in range(20)
    ]
    for r in first:
        runner.original.freeze(
            runner.path_for(tmp_path, r) / "generation/result.json",
            {"status": "request_error"},
        )
    for r in second:
        path = runner.path_for(tmp_path, r)
        runner.original.freeze(path / "generation/started.json", {"at": 1})
        runner.original.freeze(
            path / "generation/raw.json", raw().model_dump(mode="json")
        )
        runner.original.freeze(path / "generation/result.json", {"status": "completed"})
    assert not runner.generation_recovery_gate(tmp_path, first + second)["passed"]
    assert runner.generation_recovery_gate(
        tmp_path, first + second, phase=runner.generation_recovery_v2.PHASE
    )["passed"]


def second_recovery_evidence(tmp_path, monkeypatch):
    records = [
        row(assignment_id=f"first-{i}", phase=runner.generation_recovery.PHASE)
        for i in range(20)
    ]
    cohort = []
    for i, r in enumerate(records):
        result = {"status": "request_error", **({"http_status": 412} if i == 0 else {})}
        transport = (
            {"attempts": 1, "http_status": 412, "state": "not_retried"}
            if i == 0
            else {"attempts": 0, "state": "pending_admission"}
        )
        path = runner.path_for(tmp_path, r) / "generation"
        runner.original.freeze(path / "result.json", result)
        runner.original.freeze(path / "transport.json", transport)
        cohort.append(
            {
                "assignment_id": r["assignment_id"],
                "record": r,
                "result": result,
                "transport": transport,
                "raw_saved": False,
            }
        )
    first = tmp_path / runner.GENERATION_RECOVERY_DIRECTORY
    runner.original.freeze(first / "records.json", records)
    runner.original.freeze(first / "baseline-measured-rows.json", [])
    runner.original.freeze(
        tmp_path
        / runner.GENERATION_RECOVERY_V2_DIRECTORY
        / "failed-generation-cohort.json",
        cohort,
    )
    monkeypatch.setattr(
        runner.generation_recovery_v2, "build_records", lambda *args: [{"fresh": True}]
    )
    return records, cohort, [runner.measure(tmp_path, r) for r in records]


@pytest.mark.parametrize("change", ["accepted", "ambiguous", "raw", "result"])
def test_second_recovery_requires_actual_rejected_or_unsent_evidence(
    tmp_path, monkeypatch, change
):
    records, cohort, measured = second_recovery_evidence(tmp_path, monkeypatch)
    assert runner.validate_recovery_v2_cohort(tmp_path, [], measured) == [
        {"fresh": True}
    ]
    path = runner.path_for(tmp_path, records[0]) / "generation"
    if change == "raw":
        runner.original.freeze(path / "raw.json", raw().model_dump(mode="json"))
    elif change == "result":
        (path / "result.json").write_text(json.dumps({"status": "completed"}))
    else:
        transport = {
            "attempts": 1,
            "state": "not_retried",
            "http_status": 200 if change == "accepted" else 500,
        }
        (path / "transport.json").write_text(json.dumps(transport))
        # Even a self-consistent edited cohort cannot convert an accepted/ambiguous call.
        cohort[0]["transport"] = transport
        (
            tmp_path
            / runner.GENERATION_RECOVERY_V2_DIRECTORY
            / "failed-generation-cohort.json"
        ).write_text(json.dumps(cohort))
    with pytest.raises(ValueError, match="rejected or unsent"):
        runner.validate_recovery_v2_cohort(tmp_path, [], measured)


def billing_admission_evidence(tmp_path, *, mutation=None):
    base = study.base(tmp_path)
    records = [
        row(
            assignment_id=f"billing-{i}",
            profile="deepseek" if i < 2 else "qwen",
            phase=runner.generation_recovery_v2.PHASE,
        )
        for i in range(20)
    ]
    events = []
    for i, record in enumerate(records):
        path = runner.path_for(base, record) / "generation"
        profile = study.PROFILES[record["profile"]]
        transport = {
            "profile": record["profile"],
            "provider": profile["provider"],
            "model": profile["model"].split("/", 1)[1],
            "ticket": f"ticket{i}",
            "request_sha256": f"request{i}",
            "attempts": 1,
            "state": "not_retried" if i == 0 else "accepted",
            "http_status": 412 if i == 0 else 200,
        }
        runner.original.freeze(path / "started.json", {"at": i})
        runner.original.freeze(path / "transport.json", transport)
        runner.original.freeze(
            path / "result.json",
            {"status": "request_error", "http_status": 412}
            if i == 0
            else {"status": "completed"},
        )
        if i:
            runner.original.freeze(path / "raw.json", raw().model_dump(mode="json"))
        sent = 5 if i == 0 else 20 + i
        if i == 1 and mutation == "early_send":
            sent = 6  # Success arrives later, but was sent before the rejection.
        events.extend(
            [
                {**transport, "event": "send_started", "at": sent},
                {
                    **transport,
                    "event": "response_headers",
                    "at": 10 if i == 0 else 30 + i,
                },
            ]
        )
    if mutation == "wrong_model":
        for event in events:
            if event["ticket"] == "ticket1":
                event["model"] = "different-model"
    if mutation == "duplicate":
        events.append(dict(events[3]))
    gate = runner.generation_recovery_gate(
        base, records, phase=runner.generation_recovery_v2.PHASE
    )
    gate_path = (
        base / runner.GENERATION_RECOVERY_V2_DIRECTORY / "fresh-generation-gate.json"
    )
    runner.original.freeze(gate_path, gate)
    parent = base / runner.GENERATION_RECOVERY_V2_MANIFEST
    runner.original.freeze(parent, {"fixed": "second cohort"})
    directory = base / runner.RESTORED_BILLING_DIRECTORY
    runner.original.freeze(directory / "original-fresh-generation-gate.json", gate)
    directory.joinpath("transport-events.jsonl").write_text(
        "\n".join(json.dumps(x) for x in events) + "\n"
    )
    runner.original.freeze(
        directory / "audit.json",
        {
            "parent_manifest_sha256": runner.digest(parent.read_bytes()),
            "original_gate_sha256": runner.digest(gate_path.read_bytes()),
            "retained_artifacts": {
                str(p.relative_to(base)): runner.digest(p.read_bytes())
                for p in (base / "requests").rglob("*.json")
            },
        },
    )
    bindings = {
        str(p.relative_to(base)): runner.digest(p.read_bytes())
        for p in directory.iterdir()
        if p.is_file()
    }
    if mutation == "missing_binding":
        del bindings[str((directory / "transport-events.jsonl").relative_to(base))]
    amendment_path = directory / "amendment.json"
    runner.original.freeze(
        amendment_path,
        {
            "generation_recovery_v2_manifest_sha256": runner.digest(
                parent.read_bytes()
            ),
            "audit_bindings": bindings,
        },
    )
    runner.original.freeze(
        base / "operations/recovery-state.json",
        {
            "source_amendment": str(amendment_path.relative_to(base)),
            "source_amendment_sha256": runner.digest(amendment_path.read_bytes()),
            "retry_blocked": mutation == "blocked",
        },
    )
    return base, records, gate_path


def test_billing_admission_keeps_failed_gate_and_proves_later_same_model(tmp_path):
    base, records, gate_path = billing_admission_evidence(tmp_path)
    before = gate_path.read_bytes()
    proof = runner.restored_billing_admission(base, records)
    assert proof is not None
    assert proof["later_same_model_successes"] == ["billing-1"]
    assert proof["retained_rejection_assignment_id"] == "billing-0"
    assert proof["new_recovery_requests_authorized"] is False
    assert not runner.generation_recovery_gate(
        base, records, phase=runner.generation_recovery_v2.PHASE
    )["passed"]
    assert gate_path.read_bytes() == before


@pytest.mark.parametrize("mutation", ["early_send", "wrong_model", "duplicate"])
def test_billing_admission_rejects_absent_or_ambiguous_later_same_model_proof(
    tmp_path, mutation
):
    base, records, _ = billing_admission_evidence(tmp_path, mutation=mutation)
    with pytest.raises(ValueError, match="No saved same-model success"):
        runner.restored_billing_admission(base, records)


def test_billing_admission_requires_bound_evidence_and_preserves_retry_block(tmp_path):
    base, records, _ = billing_admission_evidence(tmp_path, mutation="missing_binding")
    assert runner.restored_billing_admission(base, records) is None
    # A retry block remains binding even when the proof would otherwise be valid.
    other, other_records, _ = billing_admission_evidence(
        tmp_path / "other", mutation="blocked"
    )
    with pytest.raises(ValueError, match="blocked pending new human"):
        runner.restored_billing_admission(other, other_records)


def test_billing_admission_rejects_modified_saved_response(tmp_path):
    base, records, _ = billing_admission_evidence(tmp_path)
    runner.generation_path(base, records[1]).write_text("{}")
    with pytest.raises(ValueError, match="Audited recovery request changed"):
        runner.restored_billing_admission(base, records)


def test_billing_proof_cannot_bypass_retry_or_scientific_gates(tmp_path, monkeypatch):
    base, records, _ = billing_admission_evidence(tmp_path)
    runner.original.freeze(base / runner.GENERATION_RECOVERY_MANIFEST, {"frozen": "v1"})
    monkeypatch.setattr(runner, "load", lambda root, retained: (manifest(), records))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    monkeypatch.setattr(
        runner,
        "main_readiness",
        lambda *args: {"main_ready_by_family": {"naming": False, "quotes": True}},
    )
    with pytest.raises(ValueError, match="scientific, budget, or naming-policy gate"):
        runner.collect(tmp_path, "main")
    state_path = base / "operations/recovery-state.json"
    state = json.loads(state_path.read_text())
    state["retry_blocked"] = True
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="blocked pending new human"):
        runner.collect(tmp_path, "main")
    assert not (base / "operations/launches").exists()
