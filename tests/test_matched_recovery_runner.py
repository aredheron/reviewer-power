"""Recovery is additive, same-limit and serialized with current paid collection."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import inspect_matched_recovery as runner
from grader_power import matched_recovery as study
from grader_power.spending import SpendingLimit, nanodollars

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records():
    return study.recovery_records(ROOT)


def pipeline(tmp_path, monkeypatch, records):
    monkeypatch.setattr(runner, "main_tasks", lambda root: [])
    monkeypatch.setattr(runner, "load_pilot_tasks", lambda root: [])
    m = {
        "profiles": study.PROFILES,
        "judge_profile": study.JUDGE_PROFILE,
        "records": records,
    }
    return runner.Pipeline(tmp_path, m, records, "main")


def output():
    r = ModelOutput.from_content("test", "safe")
    r.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    return r


def test_frozen_new_ids_preserve_original_prompts_and_limits(records):
    original = {
        r["assignment_id"]: r
        for r in json.loads(
            (
                ROOT / "runs/reviewer-observation/matched-review-v1/assignments.json"
            ).read_text()
        )
    }
    assert len(records) == 147
    assert sum(r["recovery_stage"] == "generation" for r in records) == 130
    for row in records:
        old = original[row["replaces"]]
        assert row["assignment_id"] != row["replaces"]
        for key in (
            "condition",
            "messages",
            "profile",
            "repetition_index",
            "source_task_id",
            "prompt_sha256",
        ):
            assert row[key] == old[key]


def test_constructor_never_opens_parent_ledger(tmp_path, monkeypatch, records):
    p = pipeline(tmp_path, monkeypatch, records[:1])
    assert p.base == study.base(tmp_path)
    assert not (tmp_path / "runs/reviewer-observation/matched-review-v1").exists()
    assert p.ledger.snapshot()["committed"] == 0


def test_saved_and_failed_fresh_attempts_are_not_replayed(
    tmp_path, monkeypatch, records
):
    row = next(r for r in records if r["recovery_stage"] == "generation")
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    client.generate.return_value = output()
    p.models[row["profile"]] = client

    async def work():
        await p.request(row, "generation", [ChatMessageUser(content="test")])
        await p.request(row, "generation", [ChatMessageUser(content="test")])

    asyncio.run(work())
    assert client.generate.await_count == 1


@pytest.mark.parametrize("mode", ["timeout", "missing_usage"])
def test_unknown_reservations_preserved(tmp_path, monkeypatch, records, mode):
    row = records[0]
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    p.models[row["profile"]] = client
    if mode == "timeout":
        client.generate.side_effect = TimeoutError("uncertain")
    else:
        client.generate.return_value = ModelOutput.from_content("test", "safe")
    asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert p.ledger.snapshot()["outstanding"] > 0
    assert (runner.path_for(p.base, row) / "generation/started.json").exists()


def test_judge_only_never_generates_or_rescores(tmp_path, monkeypatch, records):
    row = next(r for r in records if r["recovery_stage"] == "judge")
    p = pipeline(tmp_path, monkeypatch, [row])
    p.request = AsyncMock()
    asyncio.run(p.generation_stage(row))
    assert p.judging.get_nowait() == row
    p.request.assert_not_called()
    assert not (runner.path_for(p.base, row) / "generation/raw.json").exists()


def test_recovery_cap_is_independent_and_strict(tmp_path):
    m = {"run": study.RUN}
    ledger = runner.ledger_for(study.base(tmp_path), m)
    ledger.reserve({"unknown": nanodollars(15)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"next": 1})
    assert ledger.snapshot()["outstanding"] == nanodollars(15)


def test_combined_forecast_blocks_new_dispatch(tmp_path, monkeypatch, records):
    p = pipeline(tmp_path, monkeypatch, records[:1])
    p.cost_forecast = {"median_total_usd": 1, "combined_median_total_usd": 500.01}
    with pytest.raises(RuntimeError):
        p.check()
    assert p.ledger.snapshot()["committed"] == 0


def test_scientific_originals_are_not_mutated_by_designation(records):
    before = copy.deepcopy(records)
    for r in records:
        assert r["repetition_index"] in range(8)
    assert records == before


def test_terminal_missing_response_is_not_an_unstarted_judgment(tmp_path, records):
    row = next(r for r in records if r["recovery_stage"] == "generation")
    path = runner.path_for(tmp_path, row)
    assert runner.recovery_pending(tmp_path, row)
    (path / "generation").mkdir(parents=True)
    (path / "generation/result.json").write_text('{"status":"error"}')
    assert not runner.recovery_pending(tmp_path, row)
    (path / "generation/raw.json").write_text("{}")
    assert runner.recovery_pending(tmp_path, row)


def test_current_collector_blocks_recovery_before_paid_initialization(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "load", lambda root: ({}, []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    monkeypatch.setattr(runner, "process_group_members", lambda pid: {pid})
    p = tmp_path / "runs/extension-pilot/coordination/owner.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"state": "released", "pgid": 7}))
    with pytest.raises(ValueError, match="every paid child must exit"):
        runner.collect(tmp_path)
    assert not (study.base(tmp_path) / "spending.sqlite3").exists()


def test_current_gpt_ds_completion_does_not_authorize_skipping_glm_quotes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(runner, "load", lambda root: ({}, []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    monkeypatch.setattr(runner, "process_group_members", lambda pid: set())
    p = tmp_path / "runs/extension-pilot/coordination/owner.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"state": "released", "pgid": 7}))
    b = tmp_path / "runs/reviewer-observation/matched-syntax-v1"
    (b / "operations").mkdir(parents=True)
    (b / "operations/status.json").write_text(
        '{"state":"qualified_main_complete_others_held"}'
    )
    (b / "assignments.json").write_text(
        json.dumps(
            [
                {
                    "phase": "main",
                    "study": "quotes",
                    "profile": "glm",
                    "assignment_id": "not-started",
                }
            ]
        )
    )
    with pytest.raises(ValueError, match="including GLM quotes"):
        runner.collect(tmp_path)
    assert not (study.base(tmp_path) / "spending.sqlite3").exists()


@pytest.mark.parametrize("failure", ["failure", "error", "skipped"])
def test_verification_rejects_nonpassing_reports(tmp_path, failure):
    p = tmp_path / "tests.xml"
    p.write_text(f"<testsuite><testcase><{failure}/></testcase></testsuite>")
    assert not runner.passed_report(p)


def test_budget_forecast_does_not_add_already_reserved_stages_again(
    tmp_path, monkeypatch, records
):
    row = records[0]
    m = {"records": [row], "forecast_costs": {row["profile"]: [0.1, 0.2]}}
    monkeypatch.setattr(runner, "other_committed", lambda root: 180)
    ledger = runner.ledger_for(study.base(tmp_path), m)
    ledger.reserve({row["assignment_id"] + "/generation": nanodollars(0.3)})
    assert runner.forecast(tmp_path, m)["median_total_usd"] == pytest.approx(0.5)
    ledger.settle({row["assignment_id"] + "/generation": nanodollars(0.1)})
    ledger.reserve({row["assignment_id"] + "/judge": nanodollars(0.4)})
    assert runner.forecast(tmp_path, m)["median_total_usd"] == pytest.approx(0.5)
