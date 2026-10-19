"""Budget, immutable request identity and operational guard regressions for matched review."""

import asyncio
import copy
import json
import sys
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_matched_syntax as runner
from grader_power import matched_syntax as study
from grader_power.spending import SpendingLimit, nanodollars


@pytest.fixture(scope="module")
def records():
    return study.schedule(ROOT)[0]


def pipeline(tmp_path, monkeypatch, records):
    monkeypatch.setattr(runner, "load_bundles", lambda root: ([], {}))
    manifest = {
        "judge_profile": study.JUDGE_PROFILE,
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "cost_floor_per_response": {
            family: {profile: [0.01, 0.02] for profile in study.PROFILES}
            for family in study.STYLES
        },
    }
    return runner.Pipeline(tmp_path, manifest, records, "pilot")


def output(
    text: str = "safe", *, stop: Literal["stop", "max_tokens"] = "stop"
) -> ModelOutput:
    value = ModelOutput.from_content("test", text)
    value.choices[0].stop_reason = stop
    value.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    return value


def test_provider_cache_accounting_counts_each_input_once():
    anthropic = {
        "input_tokens": 100,
        "input_tokens_cache_read": 200,
        "input_tokens_cache_write": 300,
        "output_tokens": 400,
    }
    assert runner.request_cost(anthropic, study.JUDGE_PROFILE) == nanodollars(0.00542)
    assert runner.request_cost(
        anthropic, study.JUDGE_PROFILE, no_cache=True
    ) == nanodollars(0.0058)
    fireworks = {
        "input_tokens": 1000,
        "input_tokens_cache_read": 600,
        "input_tokens_cache_write": None,
        "output_tokens": 100,
    }
    assert runner.request_cost(fireworks, study.PROFILES["glm"]) == nanodollars(
        0.001996
    )
    assert runner.request_cost(
        fireworks, study.PROFILES["glm"], no_cache=True
    ) == nanodollars(0.00268)


@pytest.mark.parametrize(
    "usage",
    [
        {"input_tokens": 1, "output_tokens": None},
        {"input_tokens": True, "output_tokens": 1},
        {"input_tokens": -1, "output_tokens": 1},
        {"input_tokens": 1, "input_tokens_cache_write": 2, "output_tokens": 1},
    ],
)
def test_incomplete_or_inconsistent_usage_is_not_zero_cost(usage):
    with pytest.raises(ValueError):
        runner.request_cost(usage, study.PROFILES["glm"])


def test_saved_request_never_replayed(tmp_path, monkeypatch, records):
    row = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    client.generate.return_value = output()
    p.models[row["profile"]] = client

    async def work():
        first = await p.request(row, "generation", [ChatMessageUser(content="test")])
        assert first == await p.request(
            row, "generation", [ChatMessageUser(content="test")]
        )

    asyncio.run(work())
    assert client.generate.await_count == 1
    balance = p.ledger.snapshot()
    assert balance["spent"] > 0 and balance["outstanding"] == 0


@pytest.mark.parametrize("failure", ["timeout", "missing_usage", "credential"])
def test_failed_started_request_is_not_replayed_and_retains_reservation(
    tmp_path, monkeypatch, records, failure
):
    row = records[0]
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    p.models[row["profile"]] = client
    p.keys.add("secret-test-marker")
    if failure == "timeout":
        client.generate.side_effect = TimeoutError("uncertain acceptance")
    else:
        client.generate.return_value = ModelOutput.from_content(
            "test", "secret-test-marker" if failure == "credential" else "safe"
        )
    asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["outstanding"] > 0
    if not p.stop.is_set():
        assert (
            asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
            is None
        )
        assert client.generate.await_count == 1
    if failure == "credential":
        assert not (runner.path_for(p.base, row) / "generation/raw.json").exists()


def test_saved_capped_outputs_do_not_trigger_operational_guard(
    tmp_path, monkeypatch, records
):
    selected = records[:5]
    p = pipeline(tmp_path, monkeypatch, selected)
    for profile in study.PROFILES:
        client = AsyncMock()
        client.generate.return_value = output(stop="max_tokens")
        p.models[profile] = client

    async def work():
        for row in selected:
            assert (
                await p.request(row, "generation", [ChatMessageUser(content="test")])
                is not None
            )

    asyncio.run(work())
    assert not p.stop.is_set()
    assert len(p.history) == 5 and all(not r["failed"] for r in p.history)
    assert all(r["generation_status"] == "incomplete" for r in p.measurements.values())


def test_five_failed_stages_still_stop_admission(tmp_path, monkeypatch, records):
    p = pipeline(tmp_path, monkeypatch, records[:5])
    for row in records[:5]:
        p.event(row, "judge", True, "ValidationError")
    assert p.stop.is_set()
    with pytest.raises(RuntimeError):
        p.check()


def test_median_threshold_blocks_dispatch_before_any_reservation(
    tmp_path, monkeypatch, records
):
    row = records[0]
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    p.models[row["profile"]] = client
    p.cost_forecast = {"median_total_usd": study.MEDIAN_LIMIT + 0.01}
    with pytest.raises(RuntimeError, match="forecast"):
        asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert client.generate.await_count == 0
    assert p.ledger.snapshot()["committed"] == 0


def test_exact_median_limit_allows_request(tmp_path, monkeypatch, records):
    row = records[0]
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    client.generate.return_value = output()
    p.models[row["profile"]] = client
    p.cost_forecast = {"median_total_usd": study.MEDIAN_LIMIT}
    asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert client.generate.await_count == 1


def test_hard_cap_preserves_unknown_reserves_without_prior_parent_spend(tmp_path):
    manifest = {"run": study.RUN}
    ledger = runner.ledger_for(study.base(tmp_path), manifest)
    ledger.reserve({"unknown/request": nanodollars(study.CAP)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"next/request": 1})
    balance = runner.read_ledger(study.base(tmp_path), manifest)
    assert balance["outstanding"] == nanodollars(study.CAP) and balance["spent"] == 0
    with pytest.raises(ValueError, match="scope"):
        runner.read_ledger(study.base(tmp_path), {"other": "scope"})


def test_forecast_includes_all_cohorts_but_only_coding_judgments(tmp_path, records):
    manifest = {
        "cost_floor_per_response": {
            f: {p: [0.001, 0.002] for p in study.PROFILES} for f in study.STYLES
        }
    }
    forecast = runner.median_forecast(tmp_path, manifest, records, draws=100)
    expected = 21600 * 0.001 + 21168 * 0.002
    assert forecast["median_total_usd"] == pytest.approx(expected)
    ledger = runner.ledger_for(study.base(tmp_path), manifest)
    ledger.reserve({"unknown/noncohort": nanodollars(5)})
    updated = runner.median_forecast(tmp_path, manifest, records, draws=100)
    assert updated["median_total_usd"] == pytest.approx(expected + 5)
    assert not updated["budget_notification_required"]


def test_completed_missing_generations_do_not_add_future_judgment_cost(
    tmp_path, records
):
    row = copy.deepcopy(records[0])
    row["kind"] = "core"
    manifest = {
        "cost_floor_per_response": {
            f: {p: [0.01, 0.02] for p in study.PROFILES} for f in study.STYLES
        }
    }
    measurements = [
        runner.measure(study.base(tmp_path), row)
        | {"started": True, "attempt_complete": True, "raw_saved": False}
    ]
    forecast = runner.median_forecast(
        tmp_path, manifest, [row], measurements, draws=100
    )
    assert forecast["median_total_usd"] == 0


def test_only_passed_models_queue_and_short_tail_drains(tmp_path, monkeypatch, records):
    selected = [r for r in records if r["phase"] == "main"][:12]
    p = pipeline(tmp_path, monkeypatch, selected)
    p.phase = "main"
    p.eligible_models = [("quotes", "glm")]
    seen = []

    async def generation(row):
        seen.append(row["assignment_id"])

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert set(seen) == {
        r["assignment_id"]
        for r in selected
        if (r["study"], r["profile"]) == ("quotes", "glm")
    }


def test_factual_truncation_does_not_pass_even_when_json_is_complete(
    tmp_path, monkeypatch, records
):
    row = next(r for r in records if r["kind"] == "comprehension_probe")
    p = pipeline(tmp_path, monkeypatch, [row])
    client = AsyncMock()
    client.generate.return_value = output(
        json.dumps(study.expected_facts(row)), stop="max_tokens"
    )
    p.models[row["profile"]] = client
    asyncio.run(p.generation_stage(row))
    facts = json.loads((runner.path_for(p.base, row) / "probe.json").read_text())
    assert facts["incomplete_response"]
    assert not facts["fence_only"]["all_correct"] and not facts["strict"]["all_correct"]


def test_matching_global_lock_accepts_released_prior_owner(tmp_path, monkeypatch):
    from grader_power import matched_syntax_lock as lock

    coordination = tmp_path / "runs/extension-pilot/coordination"
    coordination.mkdir(parents=True)
    (coordination / "owner.json").write_text(
        json.dumps(
            {"owner": "expanded-preferences-v1", "state": "released", "pgid": 999}
        )
    )
    monkeypatch.setattr(lock.os, "getpid", lambda: 1234)
    monkeypatch.setattr(lock.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(
        lock, "process_group_members", lambda group: {1234} if group == 1234 else set()
    )
    with lock.collection_lock(tmp_path, owner=study.RUN, run_id=study.RUN):
        active = json.loads((coordination / "owner.json").read_text())
        assert active["state"] == "active" and active["owner"] == study.RUN
    assert json.loads((coordination / "owner.json").read_text())["state"] == "released"


def test_active_orphan_group_blocks_new_collection(tmp_path, monkeypatch):
    from grader_power import matched_syntax_lock as lock

    coordination = tmp_path / "runs/extension-pilot/coordination"
    coordination.mkdir(parents=True)
    old = {"owner": "expanded-preferences-v1", "state": "active", "pgid": 999}
    (coordination / "owner.json").write_text(json.dumps(old))
    monkeypatch.setattr(lock.os, "getpid", lambda: 1234)
    monkeypatch.setattr(lock.os, "getpgrp", lambda: 1234)
    monkeypatch.setattr(lock, "process_group_members", lambda group: {1000})
    with (
        pytest.raises(RuntimeError, match="surviving"),
        lock.collection_lock(tmp_path, owner=study.RUN, run_id=study.RUN),
    ):
        pytest.fail("An orphan paid descendant must block the new run")
    assert json.loads((coordination / "owner.json").read_text()) == old


def test_human_stop_blocks_unstarted_requests(tmp_path, monkeypatch, records):
    row = records[0]
    p = pipeline(tmp_path, monkeypatch, [row])
    p.operations.mkdir(parents=True)
    (p.operations / "STOP").write_text("human stop")
    with pytest.raises(RuntimeError, match="Human STOP"):
        asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert not (runner.path_for(p.base, row) / "generation/started.json").exists()
    assert p.ledger.snapshot()["committed"] == 0


@pytest.mark.parametrize("settled", [None, 0.003])
def test_stale_inflight_measurements_cannot_doublecount_reserved_requests(
    tmp_path, records, settled
):
    row = copy.deepcopy(records[0])
    row["kind"] = "core"
    manifest = {
        "cost_floor_per_response": {
            f: {p: [0.01, 0.02] for p in study.PROFILES} for f in study.STYLES
        }
    }
    ledger = runner.ledger_for(study.base(tmp_path), manifest)
    identity = row["assignment_id"] + "/generation"
    ledger.reserve({identity: nanodollars(0.05)})
    if settled is not None:
        ledger.settle({identity: nanodollars(settled)})
    stale = [runner.measure(study.base(tmp_path), row)]
    assert not stale[0]["started"]
    forecast = runner.median_forecast(tmp_path, manifest, [row], stale, draws=100)
    assert forecast["median_total_usd"] == pytest.approx(
        (0.05 if settled is None else settled) + 0.02
    )
    ledger.reserve({row["assignment_id"] + "/judge": nanodollars(0.08)})
    updated = runner.median_forecast(tmp_path, manifest, [row], stale, draws=100)
    assert updated["median_total_usd"] == pytest.approx(
        (0.05 if settled is None else settled) + 0.08
    )


def test_both_sandbox_namespaces_are_inspected(monkeypatch):
    from types import SimpleNamespace

    seen = []

    def inspect_container(command, **kwargs):
        seen.append(command[-1])
        assert kwargs["check"] is True
        return SimpleNamespace(
            stdout="agent-container\n" if ".agent=" in command[-1] else ""
        )

    monkeypatch.setattr(runner.subprocess, "run", inspect_container)
    assert runner.sandbox_containers() == {
        "oracle_containers": [],
        "agent_containers": ["agent-container"],
    }
    assert seen == ["label=grader-power.oracle=true", "label=grader-power.agent=true"]


def test_installed_openai_adapter_cache_normalization_matches_billing():
    from inspect_ai.model._openai import model_output_from_openai
    from openai.types.chat import ChatCompletion

    response = ChatCompletion.model_validate(
        {
            "id": "normalization-fixture",
            "created": 0,
            "model": "glm",
            "object": "chat.completion",
            "choices": [],
            "usage": {
                "prompt_tokens": 522,
                "completion_tokens": 197,
                "total_tokens": 719,
                "prompt_tokens_details": {
                    "cached_tokens": 521,
                    "cache_write_tokens": 0,
                },
            },
        }
    )
    normalized = model_output_from_openai(response, [])
    assert normalized.usage is not None
    assert normalized.usage.input_tokens == 1
    assert normalized.usage.input_tokens_cache_read == 521
    assert runner.request_cost(
        normalized.usage.model_dump(), study.PROFILES["glm"]
    ) == nanodollars(0.00100366)
