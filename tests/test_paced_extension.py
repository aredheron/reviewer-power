"""Exercise the amended worker with real ledgers and mocked provider boundaries."""

import asyncio
import importlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx2 as httpx
import pytest
from inspect_ai import eval
from inspect_ai.model import ChatMessageUser, GenerateConfig, get_model

from grader_power.pacing import PacingDeferred, SharedPacer, WaitClock, paced_http
from grader_power.spending import SpendingLedger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
paced = importlib.import_module("evals.inspect_paced_extension")
fixture = importlib.import_module("test_inspect_agent")


def test_recovery_preserves_original_stage_timing(tmp_path: Path) -> None:
    clock = WaitClock()
    clock.waited = 598.0
    paced.save_timing(tmp_path, clock)
    original = (tmp_path / "timing.json").read_bytes()
    paced.save_timing(tmp_path, WaitClock())
    assert (tmp_path / "timing.json").read_bytes() == original
    revisits = list(tmp_path.glob("timing-revisit-*.json"))
    assert len(revisits) == 1
    assert json.loads(revisits[0].read_text())["transport_wait_seconds"] == 0


def test_ramp_requires_progress_and_reduces_on_throttles() -> None:
    state = {
        "workers": 4,
        "last_change": 0,
        "completed_at_change": 0,
        "throttles": 0,
        "errors": 0,
    }
    assert not paced.ramp(state, now=900, throttles=0, completed=0, errors=0)
    assert paced.ramp(state, now=900, throttles=0, completed=1, errors=0)
    assert state["workers"] == 5
    assert not paced.ramp(state, now=1000, throttles=0, completed=2, errors=0)
    assert paced.ramp(state, now=1800, throttles=0, completed=3, errors=0)
    assert state["workers"] == 6
    assert paced.ramp(state, now=1801, throttles=1, completed=3, errors=0)
    assert state["workers"] == 3
    assert not paced.ramp(state, now=1802, throttles=1, completed=3, errors=0)


def test_judgment_failure_does_not_reduce_workers_or_restart_ramp_clock() -> None:
    """An invalid judge answer is an outcome error, not provider throttling."""
    state = {
        "workers": 10,
        "last_change": 0,
        "completed_at_change": 0,
        "throttles": 0,
        "errors": 0,
    }
    assert not paced.ramp(state, now=600, throttles=0, completed=1, errors=1)
    assert state["workers"] == 10
    assert state["last_change"] == 0
    assert state["errors"] == 1
    assert paced.ramp(state, now=900, throttles=0, completed=2, errors=1)
    assert state["workers"] == 11
    assert paced.ramp(state, now=1800, throttles=0, completed=3, errors=2)
    assert state["workers"] == 12
    assert not paced.ramp(state, now=2700, throttles=0, completed=4, errors=2)
    assert state["workers"] == 12
    assert paced.ramp(state, now=2701, throttles=1, completed=4, errors=2)
    assert state["workers"] == 6


def test_throttle_backoff_has_a_one_worker_floor() -> None:
    state = {
        "workers": 1,
        "last_change": 0,
        "completed_at_change": 0,
        "throttles": 0,
        "errors": 0,
    }
    assert paced.ramp(state, now=1, throttles=1, completed=0, errors=1)
    assert state["workers"] == 1


@pytest.mark.parametrize("provider", ["fireworks", "anthropic"])
def test_real_inspect_sdk_retries_only_throttle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    """The pinned SDKs actually use httpx2, not the separate httpx package."""
    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"type": "rate_limit_error", "message": "rate limit"}},
            )
        if provider == "fireworks":
            payload = {
                "id": "synthetic",
                "object": "chat.completion",
                "model": "pacing-test",
                "created": 0,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            }
        else:
            payload = {
                "id": "synthetic",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda *a: transport)

    async def run() -> None:
        with (
            paced.runner.pilot.zero_retry_clients(),
            paced_http(
                SharedPacer(tmp_path), tmp_path / "attempt", WaitClock(), lambda: None
            ),
        ):
            model = get_model(
                "fireworks/pacing-test"
                if provider == "fireworks"
                else "anthropic/claude-sonnet-5-5",
                api_key="synthetic-not-a-credential",
                config=GenerateConfig(max_retries=0, max_tokens=256),
                memoize=False,
            )
            paced.runner.pilot.resolved_retry_policy(model)
            result = await model.generate(
                [ChatMessageUser(content="synthetic")], tools=[], tool_choice="none"
            )
            assert result.completion == "done"

    asyncio.run(run())
    assert len(bodies) == 2 and bodies[0] == bodies[1]


def test_mode_b_retry_does_not_replay_tools_or_consume_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run native ReAct across one rejected call after an already executed tool."""
    agent = paced.inspect_agent
    monkeypatch.setattr(agent, "DockerWorkspace", fixture.FakeWorkspace)
    monkeypatch.setattr("inspect_ai._eval.context.init_dotenv", lambda: None)
    monkeypatch.setattr("inspect_ai._eval.task.log.git_context", lambda: None)
    monkeypatch.setattr(
        agent,
        "score_response",
        lambda raw, region, **kwargs: fixture.score_response(raw, region),
    )
    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        number = len(bodies)
        if number == 2:
            return httpx.Response(429, headers={"retry-after": "0"}, text="rate limit")
        if number in (1, 3):
            tool = "list_files" if number == 1 else "write_file"
            args = (
                {} if number == 1 else {"path": "src/task.py", "content": fixture.CODE}
            )
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": str(number),
                        "type": "function",
                        "function": {"name": tool, "arguments": json.dumps(args)},
                    }
                ],
            }
            stop = "tool_calls"
        else:
            message, stop = {"role": "assistant", "content": "done"}, "stop"
        return httpx.Response(
            200,
            json={
                "id": str(number),
                "object": "chat.completion",
                "model": "pacing-test",
                "created": 0,
                "choices": [{"index": 0, "message": message, "finish_reason": stop}],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "total_tokens": 11,
                },
            },
        )

    transport = httpx.MockTransport(handle)
    monkeypatch.setattr(httpx.AsyncClient, "_transport_for_url", lambda *a: transport)
    clock = WaitClock()
    monkeypatch.setattr(
        agent,
        "asyncio",
        SimpleNamespace(**{**vars(asyncio), "wait_for": clock.wait_for}),
    )
    monkeypatch.setattr(agent, "time", SimpleNamespace(monotonic=clock.monotonic))
    item = fixture.assigned()
    config = GenerateConfig(max_tokens=256, max_retries=0)
    with (
        paced.runner.pilot.zero_retry_clients(),
        paced_http(
            SharedPacer(tmp_path / "pacing"),
            tmp_path / "transport-test",
            clock,
            lambda: None,
        ),
    ):
        model = get_model(
            "fireworks/pacing-test",
            api_key="synthetic-not-a-credential",
            config=config,
            memoize=False,
        )
        log = eval(
            agent.build_task([item], tmp_path, fixture.IMAGE, config, lambda: None),
            model=model,
            max_samples=1,
            log_dir=str(tmp_path / "inspect"),
            display="none",
            retry_on_error=0,
            fail_on_error=False,
            notification=False,
        )[0]
    assert log.status == "success"
    path = tmp_path / "attempts" / item.record["assignment_id"]
    episode = json.loads((path / "episode.json").read_text())
    assert episode["status"] == "completed"
    assert episode["model_dispatches"] == 3
    assert episode["tool_calls"] == 2
    assert (path / "final.py").read_bytes() == fixture.CODE.encode()
    assert len(bodies) == 4 and bodies[1] == bodies[2]
    assert clock.waited > 0


def test_worker_preserves_global_scope_and_never_reissues_started(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = paced.runner
    row = fixture.assigned(mode="A", source_task_id="HumanEval/999")
    name = "synthetic-A"
    base = runner.directory(tmp_path)
    cohort = base / "blocks" / name
    frozen = {"oracle_image": "synthetic"}
    runner.original.freeze(cohort / "manifest.json", frozen)
    bound = sum(runner.checks.bounds(row).values())
    item = {
        "name": name,
        "mode": "A",
        "maximum_nanodollars": bound,
        "manifest_sha256": runner.digest(runner.json_text(frozen)),
    }
    manifest = {
        "blocks": [item, {"name": "unrelated"}],
        "code_sha256": {},
        "plan": {"api_cap_usd": 250},
    }
    monkeypatch.setattr(paced, "verify", lambda *a: ({}, manifest, {name: (row,)}))
    monkeypatch.setattr(paced, "child_permission", lambda *a: nullcontext())
    monkeypatch.setattr(paced, "guard", lambda *a: None)
    monkeypatch.setattr(runner, "sources", lambda _: {})
    monkeypatch.setattr(runner, "measure", lambda *a: [])
    monkeypatch.setattr(runner.extension, "access_profiles", lambda _: {})
    monkeypatch.setattr(runner.extension, "generation", lambda *a: None)
    monkeypatch.setattr(runner.extension, "enforce_judge_limit", lambda *a: None)
    calls = []
    aid = row.record["assignment_id"]
    attempt = cohort / runner.PROFILE / "attempts" / aid

    def collect(*args: Any, **kwargs: Any) -> list[Any]:
        calls.append("generation")
        runner.original.freeze(attempt / "started.json", {})
        runner.original.freeze(
            attempt / "raw.json", {"usage": {"input_tokens": 10, "output_tokens": 1}}
        )
        return []

    async def judge(*args: Any) -> None:
        path = cohort / runner.PROFILE / "judge" / aid
        if (path / "started.json").exists():
            return
        calls.append("judge")
        runner.original.freeze(path / "started.json", {})
        runner.original.freeze(
            path / "raw.json", {"usage": {"input_tokens": 10, "output_tokens": 1}}
        )
        runner.original.freeze(path / "result.json", {"status": "classified"})

    monkeypatch.setattr(runner.pilot, "collect_profile", collect)
    monkeypatch.setattr(runner.original, "collect_judgments", judge)
    # Create the real global scope before entering the worker's one-block view.
    ledger = SpendingLedger(
        base / "spending.json",
        cap=runner.nanodollars(250),
        scope=runner.digest(runner.json_text(manifest)),
    )
    ledger.reserve({name: bound})
    paced.worker(tmp_path, tmp_path / "unused", name, 123)
    before = (attempt / "raw.json").read_bytes()
    paced.worker(tmp_path, tmp_path / "unused", name, 123)
    assert calls == ["generation", "judge"]
    assert (attempt / "raw.json").read_bytes() == before
    assert ledger.snapshot()["entries"][name]["settled"] is not None
    assert (cohort / "complete.json").exists()


def test_deferred_work_blocks_restart(tmp_path: Path) -> None:
    directory = tmp_path / "operations" / paced.VERSION
    p = directory / "deferred/one.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"state":"pending_rate_limit"}')
    history = tmp_path / "operations/request-events.jsonl"
    history.write_bytes(b"")
    amendment = {
        "recovery": {
            "reviewed_prefix_bytes": 0,
            "reviewed_prefix_sha256": paced.runner.digest(b""),
        }
    }
    with pytest.raises(PacingDeferred):
        paced.guard(tmp_path, amendment)
