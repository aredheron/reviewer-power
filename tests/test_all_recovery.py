"""Cross-study recovery preserves frozen inputs, abstention, budgets and no-replay."""

import asyncio
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import httpx2 as httpx
import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_all_recovery as runner


@pytest.fixture(scope="module")
def catalog():
    return runner.catalog(ROOT)


def test_exact_scope_and_unchanged_generation_inputs(catalog):
    records = catalog["records"]
    assert Counter(r["recovery_kind"] for r in records) == {
        "generation": 204,
        "judge": 96,
    }
    assert Counter(
        r["mode"] for r in records if r["recovery_kind"] == "generation"
    ) == {"A": 163, "B": 41}
    assert len({r["assignment_id"] for r in records}) == 300
    assert all(r["assignment_id"] != r["replaces"] for r in records)
    assert all(r["replaces"] != runner.STUB for r in records)
    for r in records:
        if r["recovery_kind"] == "generation":
            old = r["original_record"]
            for field in (
                "messages",
                "condition",
                "source_task_id",
                "repetition_index",
            ):
                assert r[field] == old[field]
            assert catalog["profiles"][r["profile"]]["generation"]["max_tokens"] == 4096
            if r["mode"] == "B":
                for field in (
                    "visible_files",
                    "episode_limits",
                    "coding_request",
                    "target_path",
                ):
                    assert r[field] == old[field]
                assert r["episode_limits"]["max_model_turns"] == 16
        else:
            assert r["source_raw_sha256"] == runner.digest(
                runner.json_text(runner.read(ROOT / r["source_raw"]))
            )
    runner.prior.verify_hashes(ROOT, catalog["retained_evidence"])


def test_binary_and_three_style_parsers_are_distinct(catalog):
    for r in catalog["records"]:
        if r["origin_study"] in (
            "simple_replication",
            "observation",
            "strong",
            "extension",
        ):
            assert r["design"] == "mapping"
        elif r["origin_study"] == "syntax":
            assert r["design"] in ("naming", "quotes")
    assert {
        r["design"]
        for r in catalog["records"]
        if r["origin_study"] == "compact_replication"
    } == {"mapping", "reversal"}


def test_agent_reservation_covers_original_full_limits(catalog):
    r = next(
        r
        for r in catalog["records"]
        if r["recovery_kind"] == "generation" and r["mode"] == "B"
    )
    assert runner.agent_bound(r) == runner.nanodollars(
        ((98304 + 2048 * 16) * 0.3 + 4096 * 16 * 0.6) / 1e6
    )


def packet():
    return {
        "entry_point": "f",
        "transformation": "double items",
        "input_contract": "integer list",
        "code": "def f(items):\n    return [x * 2 for x in items]\n",
    }


def judgment():
    return {
        "status": "classified",
        "target_forms": ["list_comprehension"],
        "evidence": [
            {
                "start_line": 2,
                "end_line": 2,
                "quote": "    return [x * 2 for x in items]",
            }
        ],
        "flags": [],
        "rationale": "The comprehension builds the returned list.",
    }


def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "main_tasks", lambda _: [])
    monkeypatch.setattr(runner, "load_bundles", lambda _: ([], {}))
    r = {
        "assignment_id": "fresh",
        "replaces": "old",
        "original_run": "runs/old",
        "origin_study": "observation",
        "phase": "recovery",
        "profile": "gpt-oss-120b",
        "mode": "A",
        "recovery_kind": "judge",
        "design": "mapping",
        "study": "binary",
        "source_raw_sha256": "source",
        "packet": packet(),
    }
    spec = {
        "input_rate": 0.15,
        "output_rate": 0.6,
        "model": "fireworks/test",
        "provider": "fireworks",
    }
    m = {
        "records": [r],
        "profiles": {
            "gpt-oss-120b": spec,
            "sonnet": {
                **spec,
                "input_rate": 4,
                "output_rate": 20,
                "model": "anthropic/test",
                "provider": "anthropic",
            },
        },
        "judges": {"binary": {"system_prompt": "original semantic rubric"}},
    }
    return runner.Pipeline(tmp_path, m), r


@pytest.mark.parametrize("missing", [False, True])
def test_no_replay_and_uncertain_cost_retained(tmp_path, monkeypatch, missing):
    p, r = pipeline(tmp_path, monkeypatch)
    r["recovery_kind"] = "generation"
    client = AsyncMock()
    if missing:
        client.generate.side_effect = TimeoutError("unknown acceptance")
    else:
        output = ModelOutput.from_content("test", "response")
        output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
        client.generate.return_value = output
    p.models[r["profile"]] = client

    async def work():
        a = await p.request(r, "generation", [ChatMessageUser(content="unchanged")])
        b = await p.request(r, "generation", [ChatMessageUser(content="unchanged")])
        assert a == b

    asyncio.run(work())
    assert client.generate.await_count == 1
    balance = p.ledger.snapshot()
    assert balance["cap"] == runner.nanodollars(30)
    assert (balance["outstanding"] > 0) == missing
    assert (balance["spent"] > 0) != missing


@pytest.mark.parametrize(
    "kind", ["valid", "abstain", "malformed", "extra", "bad_quote"]
)
def test_one_blinded_judgment_and_strict_failure_retention(tmp_path, monkeypatch, kind):
    p, r = pipeline(tmp_path, monkeypatch)
    j = judgment()
    if kind == "abstain":
        j.update(
            status="abstain",
            target_forms=[],
            evidence=[],
            rationale="Ambiguous construction.",
        )
    if kind == "extra":
        j["label_guess"] = "loops"
    if kind == "bad_quote":
        j["evidence"][0]["quote"] = "invented"
    output = ModelOutput.from_content(
        "test", "{" if kind == "malformed" else json.dumps(j)
    )
    output.usage = ModelUsage(input_tokens=100, output_tokens=50, total_tokens=150)
    client = AsyncMock()
    client.generate.return_value = output
    p.models["judge/binary"] = client
    original = tmp_path / "runs/old/failure.json"
    original.parent.mkdir(parents=True)
    original.write_text('{"original":true}')
    before = original.read_bytes()

    async def work():
        await p.judging_stage(r)
        await p.judging_stage(r)

    asyncio.run(work())
    assert client.generate.await_count == 1
    result = runner.read(p.base / "requests/fresh/judge/result.json")
    assert (result["label"] is not None) == (kind == "valid")
    assert result["status"] == (
        "classified"
        if kind == "valid"
        else "abstain"
        if kind == "abstain"
        else "judge_error"
    )
    assert not (p.base / "requests/fresh/generation").exists()
    assert original.read_bytes() == before
    sent = client.generate.call_args.args[0]
    assert len(sent) == 2 and "original semantic rubric" == sent[0].content
    assert "gpt-oss" not in sent[1].content and "user_preference" not in sent[1].content


def test_parent_stop_blocks_before_any_paid_call(tmp_path, monkeypatch):
    p, _ = pipeline(tmp_path, monkeypatch)
    path = tmp_path / "runs/old/operations/STOP"
    path.parent.mkdir(parents=True)
    path.write_text("human stop")
    with pytest.raises(RuntimeError, match="human STOP"):
        p.check()


def test_five_in_48_stops_and_preserves_history(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    for i in range(5):
        item = {**copy.deepcopy(r), "assignment_id": str(i)}
        p.event(item, "judge", True, "ValidationError")
    assert p.stop.is_set()
    assert len((p.operations / "stage-events.jsonl").read_text().splitlines()) == 5


def test_judge_only_record_cannot_generate(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="Misrouted"):
        asyncio.run(p.generation_stage(r))


def test_failed_fresh_generation_never_falls_back(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    r["recovery_kind"] = "generation"
    parent = {
        "assignment_id": "old",
        "label": "loop_only",
        "judge_result": True,
        "raw_saved": True,
        "recovery_assignment_id": "earlier",
    }
    runner.freeze(
        p.base / "requests/fresh/generation/result.json", {"status": "request_error"}
    )
    result = runner.overlay_row(tmp_path, parent, r)
    assert result["label"] is None and result["raw_saved"] is False
    assert result["prior_generation_recovery_id"] == "earlier"
    assert parent["label"] == "loop_only"


def test_agent_restarts_cleanly_once_and_settles_all_dispatches(
    tmp_path, monkeypatch, catalog
):
    tasks = runner.main_tasks(ROOT)
    r = copy.deepcopy(
        next(
            r
            for r in catalog["records"]
            if r["recovery_kind"] == "generation" and r["mode"] == "B"
        )
    )
    monkeypatch.setattr(runner, "main_tasks", lambda _: tasks)
    monkeypatch.setattr(runner, "load_bundles", lambda _: ([], {}))
    manifest = {**catalog, "records": [r]}
    p = runner.Pipeline(tmp_path, manifest)
    monkeypatch.setattr(runner, "verify", lambda _: manifest)
    calls = []

    def collect(root, cohort, assignments, profile, image, values, **kwargs):
        assigned = assignments[0]
        calls.append(assigned.record)
        assert assigned.record["assignment_id"] != r["replaces"]
        assert assigned.record["visible_files"] == r["visible_files"]
        assert kwargs["generation_override"].max_tokens == 4096
        path = cohort / profile / "attempts" / assigned.record["assignment_id"]
        output = ModelOutput.from_content("test", "final message")
        output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
        raw = output.model_dump(mode="json")
        runner.freeze(path / "provider/0001.started.json", {"intent": True})
        runner.freeze(path / "provider/0001.raw.json", raw)
        runner.freeze(path / "raw.json", raw)
        runner.freeze(
            path / "episode.json",
            {"status": "completed", "cleanup_verified": True, "model_dispatches": 1},
        )
        (path / "final.py").write_text(assigned.task.visible.stub)
        runner.freeze(
            path / "scoring.json",
            {"correctness": {"status": "ok", "full_correct": False}},
        )

    monkeypatch.setattr(runner.prior.pilot, "collect_profile", collect)
    runner.collect_agents(p)
    runner.collect_agents(p)
    assert len(calls) == 1
    assert p.ledger.snapshot()["spent"] == runner.nanodollars(
        (100 * 0.15 + 20 * 0.6) / 1e6
    )
    assert p.ledger.snapshot()["outstanding"] == 0
    state = runner.outcome(p.base, r)
    assert state["attempt_complete"] and state["raw_saved"] and state["scored"]


def test_each_agent_turn_has_a_distinct_transport_context(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    r["agent_profile"] = {"model": "fireworks/test"}
    calls = []

    episode = object.__new__(runner.agent.Episode)
    episode.calls = 0

    async def handler(request):
        calls.append(request)
        return httpx.Response(200, json={})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:

            async def dispatch(episode, original, inputs, tools, choice, config):
                episode.calls += 1
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={
                        "model": "test",
                        "messages": [{"role": "user", "content": str(episode.calls)}],
                        "max_tokens": 100,
                    },
                )
                output = ModelOutput.from_content("test", "turn complete")
                output.usage = ModelUsage(
                    input_tokens=20, output_tokens=10, total_tokens=30
                )
                return output

            with runner.concurrent_transport():
                for _ in range(2):
                    await runner.agent_dispatch(
                        p,
                        r,
                        runner.WaitClock(),
                        dispatch,
                        episode,
                        None,
                        [],
                        [],
                        "auto",
                        None,
                    )

    asyncio.run(exercise())
    assert len(calls) == 2
    folder = p.base / "requests/fresh/generation/dispatches"
    first, second = [
        runner.read(folder / i / "transport.json") for i in ("0001", "0002")
    ]
    assert first["ticket"] != second["ticket"]
    assert first["attempts"] == second["attempts"] == 1
    assert first["state"] == second["state"] == "accepted"


def test_same_limit_cap_is_terminal_but_not_operational_failure(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    r.update(mode="A", recovery_kind="generation")
    q = p.base / "requests/fresh/generation"
    runner.freeze(q / "result.json", {"status": "incomplete"})
    runner.freeze(q / "raw.json", {"choices": [{"stop_reason": "max_tokens"}]})
    p.event(r, "generation", True)
    assert p.history[-1]["failed"] is True
    assert p.history[-1]["operational_failed"] is False
    assert not runner.expected_token_cap(p.base, r, "judge")
    assert not runner.expected_token_cap(p.base, {**r, "mode": "B"}, "generation")
    (q / "result.json").write_text('{"status":"request_error"}')
    assert not runner.expected_token_cap(p.base, r, "generation")


def test_approved_range_repair_changes_only_reversed_endpoints():
    b = runner.base(ROOT)
    approval = runner.read(
        b / "operations/recoveries/terminal-outcomes-v1/approval.json"
    )
    for aid in approval["range_repair_ids"]:
        p = b / "requests" / aid / "judge"
        raw = ModelOutput.model_validate(runner.read(p / "raw.json"))
        original = json.loads(raw.completion)
        repair = runner.repair_reversed_ranges(
            raw.completion, runner.read(p / "packet.json")
        )
        corrected = repair["corrected"]
        assert {k: v for k, v in original.items() if k != "evidence"} == {
            k: v for k, v in corrected.items() if k != "evidence"
        }
        assert repair["label"] == repair["audit_label"]
        for before, after in zip(
            original["evidence"], corrected["evidence"], strict=True
        ):
            assert sorted(before.values()) == sorted(after.values())
        corrected["rationale"] = ""
        with pytest.raises(ValueError):
            runner.repair_reversed_ranges(
                json.dumps(corrected), runner.read(p / "packet.json")
            )
