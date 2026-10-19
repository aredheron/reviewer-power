"""Approved mechanical repair boundaries and isolated, non-replaying recovery."""

import asyncio
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals import inspect_outcome_recovery as runner
from grader_power.spending import SpendingLimit, nanodollars


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
                "start_line": 1,
                "end_line": 1,
                "quote": "    return [x * 2 for x in items]",
            }
        ],
        "flags": [],
        "rationale": "A comprehension performs the transformation.",
    }


def test_exact_quote_range_and_empty_metadata_keep_semantics():
    j = judgment()
    j["note"] = ""
    j["evidence"][0]["start_line_note"] = "source numbering"
    before = copy.deepcopy(j)
    repaired = runner.normalize_judgment(json.dumps(j), packet(), "mapping")
    assert repaired["label"] == "comprehension_only"
    assert repaired["judgment"]["evidence"][0]["start_line"] == 2
    assert repaired["judgment"]["rationale"] == before["rationale"]
    assert (
        repaired["judgment"]["evidence"][0]["quote"] == before["evidence"][0]["quote"]
    )
    assert j == before


def test_identical_duplicate_only():
    raw = json.dumps(judgment()).replace(
        '"start_line": 1', '"start_line": 1, "start_line": 1'
    )
    assert (
        runner.normalize_judgment(raw, packet(), "mapping")["label"]
        == "comprehension_only"
    )
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        runner.normalize_judgment(
            raw.replace(
                '"start_line": 1, "start_line": 1', '"start_line": 1, "start_line": 2'
            ),
            packet(),
            "mapping",
        )


@pytest.mark.parametrize(
    "damage", ["quote", "ambiguous_quote", "semantic_extra", "malformed"]
)
def test_unapproved_repairs_fail_closed(damage):
    j, p = judgment(), packet()
    if damage == "quote":
        j["evidence"][0]["quote"] = "    return [x * 3 for x in items]"
    elif damage == "ambiguous_quote":
        p["code"] += "    return [x * 2 for x in items]\n"
    elif damage == "semantic_extra":
        j["unrecognized_judgment"] = "loops"
    raw = "{" if damage == "malformed" else json.dumps(j)
    with pytest.raises((ValueError, TypeError)):
        runner.normalize_judgment(raw, p, "mapping")


@pytest.fixture(scope="module")
def frozen_recovery():
    audit = runner.read(ROOT / runner.AUDIT)
    repairs = runner.repair_catalog(ROOT, audit)
    records, configuration, bindings = runner.build_replacements(ROOT, audit)
    return repairs, records, configuration, bindings


def test_all_120_audited_repairs_match_without_semantic_changes(frozen_recovery):
    repairs, _, _, bindings = frozen_recovery
    assert len(repairs) == 120
    assert Counter(r["origin_study"] for r in repairs) == {
        "extension": 17,
        "framing_names": 31,
        "reward_conflict": 18,
        "three_style": 34,
        "reversal": 2,
        "compact_replication": 16,
        "simple_replication": 2,
    }
    assert all(r["original_judge_result"].get("label") is None for r in repairs)
    runner.verify_hashes(ROOT, bindings)


def test_exact_80_missing_mode_a_inputs_and_new_ids(frozen_recovery):
    _, records, config, _ = frozen_recovery
    assert len(records) == len({r["assignment_id"] for r in records}) == 80
    assert Counter(r["profile"] for r in records) == {
        "gpt-oss-120b": 37,
        "qwen": 13,
        "sonnet": 11,
        "glm": 11,
        "deepseek": 8,
    }
    for r in records:
        assert r["assignment_id"] != r["replaces"]
        assert r["kind"] == "core" and r["mode"] == "A"
        assert r["phase"] == "recovery" and r["recovery_attempt"] == 1
        assert config["profiles"][r["profile"]]["generation"]["max_retries"] == 0
        assert config["profiles"][r["profile"]]["generation"]["max_tokens"] == 4096


def pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "main_tasks", lambda root: [])
    r = {
        "assignment_id": "fresh",
        "replaces": "old",
        "original_run": "runs/old",
        "phase": "recovery",
        "profile": "gpt-oss-120b",
        "study": "mapping",
        "messages": [{"role": "user", "content": "test"}],
    }
    m = {
        "records": [r],
        "profiles": {
            r["profile"]: {
                "input_rate": 0.15,
                "output_rate": 0.6,
                "model": "fireworks/gpt-oss-test",
                "provider": "fireworks",
            }
        },
    }
    return runner.RecoveryPipeline(tmp_path, m), r


@pytest.mark.parametrize("missing", [False, True])
def test_request_never_replays_and_unknown_reservation_is_retained(
    tmp_path, monkeypatch, missing
):
    p, r = pipeline(tmp_path, monkeypatch)
    client = AsyncMock()
    p.models[r["profile"]] = client
    if missing:
        client.generate.side_effect = TimeoutError("ambiguous acceptance")
    else:
        output = ModelOutput.from_content("test", "safe content")
        output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
        client.generate.return_value = output

    async def work():
        first = await p.request(r, "generation", [ChatMessageUser(content="test")])
        again = await p.request(r, "generation", [ChatMessageUser(content="test")])
        assert first == again

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["cap"] == nanodollars(5)
    assert (p.ledger.snapshot()["outstanding"] > 0) == missing
    assert not (tmp_path / "runs/old").exists()


def test_budget_blocks_before_intent_and_http(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    p.ledger.reserve({"retained-unknown": nanodollars(5)})
    client = AsyncMock()
    p.models[r["profile"]] = client
    with pytest.raises(SpendingLimit):
        asyncio.run(p.request(r, "generation", [ChatMessageUser(content="test")]))
    client.generate.assert_not_awaited()
    assert not runner.measure(p.base, r)["started"]


def test_human_stop_and_five_failure_guard(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    stop = tmp_path / r["original_run"] / "operations/STOP"
    stop.parent.mkdir(parents=True)
    stop.touch()
    with pytest.raises(RuntimeError, match="Original-study human STOP"):
        p.check()
    assert p.stop.is_set()
    p2, r2 = pipeline(tmp_path / "separate", monkeypatch)
    for i in range(5):
        p2.event({**r2, "assignment_id": str(i)}, "generation", True, "TimeoutError")
    assert p2.stop.is_set()
    assert len(p2.history) == 5


def test_small_queue_drains_and_does_not_spawn_duplicate_lane(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    stage = AsyncMock()
    monkeypatch.setattr(p, "generation_stage", stage)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    stage.assert_awaited_once_with(r)


def test_pacing_copy_preserves_reported_caps_and_cooldowns(tmp_path):
    source = tmp_path / "old.json"
    state = {
        "fireworks": {
            "limits": {"output": 123},
            "verified_limits": {"output": {"value": 123}},
            "cooldown_until": 9999999999,
            "last_429": 100,
            "queue": [],
            "window": [],
        }
    }
    runner.write_json(source, state)
    runner.seed_pacing(tmp_path, {"pacing_sources": {"gpt-oss-120b": "old.json"}})
    target = runner.base(tmp_path) / "operations/pacing/gpt-oss-120b/rate-state.json"
    assert runner.read(target) == runner.read(source) == state
    runner.write_json(target, {"updated": True})
    runner.seed_pacing(tmp_path, {"pacing_sources": {"gpt-oss-120b": "old.json"}})
    assert runner.read(target) == {"updated": True}
