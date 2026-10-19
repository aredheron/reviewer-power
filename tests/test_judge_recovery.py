"""Strict line citations, unchanged scientific inputs and bounded judge-only recovery."""

import asyncio
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_judge_recovery as runner
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
        "evidence": [{"start_line": 2, "end_line": 2}],
        "flags": [],
        "rationale": "The comprehension builds the returned list.",
    }


@pytest.mark.parametrize("design", ["mapping", "reversal"])
def test_ranges_extract_exact_source_and_preserve_original_semantics(design):
    j = judgment()
    result = runner.parse_ranges(json.dumps(j), packet(), design)
    assert result.label == "comprehension_only"
    assert result.evidence[0].quote == "    return [x * 2 for x in items]"
    assert (
        result.rationale == j["rationale"] and result.target_forms == j["target_forms"]
    )
    assert "quote" not in j["evidence"][0]
    old = (
        ROOT
        / (
            "prompts/style_judge/system.txt"
            if design == "mapping"
            else "prompts/style_judge/three-style-v2.txt"
        )
    ).read_text()
    new = runner.range_prompt(old)
    assert old.split("OUTPUT CONTRACT\n")[0] == new.split("OUTPUT CONTRACT\n")[0]
    assert '"quote"' not in new and '"start_line"' in new


@pytest.mark.parametrize(
    "damage",
    [
        "extra",
        "quote",
        "bool",
        "string",
        "zero",
        "reverse",
        "too_long",
        "duplicate",
        "prose",
        "unknown_form",
    ],
)
def test_invalid_ranges_and_schema_remain_terminal(damage):
    j = judgment()
    e = j["evidence"][0]
    if damage == "extra":
        j["helpful_extra"] = ""
    elif damage == "quote":
        e["quote"] = "fabricated"
    elif damage == "bool":
        e["start_line"] = True
    elif damage == "string":
        e["start_line"] = "2"
    elif damage == "zero":
        e["start_line"] = 0
    elif damage == "reverse":
        e["end_line"] = 1
    elif damage == "too_long":
        e["end_line"] = 50
    elif damage == "unknown_form":
        j["target_forms"] = ["good"]
    text = json.dumps(j)
    if damage == "duplicate":
        text = text.replace('"start_line": 2', '"start_line": 2, "start_line": 2')
    if damage == "prose":
        text += "\nA second answer."
    with pytest.raises((ValueError, TypeError)):
        runner.parse_ranges(text, packet(), "mapping")


def test_abstention_not_forced_to_resolved():
    j = {
        "status": "abstain",
        "target_forms": [],
        "evidence": [],
        "flags": [],
        "rationale": "Attribution ambiguous.",
    }
    assert runner.parse_ranges(json.dumps(j), packet(), "mapping").label is None


@pytest.fixture(scope="module")
def catalog():
    return runner.catalog(ROOT)


def test_exact_authorized_catalog_preserves_code_and_has_no_generation_inputs(catalog):
    records, bindings, _ = catalog
    assert len(records) == len({r["assignment_id"] for r in records}) == 90
    assert Counter(r["category"] for r in records) == runner.CATEGORIES
    assert sum(r["needs_scoring"] for r in records) == 7
    assert sum(r["line_ranges"] for r in records) == 40
    for r in records:
        assert "messages" not in r and r["assignment_id"] != r["replaces"]
        assert set(r["packet"]) == {
            "code",
            "entry_point",
            "transformation",
            "input_contract",
        }
        assert (
            runner.digest(runner.json_text(runner.read(ROOT / r["source_raw"])))
            == r["source_raw_sha256"]
        )
    runner.prior.verify_hashes(ROOT, bindings)


def pipeline(tmp_path, monkeypatch, *, ranges=True):
    monkeypatch.setattr(runner, "main_tasks", lambda root: [])
    r = {
        "assignment_id": "new",
        "replaces": "old",
        "original_run": "runs/old",
        "study": "mapping",
        "profile": "gpt-oss-120b",
        "needs_scoring": False,
        "packet": packet(),
        "line_ranges": ranges,
        "source_raw_sha256": "bound",
        "phase": "recovery",
    }
    config = {
        "input_rate": 2,
        "output_rate": 10,
        "model": "anthropic/claude-sonnet-5-5",
        "provider": "anthropic",
    }
    m = {
        "records": [r],
        "profiles": {"sonnet": config},
        "judges": {"mapping": {"system_prompt": "frozen rubric"}},
        "range_prompts": {"mapping": "frozen rubric, range serialization"},
    }
    return runner.JudgePipeline(tmp_path, m), r


def test_new_judgment_keeps_old_request_and_never_retries(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    old = tmp_path / "runs/old/failure.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"original":true}')
    client = AsyncMock()
    output = ModelOutput.from_content("test", json.dumps(judgment()))
    output.usage = ModelUsage(input_tokens=100, output_tokens=30, total_tokens=130)
    client.generate.return_value = output
    p.models["judge/mapping"] = client

    async def work():
        await p.judging_stage(r)
        await p.judging_stage(r)

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert runner.outcome(p.base, r)["label"] == "comprehension_only"
    assert old.read_text() == '{"original":true}'
    assert not (p.base / "requests/new/generation").exists()
    assert p.ledger.snapshot()["cap"] == nanodollars(5)
    assert p.ledger.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("failure", ["timeout", "invalid"])
def test_new_failure_terminal_and_unknown_usage_retained(
    tmp_path, monkeypatch, failure
):
    p, r = pipeline(tmp_path, monkeypatch)
    client = AsyncMock()
    p.models["judge/mapping"] = client
    if failure == "timeout":
        client.generate.side_effect = TimeoutError("ambiguous")
    else:
        output = ModelOutput.from_content("test", "{")
        output.usage = ModelUsage(input_tokens=100, output_tokens=30, total_tokens=130)
        client.generate.return_value = output

    async def work():
        await p.judging_stage(r)
        await p.judging_stage(r)

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert runner.outcome(p.base, r)["label"] is None
    assert (p.ledger.snapshot()["outstanding"] > 0) == (failure == "timeout")


def test_budget_before_send_and_generation_prohibited(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    p.ledger.reserve({"unknown": nanodollars(5)})
    p.models["judge/mapping"] = AsyncMock()
    with pytest.raises(SpendingLimit):
        asyncio.run(p.judging_stage(r))
    p.models["judge/mapping"].generate.assert_not_awaited()
    assert not runner.outcome(p.base, r)["judge_started"]
    with pytest.raises(RuntimeError, match="outside"):
        asyncio.run(p.generation_stage(r))


def test_ninety_records_feed_bounded_queue_without_loss(tmp_path, monkeypatch):
    p, r = pipeline(tmp_path, monkeypatch)
    p.records = [
        {**r, "assignment_id": str(i), "needs_scoring": i < 7} for i in range(90)
    ]
    seen = []

    async def score(record):
        await p.judging.put(record)

    async def judge(record):
        seen.append(record["assignment_id"])
        await asyncio.sleep(0)

    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "scoring_stage", score)
    monkeypatch.setattr(p, "judging_stage", judge)
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert Counter(seen) == Counter(str(i) for i in range(90))


def test_all_prospective_analyses_accept_prior_recovery_denominators():
    for name in runner.RELATIVE_RUNS:
        rows = runner.read(
            runner.prior.base(ROOT)
            / "analysis/replacements"
            / name
            / "measured-rows.json"
        )
        before = copy.deepcopy(rows)
        result = runner.statistics(name, rows)
        assert rows == before
        assert isinstance(result, dict) and result


def test_analysis_preserves_model_scoped_ids(tmp_path, monkeypatch):
    source = tmp_path / "input/original_main/measured-rows.json"
    source.parent.mkdir(parents=True)
    rows = [
        {"assignment_id": "same", "profile": model, "label": "loop_only"}
        for model in ("gpt-oss-120b", "claude-sonnet")
    ]
    source.write_text(json.dumps(rows))
    manifest = {"records": [], "analysis_input": "input"}
    monkeypatch.setattr(runner, "verify", lambda root: manifest)
    monkeypatch.setattr(runner, "RELATIVE_RUNS", {"original_main": "main/main-32-v1"})
    monkeypatch.setattr(
        runner, "statistics", lambda study, rows: {"assigned": len(rows)}
    )

    class FakePipeline:
        def __init__(self, root, manifest):
            self.ledger = type("Ledger", (), {"snapshot": lambda self: {}})()

    monkeypatch.setattr(runner, "JudgePipeline", FakePipeline)
    runner.analyze(tmp_path)
    assert (
        runner.read(runner.base(tmp_path) / "analysis/original_main/measured-rows.json")
        == rows
    )
