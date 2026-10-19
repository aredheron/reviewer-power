"""Qualification, immutable historical controls, accounting and missingness checks."""

import asyncio
import copy
import json
import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_compact_replication as runner
from grader_power import compact_replication as study
from grader_power.replication_spending import RequestLedger
from grader_power.spending import SpendingLimit


@pytest.fixture(scope="module")
def records():
    return study.schedule(ROOT)[0]


def test_fresh_ids_exact_historical_prompts_balanced_matrix(records):
    historical, bindings = study.historical_rows(ROOT)
    originals = {(r["study"], r["phase"], r["assignment_id"]): r for r in historical}
    assert (
        bindings and len(records) == len({r["assignment_id"] for r in records}) == 10880
    )
    for profile in study.PROFILES:
        main = [r for r in records if r["phase"] == "main" and r["profile"] == profile]
        assert Counter(r["study"] for r in main) == {"mapping": 1536, "reversal": 1024}
        assert len({r["source_task_id"] for r in main}) == 32
        assert {r["repetition_index"] for r in main} == {0, 1}
    for r in records:
        if r["kind"] != "comprehension_probe":
            old = originals[r["study"], r["phase"], r["historical_assignment_id"]]
            assert r["messages"] == old["messages"]
            assert r["condition"] == old["condition"]
    selection = json.loads((ROOT / "manifests/main_selection.json").read_text())
    assert not set(selection["confirmation"]) & {r["source_task_id"] for r in records}


def measured_pilot(records):
    return [
        {
            **r,
            "raw_saved": True,
            "full_correct": True,
            "label": "neither",
            "probe_correct": True,
        }
        for r in records
        if r["phase"] == "pilot"
    ]


def test_readiness_never_depends_on_observed_preference(records):
    rows = measured_pilot(records)
    assert study.pilot_gate(rows)["passed"]
    for r in rows:
        r["label"] = "loop_only"
    assert study.pilot_gate(rows)["passed"]
    rows[0]["raw_saved"] = False
    assert not study.pilot_gate(rows)["passed"]


@pytest.mark.parametrize("field", ["full_correct", "label", "probe_correct"])
def test_gate_catches_individual_model_design_failures(records, field):
    rows = measured_pilot(records)
    for r in rows:
        if r["profile"] == "qwen" and r["study"] == "reversal":
            r[field] = None if field == "label" else False
    gate = study.pilot_gate(rows)
    assert not gate["passed"]
    assert [g["profile"] for g in gate["groups"] if not g["passed"]] == ["qwen"]


def test_inspection_selection_is_outcome_independent(records):
    chosen = study.audit_selection(records)
    assert len(chosen) == len(set(chosen)) == 48
    changed = [{**r, "label": "loop_only"} for r in reversed(records)]
    assert study.audit_selection(changed) == chosen
    assert Counter(r["profile"] for r in records if r["assignment_id"] in chosen) == {
        k: 12 for k in study.PROFILES
    }


def test_transactional_budget_concurrent_admission_and_unknown_retention(tmp_path):
    ledger = RequestLedger(tmp_path / "spending.sqlite3", cap=100, scope="fixed")
    ledger.reserve({"unknown": 30})

    def admit(i):
        try:
            ledger.reserve({str(i): 20})
            return True
        except SpendingLimit:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(admit, range(8))) == 3
    assert ledger.snapshot()["outstanding"] == 90
    assert (
        RequestLedger(ledger.path, cap=100, scope="fixed").snapshot()["outstanding"]
        == 90
    )
    with pytest.raises(ValueError, match="scope"):
        RequestLedger(ledger.path, cap=200, scope="fixed").snapshot()
    with sqlite3.connect(ledger.path) as db:
        assert db.execute("SELECT COUNT(*) FROM entries").fetchone()[0] == 4
        assert db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 4


def test_settlement_overrun_blocks_all_further_work(tmp_path):
    ledger = RequestLedger(tmp_path / "ledger.sqlite3", cap=100, scope="fixed")
    ledger.reserve({"one": 20})
    ledger.settle({"one": 25})
    assert ledger.snapshot()["spent"] == 25
    with pytest.raises(SpendingLimit):
        ledger.reserve({"two": 1})
    with pytest.raises(ValueError):
        ledger.settle({"one": 19})
    with pytest.raises(ValueError):
        ledger.settle({"missing": 1})


def test_analysis_retains_every_assignment_and_missingness(records):
    rows = [
        {
            **r,
            "label": "loop_only",
            "raw_saved": True,
            "full_correct": True,
            "started": True,
            "scored": True,
            "judge_started": True,
            "judge_result": True,
        }
        for r in records
        if r["phase"] == "main"
    ]
    result = study.analyze(rows, draws=50)
    assert result["assigned"] == 10240 and result["holm_family_size"] == 48
    damaged = copy.deepcopy(rows)
    damaged[0]["label"] = None
    missing = study.analyze(damaged, draws=50)
    for model in missing["models"].values():
        for comparison in (
            model["mapping"]["contrasts"] + model["reversal"]["contrasts"]
        ):
            assert comparison["primary_inference"] is None
    with pytest.raises(ValueError, match="every assigned"):
        study.analyze(rows[:-1], draws=20)


def pipeline(tmp_path, monkeypatch, record):
    monkeypatch.setattr(runner, "main_tasks", lambda root: [])
    monkeypatch.setattr(runner, "load_pilot_tasks", lambda root: [])
    manifest = {
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        }
    }
    return runner.Pipeline(tmp_path, manifest, [record], "pilot")


def test_saved_request_is_never_replayed_and_usage_is_settled(
    tmp_path, monkeypatch, records
):
    r = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, r)
    output = ModelOutput.from_content("test", "safe content")
    output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    client = AsyncMock()
    client.generate.return_value = output
    p.models[r["profile"]] = client

    async def work():
        first = await p.request(r, "generation", [ChatMessageUser(content="test")])
        again = await p.request(r, "generation", [ChatMessageUser(content="test")])
        assert first == again

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["spent"] > 0
    assert p.ledger.snapshot()["outstanding"] == 0
    assert runner.measure(p.base, r)["attempt_complete"]


@pytest.mark.parametrize("failure", ["timeout", "credential", "missing_usage"])
def test_terminal_failures_retain_reservations_without_replay(
    tmp_path, monkeypatch, records, failure
):
    r = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, r)
    client = AsyncMock()
    p.models[r["profile"]] = client
    p.keys.add("secret-marker-for-test")
    if failure == "timeout":
        client.generate.side_effect = TimeoutError("unknown acceptance")
    else:
        client.generate.return_value = ModelOutput.from_content(
            "test", "secret-marker-for-test" if failure == "credential" else "safe"
        )
    asyncio.run(p.request(r, "generation", [ChatMessageUser(content="test")]))
    assert p.ledger.snapshot()["outstanding"] > 0
    assert p.ledger.snapshot()["spent"] == 0
    assert client.generate.await_count == 1
    if failure != "timeout":
        assert p.stop.is_set()
    if failure == "credential":
        assert not (runner.path_for(p.base, r) / "generation/raw.json").exists()
        for path in p.base.rglob("*.json"):
            assert "secret-marker-for-test" not in path.read_text()


def test_phase_drains_tail_below_worker_pool_size(tmp_path, monkeypatch, records):
    selected = [r for r in records if r["phase"] == "pilot"][:5]
    p = pipeline(tmp_path, monkeypatch, selected[0])
    p.records = selected
    seen = []

    async def generation(r):
        seen.append(r["assignment_id"])
        await asyncio.sleep(0)

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert set(seen) == {r["assignment_id"] for r in selected}
    assert not p.stop.is_set()


def test_pause_drains_without_starting_queued_work(tmp_path, monkeypatch, records):
    r = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, r)
    p.stop.set()
    generate = AsyncMock()
    monkeypatch.setattr(p, "generation_stage", generate)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    with pytest.raises(RuntimeError):
        asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    generate.assert_not_awaited()


def test_operational_amendment_binds_reports_science_and_retained_outcomes(
    tmp_path, monkeypatch
):
    b = study.base(tmp_path)
    name = "evals/inspect_compact_replication.py"
    old, new = runner.digest("old"), runner.digest("new")
    runner.original.freeze(b / "assignments.json", [])
    manifest = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "code_sha256": {name: old},
        "assignments_sha256": runner.digest((b / "assignments.json").read_bytes()),
        "verification_reports": {},
        "historical_bindings": {},
    }
    runner.original.freeze(b / "manifest.json", manifest)
    p = b / "source-snapshot" / name
    p.parent.mkdir(parents=True)
    p.write_text("old")
    (b / "check.xml").write_text('<testsuite><testcase name="passed"/></testsuite>')
    (b / "raw.json").write_text("{}")
    amendment = {
        "manifest_sha256": runner.digest(runner.json_text(manifest)),
        "code_sha256": {name: new},
        "audit_bindings": {},
        "verification_reports": {
            "check.xml": runner.digest((b / "check.xml").read_bytes())
        },
        "retained_artifacts": {
            "raw.json": runner.digest((b / "raw.json").read_bytes())
        },
    }
    runner.original.freeze(b / "amendment.json", amendment)
    runner.write_json(
        b / "operations/recovery-state.json",
        {
            "source_amendment": "amendment.json",
            "source_amendment_sha256": runner.digest(
                (b / "amendment.json").read_bytes()
            ),
        },
    )
    monkeypatch.setattr(runner, "sources", lambda root: {name: new})
    assert runner.load(tmp_path, retained=True)[0] == manifest
    (b / "raw.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Retained outcome"):
        runner.load(tmp_path, retained=True)
    monkeypatch.setattr(
        runner, "sources", lambda root: {"src/grader_power/compact_replication.py": new}
    )
    with pytest.raises(ValueError, match="frozen science"):
        runner.load(tmp_path)


def cleanup_fixture(tmp_path, records):
    """Create five verifiably unsent and two invalid-judge original pilot records."""
    b = study.base(tmp_path)
    code = [
        r
        for r in records
        if r["phase"] == "pilot" and r["profile"] == "deepseek" and r["kind"] == "core"
    ][:4]
    fact = next(
        r
        for r in records
        if r["phase"] == "pilot"
        and r["profile"] == "deepseek"
        and r["kind"] == "comprehension_probe"
    )
    judges = [
        r
        for r in records
        if r["phase"] == "pilot" and r["profile"] == "glm" and r["kind"] == "core"
    ][:2]
    requests = []
    for r in [*code, fact]:
        p = runner.path_for(b, r)
        runner.write_json(
            p / "generation/transport.json",
            {"attempts": 0, "state": "pending_admission"},
        )
        requests.append({"original_id": r["assignment_id"], "stage": "generation"})
    for r in judges:
        p = runner.path_for(b, r)
        runner.write_json(p / "generation/raw.json", {"preserved": True})
        runner.write_json(p / "judge/raw.json", {"invalid": True})
        runner.write_json(
            p / "judge/result.json", {"status": "judge_error", "label": None}
        )
        requests.append({"original_id": r["assignment_id"], "stage": "judge"})
    name = "operations/recoveries/pilot-cleanup-v1/supplement.json"
    runner.write_json(b / name, {"requests": requests})
    runner.write_json(
        b / "amendment.json",
        {"audit_bindings": {name: runner.digest((b / name).read_bytes())}},
    )
    runner.write_json(
        b / "operations/recovery-state.json", {"source_amendment": "amendment.json"}
    )
    return b, runner.cleanup_records(b, records)


def test_cleanup_preserves_science_and_rejects_sent_or_unbound_supplements(
    tmp_path, records
):
    before = copy.deepcopy(records)
    b, additions = cleanup_fixture(tmp_path, records)
    assert records == before
    assert len(additions) == 7
    assert not {r["assignment_id"] for r in additions} & {
        r["assignment_id"] for r in records
    }
    by_id = {r["assignment_id"]: r for r in records}
    for r in additions:
        old = by_id[r["supplements"]]
        assert r["messages"] == old["messages"] and r["condition"] == old["condition"]
    first = additions[0]
    p = b / "requests" / first["supplements"] / "generation/transport.json"
    runner.write_json(p, {"attempts": 1, "state": "uncertain_failure"})
    with pytest.raises(ValueError, match="was sent"):
        runner.cleanup_records(b, records)
    runner.write_json(p, {"attempts": 0, "state": "pending_admission"})
    (b / "operations/recoveries/pilot-cleanup-v1/supplement.json").write_text("{}")
    with pytest.raises(ValueError, match="not bound"):
        runner.cleanup_records(b, records)


def test_qualified_rows_use_fence_exact_results_and_preserve_original_failures(
    tmp_path, records
):
    b, additions = cleanup_fixture(tmp_path, records)
    fresh_fact = next(r for r in additions if r["kind"] == "comprehension_probe")
    runner.write_json(
        runner.path_for(b, fresh_fact) / "probe.json",
        {"strict": {"all_correct": False}, "fence_only": {"all_correct": True}},
    )
    fresh_judge = next(r for r in additions if r["supplement_stage"] == "judge")
    runner.write_json(
        runner.path_for(b, fresh_judge) / "judge/result.json",
        {"status": "classified", "label": "mixed"},
    )
    rows = runner.qualified_pilot_rows(b, records + additions)
    assert len(rows) == 640
    by_id = {r["assignment_id"]: r for r in rows}
    assert by_id[fresh_fact["supplements"]]["probe_correct"] is True
    assert by_id[fresh_fact["supplements"]]["probe_correct_strict"] is False
    assert by_id[fresh_judge["supplements"]]["label"] == "mixed"
    old = next(r for r in records if r["assignment_id"] == fresh_judge["supplements"])
    assert runner.measure(b, old)["label"] is None
    assert (
        runner.generation_path(b, fresh_judge)
        == runner.path_for(b, old) / "generation/raw.json"
    )
    assert not runner.measure(b, fresh_judge)["raw_saved"]
    assert not study.pilot_gate(rows)["passed"]


def test_cleanup_queues_only_fresh_generations_and_judge_only_records(
    tmp_path, monkeypatch, records
):
    _, additions = cleanup_fixture(tmp_path, records)
    p = pipeline(tmp_path, monkeypatch, additions[0])
    p.phase, p.records = "pilot_cleanup", additions
    generated, judged = [], []

    async def generation(r):
        generated.append(r["assignment_id"])

    async def judge(r):
        judged.append(r["assignment_id"])

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "judging_stage", judge)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert len(generated) == 5 and len(judged) == 2
    assert set(generated) | set(judged) == {r["assignment_id"] for r in additions}


def test_judge_scaling_drains_full_upstream_queues(tmp_path, monkeypatch, records):
    p = pipeline(tmp_path, monkeypatch, records[0])
    for _ in range(64):
        p.scoring.put_nowait(records[0])
        p.judging.put_nowait(records[0])
    for q in p.queues.values():
        q.put_nowait(records[0])
    report = {
        "pacing": {
            k: {v["provider"]: {"utilization": {"output": 0.1}, "queued": 0}}
            for k, v in study.PROFILES.items()
        }
    }
    p.tune(report)
    assert p.targets["judge"] == 12
    assert all(p.targets[k] == 8 for k in study.PROFILES)
    for _ in range(10):
        p.tune(report)
    assert p.targets["judge"] == 32
    p.targets["judge"] = 8
    report["pacing"]["sonnet"]["anthropic"]["queued"] = 1
    p.tune(report)
    assert p.targets["judge"] == 8
