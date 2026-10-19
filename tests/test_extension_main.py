"""Main scope and recovery tests with no external provider requests."""

import importlib
import json
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytest

from grader_power.spending import nanodollars

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
runner = importlib.import_module("evals.inspect_extension_main")


@pytest.fixture(scope="module")
def schedule() -> list[Any]:
    return runner.blocks(ROOT, json.loads((ROOT / runner.PLAN).read_text()))


def test_frozen_matrix_balanced_blocks_and_locked_tasks(schedule: list[Any]) -> None:
    rows = [r for _, block in schedule for r in block]
    assert len(rows) == len({r.record["assignment_id"] for r in rows}) == 12288
    assert len(schedule) == 768
    assert Counter(r.record["mode"] for r in rows) == {"A": 5376, "B": 6912}
    selection = json.loads((ROOT / "manifests/main_selection.json").read_text())
    assert {r.record["source_task_id"] for r in rows} == set(selection["exploratory"])
    assert not {r.record["source_task_id"] for r in rows} & set(
        selection["confirmation"]
    )
    for i in range(0, len(schedule), 2):
        a, b = schedule[i][1], schedule[i + 1][1]
        assert a[0].record["source_task_id"] == b[0].record["source_task_id"]
        assert a[0].record["repetition_index"] == b[0].record["repetition_index"]
        assert {a[0].record["mode"], b[0].record["mode"]} == {"A", "B"}
    for row in rows:
        if row.record["scenario"] == "no_grader":
            assert row.record["condition"]["grader_preference"] is None
            assert not any(
                f["path"].startswith(".ci/") for f in row.record["visible_files"]
            )
        assert row.record["kind"] == "core"
    assert max(
        sum(sum(runner.checks.bounds(r).values()) for r in block)
        for _, block in schedule
    ) < nanodollars(5)


def test_error_guard_deduplicates_reconciled_stages_and_honors_stop(
    tmp_path: Path,
) -> None:
    for i in range(4):
        runner.record_stage(tmp_path, str(i), "generation", True)
        runner.record_stage(tmp_path, str(i), "generation", True)
    runner.guard(tmp_path)
    runner.record_stage(tmp_path, "4", "judge", True)
    with pytest.raises(RuntimeError, match="Five failed"):
        runner.guard(tmp_path)
    assert (
        len((tmp_path / "operations/request-events.jsonl").read_text().splitlines())
        == 5
    )
    (tmp_path / "operations/STOP").touch()
    with pytest.raises(RuntimeError, match="STOP"):
        runner.guard(tmp_path)


def test_scope_change_rejected_before_task_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = json.loads((ROOT / runner.PLAN).read_text())
    plan["source_task_ids"][0] = plan["locked_confirmation_task_ids"][0]
    monkeypatch.setattr(
        runner,
        "main_tasks",
        lambda _: pytest.fail("Should reject before loading tasks"),
    )
    with pytest.raises(ValueError, match="approved 32 tasks"):
        runner.blocks(ROOT, plan)


def test_crash_resume_skips_started_and_releases_only_known_costs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, schedule: list[Any]
) -> None:
    """Use real ledgers and controller; replace only adapters/expensive measurements."""
    name, original_rows = next(
        (n, r) for n, r in schedule if r[0].record["mode"] == "A"
    )
    rows = original_rows[:2]
    base = runner.directory(tmp_path)
    cohort = base / "blocks" / name
    source = cohort / runner.PROFILE / "attempts"
    frozen = {"oracle_image": "synthetic"}
    runner.original.freeze(cohort / "manifest.json", frozen)
    bound = sum(sum(runner.checks.bounds(r).values()) for r in rows)
    item = {
        "name": name,
        "mode": "A",
        "maximum_nanodollars": bound,
        "manifest_sha256": runner.digest(runner.json_text(frozen)),
    }
    manifest = {"blocks": [item], "code_sha256": {}, "plan": {"api_cap_usd": 10}}
    monkeypatch.setattr(runner, "load", lambda _: (manifest, {name: rows}))
    monkeypatch.setattr(runner, "sources", lambda _: {})
    monkeypatch.setattr(runner, "collection_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(runner, "status", lambda *a, **kw: None)
    monkeypatch.setattr(runner, "measure", lambda *a: [])
    monkeypatch.setattr(runner, "analyze", lambda _: {})
    monkeypatch.setattr(runner.extension, "access_profiles", lambda _: {})
    monkeypatch.setattr(runner.extension, "generation", lambda *a: None)
    monkeypatch.setattr(runner.extension, "enforce_judge_limit", lambda *a: None)
    generated, judged = [], []
    usage = {"input_tokens": 100, "output_tokens": 10}

    def collect(*args: Any, **kwargs: Any) -> list[Any]:
        row = next(
            r
            for r in rows
            if not (source / r.record["assignment_id"] / "started.json").exists()
        )
        aid = row.record["assignment_id"]
        generated.append(aid)
        runner.original.freeze(source / aid / "started.json", {})
        if len(generated) == 1:
            raise RuntimeError("simulated crash after intent")
        runner.original.freeze(source / aid / "raw.json", {"usage": usage})
        return []

    async def judge(*args: Any) -> None:
        row = args[2][0]
        aid = row.record["assignment_id"]
        if not (source / aid / "raw.json").exists():
            return
        judged.append(aid)
        path = cohort / runner.PROFILE / "judge" / aid
        runner.original.freeze(path / "started.json", {})
        runner.original.freeze(path / "raw.json", {"usage": usage})
        runner.original.freeze(path / "result.json", {"status": "classified"})

    monkeypatch.setattr(runner.pilot, "collect_profile", collect)
    monkeypatch.setattr(runner.original, "collect_judgments", judge)
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.collect(tmp_path)
    runner.collect(tmp_path)
    assert generated == [r.record["assignment_id"] for r in rows]
    assert judged == [rows[1].record["assignment_id"]]
    ledger = json.loads((cohort / "spending.json").read_text())
    assert ledger["outstanding"] == runner.checks.bounds(rows[0])["generation"]
    assert (cohort / "complete.json").exists()
    runner.collect(tmp_path)
    assert len(generated) == 2 and len(judged) == 1
