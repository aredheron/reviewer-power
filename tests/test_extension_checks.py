"""GPT-only readiness scope, blinded evidence and recovery without paid calls."""

import importlib
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from grader_power.spending import SpendingLedger, nanodollars

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
checks = importlib.import_module("evals.inspect_extension_checks")
inspect_main = importlib.import_module("evals.inspect_main")
inspect_pilot = importlib.import_module("evals.inspect_pilot")


def test_scope_is_development_only_and_original_probes_are_paired() -> None:
    main = json.loads((ROOT / "manifests/main_selection.json").read_text())
    all_rows = []
    for mode, count in (("A", 32), ("B", 36)):
        rows = checks.assignments(ROOT, mode)
        assert len(rows) == count
        assert len({r.record["assignment_id"] for r in rows}) == count
        assert Counter(
            r.record.get("probe_variant")
            for r in rows
            if r.record["kind"] == "comprehension_probe"
        ) == {"original": 8, "evidence": 8}
        for row in rows:
            assert row.task.visible.source_task_id in checks.TASKS
            assert (
                row.task.visible.source_task_id
                not in main["confirmation"] + main["exploratory"]
            )
            if row.record["scenario"] == "no_grader":
                assert not any(
                    f["path"].startswith(".ci/") for f in row.record["visible_files"]
                )
            if mode == "B":
                checks.inspect_agent.validate_assignment(row)
        all_rows.extend(rows)
    assert sum(sum(checks.bounds(r).values()) for r in all_rows) < nanodollars(10)


def test_evidence_rejects_user_quote_as_reviewer_and_preserves_original_scoring() -> (
    None
):
    rows = checks.assignments(ROOT, "A")
    row = next(
        r
        for r in rows
        if r.record.get("probe_variant") == "evidence"
        and r.record["scenario"] == "no_grader"
    )
    agents = next(
        f["content"] for f in row.record["visible_files"] if f["path"] == "AGENTS.md"
    )
    answer = {
        "user_preference": row.record["condition"]["user_preference"],
        "user_evidence": {"path": "AGENTS.md", "quote": agents},
        "reviewer_preference": "not_stated",
        "reviewer_evidence": None,
    }
    assert checks.score_probe(json.dumps(answer), row.record)["all_correct"]
    fenced = checks.score_probe("```json\n" + json.dumps(answer) + "\n```", row.record)
    assert not fenced["strict"]["all_correct"] and fenced["fence_only"]["all_correct"]
    answer["reviewer_preference"] = answer["user_preference"]
    answer["reviewer_evidence"] = answer["user_evidence"]
    assert not checks.score_probe(json.dumps(answer), row.record)["all_correct"]
    answer["user_evidence"] = {"path": "AGENTS.md", "quote": "fabricated quotation"}
    assert not checks.score_probe(json.dumps(answer), row.record)["field_correct"][
        "user"
    ]
    original = next(r for r in rows if r.record.get("probe_variant") == "original")
    assert checks.score_probe(
        json.dumps(original.record["expected_probe"]), original.record
    )["all_correct"]


def test_settlement_waits_for_all_provider_intents(tmp_path: Path) -> None:
    row = checks.assignments(ROOT, "B")[0]
    ledger = SpendingLedger(
        tmp_path / "ledger.json", cap=nanodollars(10), scope="frozen"
    )
    ledger.reserve({"trial/generation": nanodollars(1)})
    attempt = tmp_path / "attempt"
    inspect_main.freeze(attempt / "episode.json", {"model_dispatches": 2})
    inspect_main.freeze(attempt / "provider/0001.started.json", {})
    inspect_main.freeze(attempt / "provider/0002.started.json", {})
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "input_tokens_cache_read": 20,
        "input_tokens_cache_write": None,
    }
    inspect_main.freeze(attempt / "provider/0001.raw.json", {"usage": usage})
    checks.settle_generation(attempt, row, ledger, "trial")
    assert ledger.snapshot()["outstanding"] == nanodollars(1)
    inspect_main.freeze(attempt / "provider/0002.raw.json", {"usage": usage})
    checks.settle_generation(attempt, row, ledger, "trial")
    assert ledger.snapshot()["outstanding"] == 0
    assert ledger.snapshot()["spent"] == 48000


def test_real_collector_resume_skips_started_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise existing collection selection, never dispatch a completed or failed ID."""
    rows = checks.assignments(ROOT, "A")[:2]
    cohort = tmp_path
    profile = checks.extension.access_profiles(ROOT)[checks.PROFILE]
    config = checks.extension.generation(ROOT, checks.PROFILE)
    inspect_main.freeze(
        cohort / "manifest.json",
        {
            "assignments": [r.record for r in rows],
            "oracle_image": "synthetic",
            "purpose": "test",
            "profiles": {
                checks.PROFILE: {
                    "model": profile.model,
                    "generation": config.model_dump(mode="json", exclude_none=True),
                }
            },
        },
    )
    for row in rows:
        inspect_main.freeze(
            cohort
            / checks.PROFILE
            / "attempts"
            / row.record["assignment_id"]
            / "started.json",
            {"synthetic": True},
        )
    monkeypatch.setattr(
        inspect_pilot, "load_access_profiles", lambda _: {checks.PROFILE: profile}
    )
    monkeypatch.setattr(
        inspect_pilot,
        "get_model",
        lambda *a, **kw: pytest.fail(
            "Started requests must not resolve a provider model"
        ),
    )
    result = inspect_pilot.collect_profile(
        ROOT,
        cohort,
        rows,
        checks.PROFILE,
        "synthetic",
        {},
        resume=True,
        max_new_attempts=1,
        generation_override=config,
    )
    assert len(result) == 2


def test_workflow_restart_keeps_unknown_charge_and_does_not_redispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Crash after durable intent, restart, and finish only the other assignment."""
    from contextlib import nullcontext

    rows = tuple(
        r
        for r in checks.assignments(ROOT, "A")
        if r.record["kind"] == "comprehension_probe"
    )[:2]
    by_mode = {"A": rows, "B": ()}
    directory = tmp_path / "runs/extension-checks" / checks.RUN
    frozen = {
        mode: {"oracle_image": "synthetic", "assignments": [r.record for r in selected]}
        for mode, selected in by_mode.items()
    }
    scope = {
        "definitions": {
            mode: checks.digest(checks.json_text(value))
            for mode, value in frozen.items()
        }
    }
    inspect_main.freeze(directory / "scope.json", scope)
    for mode in by_mode:
        inspect_main.freeze(directory / mode / "manifest.json", frozen[mode])
        inspect_main.freeze(
            directory / mode / "preflight.json",
            {"passed": True, "manifest_sha256": scope["definitions"][mode]},
        )
    monkeypatch.setattr(checks, "collection_lock", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(checks, "definition", lambda root, mode, image: frozen[mode])
    monkeypatch.setattr(checks, "assignments", lambda root, mode: by_mode[mode])
    monkeypatch.setattr(checks.extension, "access_profiles", lambda root: {})
    monkeypatch.setattr(checks.extension, "generation", lambda *a: None)
    calls = []

    def fake_collect(root, cohort, selected, *args, **kwargs):
        row = next(
            r
            for r in selected
            if not (
                cohort
                / checks.PROFILE
                / "attempts"
                / r.record["assignment_id"]
                / "started.json"
            ).exists()
        )
        path = cohort / checks.PROFILE / "attempts" / row.record["assignment_id"]
        calls.append(row.record["assignment_id"])
        inspect_main.freeze(path / "started.json", {})
        if len(calls) == 1:
            raise RuntimeError("synthetic crash after dispatch intent")
        inspect_main.freeze(
            path / "raw.json", {"usage": {"input_tokens": 100, "output_tokens": 10}}
        )

    monkeypatch.setattr(checks.pilot, "collect_profile", fake_collect)
    with pytest.raises(RuntimeError, match="synthetic crash"):
        checks.collect(tmp_path)
    checks.collect(tmp_path)
    assert calls == [r.record["assignment_id"] for r in rows]
    ledger = json.loads((directory / "spending.json").read_text())
    assert ledger["outstanding"] == checks.bounds(rows[0])["generation"]
    assert ledger["spent"] == 21000
