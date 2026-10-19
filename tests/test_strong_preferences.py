"""Fresh assignment scope, qualified admission, missingness and no-replay regressions."""

import asyncio
import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evals import inspect_strong_preferences as runner
from grader_power import strong_preference_replication as study
from grader_power.spending import SpendingLimit, nanodollars


@pytest.fixture(scope="module")
def records():
    return study.schedule(ROOT)[0]


def test_exact_fresh_scope_and_unchanged_coding_prompts(records):
    old = json.loads((ROOT / study.PARENT / "assignments.json").read_text())
    byid = {r["assignment_id"]: r for r in old}
    assert len(records) == 3968 and len({r["assignment_id"] for r in records}) == 3968
    assert not {r["assignment_id"] for r in records} & set(byid)
    assert Counter((r["phase"], r["profile"]) for r in records) == {
        ("main", "glm"): 2304,
        ("main", "minimax"): 768,
        ("main", "kimi"): 768,
        ("pilot", "minimax"): 64,
        ("pilot", "kimi"): 64,
    }
    for r in records:
        if r["kind"] == "core":
            assert r["messages"] == byid[r["historical_assignment_id"]]["messages"]
    assert {r["repetition_index"] for r in records if r["profile"] == "glm"} == set(
        range(3, 12)
    )
    assert nanodollars(study.CAP) + nanodollars(study.PARENT_RECORDED) == nanodollars(
        375
    )


def ready_rows(records):
    rows = []
    for r in records:
        if r["phase"] != "pilot":
            continue
        rows.append(
            r
            | {
                "raw_saved": True,
                "full_correct": True,
                "label": "loop_only",
                "probe_correct": True,
            }
        )
    rows += [r | {"profile": "glm"} for r in rows if r["profile"] == "minimax"]
    return rows


def test_readiness_is_per_model_complete_and_not_selected_by_style(records):
    rows = ready_rows(records)
    gate = study.pilot_gate(rows)
    assert gate["passed"]
    for row in rows:
        if (
            row["profile"] == "kimi"
            and row["kind"] == "comprehension_probe"
            and row["condition"]["arm"] == "strong_shadow"
        ):
            row["probe_correct"] = False
    gate = study.pilot_gate(rows)
    assert not gate["passed"]
    assert [g["profile"] for g in gate["groups"] if g["passed"]] == ["glm", "minimax"]
    rows = ready_rows(records)
    rows[-1] = rows[-2]
    assert not study.pilot_gate(rows)["passed"]


def test_factual_validation_retains_strict_schema(records):
    r = next(r for r in records if r["kind"] == "comprehension_probe")
    text = json.dumps(study.expected_facts(r))
    assert study.score_probe(text, r)["strict"]["all_correct"]
    assert study.score_probe("```json\n" + text + "\n```", r)["fence_only"][
        "all_correct"
    ]
    assert not study.score_probe(text + " explanation", r)["fence_only"]["all_correct"]
    assert not study.score_probe(text[:-1] + ',"user_preference":"loops"}', r)[
        "strict"
    ]["all_correct"]


def test_analysis_complete_assignment_family_and_missingness(records):
    rows = [
        r
        | {
            "label": "loop_only"
            if r["condition"]["user_preference"] == "loops"
            else "comprehension_only",
            "full_correct": True,
        }
        for r in records
        if r["phase"] == "main"
    ]
    analysis = study.analyze(rows, draws=100)
    assert (
        analysis["assigned"] == 3840
        and len(analysis["comparisons"]) == 15
        and analysis["missing_labels"] == 0
    )
    assert all(c["estimate"] == 0 for c in analysis["comparisons"])
    rows[0]["label"] = None
    analysis = study.analyze(rows, draws=100)
    assert analysis["missing_labels"] == 1 and all(
        c["p_value"] is None for c in analysis["comparisons"]
    )
    with pytest.raises(ValueError):
        study.analyze(rows[:-1], draws=100)
    with pytest.raises(ValueError):
        study.analyze(rows[:-1] + [rows[0]], draws=100)


def pipeline(tmp_path, monkeypatch, records):
    monkeypatch.setattr(runner, "main_tasks", lambda root: [])
    monkeypatch.setattr(runner, "load_pilot_tasks", lambda root: [])
    manifest = {
        "judge_profile": study.JUDGE_PROFILE,
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
    }
    return runner.Pipeline(tmp_path, manifest, records, "pilot")


def test_saved_request_never_replayed(tmp_path, monkeypatch, records):
    r = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, [r])
    output = ModelOutput.from_content("test", "safe")
    output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    client = AsyncMock()
    client.generate.return_value = output
    p.models[r["profile"]] = client

    async def work():
        assert await p.request(
            r, "generation", [ChatMessageUser(content="test")]
        ) == await p.request(r, "generation", [ChatMessageUser(content="test")])

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["spent"] > 0 and p.ledger.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("error", ["timeout", "missing_usage", "credential"])
def test_terminal_failures_preserve_reserves(tmp_path, monkeypatch, records, error):
    r = next(r for r in records if r["phase"] == "pilot")
    p = pipeline(tmp_path, monkeypatch, [r])
    client = AsyncMock()
    p.models[r["profile"]] = client
    p.keys.add("secret-test-marker")
    if error == "timeout":
        client.generate.side_effect = TimeoutError("uncertain acceptance")
    else:
        client.generate.return_value = ModelOutput.from_content(
            "test", "secret-test-marker" if error == "credential" else "safe"
        )
    asyncio.run(p.request(r, "generation", [ChatMessageUser(content="test")]))
    assert client.generate.await_count == 1 and p.ledger.snapshot()["outstanding"] > 0
    if error == "credential":
        assert not (runner.path_for(p.base, r) / "generation/raw.json").exists()


def test_short_tail_drains_and_held_models_never_queue(tmp_path, monkeypatch, records):
    selected = [r for r in records if r["phase"] == "main"][:8]
    p = pipeline(tmp_path, monkeypatch, selected)
    p.phase = "main"
    p.eligible_models = ["glm"]
    seen = []

    async def generation(r):
        seen.append(r["assignment_id"])

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert set(seen) == {r["assignment_id"] for r in selected if r["profile"] == "glm"}


def test_five_failed_stages_stop_admission(tmp_path, monkeypatch, records):
    selected = records[:5]
    p = pipeline(tmp_path, monkeypatch, selected)
    for row in selected:
        p.event(row, "generation", True, "test")
    assert p.stop.is_set()
    with pytest.raises(RuntimeError):
        p.check()


def test_model_caps_respect_numeric_provider_evidence(tmp_path):
    for profile, limit in [("minimax", 432000), ("kimi", 216000)]:
        pacer = runner.ModelPacer(tmp_path / profile, profile=profile)
        lane = {
            "limits": {"input": 1e9, "uncached": 1e9, "output": 1e9, "requests": 6000},
            "verified_limits": {"input": 1e9},
            "window": [],
            "queue": [],
            "last_429": 0,
            "last_adjustment": 1000,
            "last_calibration": 1000,
            "calibration_probes": 8,
            "fraction": 0.85,
            "cooldown_until": 0,
        }
        # Isolate the new published-tier bound from inherited adaptive heuristics.
        from unittest.mock import patch

        with patch.object(runner.SharedModelPacer, "adjust"):
            pacer.adjust("fireworks", lane, 1000)
        assert lane["limits"]["output"] == limit and lane["limits"]["input"] == 1e9


def test_combined_ceiling_keeps_parent_spend_and_unknown_reservations(tmp_path):
    manifest = {"run": study.RUN}
    ledger = runner.ledger_for(study.base(tmp_path), manifest)
    ledger.reserve({"unknown": nanodollars(study.CAP)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"extra": 1})
    balance = ledger.snapshot()
    assert balance["outstanding"] + nanodollars(study.PARENT_RECORDED) == nanodollars(
        375
    )
    assert balance["spent"] == 0


def test_forecast_includes_all_assigned_generations_and_only_code_judges(
    tmp_path, records
):
    manifest = {
        "cost_floor_per_response": {profile: [0.01, 0.02] for profile in study.PROFILES}
    }
    forecast = runner.median_forecast(tmp_path, manifest, records, draws=100)
    assert forecast["median_incremental_usd"] == pytest.approx(
        3968 * 0.01 + 3904 * 0.02
    )
    assert forecast["median_total_usd"] == pytest.approx(study.PARENT_RECORDED + 117.76)
    assert not forecast["budget_notification_required"]
    ledger = runner.ledger_for(study.base(tmp_path), manifest)
    ledger.reserve({"unknown/noncohort": nanodollars(5)})
    updated = runner.median_forecast(tmp_path, manifest, records, draws=100)
    assert updated["median_total_usd"] == pytest.approx(
        forecast["median_total_usd"] + 5
    )
    assert updated["budget_notification_required"]
