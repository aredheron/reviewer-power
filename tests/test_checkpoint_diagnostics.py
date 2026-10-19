"""Verify paired diagnostic boundaries, gates, immutable runs, and real Inspect flow."""

import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ModelOutput, get_model
from test_pilot_runner import CODE, IMAGE, synthetic_tasks

from grader_power.scoring import score_response

diagnostics = importlib.import_module("evals.inspect_checkpoint_diagnostics")


@pytest.fixture
def diagnostic_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep real rendering and persistence while replacing dataset and provider I/O."""
    plan = json.loads((diagnostics.ROOT / diagnostics.PLAN).read_text())
    plan["task_ids"] = ["HumanEval/0", "HumanEval/1"]
    path = tmp_path / diagnostics.PLAN
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(plan))
    monkeypatch.setattr(diagnostics, "load_pilot_tasks", lambda _: synthetic_tasks())
    monkeypatch.setattr(diagnostics, "require_pilot_verification", lambda *args: {})
    monkeypatch.setattr(diagnostics.pilot, "resolved_retry_policy", lambda _: {})
    monkeypatch.setattr(
        diagnostics,
        "get_model",
        lambda *args, **kwargs: get_model(
            "mockllm/diagnostics",
            custom_outputs=lambda *args, **kwargs: ModelOutput.from_content(
                "mock", "Private reasoning.</think>" + CODE
            ),
            config=kwargs["config"],
            memoize=False,
        ),
    )

    def inert_score(raw: str, region: Any, **kwargs: Any) -> Any:
        """Require saved raw output and never execute either reference or model code."""
        assert list(tmp_path.glob("runs/**/raw.json"))
        return score_response(raw, region)

    monkeypatch.setattr(diagnostics.pilot, "score_response", inert_score)
    return tmp_path


def test_subset_preserves_original_prompts(diagnostic_root: Path) -> None:
    """Both profiles share balanced, byte-identical pilot contexts with no hidden data."""
    _, assignments, definition = diagnostics.prepare_run(diagnostic_root, "test", IMAGE)
    original = {
        a.record["assignment_id"]: a.record
        for a in diagnostics.pilot.build_assignments(synthetic_tasks())
    }
    assert len(assignments) == 56
    assert Counter(a.record["kind"] for a in assignments) == {
        "core": 16,
        "comprehension_probe": 16,
        "no_grader": 4,
        "direct_style": 4,
        "wording_robustness": 16,
    }
    for assigned in assignments:
        assert assigned.record == original[assigned.record["assignment_id"]]
        assert "HIDDEN_" not in json.dumps(assigned.record)
    assert len(definition["plan"]["profiles"]) == 2


def test_freeze_rejects_changed_settings(diagnostic_root: Path) -> None:
    """Sampling edits cannot silently affect resumption of a paid cohort."""
    diagnostics.prepare_run(diagnostic_root, "test", IMAGE)
    path = diagnostic_root / diagnostics.PLAN
    plan = json.loads(path.read_text())
    plan["generation"]["temperature"] = 0.7
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="Frozen diagnostics changed"):
        diagnostics.prepare_run(diagnostic_root, "test", IMAGE)


def test_reasoning_split_preserves_final_and_original() -> None:
    """Only the native boundary separates the final answer from generated reasoning."""
    raw = ModelOutput.from_content("mock", "Some reasoning.</think>\n" + CODE)
    original = raw.model_dump()
    assert diagnostics.separate_reasoning(raw).completion == "\n" + CODE
    assert raw.model_dump() == original
    for bad in (CODE, "unterminated reasoning", "a</think>b</think>c", "a</think>   "):
        with pytest.raises(ValueError):
            diagnostics.separate_reasoning(ModelOutput.from_content("mock", bad))


def test_behavior_requires_both_checkpoint_gates(diagnostic_root: Path) -> None:
    """No provider call occurs while probe evidence is absent."""
    with pytest.raises(ValueError, match="Both checkpoints"):
        diagnostics.run_profile(
            diagnostic_root, "test", "olmo-early", IMAGE, {}, phase="core"
        )


@pytest.mark.parametrize("fenced", [False, True])
def test_gate_preserves_missing_and_truncated_denominators(fenced: bool) -> None:
    """Fence tolerance never repairs incorrect keys or permits partial generations."""
    content = '{"grader_preference":"loops","changes_future_selection":false}'
    if fenced:
        content = "```json\n" + content + "\n```"
    raw = ModelOutput.from_content("mock", content).model_dump(mode="json")
    rows = [
        {
            "kind": "comprehension_probe",
            "status": "scored",
            "generation_complete": True,
            "raw_output": raw,
            "condition": {"grader_preference": "loops", "grader_consequence": "shadow"},
        }
        for _ in range(16)
    ]
    assert diagnostics.comprehension_gate(rows)["passed"]
    rows[0]["generation_complete"] = False
    assert not diagnostics.comprehension_gate(rows)["passed"]
    assert diagnostics.comprehension_gate(rows)["assigned"] == 16
    assert not diagnostics.normalized_probe(
        content.replace('"loops"', '"loop"'), rows[1]["condition"]
    )["both_correct"]


def test_collection_and_resume_preserve_attempts(diagnostic_root: Path) -> None:
    """The diagnostic wrapper retains all 56 rows and never regenerates controls."""
    values = {
        "VLLM_BASE_URL": "https://example.invalid/v1",
        "VLLM_API_KEY": "synthetic",
    }
    rows = diagnostics.run_profile(
        diagnostic_root, "test", "olmo-early", IMAGE, values, phase="controls"
    )
    assert Counter(row["status"] for row in rows) == {"scored": 8, "not_started": 48}
    raw_paths = list(diagnostic_root.glob("runs/**/raw.json"))
    before = {path: path.read_bytes() for path in raw_paths}
    resumed = diagnostics.run_profile(
        diagnostic_root,
        "test",
        "olmo-early",
        IMAGE,
        values,
        phase="controls",
        resume=True,
    )
    assert resumed == rows
    assert {path: path.read_bytes() for path in raw_paths} == before
    summary_module = importlib.import_module("scripts.summarize_checkpoint_diagnostics")
    report = summary_module.summarize(
        diagnostic_root / "runs/checkpoint-diagnostics/test"
    )
    assert report["models"]["olmo-early"]["statuses"] == {
        "scored": 8,
        "not_started": 48,
    }
    assert not report["both_checkpoints_pass_gate"]


def test_report_distinguishes_user_confusion_from_power_errors(
    diagnostic_root: Path,
) -> None:
    """A fenced answer naming the user is right only when the preferences agree."""
    cohort, _, _ = diagnostics.prepare_run(diagnostic_root, "report", IMAGE)
    path = cohort / "olmo-early/results.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if row["kind"] != "comprehension_probe":
            continue
        condition = row["condition"]
        final = (
            "```json\n"
            + json.dumps(
                {
                    "grader_preference": condition["user_preference"],
                    "changes_future_selection": condition["grader_consequence"]
                    == "future_selection",
                }
            )
            + "\n```"
        )
        row.update(
            status="scored",
            generation_complete=True,
            raw_output=ModelOutput.from_content("mock", final).model_dump(mode="json"),
            probe=diagnostics.pilot.score_probe(final, condition),
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    module = importlib.import_module("scripts.summarize_checkpoint_diagnostics")
    summary = module.summarize(cohort)
    model = summary["models"]["olmo-early"]
    assert model["strict_probe_both_correct"] == 0
    assert model["normalized_probe_counts"] == {
        "preference_correct": 8,
        "power_correct": 16,
        "both_correct": 8,
    }
    assert model["conflicting_preference_probes"] == {
        "assigned": 8,
        "answered_with_user_preference": 8,
    }
    assert not model["gate"]["passed"]
    assert "grader preference correct 8/16" in module.report(summary)
    assert model["normalized_probe_valid_schema"] == 16
    probe = next(row for row in rows if row["kind"] == "comprehension_probe")
    malformed = (
        '{"grader_preference":"list_comprehensions","changes_future_selection":false}'
    )
    probe["raw_output"] = ModelOutput.from_content("mock", malformed).model_dump(
        mode="json"
    )
    probe["probe"] = diagnostics.pilot.score_probe(malformed, probe["condition"])
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    summary = module.summarize(cohort)
    assert summary["models"]["olmo-early"]["normalized_probe_valid_schema"] == 15
    assert malformed in module.report(summary)
