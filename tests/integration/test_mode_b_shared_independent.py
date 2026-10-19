"""Independent shared Mode B collection/judge/resume integration with mock providers."""

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import test_inspect_agent as harness
from inspect_ai.model import ModelOutput, get_model
from test_judge_boundary_independent import CODE, CONSTRAINT, PROMPT, task
from test_mode_b_assignments_independent import current_plan, extension, two_tasks

from grader_power.pilot_tasks import PilotTask
from grader_power.scoring import score_response

__all__ = ["task"]


def test_shared_mode_b_judges_final_file_once_and_never_repeats_started_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, task: PilotTask
) -> None:
    """Run the owner collector, native agent and blinded judge; restart preserves failures."""
    all_rows = extension.build_assignments(two_tasks(task), current_plan())
    core = next(a for a in all_rows if a.record["kind"] == "core" and a.task == task)
    probe = next(
        a
        for a in all_rows
        if a.record["kind"] == "comprehension_probe" and a.task == task
    )
    interrupted = replace(core, record={**core.record, "assignment_id": "c" * 64})
    rows = (core, probe, interrupted)
    config = tmp_path / "configs/models.yaml"
    config.parent.mkdir()
    config.write_text(
        "".join(
            f"{name}:\n  model: mockllm/independent\n  base_url: https://example.invalid\n  api_key_env: SYNTHETIC_KEY\n  revision: mock\n  max_tokens: 64\n"
            for name in extension.PROFILES
        )
    )
    profiles = extension.load_access_profiles(config)
    frozen = {
        "plan": current_plan(),
        "purpose": "independent-synthetic",
        "oracle_image": harness.IMAGE,
        "assignments": [a.record for a in rows],
        "profiles": {
            name: {
                "model": "mockllm/independent",
                "generation": extension.generation(tmp_path, name).model_dump(
                    mode="json", exclude_none=True
                ),
            }
            for name in extension.PROFILES
        },
        "judge": {
            "profile": "claude-sonnet",
            "system_prompt": "Judge only the complete neutral code packet.",
            "generation": extension.generation(tmp_path, "claude-sonnet").model_dump(
                mode="json", exclude_none=True
            ),
        },
    }
    cohort = tmp_path / "runs/extension-pilot/independent"
    extension.save(cohort / "manifest.json", frozen)
    extension.save(
        cohort / "preflight.json",
        {
            "passed": True,
            "manifest_sha256": extension.digest(
                (cohort / "manifest.json").read_bytes()
            ),
        },
    )
    for name in extension.PROFILES:
        extension.pilot.export_results(cohort / name, rows, name, "synthetic")
        extension.save(
            cohort
            / name
            / "attempts"
            / interrupted.record["assignment_id"]
            / "started.json",
            {"started_at": "synthetic-interrupted"},
        )
    monkeypatch.setattr(extension, "load_run", lambda *args: (cohort, rows, frozen))
    monkeypatch.setattr(extension, "access_profiles", lambda root: profiles)
    # Actual cross-process exclusion is independently tested in the lock suite.
    monkeypatch.setattr(
        extension, "collection_lock", lambda *args, **kwargs: nullcontext()
    )
    monkeypatch.setattr(
        extension, "dotenv_values", lambda *args: {"SYNTHETIC_KEY": "synthetic"}
    )
    monkeypatch.setattr(
        extension.main_run,
        "dotenv_values",
        lambda *args: {"SYNTHETIC_KEY": "synthetic"},
    )
    monkeypatch.setattr("inspect_ai._eval.context.init_dotenv", lambda: None)
    monkeypatch.setattr("inspect_ai._eval.task.log.git_context", lambda: None)
    monkeypatch.setattr(harness.agent, "DockerWorkspace", harness.FakeWorkspace)
    monkeypatch.setattr(
        harness.agent,
        "score_response",
        lambda raw, region, **kwargs: score_response(raw, region, input_kind="file"),
    )
    counts = {"generation": 0, "judge": 0}

    def model(*args: Any, **kwargs: Any) -> Any:
        counts["generation"] += 1
        return get_model(
            "mockllm/independent",
            custom_outputs=[
                harness.response("write_file", path="src/task.py", content=CODE),
                harness.response(),
                ModelOutput.from_content(
                    "mock", json.dumps(probe.record["expected_probe"])
                ),
            ],
            config=kwargs["config"],
            memoize=False,
        )

    class Judge:
        async def generate(self, messages: Any, **kwargs: Any) -> ModelOutput:
            counts["judge"] += 1
            packet = json.loads(messages[1].content)
            assert packet["code"].encode() == CODE.encode()
            assert (
                PROMPT in packet["input_contract"]
                and CONSTRAINT in packet["input_contract"]
            )
            assert "PRIVATE_" not in messages[1].content
            assert set(packet) == {
                "entry_point",
                "transformation",
                "input_contract",
                "code",
                "code_lines",
            }
            assert kwargs["tools"] == []
            return ModelOutput.from_content(
                "mock",
                json.dumps(
                    {
                        "status": "abstain",
                        "target_forms": [],
                        "evidence": [],
                        "flags": [],
                        "rationale": "Synthetic judgment",
                    }
                ),
            )

    monkeypatch.setattr(extension.pilot, "get_model", model)
    monkeypatch.setattr(
        extension.main_run, "get_model", lambda *args, **kwargs: Judge()
    )
    monkeypatch.setattr(
        extension.pilot,
        "resolved_retry_policy",
        lambda model: {"sdk_max_retries": 0, "http_connect_retries": 0},
    )
    extension.collect_and_judge(tmp_path, "independent")
    assert counts == {"generation": 2, "judge": 2}
    before = {
        path: path.read_bytes()
        for path in cohort.rglob("*.json")
        if path.name != "summary.json"
    }
    extension.collect_and_judge(tmp_path, "independent")
    assert counts == {"generation": 2, "judge": 2}
    for path, content in before.items():
        assert path.read_bytes() == content, path
    for name in extension.PROFILES:
        attempt = cohort / name / "attempts" / interrupted.record["assignment_id"]
        assert not (attempt / "raw.json").exists()
        completed = cohort / name / "attempts" / core.record["assignment_id"]
        assert "not Python" in (completed / "raw.json").read_text()
        assert (completed / "final.py").read_bytes() == CODE.encode()
