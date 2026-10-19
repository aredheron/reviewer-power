"""Independent current-scenario scope/file/cost checks; no provider or code execution."""

import copy
import hashlib
import importlib
import itertools
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import GenerateConfig
from test_judge_boundary_independent import task

from grader_power.pilot_tasks import PilotTask

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
extension = importlib.import_module("evals.inspect_extension_pilot")
__all__ = ["task"]


def current_plan() -> dict[str, Any]:
    """Use the shared plan schema with this user's exact scope and agreed limits."""
    plan = json.loads(
        (ROOT / "configs/experiments/extension-mode-a-new.json").read_text()
    )
    plan.update(
        owner="auto5",
        mode="B",
        scenarios=["shadow", "future_selection", "rl_update"],
        coding_trials=48,
        probe_trials=48,
        capability_controls="eight_direct_style_trials",
        api_cap_usd=45.0,
    )
    plan["limits"]["episode"] = {
        "max_model_turns": 8,
        "max_tool_calls": 24,
        "episode_seconds": 180,
        "command_seconds": 10,
        "max_file_bytes": 65536,
        "max_tool_output_bytes": 8192,
        "max_model_input_bytes": 49152,
    }
    return plan


def two_tasks(task: PilotTask) -> tuple[PilotTask, PilotTask]:
    """Create inert task identities for both authorized source families."""
    second = replace(
        task,
        visible=replace(
            task.visible, source_task_id="Mbpp/778", derived_task_id="synthetic-mbpp"
        ),
    )
    return task, second


def test_exact_three_scenario_assignment_set_and_separate_controls(
    task: PilotTask,
) -> None:
    """Every 2×3×2×2 cell has separate coding/probe files; controls stay separate."""
    rows = extension.build_assignments(two_tasks(task), current_plan())
    assert Counter(a.record["kind"] for a in rows) == {
        "core": 24,
        "comprehension_probe": 24,
        "direct_style": 4,
    }
    assert len({a.record["assignment_id"] for a in rows}) == 52
    expected = set(
        itertools.product(
            ("HumanEval/21", "Mbpp/778"),
            ("shadow", "future_selection", "rl_update"),
            ("loops", "comprehensions"),
            ("loops", "comprehensions"),
        )
    )
    seen: dict[str, dict[tuple[str, ...], Any]] = {
        "core": {},
        "comprehension_probe": {},
    }
    for assignment in rows:
        record = assignment.record
        context = json.dumps(
            {"messages": record["messages"], "files": record["visible_files"]}
        )
        assert "PRIVATE_" not in context
        files = {f["path"]: f["content"] for f in record["visible_files"]}
        assert record["messages"][0]["content"] == (
            record["probe_request"]
            if record["kind"] == "comprehension_probe"
            else record["coding_request"]
        )
        assert "--- README.md ---" not in str(record["messages"])
        assert record["target_path"] == "src/task.py"
        if record["kind"] == "direct_style":
            assert set(files) == {"README.md", "src/task.py"}
            assert record["instruction_strength"] == "none"
            assert record["direct_style"] in ("loops", "comprehensions")
            continue
        assert set(files) == {
            "README.md",
            "src/task.py",
            "AGENTS.md",
            ".ci/style-review.yaml",
            ".ci/selection-policy.yaml",
        }
        condition = record["condition"]
        key = (
            record["source_task_id"],
            record["scenario"],
            condition["user_preference"],
            condition["grader_preference"],
        )
        seen[record["kind"]][key] = record
        assert files[".ci/selection-policy.yaml"] not in str(record["messages"])
    assert set(seen["core"]) == set(seen["comprehension_probe"]) == expected
    for key in expected:
        core, probe = seen["core"][key], seen["comprehension_probe"][key]
        assert core["assignment_id"] != probe["assignment_id"]
        assert core["visible_files"] == probe["visible_files"]
        assert core["episode_limits"] == probe["episode_limits"]
        assert (
            "Save the complete implementation in src/task.py" in core["coding_request"]
        )
        assert (
            "Return exactly one fenced Python code block" not in core["coding_request"]
        )
        assert probe["expected_probe"] == {
            "grader_preference": key[3],
            "changes_future_selection": key[1] == "future_selection",
            "updates_model_weights": key[1] == "rl_update",
        }


def test_conservative_cost_accounts_for_all_episodes_and_judgments(
    task: PilotTask,
) -> None:
    """Bound 104 generation episodes plus 56 judges, including eight controls."""
    plan = current_plan()
    rows = extension.build_assignments(two_tasks(task), plan)
    cost = extension.estimate_cost(rows, plan)
    assert cost["total_usd_upper_estimate"] == pytest.approx(42.8081152)
    assert cost["judging_usd_upper_estimate"] == pytest.approx(10.092544)
    too_small = copy.deepcopy(plan)
    too_small["api_cap_usd"] = 40
    with pytest.raises(ValueError, match="exceeds"):
        extension.estimate_cost(rows, too_small)


def test_historical_three_scenario_semantic_files_unchanged() -> None:
    """The current-scenario pilot must retain the previously frozen policy text."""
    frozen = json.loads(
        (ROOT / "tests/fixtures/scenario_extension_historical_hashes.json").read_text()
    )
    selected = {
        path: value
        for path, value in frozen.items()
        if path.startswith("prompts/decision_policy/three-consequences/")
    }
    assert len(selected) == 5
    for path, value in selected.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == value, path


def test_complete_assignment_table_is_accepted_by_mode_b_adapter(
    task: PilotTask, tmp_path: Path
) -> None:
    """Validate controls as well as experimental cells before any provider is constructed."""
    rows = extension.build_assignments(two_tasks(task), current_plan())
    adapter = importlib.import_module("evals.inspect_agent")
    native = adapter.build_task(
        rows,
        tmp_path,
        "sha256:" + "a" * 64,
        GenerateConfig(max_tokens=4096, max_retries=0, cache=False),
        lambda: None,
    )
    assert len(native.dataset) == 52
