"""Opt-in end-to-end routing checks for the expanded experiment's task adapters."""

import ast
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from grader_power.expanded_preference_scoring import score_task
from grader_power.expanded_preference_tasks import load_tasks

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def task_map() -> Iterator[dict[str, dict]]:
    """Verify actual task contracts and require clean sandbox boundaries."""
    if os.environ.get("GRADER_POWER_MODE_B_DOCKER_TESTS") != "1":
        pytest.skip("Expanded-task Docker verification is explicitly opt-in")
    for label in ("grader-power.oracle=true", "grader-power.agent=true"):
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=" + label],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        assert not result.stdout.strip()
    yield {t["task_id"]: t for t in load_tasks(ROOT)}
    for label in ("grader-power.oracle=true", "grader-power.agent=true"):
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=" + label],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        assert not result.stdout.strip()


@pytest.mark.parametrize(
    "task_id",
    [
        "HumanEval/70",
        "Mbpp/612",
        "MbppFull/157",
        "MbppFull/533",
        "APPS/train/3534",
        "APPS/train/3796",
    ],
)
@pytest.mark.parametrize("style", ["loop", "comprehension"])
def test_verified_reference_through_experiment_adapter(
    task_map: dict[str, dict], task_id: str, style: str
) -> None:
    task = task_map[task_id]
    result = score_task(
        ROOT,
        task,
        "```python\n" + task["implementations"][style] + "\n```",
        (ROOT / "data/sandbox/oracle-image.id").read_text().strip(),
    )
    assert result["correctness"]["full_correct"] is True
    if task["dataset"] == "mbpp_full":
        assert result["correctness"]["public_example_passed"] is True
        assert result["correctness"]["withheld_all_passed"] is True
        assert result["correctness"]["withheld_assertions"] >= 2


def test_memorized_public_example_fails_withheld_assertions(
    task_map: dict[str, dict],
) -> None:
    task = task_map["MbppFull/533"]
    node = ast.parse(task["source"]["test_list"][0]).body[0]
    assert isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare)
    code = (
        task["stub"].split("    # TODO")[0]
        + "    return "
        + ast.unparse(node.test.comparators[0])
        + "\n"
    )
    result = score_task(
        ROOT, task, code, (ROOT / "data/sandbox/oracle-image.id").read_text().strip()
    )
    assert result["correctness"]["public_example_passed"] is True
    assert result["correctness"]["withheld_all_passed"] is False
    assert result["correctness"]["full_correct"] is False


def test_wrong_apps_output_is_not_coerced_to_pass(task_map: dict[str, dict]) -> None:
    task = task_map["APPS/train/3796"]
    code = task["stub"].split("    # TODO")[0] + "    return []\n"
    result = score_task(
        ROOT, task, code, (ROOT / "data/sandbox/oracle-image.id").read_text().strip()
    )
    assert result["correctness"]["full_correct"] is False
    assert result["correctness"]["cleanup_verified"] is True
