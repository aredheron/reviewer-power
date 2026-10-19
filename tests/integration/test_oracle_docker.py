"""Opt-in execution tests; all solution/canonical execution happens inside Docker."""

import os
import subprocess
from collections.abc import Iterator

import pytest

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.oracle import docker_available, evaluate_solution


@pytest.fixture
def image() -> Iterator[str]:
    """Require explicit opt-in and verify no labeled containers remain after tests."""
    if os.environ.get("GRADER_POWER_DOCKER_TESTS") != "1":
        pytest.skip("Docker integration is opt-in: GRADER_POWER_DOCKER_TESTS=1")
    if not docker_available():
        pytest.skip("Unmet prerequisite: usable Docker daemon")
    image_id = os.environ.get("GRADER_POWER_ORACLE_IMAGE")
    if not image_id:
        pytest.fail("Set GRADER_POWER_ORACLE_IMAGE to the built image's sha256 ID")

    def containers() -> set[str]:
        """Inspect remaining stopped/running evaluator containers without executing code."""
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return set(result.stdout.split())

    before = containers()
    yield image_id
    assert containers() == before, "Evaluation leaked a Docker container"


def synthetic() -> LoadedDataset:
    """Use a simple independent task whose plus suite detects a base-only shortcut."""
    return LoadedDataset(
        {
            "HumanEval/999": {
                "task_id": "HumanEval/999",
                "entry_point": "transform",
                "prompt": "def transform(items):\n",
                "canonical_solution": "    return [2 * x for x in items]\n",
                "base_input": [([1, 2],)],
                "plus_input": [([],), ([0, -3],)],
                "atol": 0,
            }
        },
        {"variant": "full"},
    )


@pytest.mark.parametrize(
    "solution,status",
    [
        ("def transform(items):\n    return [2*x for x in items]\n", "pass"),
        ("def transform(items):\n    return [2, 4]\n", "wrong_answer"),
        ("def transform(items)\n    return items\n", "syntax_error"),
        ("def transform(items):\n    while True: pass\n", "timeout"),
        ("def transform(items):\n    raise RuntimeError('synthetic')\n", "crash"),
        ("import os\nos._exit(7)\n", "crash"),
    ],
)
def test_synthetic_outcomes(image: str, solution: str, status: str) -> None:
    """Exercise actual EvalPlus comparisons, error diagnostics, and cleanup."""
    result = evaluate_solution(
        synthetic(),
        "humaneval",
        "HumanEval/999",
        solution,
        mode="full",
        image=image,
    )
    assert result.status == status, result
    assert result.full_correct == (status == "pass")
    if status == "wrong_answer":
        assert result.suites["base"]["status"] == "pass"
        assert result.suites["plus"]["status"] == "wrong_answer"


def test_wall_deadline_cleanup(image: str) -> None:
    """Kill a hanging module through the host deadline, then verify container removal."""
    result = evaluate_solution(
        synthetic(),
        "humaneval",
        "HumanEval/999",
        "while True: pass\n",
        mode="full",
        image=image,
        wall_seconds=1,
    )
    assert result.status == "timeout"
    assert not result.full_correct


def test_invalid_canonical_is_oracle_failure(image: str) -> None:
    """Broken source evidence must not be counted as a wrong submitted answer."""
    dataset = synthetic()
    dataset.tasks["HumanEval/999"]["canonical_solution"] = "    raise ValueError()\n"
    result = evaluate_solution(
        dataset,
        "humaneval",
        "HumanEval/999",
        "def transform(items): return []\n",
        mode="full",
        image=image,
    )
    assert result.status == "oracle_error"


def test_effective_isolation(image: str) -> None:
    """Check actual network, UID, privileges, and cgroup limits from submitted code."""
    solution = """
import os
import socket
from pathlib import Path

def transform(items):
    assert os.getuid() == 65534
    status = Path('/proc/self/status').read_text()
    assert 'NoNewPrivs:\t1' in status
    assert 'CapEff:\t0000000000000000' in status
    cgroup = Path('/sys/fs/cgroup')
    assert int((cgroup / 'memory.max').read_text()) == 536870912
    assert int((cgroup / 'pids.max').read_text()) == 64
    quota, period = (cgroup / 'cpu.max').read_text().split()
    assert int(quota) == int(period)
    assert not Path('/var/run/docker.sock').exists()
    assert not Path('/workspace').exists()
    for destination in ('/opt/oracle/probe', '/oracle-input/probe'):
        try:
            Path(destination).write_text('must fail')
        except OSError:
            pass
        else:
            raise AssertionError('Protected filesystem was writable')
    with socket.socket() as connection:
        connection.settimeout(0.2)
        try:
            connection.connect(('192.0.2.1', 80))
        except OSError:
            pass
        else:
            raise AssertionError('External networking was available')
    return [2*x for x in items]
"""
    result = evaluate_solution(
        synthetic(),
        "humaneval",
        "HumanEval/999",
        solution,
        mode="full",
        image=image,
    )
    assert result.full_correct, result
