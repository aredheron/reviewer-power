"""Opt-in original-assertion tests through the existing verified Docker boundary."""

import os
from pathlib import Path

import pytest

from grader_power.datasets.mbpp_full import evaluate_assertions


@pytest.mark.parametrize(
    "code,passed,error",
    [
        ("def f(x): return x * 2", True, None),
        ("def f(x): return 6", False, "AssertionError"),
        ("def f(x) return x", False, "SyntaxError"),
        ("def f(x): raise RuntimeError()", False, "RuntimeError"),
    ],
)
def test_supplied_tests_setup_challenge_and_failures(
    code: str, passed: bool, error: str | None
) -> None:
    """Require real assertions, complete diagnostics and verified cleanup."""
    if os.environ.get("GRADER_POWER_MODE_B_DOCKER_TESTS") != "1":
        pytest.skip("Docker assertion verification is explicitly opt-in")
    root = Path(__file__).resolve().parents[2]
    source = {
        "task_id": 1,
        "code": "def f(x): return x * 2",
        "test_setup_code": "x = 3",
        "test_list": ["assert f(x) == 6", "assert f(0) == 0", "assert f(-2) == -4"],
        "challenge_test_list": ["assert f(10) == 20"],
    }
    result = evaluate_assertions(
        source, code, (root / "data/sandbox/oracle-image.id").read_text().strip()
    )
    assert result["passed"] is passed
    assert result["supplied_tests"] == 3 and result["challenge_tests"] == 1
    assert len(result["details"]) == 4
    assert result["cleanup_verified"]
    if error is not None:
        assert any(row["error_type"] == error for row in result["details"])
