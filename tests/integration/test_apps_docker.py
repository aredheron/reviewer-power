"""Opt-in callable APPS checks through the unchanged verified Docker boundary."""

import json
import os
from pathlib import Path

import pytest

from grader_power.datasets.apps_source import evaluate


@pytest.mark.parametrize(
    "code,passed,error",
    [
        (
            "seen = []\ndef f(values):\n    seen.append(1)\n    values.append(len(seen))\n    return values",
            True,
            None,
        ),
        ("def f(values): return [[1, 2, 1]]", False, "AssertionError"),
        ("def f(values): return (1, 2, 1)", False, "AssertionError"),
        ("def f(values) return values", False, "SyntaxError"),
        ("def f(values): raise RuntimeError()", False, "RuntimeError"),
    ],
)
def test_callable_value_depth_type_failure_and_case_isolation(
    code: str, passed: bool, error: str | None
) -> None:
    """Repeated mutating cases must each get fresh arguments and module globals."""
    if os.environ.get("GRADER_POWER_MODE_B_DOCKER_TESTS") != "1":
        pytest.skip("Docker verification is explicitly opt-in")
    root = Path(__file__).resolve().parents[2]
    source = {
        "id": 1,
        "input_output": json.dumps(
            {
                "fn_name": "f",
                "inputs": [[[1, 2]], [[1, 2]]],
                "outputs": [[[1, 2, 1]], [[1, 2, 1]]],
            }
        ),
    }
    result = evaluate(
        source, code, (root / "data/sandbox/oracle-image.id").read_text().strip()
    )
    assert result["passed"] is passed
    assert result["cleanup_verified"] and result["supplied_cases"] == 2
    assert all(
        row == {"passed": passed, "error_type": error} for row in result["details"]
    )
