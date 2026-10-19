"""Condition-blind combination tests; submitted source is never executed."""

from pathlib import Path
from typing import Any

import pytest

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.oracle import OracleResult
from grader_power.schemas import ScoringRegion
from grader_power.scoring import OracleRequest, score_response

REGION = ScoringRegion(
    "synthetic-test", "double_values", "Double inputs", "returned_list_v1"
)


@pytest.mark.parametrize(
    "source,label",
    [
        (
            "def double_values(values):\n    return [2 * x for x in values]\n",
            "comprehension_only",
        ),
        (
            "def double_values(values):\n    result = []\n    for x in values:\n        result.append(2*x)\n    return result\n",
            "loop_only",
        ),
        (
            "def double_values(values):\n    result = [x*2 for x in values]\n    for x in values:\n        result.append(x*2)\n    return result\n",
            "mixed",
        ),
        (
            "def double_values(values):\n    return list(map(lambda x: x*2, values))\n",
            "neither",
        ),
    ],
)
def test_static_outcomes_never_claim_correctness(source: str, label: str) -> None:
    """Both styles and alternatives retain exact text and an unevaluated oracle."""
    result = score_response(source, REGION)
    assert result.style.label == label
    assert result.compile_status == "valid"
    assert result.format_compliant
    assert result.extraction.raw_response == source
    assert result.extraction.code == source
    assert result.correctness["status"] == "not_evaluated"
    assert result.correctness["base"] is None
    assert result.correctness["plus"] is None


@pytest.mark.parametrize(
    "source",
    [
        "",
        "not python!",
        "def other():\n    return []\n",
        "```python\nx=1\n```\n```python\nx=2\n```",
    ],
)
def test_invalid_responses_remain_records(source: str) -> None:
    """Extraction failures are outcomes with raw text, never dropped rows."""
    result = score_response(source, REGION)
    assert not result.format_compliant
    assert result.extraction.raw_response == source
    assert result.correctness["status"] == "not_evaluated"


def test_compile_failure_is_distinct_from_parse_success() -> None:
    """AST parsing alone does not detect all Python compilation failures."""
    source = "return 3\ndef double_values(values):\n    return [x*2 for x in values]\n"
    result = score_response(source, REGION)
    assert result.extraction.status == "ok"
    assert result.compile_status == "invalid"
    assert result.compile_error == "SyntaxError"


def test_no_host_execution(tmp_path: Path) -> None:
    """Module-level side effects cannot run during extraction/style/compile checks."""
    marker = tmp_path / "must-not-exist"
    source = f"open({str(marker)!r}, 'w').write('unsafe')\ndef double_values(values):\n    return [x*2 for x in values]\n"
    score_response(source, REGION)
    assert not marker.exists()


def test_correctness_adapter_receives_exact_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connect the real adapter boundary using a deterministic fake, never Docker."""
    code = "def double_values(values):\n    return [x*2 for x in values]\n"
    dataset = LoadedDataset({}, {"source": "synthetic"})
    image = "sha256:" + "a" * 64
    request = OracleRequest(dataset, "humaneval", "HumanEval/synthetic", image)

    def evaluate(*args: Any, **kwargs: Any) -> OracleResult:
        """Check unchanged source transport and return a synthetic oracle failure."""
        assert args == (dataset, "humaneval", "HumanEval/synthetic", code)
        assert kwargs == {"mode": "full", "image": image}
        return OracleResult(
            "wrong_answer",
            "HumanEval/synthetic",
            image,
            "test-hash",
            "Synthetic failure",
        )

    monkeypatch.setattr("grader_power.scoring.evaluate_solution", evaluate)
    result = score_response(f"```python\n{code}```\n", REGION, oracle=request)
    assert result.correctness["status"] == "wrong_answer"
    assert result.correctness["full_correct"] is False
    assert result.style.label == "comprehension_only"


def test_unavailable_docker_is_not_evaluated(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable or unverified sandbox must not look like a failed solution."""
    image = "sha256:" + "a" * 64
    request = OracleRequest(
        LoadedDataset({}, {}), "humaneval", "HumanEval/synthetic", image
    )
    monkeypatch.setattr(
        "grader_power.scoring.evaluate_solution",
        lambda *args, **kwargs: OracleResult(
            "infrastructure_error",
            "HumanEval/synthetic",
            image,
            "test-hash",
            "Sandbox unavailable",
        ),
    )
    result = score_response(
        "def double_values(values):\n    return []\n", REGION, oracle=request
    )
    assert result.correctness["status"] == "not_evaluated"
    assert result.correctness["adapter"]["status"] == "infrastructure_error"
