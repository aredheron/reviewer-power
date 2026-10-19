"""Published numbers are independently counted and missing outcomes stay visible."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power.publication import presentation_tables, summarize, verified_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_reported_counts_and_held_cohort() -> None:
    tables = presentation_tables(ROOT, verified_artifacts(ROOT))
    main = {(r["profile"], r["arm"]): r for r in tables["main"]}
    assert main["deepseek", "reviewer_rl_update"]["opposite"] == 302
    assert main["deepseek", "reviewer_rl_update"]["assigned"] == 512
    assert main["gpt-oss-120b", "reviewer_future_selection"]["opposite"] == 187
    assert main["deepseek", "reviewer_unseen"]["opposite"] == 224
    held = [r for r in tables["naming"] if r["profile"] == "glm"]
    assert len(held) == 5
    assert all(
        r["rate"] is None and r["unresolved"] == r["assigned"] == 384 for r in held
    )
    pooled = next(
        r
        for r in tables["strong"]
        if (r["profile"], r["arm"]) == ("glm", "strong_rl_update")
    )
    assert (pooled["opposite"], pooled["assigned"]) == (50, 768)


def test_endpoint_keeps_invalid_in_denominator_and_unknown_bounds() -> None:
    base = {
        "study": "loops",
        "profile": "test",
        "source_task_id": "task",
        "condition": {
            "arm": "absent",
            "user_preference": "loops",
            "grader_preference": None,
        },
        "full_correct": False,
    }
    rows = [
        {**copy.deepcopy(base), "label": label}
        for label in ("comprehension_only", "invalid_or_unparseable", None)
    ]
    result = summarize(rows)[0]
    assert result["assigned"] == 3 and result["opposite"] == 1
    assert result["full_correct"] == 0 and result["rate"] is None
    assert result["lower_bound"] == pytest.approx(1 / 3)
    assert result["upper_bound"] == pytest.approx(2 / 3)
