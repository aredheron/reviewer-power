"""Verify RL preference contrasts preserve failed outcomes and task pairing."""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
analysis = importlib.import_module("scripts.analyze_rl_pilot")


def rows() -> list[dict[str, Any]]:
    """Two synthetic complete tasks follow the grader perfectly in all cells."""
    return [
        {
            "kind": "core",
            "source_task_id": t,
            "condition": {
                "user_preference": u,
                "grader_preference": g,
                "grader_consequence": "rl_update",
            },
            "scoring": {
                "style": {
                    "label": "comprehension_only"
                    if g == "comprehensions"
                    else "loop_only",
                    "status": "scored",
                },
                "correctness": {"full_correct": True, "status": "pass"},
            },
        }
        for t in ("HumanEval/7", "Mbpp/1")
        for u in analysis.STYLES
        for g in analysis.STYLES
    ]


def test_preference_effect_uses_paired_task_cells() -> None:
    result = analysis.core_statistics(rows())
    assert result["assigned"] == 8
    assert result["grader_preference_effect"] == 1
    assert result["interval95"] == [1, 1]
    assert result["development_excluded_effect"] == 1


def test_missing_style_counts_zero_without_removing_assignment() -> None:
    assigned = rows()
    assigned[1]["scoring"] = None
    result = analysis.core_statistics(assigned)
    assert result["assigned"] == 8
    assert result["style_counts"]["unscored"] == 1
    assert result["grader_preference_effect"] == 0.75
    assert result["full_correct"] == 7


@pytest.mark.parametrize("change", ["missing_cell", "duplicate", "other_consequence"])
def test_invalid_matrix_is_rejected(change: str) -> None:
    assigned = rows()
    if change == "missing_cell":
        assigned.pop()
    elif change == "duplicate":
        assigned.append(assigned[0])
    else:
        assigned[0]["condition"]["grader_consequence"] = "shadow"
    with pytest.raises(ValueError):
        analysis.core_statistics(assigned)
