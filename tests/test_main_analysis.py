"""Deterministic counterexamples for the main estimand, missingness and multiplicity."""

from typing import Any

import pytest

from grader_power.main_analysis import (
    CONSEQUENCES,
    PROFILES,
    STYLES,
    analyze_rows,
    holm,
)


def outcomes(repetitions: int = 2) -> list[dict[str, Any]]:
    """Construct a known selection effect of 1 and RL effect of 0 on three tasks."""
    return [
        {
            "profile": m,
            "source_task_id": t,
            "repetition": r,
            "condition": {
                "grader_consequence": p,
                "user_preference": u,
                "grader_preference": g,
            },
            "label": "comprehension_only"
            if p == "future_selection" and g == "comprehensions"
            else "loop_only",
            "full_correct": False,
        }
        for m in PROFILES
        for t in ["a", "b", "c"]
        for p in CONSEQUENCES
        for u in STYLES
        for g in STYLES
        for r in range(repetitions)
    ]


def test_paired_effect_counts_functional_failures_and_clusters_repeats() -> None:
    small = analyze_rows(outcomes(), ["a", "b", "c"], 2, draws=500)
    large = analyze_rows(outcomes(12), ["a", "b", "c"], 12, draws=500)
    assert [r["estimate"] for r in small["comparisons"]] == [1, 0, 1, 0]
    assert [r["interval95"] for r in small["comparisons"]] == [
        [1, 1],
        [0, 0],
        [1, 1],
        [0, 0],
    ]
    assert [r["p_value"] for r in small["comparisons"]] == [
        r["p_value"] for r in large["comparisons"]
    ]
    assert small["coverage"][PROFILES[0]]["full_correct"] == 0


def test_missing_is_unknown_not_loop_or_dropped() -> None:
    rows = outcomes()
    row = next(
        r
        for r in rows
        if r["profile"] == PROFILES[0]
        and r["condition"]["grader_consequence"] == "future_selection"
        and r["label"] == "comprehension_only"
    )
    row["label"] = None
    result = analyze_rows(rows, ["a", "b", "c"], 2, draws=500)
    assert result["missing_labels"] == 1
    assert all(c["p_value"] is None for c in result["comparisons"])
    assert result["comparisons"][0]["identification_bounds"] == pytest.approx(
        [1 - 1 / 12, 1]
    )
    assert result["comparisons"][1]["identification_bounds"] == [0, 0]


@pytest.mark.parametrize("change", ["duplicate", "omit", "bad_label"])
def test_malformed_denominators_fail(change: str) -> None:
    rows = outcomes()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "omit":
        rows.pop()
    else:
        rows[0]["label"] = "unknown"
    with pytest.raises(ValueError):
        analyze_rows(rows, ["a", "b", "c"], 2)


def test_holm_unsorted_ties_and_invalid_values() -> None:
    assert holm([0.04, 0.001, 0.02, 0.02]) == pytest.approx([0.06, 0.004, 0.06, 0.06])
    for invalid in [-1, float("nan"), float("inf"), 1.01]:
        with pytest.raises(ValueError):
            holm([invalid])
