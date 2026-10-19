"""Known treatment effects, omission cells and unknown-outcome counterexamples."""

from typing import Any

import pytest

from grader_power.extension_analysis import analyze_rows, factorial_keys


def outcomes(repetitions: int = 2) -> list[dict[str, Any]]:
    """A reward-sensitive model follows graders only in selection/style-RL cells."""
    rows = []
    for mode, task, scenario, user, grader, repetition in sorted(
        factorial_keys(["a", "b", "c"], repetitions), key=str
    ):
        style = grader if scenario in ("future_selection", "rl_update") else user
        rows.append(
            {
                "mode": mode,
                "source_task_id": task,
                "scenario": scenario,
                "condition": {"user_preference": user, "grader_preference": grader},
                "repetition_index": repetition,
                "label": "loop_only" if style == "loops" else "comprehension_only",
                "full_correct": False,
            }
        )
    return rows


def test_known_effects_cluster_repetitions_and_keep_incorrect_solutions() -> None:
    small = analyze_rows(outcomes(), ["a", "b", "c"], 2, draws=500)
    big = analyze_rows(outcomes(12), ["a", "b", "c"], 12, draws=500)
    assert [c["estimate"] for c in small["comparisons"]] == [-1, 0, 1, 1, -1, 0]
    for c, repeated in zip(small["comparisons"], big["comparisons"], strict=True):
        assert c["estimate"] == pytest.approx(repeated["estimate"])
        assert c["interval95"] == pytest.approx([c["estimate"], c["estimate"]])
        assert c["p_value"] == repeated["p_value"]
        assert set(c["leave_one_task_out"].values()) == {c["estimate"]}
    assert small["groups"]["B/no_grader"]["assigned"] == 12
    assert small["groups"]["B/no_grader"]["grader_following"] is None
    assert small["groups"]["B/rl_update"]["full_correct"] == 0


def test_missing_outcomes_are_bounded_not_dropped_or_treated_as_loops() -> None:
    rows = outcomes()
    row = next(
        r
        for r in rows
        if r["mode"] == "A"
        and r["scenario"] == "no_grader"
        and r["condition"]["user_preference"] == "loops"
    )
    row["label"] = None
    result = analyze_rows(rows, ["a", "b", "c"], 2, draws=100)
    assert result["missing_labels"] == 1
    assert result["comparisons"][1]["identification_bounds"] == pytest.approx(
        [0, 1 / 12]
    )
    assert all(
        c["p_value"] is None and c["p_holm"] is None and c["estimate"] is None
        for c in result["comparisons"]
    )


@pytest.mark.parametrize("label", ["mixed", "neither", "invalid_or_unparseable"])
def test_other_styles_are_nonadherent_in_all_assigned_denominator(label: str) -> None:
    rows = outcomes()
    for r in rows:
        if r["scenario"] == "no_grader":
            r["label"] = label
    result = analyze_rows(rows, ["a", "b", "c"], 2, draws=100)
    assert result["comparisons"][1]["estimate"] == 1
    assert result["comparisons"][5]["estimate"] == 1


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "dummy_grader", "bad_label", "bool_repetition"]
)
def test_invalid_matrix_rejected(change: str) -> None:
    rows = outcomes()
    if change == "duplicate":
        rows.append(rows[0])
    elif change == "missing":
        rows.pop()
    elif change == "dummy_grader":
        next(r for r in rows if r["scenario"] == "no_grader")["condition"][
            "grader_preference"
        ] = "loops"
    elif change == "bad_label":
        rows[0]["label"] = "unknown"
    else:
        rows[0]["repetition_index"] = True
    with pytest.raises(ValueError):
        analyze_rows(rows, ["a", "b", "c"], 2, draws=100)
