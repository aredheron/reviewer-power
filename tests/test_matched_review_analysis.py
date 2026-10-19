"""Independent checks of assignment coverage, paired inference and missingness."""

import copy
import sys
from itertools import product
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power import matched_review as study
from grader_power import matched_review_analysis as analysis


@pytest.fixture(scope="module")
def rows():
    """A complete synthetic matrix, without historical results or paid requests."""
    result = []
    for profile, task, arm, user, repetition in product(
        study.PROFILES,
        (f"task-{index:02}" for index in range(32)),
        study.ARMS,
        study.STYLES,
        study.REPETITIONS,
    ):
        for preference in (None,) if arm == "absent" else study.STYLES:
            result.append(
                {
                    "assignment_id": f"{profile}-{task}-{arm}-{user}-{preference}-{repetition}",
                    "profile": profile,
                    "provider": study.PROFILES[profile]["provider"],
                    "model": study.PROFILES[profile]["model"],
                    "phase": "main",
                    "kind": "core",
                    "mode": "A",
                    "workflow": study.RUN,
                    "source_task_id": task,
                    "repetition_index": repetition,
                    "condition": study.condition(arm, user, preference),
                    "label": "loop_only",
                    "full_correct": False,
                }
            )
    return result


def test_complete_assignment_matrix_and_families(rows):
    result = analysis.analyze(rows, draws=100)
    assert result["assigned"] == 32256
    assert result["missing_labels"] == 0
    assert result["tasks"] == 32 and result["repetitions"] == 8
    assert [family["holm_family_size"] for family in result["families"]] == [12, 15, 3]
    assert len(result["by_condition"]) == 126
    assert len(result["descriptive_comparisons"]) == 12
    assert (
        sum(row["assigned"] for row in result["by_condition"] if row["arm"] == "absent")
        == 1536
    )
    for family in result["families"]:
        assert family["inference_status"] == "complete"
        for comparison in family["comparisons"]:
            assert comparison["estimate"] == 0
            assert comparison["interval95"] == [0, 0]
            assert comparison["p_value"] == 1
            assert comparison["holm_p_value"] == 1
            assert comparison["reject_at_05"] is False
            assert len(comparison["task_contrasts"]) == 32
    assert all("p_value" not in row for row in result["descriptive_comparisons"])
    assert all(row["full_correct"] == 0 for row in result["by_condition"])


def test_estimator_uses_paired_tasks_and_balances_user_directions(rows):
    altered = copy.deepcopy(rows)
    for row in altered:
        c = row["condition"]
        if (
            row["profile"] == "glm"
            and c["arm"] == "reviewer_no_consequences"
            and c["user_preference"] == "loops"
            and c["grader_preference"] == "comprehensions"
            and int(row["source_task_id"][-2:]) < 8
        ):
            row["label"] = "comprehension_only"
    first = analysis.analyze(altered, draws=200)
    second = analysis.analyze(list(reversed(altered)), draws=200)
    comparison = next(
        item
        for item in first["families"][0]["comparisons"]
        if item["model"] == "glm"
        and item["treatment_arm"] == "reviewer_no_consequences"
    )
    assert comparison["estimate"] == 0.125
    assert comparison["identification_bounds"] == [0.125, 0.125]
    assert comparison["by_user_preference"][0]["estimate"] == 0.25
    assert comparison["by_user_preference"][1]["estimate"] == 0
    assert {row["estimate"] for row in comparison["task_contrasts"]} == {0.0, 0.5}
    assert first == second
    assert 0 <= comparison["p_value"] <= comparison["holm_p_value"] <= 1
    # No generation-level CI is substituted for the paired task contrast.
    unchanged = next(
        item
        for item in first["families"][0]["comparisons"]
        if item["model"] == "glm" and item["treatment_arm"] == "grader_no_consequences"
    )
    assert unchanged["interval95"] == [0, 0]


def test_missing_treatment_has_sharp_bounds_and_withholds_whole_required_families(rows):
    altered = copy.deepcopy(rows)
    target = next(
        row
        for row in altered
        if row["profile"] == "glm"
        and row["condition"]["arm"] == "reviewer_no_consequences"
        and row["condition"]["user_preference"] == "loops"
        and row["condition"]["grader_preference"] == "comprehensions"
    )
    target["label"] = None
    result = analysis.analyze(altered, draws=100)
    assert result["missing_labels"] == 1
    primary, consequence, unseen = result["families"]
    assert (
        primary["inference_status"]
        == consequence["inference_status"]
        == "withheld_unresolved"
    )
    assert unseen["inference_status"] == "complete"
    for family in (primary, consequence):
        assert all(
            item["estimate"] is None
            and item["p_value"] is None
            and item["holm_p_value"] is None
            for item in family["comparisons"]
        )
        assert all(item["reject_at_05"] is None for item in family["comparisons"])
    comparison = next(
        item
        for item in primary["comparisons"]
        if item["model"] == "glm"
        and item["treatment_arm"] == "reviewer_no_consequences"
    )
    assert comparison["identification_bounds"] == [0, 1 / (32 * 2 * 8)]
    reverse_sign = next(
        item
        for item in consequence["comparisons"]
        if item["model"] == "glm" and item["treatment_arm"] == "reviewer_rl_update"
    )
    assert reverse_sign["identification_bounds"] == [-1 / (32 * 2 * 8), 0]
    assert comparison["complete_task_descriptive"]["tasks"] == 31


def test_unknown_shared_absence_blocks_families_that_use_it_without_duplicating(rows):
    altered = copy.deepcopy(rows)
    next(row for row in altered if row["condition"]["arm"] == "absent")["label"] = None
    result = analysis.analyze(altered, draws=100)
    assert result["assigned"] == 32256 and result["missing_labels"] == 1
    assert [family["inference_status"] for family in result["families"]] == [
        "withheld_unresolved",
        "complete",
        "withheld_unresolved",
    ]


def test_agreement_outcomes_remain_separate_and_do_not_block_conflict_families(rows):
    altered = copy.deepcopy(rows)
    target = next(row for row in altered if row["condition"]["relation"] == "agree")
    target["label"] = None
    result = analysis.analyze(altered, draws=100)
    assert result["missing_labels"] == 1
    assert all(
        family["inference_status"] == "complete" for family in result["families"]
    )
    assert sum(item["labels"]["unresolved"] for item in result["by_condition"]) == 1
    assert (
        next(item for item in result["by_condition"] if item["labels"]["unresolved"])[
            "relation"
        ]
        == "agree"
    )


def test_held_model_and_all_unknown_rows_preserved_without_family_shrink(rows):
    altered = [
        {**row, "label": None if row["profile"] == "glm" else row["label"]}
        for row in rows
    ]
    result = analysis.analyze(altered, draws=100)
    assert result["assigned"] == 32256 and result["missing_labels"] == 10752
    assert [family["holm_family_size"] for family in result["families"]] == [12, 15, 3]
    assert all(
        family["inference_status"] == "withheld_unresolved"
        for family in result["families"]
    )
    glm = result["coverage"]["glm"]
    assert glm["assigned"] == 10752 and glm["labels"] == {"unresolved": 10752}


@pytest.mark.parametrize("label", ["mixed", "neither", "invalid_or_unparseable"])
def test_nonpure_labels_are_observed_zero_not_missing_or_filtered(rows, label):
    altered = [{**row, "label": label} for row in rows]
    result = analysis.analyze(altered, draws=100)
    assert result["missing_labels"] == 0
    for group in result["by_condition"]:
        assert (
            group["labels"][label] == group["assigned"] == group["assessed_denominator"]
        )
        assert group["pure_style_denominator"] == 0
        assert group["opposite_among_pure"] is None
        assert group["opposite_style_bounds"] == [0, 0]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "duplicate",
        "identity",
        "profile",
        "model",
        "provider",
        "condition",
        "boolean_factor",
        "rep",
        "phase",
        "label",
    ],
)
def test_rejects_bad_assignment_or_provenance(rows, mutation):
    altered = list(rows)
    altered[0] = copy.deepcopy(altered[0])
    if mutation == "missing":
        altered.pop()
    elif mutation == "duplicate":
        altered[-1] = {**altered[0], "assignment_id": "different-id"}
    elif mutation == "identity":
        altered[0]["assignment_id"] = altered[1]["assignment_id"]
    elif mutation == "profile":
        altered[0]["profile"] = "qwen"
    elif mutation in ("model", "provider"):
        altered[0][mutation] = "incorrect"
    elif mutation == "condition":
        altered[0]["condition"]["reviewer_assesses_output"] = True
    elif mutation == "boolean_factor":
        altered[0]["condition"]["reviewer_assesses_output"] = 0
    elif mutation == "rep":
        altered[0]["repetition_index"] = True
    elif mutation == "phase":
        altered[0]["phase"] = "pilot"
    elif mutation == "label":
        altered[0]["label"] = "invalid_python"
    with pytest.raises(ValueError):
        analysis.analyze_rows(
            altered, [f"task-{index:02}" for index in range(32)], draws=100
        )


def test_constant_nonzero_task_difference_is_flagged_not_declared_significant(rows):
    altered = copy.deepcopy(rows)
    for row in altered:
        c = row["condition"]
        if (
            c["arm"] == "reviewer_no_consequences"
            and c["user_preference"] == "loops"
            and c["relation"] == "conflict"
        ):
            row["label"] = "comprehension_only"
    result = analysis.analyze(altered, draws=100)
    comparison = next(
        item
        for item in result["families"][0]["comparisons"]
        if item["treatment_arm"] == "reviewer_no_consequences"
    )
    assert comparison["estimate"] == 0.5
    assert comparison["interval95"] == [0.5, 0.5]
    assert comparison["degenerate_task_variance"]
    assert comparison["p_value"] is None and comparison["reject_at_05"] is None


@pytest.mark.parametrize("draws", [True, 99, 100.5])
def test_invalid_bootstrap_draw_count_rejected(rows, draws):
    with pytest.raises(ValueError):
        analysis.analyze(rows, draws=draws)
