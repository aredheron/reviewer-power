"""Approved no-implementation endpoint override, keeping the strict abstention."""

import copy
from typing import Any

from grader_power import matched_review_analysis as original

IDENTITY = "9207fc0b46a266cca1a614a58e7fab3e15bb9d15472afe8c0466ca99b6003684"


def analyze(rows: list[dict[str, Any]], draws: int = 10000) -> dict[str, Any]:
    """Use unchanged paired statistics with one explicit, approved known-zero outcome.

    The existing calculator accepts only its original label vocabulary. A private
    calculation copy uses its zero-valued `neither` category; every reported label
    table restores the separate no_implementation category. No stored scientific
    row is classified as neither, and the strict abstention is retained in lineage.
    """
    calculation = copy.deepcopy(rows)
    targets = [r for r in calculation if r["assignment_id"] == IDENTITY]
    if len(targets) != 1 or targets[0]["label"] != "no_implementation":
        raise ValueError("Require exactly the approved no-implementation outcome")
    if any(
        r["label"] == "no_implementation" and r["assignment_id"] != IDENTITY
        for r in calculation
    ):
        raise ValueError("Unapproved endpoint override")
    target = targets[0]
    if (
        target.get("endpoint_value") != 0
        or target.get("original_judge_status") != "abstain"
    ):
        raise ValueError("Missing explicit endpoint/abstention provenance")
    target["label"] = "neither"
    result = original.analyze(calculation, draws=draws)
    c = target["condition"]
    for summary in result["by_condition"]:
        if summary["model"] == target["profile"] and all(
            summary[k] == c[k] for k in ("arm", "user_preference", "grader_preference")
        ):
            summary["labels"]["neither"] -= 1
            summary["labels"]["no_implementation"] = 1
    labels = result["coverage"][target["profile"]]["labels"]
    labels["neither"] -= 1
    labels["no_implementation"] = 1
    result["endpoint"] += (
        " One explicitly approved, source-audited no_implementation outcome also counts zero, separately from neither and invalid."
    )
    result["endpoint_overrides"] = [
        {
            "assignment_id": IDENTITY,
            "outcome": "no_implementation",
            "endpoint_value": 0,
            "original_judge_status": "abstain",
        }
    ]
    return result
