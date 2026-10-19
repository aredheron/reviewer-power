"""Task-level analysis of three distinct preferences with all assignments retained."""

import random
from collections import Counter, defaultdict
from itertools import product
from typing import Any

from grader_power.three_style import PURPOSES, STYLES, VARIANTS, relation

LABELS = {
    "loops": "loop_only",
    "comprehensions": "comprehension_only",
    "map_filter": "map_filter_only",
}
SEED = 2026101603
DRAWS = 10000
SOURCES = {
    "user": "user_preference",
    "reviewer": "grader_preference",
    "reward_style": "reward_preference",
}


def group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe assigned and resolved denominators without correctness filtering."""
    if not rows:
        raise ValueError("Empty assigned group")
    known = [r for r in rows if r["label"] is not None]
    output = {
        "assigned": len(rows),
        "started": sum(r["started"] for r in rows),
        "saved": sum(r["raw_saved"] for r in rows),
        "scored": sum(r["scored"] for r in rows),
        "resolved": len(known),
        "missing_generation": sum(not r["raw_saved"] for r in rows),
        "unresolved_judgment": sum(
            r["judge_started"] and r["label"] is None for r in rows
        ),
        "pending_judgment": sum(
            r["raw_saved"] and not r["judge_started"] and not r["judge_result"]
            for r in rows
        ),
        "correct": sum(r["full_correct"] is True for r in rows),
        "labels": dict(Counter(r["label"] for r in known)),
    }
    for name, key in SOURCES.items():
        n = sum(r["label"] == LABELS[r["condition"][key]] for r in known)
        output[name + "_following"] = n
        output[name + "_resolved_rate"] = n / len(known) if known else None
        output[name + "_assigned_bounds"] = [
            n / len(rows),
            (n + len(rows) - len(known)) / len(rows),
        ]
    return output


def contrast(
    rows: list[dict[str, Any]],
    name: str,
    components: list[tuple[str, str, float]],
    *,
    draws: int,
) -> dict[str, Any]:
    """Compare sources in all-different cells, equally weighting permutations/tasks."""
    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c = row["condition"]
        cells[
            (
                row["source_task_id"],
                c["grader_consequence"],
                c["user_preference"],
                c["grader_preference"],
                c["reward_preference"],
            )
        ].append(row)
    triples = [t for t in product(STYLES, repeat=3) if relation(*t) == "all_different"]
    tasks = sorted({r["source_task_id"] for r in rows})
    low, high, estimates, missing_ids = [], [], [], set()
    for task in tasks:
        lo = hi = est = 0.0
        estimable = True
        for purpose, preference, weight in components:
            for triple in triples:
                group = cells[(task, purpose, *triple)]
                if not group:
                    raise ValueError("Missing assigned factorial cell")
                known = [r for r in group if r["label"] is not None]
                successes = sum(
                    r["label"] == LABELS[r["condition"][preference]] for r in known
                )
                unknown = len(group) - len(known)
                missing_ids.update(
                    r["assignment_id"] for r in group if r["label"] is None
                )
                a, b = successes / len(group), (successes + unknown) / len(group)
                lo += weight * (a if weight >= 0 else b) / len(triples)
                hi += weight * (b if weight >= 0 else a) / len(triples)
                if known:
                    est += weight * successes / len(known) / len(triples)
                else:
                    estimable = False
        low.append(lo)
        high.append(hi)
        if estimable:
            estimates.append(est)
    result: dict[str, Any] = {
        "name": name,
        "components": components,
        "tasks": len(tasks),
        "missing_assigned_observations": len(missing_ids),
        "identification_bounds": [sum(low) / len(tasks), sum(high) / len(tasks)],
        "primary_inference": None,
        "resolved_only_sensitivity": None,
    }
    if len(estimates) != len(tasks):
        return result
    mean = sum(estimates) / len(tasks)
    rng = random.Random(SEED)
    boot = sorted(
        sum(rng.choices(estimates, k=len(tasks))) / len(tasks) for _ in range(draws)
    )
    inference = {
        "estimate": mean,
        "interval95": [
            boot[int(draws * 0.025)],
            boot[min(draws - 1, int(draws * 0.975))],
        ],
        "p_value": (1 + sum(abs(x - mean) >= abs(mean) - 1e-12 for x in boot))
        / (draws + 1),
        "degenerate_task_variance": max(estimates) == min(estimates),
        "task_estimates": dict(zip(tasks, estimates, strict=True)),
    }
    result["resolved_only_sensitivity"] = inference
    if not missing_ids:
        result["primary_inference"] = dict(inference)
    return result


def summarize(rows: list[dict[str, Any]], *, draws: int = DRAWS) -> dict[str, Any]:
    """Report all 54 cells and the predeclared four-contrast Holm family."""
    grouped: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        c = r["condition"]
        if c["variant"] != relation(
            c["user_preference"], c["grader_preference"], c["reward_preference"]
        ):
            raise ValueError("Incorrect agreement-pattern metadata")
        grouped[
            (
                c["grader_consequence"],
                c["user_preference"],
                c["grader_preference"],
                c["reward_preference"],
            )
        ].append(r)
    comparisons = []
    for a, b in (
        ("user", "reviewer"),
        ("reward_style", "user"),
        ("reward_style", "reviewer"),
    ):
        comparisons.append(
            contrast(
                rows,
                a + "_minus_" + b + "_rl_all_different",
                [("rl_update", SOURCES[a], 1.0), ("rl_update", SOURCES[b], -1.0)],
                draws=draws,
            )
        )
    comparisons.append(
        contrast(
            rows,
            "reward_style_rl_minus_shadow_all_different",
            [
                ("rl_update", "reward_preference", 1.0),
                ("shadow", "reward_preference", -1.0),
            ],
            draws=draws,
        )
    )
    if any(c["primary_inference"] is None for c in comparisons):
        for c in comparisons:
            c["primary_inference"] = None
    for key in ("primary_inference", "resolved_only_sensitivity"):
        if any(c[key] is None for c in comparisons):
            continue
        last = 0.0
        for rank, c in enumerate(sorted(comparisons, key=lambda c: c[key]["p_value"])):
            last = max(last, min(1.0, (len(comparisons) - rank) * c[key]["p_value"]))
            c[key]["p_holm"] = last
    return {
        "counts": group_summary(rows),
        "cells": [
            {
                "purpose": k[0],
                "user_preference": k[1],
                "reviewer_preference": k[2],
                "reward_preference": k[3],
                "variant": relation(*k[1:]),
                **group_summary(g),
            }
            for k, g in sorted(grouped.items())
        ],
        "groups": [
            {
                "purpose": p,
                "variant": v,
                **group_summary(
                    [
                        r
                        for r in rows
                        if r["condition"]["grader_consequence"] == p
                        and r["condition"]["variant"] == v
                    ]
                ),
            }
            for p in PURPOSES
            for v in VARIANTS
        ],
        "comparisons": comparisons,
        "bootstrap_draws": draws,
        "seed": SEED,
        "note": "32 reused exploratory tasks, not held-out confirmation. All assignments retained; mixed/other/invalid are nonadherent, never opposite-style evidence. Equal permutation and task weights. Primary four-contrast family withheld if contributing labels are missing; sensitivity analyses and conservative missingness bounds reported. Marginal task-bootstrap intervals and Holm-adjusted p-values. No actual training.",
    }
