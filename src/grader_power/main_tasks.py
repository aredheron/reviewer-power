"""Prepare a source-linked main split without exposing canonical solutions to models."""

import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from grader_power.datasets.reference_solutions import audit_references
from grader_power.pilot_tasks import PilotTask, _task_record, digest


def main_tasks(root: Path) -> tuple[PilotTask, ...]:
    """Rebuild all 40 approved reserved tasks and verify their exact reference evidence.

    The source prompt and EvalPlus preconditions are the judge input contract. Neither canonical
    implementations nor hidden tests enter that contract. Old AST coverage does
    not determine eligibility for the new semantic scoring pipeline.
    """
    selection = json.loads((root / "manifests/pilot_selection.json").read_text())
    ids = selection["remaining_task_ids"]
    if (
        len(ids) != 40
        or len(set(ids)) != 40
        or set(ids) & set(selection["selected_task_ids"])
    ):
        raise ValueError("Expected 40 unique reserved tasks disjoint from the pilot")
    evidence = json.loads(
        (root / "manifests/reference_execution_20261010.json").read_text()
    )
    pairs, datasets, audits = audit_references(root)
    by_pair = {p.task_id: p for p in pairs}
    by_audit = {a["task_id"]: a for a in audits}
    by_dataset = {key: d for d in datasets for key in d.tasks}
    tasks = []
    for task_id in sorted(ids):
        pair, dataset = by_pair[task_id], by_dataset[task_id]
        record, visible, region = _task_record(dataset, pair, by_audit[task_id], False)
        derived = task_id.lower().replace("/", "-") + "-main"
        visible = replace(visible, derived_task_id=derived)
        region = replace(region, task_id=derived)
        record.update(
            derived_task_id=derived,
            split="main_candidate",
            scoring_region=asdict(region),
            fixture_path=None,
            oracle_evidence="manifests/reference_execution_20261010.json",
            input_contract=source_contract(dataset.tasks[task_id]),
            human_review={
                "status": "approved",
                "evidence": "User reports manually checking suitability of all 40 remaining tasks.",
            },
        )
        for style in ("loop", "comprehension"):
            name = f"runs/reference-validation/all-51-20261010/{task_id.lower().replace('/', '-')}-{style}.json"
            payload = (root / name).read_bytes()
            row = json.loads(payload)
            if (
                digest(payload) != evidence["evidence_sha256"][name]
                or row["solution_sha256"] != record["reference_sha256"][style]
                or row["image"] != evidence["oracle_image"]
                or row["status"] != "pass"
                or row["full_correct"] is not True
            ):
                raise ValueError(f"Reference evidence mismatch: {task_id} {style}")
            for suite, count in record["suite_counts"].items():
                actual = row["suites"][suite]
                if (
                    actual["total"] != count
                    or actual["status"] != "pass"
                    or len(actual["details"]) != count
                    or not all(actual["details"])
                    or actual["upstream_parity"]["status"] != "matched"
                ):
                    raise ValueError("Incomplete reference suite/parity evidence")
        tasks.append(PilotTask(dataset, pair, visible, region, record))
    return tuple(tasks)


def source_contract(source: dict[str, Any]) -> str:
    """Include independent input restrictions without solutions or hidden inputs."""
    prompt, contract = source["prompt"], source.get("contract", "")
    if not isinstance(prompt, str) or not isinstance(contract, str):
        raise TypeError("Source prompt and contract must be text")
    return prompt + "\n\nEvalPlus input preconditions:\n" + contract


def split_tasks(tasks: tuple[PilotTask, ...], seed: str) -> dict[str, Any]:
    """Lock 3 HumanEval and 5 MBPP tasks, spreading family coverage before hash ties.

    The reserved pool is 15 HumanEval and 25 MBPP, so these quotas preserve its
    source proportions exactly. No generated response or treatment effect is read.
    """
    counts = Counter(t.visible.source_task_id.split("/")[0] for t in tasks)
    if (
        counts != {"HumanEval": 15, "Mbpp": 25}
        or len({t.visible.source_task_id for t in tasks}) != 40
    ):
        raise ValueError("Expected the approved 15/25 source composition")
    selected: list[str] = []
    families: Counter[str] = Counter()
    for source, quota in (("HumanEval", 3), ("Mbpp", 5)):
        pool = [t for t in tasks if t.visible.source_task_id.startswith(source + "/")]
        for _ in range(quota):
            chosen = min(
                pool,
                key=lambda t: (
                    families[t.visible.family],
                    digest(seed + ":" + t.visible.source_task_id),
                ),
            )
            selected.append(chosen.visible.source_task_id)
            families[chosen.visible.family] += 1
            pool.remove(chosen)
    return {
        "seed": seed,
        "procedure": "Source quotas 3 HumanEval/5 MBPP; choose least represented confirmation family then SHA256(seed:source_id); HumanEval quota first. No outcomes consulted.",
        "exploratory": sorted(
            t.visible.source_task_id
            for t in tasks
            if t.visible.source_task_id not in selected
        ),
        "confirmation": sorted(selected),
        "human_approval": "User reports manually checking all 40 tasks; task approval is distinct from judge calibration.",
    }
