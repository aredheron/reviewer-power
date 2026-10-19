"""Prepare and inspect the expanded task study without making model API calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.inspect_mode_a import digest, json_text
from grader_power import expanded_preference_study as study
from grader_power.datasets.mbpp_full import write_once
from grader_power.expanded_preference_tasks import load_tasks, public_inventory


def main() -> None:
    """Save reviewable source-faithful prompts, reuse coverage and a spending proposal."""
    root = Path(__file__).resolve().parents[1]
    target = study.base(root) / study.PREPARATION
    tasks = load_tasks(root)
    rows = study.schedule(root, tasks)
    artifacts = {
        "tasks-private.json": tasks,
        "tasks-public.json": public_inventory(tasks),
        "assignments-proposed.json": rows,
        "reuse-coverage.json": study.old_coverage(root),
        "budget-proposal.json": study.budget_proposal(root, tasks),
    }
    bindings = {}
    for name, value in artifacts.items():
        raw = json_text(value).encode()
        write_once(target / name, raw)
        bindings[name] = digest(raw)
    previews = set()
    for row in rows:
        if row["phase"] != "main":
            continue
        key = (
            row["source_task_id"],
            row["condition"]["arm"],
            row["condition"]["user_preference"],
        )
        if key in previews:
            continue
        previews.add(key)
        filename = "--".join(key).replace("/", "-") + ".txt"
        write_once(
            target / "prompts" / filename, row["messages"][0]["content"].encode()
        )
    if len(previews) != 864:
        raise ValueError("Incomplete task/arm/direction preview coverage")
    write_once(
        target / "preparation.json",
        json_text(
            {
                "paid_calls": 0,
                "paid_authorized": False,
                "prompt_previews": len(previews),
                "artifacts": bindings,
            }
        ).encode(),
    )
    print(
        json.dumps(
            {
                "directory": str(target),
                "tasks": len(tasks),
                "prompt_previews": len(previews),
                "fresh_main": 7776,
                "reused": 3456,
                "budget": artifacts["budget-proposal.json"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
