"""Prepare an offline three-consequence pilot preview; never call models or train."""

import argparse
import importlib
import json
import sys
from pathlib import Path

from grader_power.consequence_extension import PROBE_REQUEST, render_consequence_block
from grader_power.pilot_tasks import load_pilot_tasks, save

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pilot = importlib.import_module("evals.inspect_pilot")


def main() -> None:
    """Write a new preview manifest and exact example without touching prior runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Use a new output directory; previous previews are preserved")
    root = Path(__file__).resolve().parents[1]
    tasks = load_pilot_tasks(root)
    records = []
    example = None
    for task in tasks:
        for rendered in render_consequence_block(task.visible):
            condition = rendered.condition.model_dump()
            content = rendered.messages[0].content
            kinds = ["core"]
            if task.provenance.get("control_subset"):
                kinds.append("comprehension_probe")
            for kind in kinds:
                message = (
                    content + "\n\n" + pilot.OUTPUT_CONTRACT
                    if kind == "core"
                    else PROBE_REQUEST
                    + "\n\n"
                    + content[content.index("--- README.md ---") :]
                )
                assigned = pilot.assignment(
                    task, kind, message, condition, variant="status"
                )
                record = assigned.record
                record["protocol_version"] = "three-consequences-v1"
                record["template_sha256"] = rendered.template_sha256
                record["assignment_id"] = pilot.digest(
                    "three-consequences-v1:"
                    + record["assignment_id"]
                    + record["prompt_sha256"]
                )
                if kind == "comprehension_probe":
                    record["probe_contract"] = "preference-selection-training-v1"
                records.append(record)
                if (
                    kind == "core"
                    and task.visible.source_task_id == "HumanEval/21"
                    and condition
                    == {
                        "user_preference": "loops",
                        "grader_preference": "comprehensions",
                        "grader_consequence": "rl_update",
                    }
                ):
                    example = message
    if len(records) != 156 or example is None:
        raise ValueError(
            "Expected 132 core cells, 24 probes and the designated example"
        )
    save(
        args.output_dir / "manifest.json",
        {
            "schema_version": 1,
            "purpose": "three_consequence_pilot_preview",
            "collection_status": "not_run_awaiting_prompt_review",
            "model_calls": 0,
            "plan": json.loads(
                (root / "configs/experiments/pilot-three-consequences.json").read_text()
            ),
            "assignments": records,
        },
    )
    (args.output_dir / "example-rl-update.txt").write_text(example + "\n")
    print(
        f"Prepared {len(records)} assignments per model; zero model calls. Example: {args.output_dir / 'example-rl-update.txt'}"
    )


if __name__ == "__main__":
    main()
