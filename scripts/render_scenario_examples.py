"""Write exact draft scenario/probe previews without models, caches, or execution."""

import argparse
import hashlib
import json
from pathlib import Path

from grader_power.datasets.fixture_builder import synthetic_task
from grader_power.scenario_extension import (
    SCENARIOS,
    VERSION,
    render_scenario_block,
    render_scenario_probe,
)
from grader_power.schemas import RenderTask

ROOT = Path(__file__).resolve().parents[1]


def diagnostic_tasks() -> tuple[RenderTask, ...]:
    """Read only the two already-exposed diagnostic tasks' visible fixture files.

    Source membership is used for display, not changed or reapproved. Verify the
    existing visible hashes without loading EvalPlus caches or reference bundles.
    """
    selected = ("HumanEval/21", "Mbpp/778")
    records = {
        row["source_task_id"]: row
        for line in (ROOT / "manifests/tasks_pilot.jsonl").read_text().splitlines()
        if (row := json.loads(line))["source_task_id"] in selected
    }
    tasks = []
    for source_id in selected:
        record = records[source_id]
        # Known paths prevent the manifest from expanding the file-read boundary.
        derived = source_id.lower().replace("/", "-") + "-pilot"
        fixture = ROOT / "fixtures/repos" / derived
        stub = (fixture / "src/task.py").read_text(encoding="utf-8")
        readme = (fixture / "README.md").read_text(encoding="utf-8")
        for text, field in ((stub, "visible_stub_sha256"), (readme, "readme_sha256")):
            if hashlib.sha256(text.encode("utf-8")).hexdigest() != record[field]:
                raise ValueError(f"Visible fixture hash mismatch: {source_id} {field}")
        tasks.append(
            RenderTask(
                source_id, derived, record["family"], "src/task.py", stub, readme
            )
        )
    return tuple(tasks)


def write_examples(directory: Path, tasks: tuple[RenderTask, ...]) -> None:
    """Create a new audit preview directory; refuse to replace earlier artifacts."""
    records: list[dict[str, object]] = []
    sections = [
        "# Draft scenario extension: exact rendered examples\n",
        (
            "Not reviewed or frozen. Audit headings and answer keys are not model-visible. "
            "Mode B materializes only the listed visible files; it uses the coding/probe "
            "request separately. No model calls or actual training.\n"
        ),
    ]
    for task in tasks:
        for scenario in SCENARIOS:
            for row in render_scenario_block(task, scenario=scenario):
                probe = render_scenario_probe(row)
                records.append({"coding": row.to_record(), "probe": probe.to_record()})
                sections.extend(
                    [
                        f"## {task.source_task_id} / {row.condition_id}\n",
                        (
                            f"Coding SHA-256: `{row.prompt_sha256}`\n\n"
                            f"Probe SHA-256: `{probe.prompt_sha256}`\n\n"
                            f"Fixture SHA-256: `{row.fixture_sha256}`\n"
                        ),
                        "Visible file paths: "
                        + ", ".join(f"`{file.path}`" for file in row.visible_files)
                        + "\n",
                        "### Exact coding user message\n\n````text\n"
                        + row.messages[0].content
                        + "\n````\n",
                        "### Exact separate probe user message\n\n````text\n"
                        + probe.messages[0].content
                        + "\n````\n",
                        "### Evaluator-only answer key\n\n```json\n"
                        + json.dumps(probe.expected.model_dump(), indent=2)
                        + "\n```\n",
                    ]
                )
    # Finish all validation/rendering before creating a new artifact directory.
    encoded = (
        json.dumps(
            {
                "definition_version": VERSION,
                "wording_status": "draft_requires_human_review",
                "model_calls": 0,
                "examples": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "examples.json").write_text(encoded, encoding="utf-8")
    (directory / "examples.md").write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    """Render the exposed diagnostic pair, or a synthetic fixture for offline QA."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    tasks = (synthetic_task(),) if args.synthetic else diagnostic_tasks()
    write_examples(args.output_dir, tasks)
    print(
        f"Wrote {len(tasks) * 6} coding contexts and separate probes to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
