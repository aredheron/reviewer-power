"""Audit new preference tasks and optionally verify references in isolated Docker."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from grader_power.datasets.oracle import OracleResult, evaluate_solution
from grader_power.pilot_tasks import digest, require_pilot_verification, save
from grader_power.preference_judge import JudgeInput, audit_style
from grader_power.preference_tasks import (
    Family,
    PreferenceTask,
    load_bundles,
    task_manifest,
    visible_task,
)


def freeze(path: Path, value: Any) -> None:
    """Resume identical artifacts, rejecting attempts to replace prior evidence."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(
                f"Frozen artifact changed; choose a fresh output directory: {path}"
            )
    else:
        save(path, value)


def main() -> None:
    """Prepare public task packets; verification executes only through the oracle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--workers", type=int, default=4, choices=range(1, 5))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundles, datasets = load_bundles(root)
    manifests = [task_manifest(bundle, datasets) for bundle in bundles]
    freeze(args.output / "tasks.json", manifests)
    audits = []
    for bundle in bundles:
        for task in bundle.tasks:
            visible = visible_task(task, datasets[task.dataset].tasks[task.task_id])
            for style, code in task.implementations.items():
                packet = JudgeInput(
                    family=bundle.family,
                    entry_point=task.entry_point,
                    original_parameters=visible["original_parameters"],
                    code=code,
                )
                audit = audit_style(packet)
                audits.append(
                    {
                        "family": bundle.family,
                        "task_id": task.task_id,
                        "style": style,
                        "code_sha256": digest(code),
                        "audit": audit.model_dump(mode="json"),
                    }
                )
                if audit.label != style + "_only":
                    raise ValueError(
                        f"Reference has unexpected style: {task.task_id} {style}"
                    )
    freeze(args.output / "style-audit.json", audits)
    print(
        "Static checks: 2 families,",
        sum(len(b.tasks) for b in bundles),
        "task pairs",
        flush=True,
    )
    if not args.verify:
        return
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    jobs = [
        (bundle.family, task, style, code)
        for bundle in bundles
        for task in bundle.tasks
        for style, code in task.implementations.items()
    ]

    def check(job: tuple[Family, PreferenceTask, str, str]) -> dict[str, Any]:
        """Cache only hash-matching complete outcomes, retaining failures verbatim."""
        family, task, style, code = job
        dataset = datasets[task.dataset]
        source_hash = dataset.provenance["sha256"]
        binding = {
            "family": family,
            "task_id": task.task_id,
            "style": style,
            "code_sha256": digest(code),
            "source_sha256": source_hash,
            "image": image,
        }
        key = digest(json.dumps(binding, sort_keys=True))
        path = root / "runs/style-preferences/reference-checks" / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["binding"] != binding:
                raise ValueError("Reference cache binding changed")
        else:
            outcome = evaluate_solution(
                dataset,
                task.dataset,
                task.task_id,
                code,
                mode="full",
                image=image,
                verify_upstream=True,
            )
            record = {"binding": binding, "result": asdict(outcome)}
            save(path, record)
        print(
            f"{family} {task.task_id} {style}: {record['result']['status']}", flush=True
        )
        return record

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(executor.map(check, jobs))
    passed = all(OracleResult(**row["result"]).full_correct for row in outcomes)
    result = {
        "passed": passed,
        "implementations": len(outcomes),
        "tasks_sha256": digest((args.output / "tasks.json").read_bytes()),
        "oracle_image": image,
        "outcomes": outcomes,
    }
    freeze(args.output / "reference-verification.json", result)
    print(
        f"Full base + plus verification: {sum(OracleResult(**row['result']).full_correct for row in outcomes)}/{len(outcomes)} passed",
        flush=True,
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
