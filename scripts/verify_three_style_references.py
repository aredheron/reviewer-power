"""Verify all three-style references only inside the immutable Docker oracle."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from grader_power.datasets.oracle import OracleResult, evaluate_solution
from grader_power.main_tasks import main_tasks
from grader_power.pilot_tasks import (
    digest,
    load_pilot_tasks,
    require_pilot_verification,
    save,
)


def main() -> None:
    """Record each outcome exclusively; resume only unevaluated local references."""
    root = Path(__file__).resolve().parents[1]
    source = root / "fixtures/three_style/references.json"
    references = json.loads(source.read_text())
    base = root / "runs/three-style/three-style-v1/verification"
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    tasks = {
        t.visible.source_task_id: t
        for t in (*main_tasks(root), *load_pilot_tasks(root))
    }
    selection = json.loads((root / "manifests/main_selection.json").read_text())
    assert {r["task_id"] for r in references} == set(selection["exploratory"]) | {
        "HumanEval/87",
        "Mbpp/413",
    }
    assert not set(selection["confirmation"]) & {r["task_id"] for r in references}

    def check(item: tuple[str, str, str]) -> dict:
        task_id, style, code = item
        path = base / "reference-outcomes" / f"{task_id.replace('/', '-')}-{style}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["code_sha256"] != digest(code) or saved["image"] != image:
                raise ValueError("Reference evidence changed")
            return saved
        task = tasks[task_id]
        result = evaluate_solution(
            task.dataset,
            task.dataset_name,
            task_id,
            code,
            mode="full",
            image=image,
            verify_upstream=True,
        )
        record = {
            "task_id": task_id,
            "style": style,
            "code_sha256": digest(code),
            "image": image,
            "result": asdict(result),
        }
        save(path, record)
        print(task_id, style, result.status, flush=True)
        return record

    jobs = [
        (r["task_id"], style, code)
        for r in references
        for style, code in r["implementations"].items()
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(check, jobs))
    passed = len(outcomes) == 102 and all(
        OracleResult(**r["result"]).full_correct for r in outcomes
    )
    save(
        base / "references.json",
        {
            "passed": passed,
            "references_sha256": digest(source.read_bytes()),
            "oracle_image": image,
            "implementations": len(outcomes),
            "outcomes": outcomes,
        },
    )
    if not passed:
        raise RuntimeError(
            "Three-style reference gate failed; inspect preserved outcomes"
        )


if __name__ == "__main__":
    main()
