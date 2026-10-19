"""One real EvalPlus example across Mode A's eight conditions, using Inspect.

This bounded demonstration does not accept a task into the pilot. All execution
is delegated to the independently verified Docker oracle. Run through the CLI in
scripts/run_mode_a_example.py; credential values never enter saved configuration.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import dotenv_values
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ChatMessageUser, GenerateConfig, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.oracle import evaluate_solution
from grader_power.datasets.reference_solutions import ReferencePair, audit_references
from grader_power.inference import AccessProfile, load_access_profiles
from grader_power.render import render_pilot_block
from grader_power.schemas import RenderedCondition, RenderTask, ScoringRegion
from grader_power.scoring import OracleRequest, score_response

ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "HumanEval/7"
PURPOSE = "one_task_mode_a_demonstration_not_pilot_data"
VERIFICATION = "data/sandbox/mode-a-example-verification.json"
ORACLE_FILES = (
    "configs/sandbox/Dockerfile",
    "configs/sandbox/runner.py",
    "src/grader_power/datasets/oracle.py",
    "tests/integration/test_oracle_docker.py",
)


def digest(content: str | bytes) -> str:
    """Hash exact source or artifact bytes for provenance and stale-gate checks."""
    return hashlib.sha256(
        content.encode() if isinstance(content, str) else content
    ).hexdigest()


def json_text(value: Any) -> str:
    """Serialize deterministic, readable UTF-8 records without special encoders."""
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def save(path: Path, value: Any) -> None:
    """Write an exclusive private record; previous attempts cannot be overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as stream:
        stream.write(json_text(value))
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class Example:
    """Keep hidden oracle/reference data separate from the visible task projection."""

    dataset: LoadedDataset
    reference: ReferencePair
    visible: RenderTask
    region: ScoringRegion
    provenance: dict[str, Any]


def load_example(root: Path) -> Example:
    """Use the reviewed, hash-verified original statement without semantic edits."""
    pairs, datasets, records = audit_references(root)
    pair = next(pair for pair in pairs if pair.task_id == TASK_ID)
    record = next(row for row in records if row["task_id"] == TASK_ID)
    if record["static_status"] != "pass":
        raise ValueError("Example references must pass the independent static audit")
    dataset = next(dataset for dataset in datasets if TASK_ID in dataset.tasks)
    task = dataset.tasks[TASK_ID]
    visible = RenderTask(
        source_task_id=TASK_ID,
        derived_task_id="humaneval-7-example",
        family="filtering",
        stub_path="src/strings.py",
        stub=task["prompt"] + "    # TODO\n    raise NotImplementedError\n",
        readme="# String utilities\n\nUtilities for working with sequences of strings.\n",
    )
    fixture = root / "fixtures/repos" / visible.derived_task_id
    if (fixture / visible.stub_path).read_text() != visible.stub or (
        fixture / "README.md"
    ).read_text() != visible.readme:
        raise ValueError(
            "Visible example fixture differs from its unchanged source contract"
        )
    region = ScoringRegion(
        task_id=visible.derived_task_id,
        entry_point=pair.entry_point,
        transformation=pair.transformation,
        rule="returned_list_v1",
    )
    provenance = {
        "source_task_id": TASK_ID,
        "derived_task_id": visible.derived_task_id,
        "source": dataset.provenance,
        "original_prompt_sha256": digest(task["prompt"]),
        "visible_stub_sha256": digest(visible.stub),
        "scoring_region": asdict(region),
        "reference_sha256": {
            style: digest(getattr(pair.implementations, style))
            for style in ("loop", "comprehension")
        },
        "reference_bundle": record["reference_bundle"],
        "suite_counts": {
            suite: len(task[f"{suite}_input"]) for suite in ("base", "plus")
        },
        "human_review": "not_reviewed",
        "pilot_membership": "not_assigned",
        "adaptation": "Original prompt unchanged; append TODO and NotImplementedError only.",
    }
    return Example(dataset, pair, visible, region, provenance)


def oracle_hashes(root: Path) -> dict[str, str]:
    """Bind sandbox verification to the actual adapter, image recipe, and tests."""
    return {name: digest((root / name).read_bytes()) for name in ORACLE_FILES}


def verify_example(
    root: Path, example: Example, image: str, report: Path
) -> dict[str, Any]:
    """Require nine real isolation tests, then check both references and a bad answer.

    The independent official checker runs in the same restricted container for
    each full source suite. No canonical/reference/submission code runs locally.
    The report is evidence of a test run, not a cryptographic sandbox certificate.
    """
    cases = ET.parse(report).getroot().findall(".//testcase")
    if len(cases) != 9 or any(list(case) for case in cases):
        raise ValueError(
            "Expected all nine Docker integration cases to pass without skips"
        )
    checks = {}
    for name, code in {
        "loop": example.reference.implementations.loop,
        "comprehension": example.reference.implementations.comprehension,
        "known_wrong": "def filter_by_substring(strings, substring):\n    return []\n",
    }.items():
        outcome = evaluate_solution(
            example.dataset,
            "humaneval",
            TASK_ID,
            code,
            mode="full",
            image=image,
            verify_upstream=True,
        )
        checks[name] = {**asdict(outcome), "full_correct": outcome.full_correct}
    expected = {"loop": "pass", "comprehension": "pass", "known_wrong": "wrong_answer"}
    passed = all(checks[name]["status"] == status for name, status in expected.items())
    record = {
        "purpose": PURPOSE,
        "verified_at": datetime.now(UTC).isoformat(),
        "image": image,
        "oracle_files": oracle_hashes(root),
        "integration_report_sha256": digest(report.read_bytes()),
        "integration_cases": 9,
        "example": example.provenance,
        "checks": checks,
        "passed": passed,
    }
    save(root / VERIFICATION, record)
    if not passed:
        raise RuntimeError(
            "Example verification failed; retained the diagnostic record"
        )
    return record


def require_verification(root: Path, example: Example, image: str) -> dict[str, Any]:
    """Reject missing, failed, stale, or differently pinned verification evidence."""
    record = json.loads((root / VERIFICATION).read_text())
    if (
        record.get("passed") is not True
        or record.get("image") != image
        or record.get("oracle_files") != oracle_hashes(root)
        or record.get("example") != example.provenance
    ):
        raise ValueError(
            "Reverify the sandbox and example after an image/code/source change"
        )
    return record


def generation_config(profile: AccessProfile) -> GenerateConfig:
    """Use bounded requests; Claude sampling is provider-managed, not silently coerced."""
    config = profile.generation_config()
    if not profile.model.startswith("anthropic/"):
        config.temperature = 0
        config.top_p = 1
    return config


def build_task(
    example: Example,
    block: tuple[RenderedCondition, ...],
    directory: Path,
    image: str,
    config: GenerateConfig,
) -> Task:
    """Create Inspect samples from visible messages only; score outside their context."""

    @solver
    def retain_generation() -> Solver:
        """Persist intent before inference and exact raw output before scoring."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            path = directory / "attempts" / str(state.sample_id)
            started = time.monotonic()
            save(path / "started.json", {"started_at": datetime.now(UTC).isoformat()})
            state = await asyncio.wait_for(
                generate(state, tool_calls="none"), timeout=200
            )
            save(path / "raw.json", state.output.model_dump(mode="json"))
            save(path / "generation.json", {"wall_seconds": time.monotonic() - started})
            return state

        return solve

    @scorer(metrics=[])
    def full_score() -> Scorer:
        """Use condition-blind AST scoring and both full original test suites."""

        async def score(state: TaskState, target: Target) -> Score:
            request = OracleRequest(example.dataset, "humaneval", TASK_ID, image)
            result = await asyncio.to_thread(
                score_response, state.output.completion, example.region, oracle=request
            )
            payload = result.to_dict()
            save(
                directory / "attempts" / str(state.sample_id) / "scoring.json", payload
            )
            return Score(
                value=int(payload["correctness"].get("full_correct", False)),
                metadata=payload,
            )

        return score

    return Task(
        name="mode_a_real_task_example",
        dataset=[
            Sample(
                id=row.render_id,
                input=[ChatMessageUser(content=m.content) for m in row.messages],
                metadata={"condition_id": row.condition_id, "purpose": PURPOSE},
            )
            for row in block
        ],
        solver=retain_generation(),
        scorer=full_score(),
        config=config,
    )


def export_results(
    directory: Path,
    block: tuple[RenderedCondition, ...],
    profile_name: str,
    error_ids: set[str],
) -> list[dict[str, Any]]:
    """Retain all assigned conditions, including unstarted, interrupted, and failed ones."""
    rows = []
    for render in block:
        path = directory / "attempts" / render.render_id
        raw = (
            json.loads((path / "raw.json").read_text())
            if (path / "raw.json").exists()
            else None
        )
        scoring = (
            json.loads((path / "scoring.json").read_text())
            if (path / "scoring.json").exists()
            else None
        )
        status = (
            "scored"
            if scoring
            else "scoring_error"
            if raw and render.render_id in error_ids
            else "generation_error"
            if render.render_id in error_ids
            else ("interrupted" if (path / "started.json").exists() else "not_started")
        )
        stop_reason = (
            raw["choices"][0]["stop_reason"] if raw and raw["choices"] else None
        )
        row = {
            "profile": profile_name,
            "trial_id": profile_name + "-" + render.render_id,
            "attempt_number": 1,
            "retry_of": None,
            "purpose": PURPOSE,
            "condition": asdict(render.condition),
            "condition_id": render.condition_id,
            "prompt_sha256": render.prompt_sha256,
            "status": status,
            "stop_reason": stop_reason,
            "generation_complete": stop_reason == "stop",
            "raw_output": raw,
            "scoring": scoring,
        }
        save(path / "result.json", row)
        rows.append(row)
    with (directory / "results.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def run_profile(
    root: Path,
    run_id: str,
    profile_name: str,
    image: str,
    values: Mapping[str, str | None],
) -> list[dict[str, Any]]:
    """Run exactly one eight-cell block with no retries or reuse of an existing profile.

    A common immutable definition prevents changes between the four model runs.
    Per-profile directories are exclusive: this example runner deliberately does
    not resume generation or retry failed/billed requests. Raw responses and the
    initial assignment remain recoverable if the process is killed.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple name")
    if version("inspect-ai") != "0.3.277":
        raise RuntimeError("Revalidate Inspect logging/initialization for this version")
    example = load_example(root)
    verification = require_verification(root, example, image)
    profiles = load_access_profiles(root / "configs/models.yaml")
    profile = profiles[profile_name]
    endpoint, key = profile.resolve_access(values)
    block = render_pilot_block(
        example.visible, preflight_variant="status", seed=20261010
    )
    files = sorted(
        {
            *root.glob("src/grader_power/**/*.py"),
            *root.glob("evals/*.py"),
            *root.glob("prompts/**/*.*"),
            root / "uv.lock",
            root / "configs/models.yaml",
            root / "manifests/model_access.json",
        }
    )
    definition = {
        "purpose": PURPOSE,
        "run_id": run_id,
        "mode": "A",
        "example": example.provenance,
        "human_review": "not_reviewed",
        "pilot_membership": "not_assigned",
        "verification_sha256": digest(json_text(verification)),
        "oracle_image": image,
        "sampling_note": "Claude sampling is provider-managed; other profiles use temperature=0, top_p=1. No seeds sent to providers.",
        "profiles": {
            name: {
                "model": p.model,
                "revision": p.revision,
                "generation": generation_config(p).model_dump(
                    mode="json", exclude_none=True
                ),
            }
            for name, p in profiles.items()
        },
        "trials": [asdict(row) for row in block],
        "automatic_retries": 0,
        "file_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in files
            if p.is_file()
        },
    }
    cohort = root / "runs/mode-a-example" / run_id
    cohort.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest = cohort / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != json.loads(json_text(definition)):
            raise ValueError("Run definition changed; existing cohort is immutable")
    else:
        save(manifest, definition)
    directory = cohort / profile_name
    directory.mkdir(mode=0o700, exist_ok=False)
    config = generation_config(profile)
    error_ids: set[str] = set()
    try:
        model: Model = get_model(
            profile.model, base_url=endpoint, api_key=key, config=config, memoize=False
        )
        # Same pinned-version logging boundary as the offline demonstration.
        with (
            patch("inspect_ai._eval.context.init_dotenv"),
            patch("inspect_ai._eval.task.log.git_context", return_value=None),
        ):
            logs = eval(
                build_task(example, block, directory, image, config),
                model=model,
                log_dir=str(directory / "inspect"),
                log_format="eval",
                display="none",
                notification=False,
                log_samples=True,
                log_model_api=True,
                log_refusals=False,
                fail_on_error=False,
                retry_on_error=0,
                max_samples=1,
                epochs=1,
                score_on_error=False,
            )
        log = logs[0]
        error_ids = {
            str(sample.id) for sample in log.samples or [] if sample.error is not None
        }
        save(
            directory / "inspect.json",
            {
                "status": log.status,
                "log": str(Path(log.location).relative_to(directory)),
                "error_sample_ids": sorted(error_ids),
            },
        )
    except Exception as error:  # noqa: BLE001 -- safe reporting boundary for provider failures
        save(directory / "failure.json", {"error_type": type(error).__name__})
    finally:
        rows = export_results(directory, block, profile_name, error_ids)
    return rows


def main() -> None:
    """Separate offline preparation, isolated verification, and explicit paid generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=tuple(load_access_profiles(ROOT / "configs/models.yaml"))
    )
    parser.add_argument("--run-id")
    parser.add_argument("--oracle-image")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--integration-report", type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if args.verify and args.generate:
        parser.error("Verify first, then generate in a separate invocation")
    if args.verify:
        if not args.oracle_image or not args.integration_report:
            parser.error("--verify requires --oracle-image and --integration-report")
        record = verify_example(
            ROOT, load_example(ROOT), args.oracle_image, args.integration_report
        )
        print(json.dumps({"verified": record["passed"], "task": TASK_ID}))
    elif args.generate:
        if not args.oracle_image or not args.profile or not args.run_id:
            parser.error("--generate requires --oracle-image, --profile, and --run-id")
        values = {**dotenv_values(ROOT / ".env", interpolate=False), **os.environ}
        os.environ["PYTHON_DOTENV_DISABLED"] = "1"
        rows = run_profile(ROOT, args.run_id, args.profile, args.oracle_image, values)
        passed = sum(
            bool(row["scoring"] and row["scoring"]["correctness"].get("full_correct"))
            for row in rows
        )
        print(
            json.dumps(
                {
                    "profile": args.profile,
                    "conditions": len(rows),
                    "full_correct": passed,
                }
            )
        )
        raise SystemExit(
            0
            if all(
                row["status"] == "scored"
                and row["generation_complete"]
                and row["scoring"]["correctness"]["status"]
                not in ("not_evaluated", "infrastructure_error", "oracle_error")
                for row in rows
            )
            else 1
        )
    else:
        example = load_example(ROOT)
        print(
            json_text(
                {
                    "purpose": PURPOSE,
                    "example": example.provenance,
                    "conditions_per_model": 8,
                    "network_requests": 0,
                }
            )
        )


if __name__ == "__main__":
    main()
