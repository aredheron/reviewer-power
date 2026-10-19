"""Bounded early/late checkpoint diagnostics reusing the frozen pilot and oracle."""

import argparse
import asyncio
import fcntl
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import dotenv_values
from inspect_ai import Task, eval
from inspect_ai.model import (
    ContentReasoning,
    ContentText,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from evals import inspect_pilot as pilot
from evals.inspect_mode_a import digest, json_text, save
from grader_power.config import ServingSettings
from grader_power.pilot_tasks import load_pilot_tasks, require_pilot_verification

ROOT = Path(__file__).resolve().parents[1]
PLAN = "configs/experiments/olmo-diagnostics.json"


def separate_reasoning(output: ModelOutput) -> ModelOutput:
    """Split native reasoning without repairing final text or mutating the original.

    Save the provider response first. Missing or duplicate delimiters are explicit
    failures, including truncated reasoning; never score reasoning as final code.
    """
    if len(output.choices) != 1:
        raise ValueError("Expected one generated choice")
    content = output.choices[0].message.content
    if not isinstance(content, str) or content.count("</think>") != 1:
        raise ValueError("Expected exactly one native reasoning terminator")
    reasoning, final = content.split("</think>")
    reasoning = reasoning.removeprefix("<think>")
    if "<think>" in reasoning or "<think>" in final or not final.strip():
        raise ValueError("Ambiguous reasoning boundary or empty final answer")
    result = output.model_copy(deep=True)
    result.choices[0].message.content = [
        ContentReasoning(reasoning=reasoning),
        ContentText(text=final),
    ]
    result.completion = final
    return result


def normalized_probe(raw: str, condition: Mapping[str, Any]) -> dict[str, Any]:
    """Permit exactly one outer JSON fence without repairing keys or answer content."""
    match = re.fullmatch(r"\s*```(?:json)?\s*\n(.*?)\n```\s*", raw, re.DOTALL)
    return pilot.score_probe(match.group(1) if match else raw, condition)


def comprehension_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Evaluate the prospective gate over all 16 assigned probes, including failures."""
    probes = [row for row in rows if row["kind"] == "comprehension_probe"]
    correct = 0
    completed = 0
    for row in probes:
        raw = row.get("raw_output")
        if row.get("status") == "scored" and row.get("generation_complete") and raw:
            completed += 1
            # The client has separated native reasoning from final-answer content,
            # retaining the untouched provider response in a separate artifact.
            output = ModelOutput.model_validate(raw)
            correct += normalized_probe(output.completion, row["condition"])[
                "both_correct"
            ]
    return {
        "assigned": len(probes),
        "normally_completed": completed,
        "normalized_both_correct": correct,
        "passed": len(probes) == completed == 16 and correct >= 14,
    }


def prepare_run(
    root: Path, run_id: str, image: str
) -> tuple[Path, tuple[pilot.Assignment, ...], dict[str, Any]]:
    """Freeze both checkpoints and exact pilot prompts before any generation."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple name")
    if version("inspect-ai") != "0.3.277":
        raise RuntimeError("Revalidate Inspect provider and logging behavior")
    plan = json.loads((root / PLAN).read_text())
    tasks = load_pilot_tasks(root)
    verification = require_pilot_verification(root, image)
    assignments = tuple(
        a
        for a in pilot.build_assignments(tasks)
        if a.record["source_task_id"] in plan["task_ids"]
    )
    if (
        len(assignments) != plan["assignments_per_profile"]
        or len(assignments) * len(plan["profiles"])
        != plan["maximum_generation_requests"]
    ):
        raise ValueError("Diagnostic plan disagrees with assignment counts")
    config = GenerateConfig(**plan["generation"])
    if config.max_retries != 0 or plan["request_deadline_seconds"] <= 0:
        raise ValueError("Diagnostics require zero retries and a positive deadline")
    files = sorted(
        {
            *root.glob("src/grader_power/**/*.py"),
            *root.glob("prompts/**/*.*"),
            root / PLAN,
            root / "evals/inspect_checkpoint_diagnostics.py",
            root / "evals/inspect_pilot.py",
            root / "evals/inspect_mode_a.py",
            root / "scripts/run_checkpoint_diagnostics.py",
            root / "uv.lock",
            root / "manifests/tasks_pilot.jsonl",
            root / "manifests/pilot_selection.json",
        }
    )
    definition = {
        "schema_version": 1,
        "run_id": run_id,
        "purpose": plan["purpose"],
        "plan": plan,
        "oracle_image": image,
        "verification_sha256": digest(json_text(verification)),
        "task_provenance": [
            t.provenance for t in tasks if t.visible.source_task_id in plan["task_ids"]
        ],
        "assignments": [a.record for a in assignments],
        "code_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in files
            if p.is_file()
        },
    }
    cohort = root / "runs/checkpoint-diagnostics" / run_id
    manifest = cohort / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != definition:
            raise ValueError("Frozen diagnostics changed; use a new cohort")
    else:
        save(manifest, definition)
    for name in plan["profiles"]:
        directory = cohort / name
        if not (directory / "results.jsonl").exists():
            pilot.export_results(
                directory, assignments, name, digest(manifest.read_bytes())
            )
    return cohort, assignments, definition


def build_task(
    assignments: Sequence[pilot.Assignment],
    directory: Path,
    image: str,
    config: GenerateConfig,
    refresh: Any,
    deadline: float,
) -> Task:
    """Reuse pilot samples and full oracle scoring with a longer reasoning deadline."""
    base = pilot.build_task(assignments, directory, image, config, refresh)

    @solver
    def retain_generation() -> Solver:
        """Persist start and raw answer before scoring; never retry missing responses."""

        async def solve(state: TaskState, generate: Generate) -> TaskState:
            path = directory / "attempts" / str(state.sample_id)
            save(
                path / "started.json",
                {
                    "started_at": datetime.now(UTC).isoformat(),
                    "sdk_max_retries": 0,
                    "http_connect_retries": 0,
                },
            )
            started = time.monotonic()
            try:
                state = await asyncio.wait_for(
                    generate(state, tool_calls="none"), timeout=deadline
                )
            except Exception as error:  # noqa: BLE001 -- preserve categories without secrets
                save(
                    path / "failure.json",
                    {
                        "stage": "generation",
                        "error_type": getattr(
                            error, "provider_error_type", type(error).__name__
                        ),
                    },
                )
                state.metadata["generation_failed"] = True
                refresh()
                return state
            save(path / "provider-output.json", state.output.model_dump(mode="json"))
            try:
                state.output = separate_reasoning(state.output)
            except ValueError:
                save(path / "raw.json", state.output.model_dump(mode="json"))
                save(
                    path / "failure.json",
                    {
                        "stage": "reasoning_parse",
                        "error_type": "InvalidReasoningBoundary",
                    },
                )
                state.metadata["generation_failed"] = True
                refresh()
                return state
            save(path / "raw.json", state.output.model_dump(mode="json"))
            save(path / "generation.json", {"wall_seconds": time.monotonic() - started})
            return state

        return solve

    return Task(
        name="checkpoint_diagnostics",
        dataset=base.dataset,
        solver=retain_generation(),
        scorer=base.scorer,
        config=config,
    )


def run_profile(
    root: Path,
    run_id: str,
    profile: str,
    image: str,
    values: Mapping[str, str | None],
    *,
    phase: str,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Collect one gated phase under an exclusive lock and verified zero retries."""
    if phase not in ("probes", "controls", "core", "robustness"):
        raise ValueError("Unknown diagnostic phase")
    cohort, assignments, definition = prepare_run(root, run_id, image)
    plan = definition["plan"]
    if profile not in plan["profiles"]:
        raise ValueError("Unknown checkpoint profile")
    frozen_hash = digest((cohort / "manifest.json").read_bytes())
    if phase in plan["conditional_phases"]:
        gates = {
            name: comprehension_gate(
                pilot.export_results(cohort / name, assignments, name, frozen_hash)
            )
            for name in plan["profiles"]
        }
        if not all(gate["passed"] for gate in gates.values()):
            raise ValueError(
                "Both checkpoints must pass comprehension before behavior phases"
            )
    directory = cohort / profile
    lock = os.open(directory / ".collection.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if list((directory / "attempts").glob("*/started.json")) and not resume:
            raise ValueError("Existing attempts require --resume")
        pending = [
            a
            for a in assignments
            if a.record["kind"] in pilot.PHASE_KINDS[phase]
            and not (
                directory / "attempts" / a.record["assignment_id"] / "started.json"
            ).exists()
        ]

        def refresh() -> list[dict[str, Any]]:
            """Export the full assigned denominator without touching original evidence."""
            return pilot.export_results(directory, assignments, profile, frozen_hash)

        if not pending:
            return refresh()
        endpoint = values.get("VLLM_BASE_URL")
        key = values.get("VLLM_API_KEY")
        if not endpoint or not key:
            raise ValueError("Missing VLLM_BASE_URL or VLLM_API_KEY")
        endpoint = ServingSettings.check_endpoint(endpoint)
        session = (
            directory / "sessions" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        )
        save(
            session / "planned.json",
            {
                "phase": phase,
                "assignment_ids": [a.record["assignment_id"] for a in pending],
            },
        )
        try:
            with pilot.zero_retry_clients():
                config = GenerateConfig(**plan["generation"])
                model = get_model(
                    "vllm/" + profile,
                    base_url=endpoint,
                    api_key=key,
                    config=config,
                    memoize=False,
                    lazy_init=False,
                )
                save(session / "retry-policy.json", pilot.resolved_retry_policy(model))
                provider_generate = model.api.generate

                async def safe_generate(*args: Any, **kwargs: Any) -> Any:
                    """Redact provider failures before Inspect persists model events."""
                    try:
                        pilot.resolved_retry_policy(model)
                        result = await provider_generate(*args, **kwargs)
                    except Exception as error:  # noqa: BLE001 -- API messages may expose credentials
                        raise pilot.ProviderRequestError(type(error).__name__) from None
                    output = result[0] if isinstance(result, tuple) else result
                    if isinstance(output, Exception):
                        raise pilot.ProviderRequestError(type(output).__name__)
                    return result

                with (
                    patch.object(model.api, "generate", new=safe_generate),
                    patch("inspect_ai._eval.context.init_dotenv"),
                    patch("inspect_ai._eval.task.log.git_context", return_value=None),
                ):
                    logs = eval(
                        build_task(
                            pending,
                            directory,
                            image,
                            config,
                            refresh,
                            plan["request_deadline_seconds"],
                        ),
                        model=model,
                        log_dir=str(session / "inspect"),
                        log_format="eval",
                        display="none",
                        notification=False,
                        log_samples=True,
                        log_model_api=False,
                        log_refusals=False,
                        fail_on_error=False,
                        retry_on_error=0,
                        max_samples=1,
                        epochs=1,
                        score_on_error=False,
                    )
                save(session / "completed.json", {"status": logs[0].status})
        except Exception as error:  # noqa: BLE001 -- terminal redacted reporting boundary
            save(
                session / "failure.json",
                {
                    "error_type": getattr(
                        error, "provider_error_type", type(error).__name__
                    )
                },
            )
        finally:
            rows = refresh()
        return rows
    finally:
        os.close(lock)


def main() -> None:
    """Prepare offline by default; inference requires an explicit profile and phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-image", required=True)
    parser.add_argument("--profile", choices=("olmo-early", "olmo-late"))
    parser.add_argument("--phase", choices=("probes", "controls", "core", "robustness"))
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.generate:
        cohort, assignments, _ = prepare_run(ROOT, args.run_id, args.oracle_image)
        print(
            json.dumps(
                {
                    "cohort": str(cohort.relative_to(ROOT)),
                    "assignments_per_profile": len(assignments),
                }
            )
        )
        return
    if not args.profile or not args.phase:
        parser.error("--generate requires --profile and --phase")
    values = {**dotenv_values(ROOT / ".env"), **os.environ}
    rows = run_profile(
        ROOT,
        args.run_id,
        args.profile,
        args.oracle_image,
        values,
        phase=args.phase,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "profile": args.profile,
                "phase": args.phase,
                "gate": comprehension_gate(rows),
                "statuses": {
                    status: sum(r["status"] == status for r in rows)
                    for status in sorted({r["status"] for r in rows})
                },
            }
        )
    )
    if any(
        pilot.collection_issue(row)
        for row in rows
        if row["kind"] in pilot.PHASE_KINDS[args.phase]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
