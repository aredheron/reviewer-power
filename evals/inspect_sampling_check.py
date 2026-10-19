"""Run the authorized 96-request sampling diagnostic on two exposed pilot tasks."""

import argparse
import asyncio
import fcntl
import json
import os
import random
import re
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)

from evals import inspect_pilot as pilot
from evals.inspect_mode_a import digest, generation_config, json_text, save
from evals.inspect_rl_pilot import comprehension_gate, require_core_gate
from grader_power.code_extract import extract_code
from grader_power.consequence_extension import PROBE_REQUEST, render_consequence_block
from grader_power.inference import load_access_profiles
from grader_power.pilot_tasks import (
    PilotTask,
    load_pilot_tasks,
    require_pilot_verification,
)
from grader_power.style_judge import JudgeInput, parse_judgment, render_judge_input

PLAN = "configs/experiments/sampling-check.json"
PROFILE = "gpt-oss-120b"


def build_assignments(
    tasks: tuple[PilotTask, ...], seed: int
) -> tuple[pilot.Assignment, ...]:
    """Give repetitions distinct IDs with byte-identical prompts within each cell."""
    controls = [t for t in tasks if t.provenance.get("control_subset")]
    if len(controls) != 2:
        raise ValueError("Exactly two pilot control tasks are required")
    result = []
    for task in controls:
        for rendered in render_consequence_block(task.visible):
            content = rendered.messages[0].content
            for kind, repetitions in (("comprehension_probe", 1), ("core", 3)):
                message = (
                    PROBE_REQUEST
                    + "\n\n"
                    + content[content.index("--- README.md ---") :]
                    if kind == "comprehension_probe"
                    else content + "\n\n" + pilot.OUTPUT_CONTRACT
                )
                for repetition in range(repetitions):
                    assigned = pilot.assignment(
                        task,
                        kind,
                        message,
                        rendered.condition.model_dump(),
                        variant="status",
                    )
                    record = dict(assigned.record, repetition=repetition)
                    record["assignment_id"] = digest(
                        "sampling-check-v1:"
                        + assigned.record["assignment_id"]
                        + ":"
                        + str(repetition)
                        + ":"
                        + record["prompt_sha256"]
                    )
                    if kind == "comprehension_probe":
                        record["probe_contract"] = "preference-selection-training-v1"
                    result.append(pilot.Assignment(record, task))
    random.Random(seed).shuffle(result)
    if len({a.record["assignment_id"] for a in result}) != 96:
        raise ValueError("Expected 96 unique assignments")
    return tuple(result)


def sampling_config(root: Path) -> GenerateConfig:
    """Keep pilot generation settings, changing only the approved temperature."""
    config = generation_config(
        load_access_profiles(root / "configs/models.yaml")[PROFILE]
    )
    config.temperature = 0.7
    return config


def prepare_run(
    root: Path, run_id: str, image: str
) -> tuple[Path, tuple[pilot.Assignment, ...]]:
    """Verify references and freeze prompts, settings, judge rubric, and source hashes."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple name")
    evidence = require_pilot_verification(root, image)
    plan = json.loads((root / PLAN).read_text())
    if (
        plan["temperature"] != 0.7
        or plan["maximum_generation_requests"] != 96
        or plan["repetitions"] != 3
    ):
        raise ValueError("Unexpected sampling plan")
    tasks = load_pilot_tasks(root)
    assignments = build_assignments(tasks, plan["ordering_seed"])
    if {a.record["source_task_id"] for a in assignments} != set(plan["task_ids"]):
        raise ValueError("Only the two already exposed control tasks may be used")
    profiles = load_access_profiles(root / "configs/models.yaml")
    judge_config = generation_config(profiles["claude-sonnet"])
    judge_config.max_connections = 1
    judge_config.parallel_tool_calls = False
    files = [
        *root.glob("src/grader_power/**/*.py"),
        *root.glob("prompts/**/*.*"),
        root / PLAN,
        root / "evals/inspect_sampling_check.py",
        root / "evals/inspect_pilot.py",
        root / "evals/inspect_mode_a.py",
        root / "evals/inspect_rl_pilot.py",
        root / "scripts/run_sampling_check.py",
        root / "configs/models.yaml",
        root / "manifests/tasks_pilot.jsonl",
        root / "uv.lock",
    ]
    definition = {
        "schema_version": 1,
        "purpose": "mode_a_sampling_check",
        "run_id": run_id,
        "oracle_image": image,
        "plan": plan,
        "verification_sha256": digest(json_text(evidence)),
        "assignments": [a.record for a in assignments],
        "profiles": {
            PROFILE: {
                "model": profiles[PROFILE].model,
                "revision": profiles[PROFILE].revision,
                "generation": sampling_config(root).model_dump(
                    mode="json", exclude_none=True
                ),
            }
        },
        "judge": {
            "profile": "claude-sonnet",
            "model": profiles["claude-sonnet"].model,
            "generation": judge_config.model_dump(mode="json", exclude_none=True),
            "system_prompt": (root / "prompts/style_judge/system.txt").read_text(),
        },
        "code_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted(files)
            if p.is_file()
        },
    }
    cohort = root / "runs/sampling-check" / run_id
    manifest = cohort / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != definition:
            raise ValueError(
                "Frozen sampling cohort differs; do not modify started runs"
            )
    else:
        save(manifest, definition)
    pilot.export_results(
        cohort / PROFILE,
        assignments,
        PROFILE,
        digest(manifest.read_bytes()),
        purpose=definition["purpose"],
    )
    return cohort, assignments


def judge_packet(raw: dict[str, Any], task: PilotTask) -> JudgeInput | None:
    """Extract a blinded packet using only the original public task contract.

    Extraction failures are invalid responses, not judge abstentions. No requested
    style, condition, model identity, reference solution, or tests enter the packet.
    """
    output = ModelOutput.model_validate(raw)
    extracted = extract_code(output.completion, task.region.entry_point)
    if extracted.status != "ok" or extracted.code is None:
        return None
    source = task.dataset.tasks[task.visible.source_task_id]
    contract = source.get("contract", "")
    if not isinstance(contract, str):
        raise TypeError("The independently supplied source contract must be text")
    return JudgeInput(
        entry_point=task.region.entry_point,
        transformation=task.region.transformation,
        input_contract=source["prompt"]
        + "\n\nEvalPlus input preconditions:\n"
        + contract,
        code=extracted.code,
    )


async def judge_outputs(
    root: Path,
    cohort: Path,
    assignments: tuple[pilot.Assignment, ...],
    *,
    judge_directory: str = "judge",
) -> None:
    """Make one blinded Sonnet request per extractable core response, preserving failures.

    Started requests are never repeated, including after interruption. SDK and HTTP
    retries are disabled. A process lock prevents overlapping judge collections.
    """
    definition = json.loads((cohort / "manifest.json").read_text())
    profile = load_access_profiles(root / "configs/models.yaml")["claude-sonnet"]
    endpoint, key = profile.resolve_access(
        {**dotenv_values(root / ".env"), **os.environ}
    )
    config = GenerateConfig.model_validate(definition["judge"]["generation"])
    directory = cohort / judge_directory
    directory.mkdir(exist_ok=True)
    lock = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pilot.zero_retry_clients():
            model = get_model(
                profile.model,
                base_url=endpoint,
                api_key=key,
                config=config,
                memoize=False,
            )
            policy = pilot.resolved_retry_policy(model)
            if not (directory / "retry-policy.json").exists():
                save(directory / "retry-policy.json", policy)
            for assigned in assignments:
                if assigned.record["kind"] != "core":
                    continue
                identity = assigned.record["assignment_id"]
                path = directory / identity
                raw_path = cohort / PROFILE / "attempts" / identity / "raw.json"
                if (
                    not raw_path.exists()
                    or (path / "started.json").exists()
                    or (path / "result.json").exists()
                ):
                    continue
                raw = json.loads(raw_path.read_text())
                packet = judge_packet(raw, assigned.task)
                if packet is None:
                    save(
                        path / "result.json",
                        {
                            "status": "extraction_failed",
                            "label": "invalid_or_unparseable",
                        },
                    )
                    continue
                messages = [
                    ChatMessageSystem(content=definition["judge"]["system_prompt"]),
                    ChatMessageUser(content=render_judge_input(packet)),
                ]
                save(path / "packet.json", packet.model_dump())
                save(
                    path / "started.json",
                    {
                        "started_at": datetime.now(UTC).isoformat(),
                        "source_raw_sha256": digest(raw_path.read_bytes()),
                    },
                )
                result: dict[str, Any] = {"status": "provider_error", "label": None}
                try:
                    pilot.resolved_retry_policy(model)
                    output = await asyncio.wait_for(
                        model.generate(
                            messages, tools=[], tool_choice="none", cache=False
                        ),
                        timeout=200,
                    )
                    serialized = output.model_dump(mode="json")
                    if key in json_text(serialized):
                        raise ValueError("Credential echo rejected")
                    save(path / "raw.json", serialized)
                    if output.choices and output.choices[0].stop_reason == "stop":
                        judgment = parse_judgment(output.completion, packet)
                        result = {
                            "status": judgment.status,
                            "label": judgment.label,
                            "judgment": judgment.model_dump(mode="json"),
                        }
                    else:
                        result = {"status": "incomplete", "label": None}
                except Exception as error:  # noqa: BLE001 -- preserve errors without provider text or credentials
                    result = {
                        "status": "judge_error",
                        "label": None,
                        "error_type": type(error).__name__,
                    }
                save(path / "result.json", result)
                print(
                    json.dumps({"judge": identity[:12], "status": result["status"]}),
                    flush=True,
                )
    finally:
        os.close(lock)


def summarize(cohort: Path, *, judge_directory: str = "judge") -> dict[str, Any]:
    """Report coverage and exact-code/semantic diversity without a significance claim."""
    rows = [
        json.loads(s)
        for s in (cohort / PROFILE / "results.jsonl").read_text().splitlines()
    ]
    core = [r for r in rows if r["kind"] == "core"]
    cells: dict[str, list[dict[str, Any]]] = {}
    for row in core:
        condition = row["condition"]
        key = "|".join(
            [
                row["source_task_id"],
                condition["grader_consequence"],
                condition["user_preference"],
                condition["grader_preference"],
            ]
        )
        cells.setdefault(key, []).append(row)
    details = {}
    judge_counts: Counter[str] = Counter()
    for key, cell in sorted(cells.items()):
        codes = []
        labels = []
        for row in cell:
            path = cohort / judge_directory / row["assignment_id"]
            packet = path / "packet.json"
            if packet.exists():
                codes.append(digest(json.loads(packet.read_text())["code"]))
            result_path = path / "result.json"
            result = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"status": "not_judged", "label": None}
            )
            judge_counts[result["status"]] += 1
            labels.append(result["label"])
        details[key] = {
            "assigned": len(cell),
            "extracted": len(codes),
            "distinct_code": len(set(codes)),
            "labels": labels,
            "distinct_resolved_styles": len({x for x in labels if x is not None}),
        }
    return {
        "comprehension": comprehension_gate(rows),
        "core_assigned": len(core),
        "statuses": dict(Counter(r["status"] for r in core)),
        "stop_reasons": dict(Counter(r["stop_reason"] or "missing" for r in core)),
        "full_correct": sum(
            bool((r.get("scoring") or {}).get("correctness", {}).get("full_correct"))
            for r in core
        ),
        "ast_labels": dict(
            Counter(
                (r.get("scoring") or {}).get("style", {}).get("label") or "unresolved"
                for r in core
            )
        ),
        "judge_statuses": dict(judge_counts),
        "cells": details,
        "cells_with_code_variation": sum(
            v["distinct_code"] > 1 for v in details.values()
        ),
        "cells_with_style_variation": sum(
            v["distinct_resolved_styles"] > 1 for v in details.values()
        ),
    }


def correct_contracts(root: Path, run_id: str, image: str) -> None:
    """Append a uniform rescore with full source preconditions; never regenerate code.

    Verify old source bytes (including the archived original runner), all current
    task/prompt identities, and original reference evidence before making calls.
    This amendment is specific to the incomplete-contract diagnostic cohort.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Invalid run ID")
    cohort = root / "runs/sampling-check" / run_id
    manifest = cohort / "manifest.json"
    definition = json.loads(manifest.read_text())
    evidence = require_pilot_verification(root, image)
    if definition["oracle_image"] != image or definition[
        "verification_sha256"
    ] != digest(json_text(evidence)):
        raise ValueError("Original oracle evidence changed")
    for name, expected in definition["code_sha256"].items():
        path = root / name
        if name == "evals/inspect_sampling_check.py":
            path = cohort / "source-at-collection/inspect_sampling_check.py"
        if digest(path.read_bytes()) != expected:
            raise ValueError("Original frozen source evidence changed")
    assignments = build_assignments(
        load_pilot_tasks(root), definition["plan"]["ordering_seed"]
    )
    if [a.record for a in assignments] != definition["assignments"]:
        raise ValueError("Original generation prompts or assignments changed")
    correction = {
        "reason": "Original judge inputs omitted the separate EvalPlus contract field; add all source preconditions uniformly for both tasks, preserving every original judgment.",
        "base_manifest_sha256": digest(manifest.read_bytes()),
        "runner_sha256": digest(Path(__file__).read_bytes()),
        "generation_requests": 0,
        "maximum_judge_requests": 72,
        "source_contracts": {
            a.task.visible.source_task_id: a.task.dataset.tasks[
                a.task.visible.source_task_id
            ].get("contract", "")
            for a in assignments
        },
    }
    directory = "judge-with-source-contract"
    path = cohort / directory / "amendment.json"
    if path.exists():
        if json.loads(path.read_text()) != correction:
            raise ValueError("Contract correction definition changed")
    else:
        save(path, correction)
    asyncio.run(judge_outputs(root, cohort, assignments, judge_directory=directory))
    pilot.replace_private(
        cohort / "summary-with-source-contract.json",
        json_text(summarize(cohort, judge_directory=directory)),
    )


def main() -> None:
    """Prepare offline, collect gated probes/core, judge, or summarize a frozen cohort."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-image", required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "prepare",
            "probes",
            "core",
            "judge",
            "summarize",
            "judge-contract-correction",
        ],
        default="prepare",
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "judge-contract-correction":
        if not args.generate:
            parser.error("Paid phases require --generate")
        correct_contracts(root, args.run_id, args.oracle_image)
        return
    cohort, assignments = prepare_run(root, args.run_id, args.oracle_image)
    if args.phase in {"probes", "core", "judge"}:
        if not args.generate:
            parser.error("Paid phases require --generate")
        if args.phase == "judge":
            asyncio.run(judge_outputs(root, cohort, assignments))
        else:
            if args.phase == "core":
                require_core_gate(
                    [
                        json.loads(s)
                        for s in (cohort / PROFILE / "results.jsonl")
                        .read_text()
                        .splitlines()
                    ]
                )
            pilot.collect_profile(
                root,
                cohort,
                assignments,
                PROFILE,
                args.oracle_image,
                {**dotenv_values(root / ".env"), **os.environ},
                phase=args.phase,
                resume=args.resume,
                generation_override=sampling_config(root),
            )
    summary = summarize(cohort)
    pilot.replace_private(cohort / "summary.json", json_text(summary))
    print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=2))


if __name__ == "__main__":
    main()
