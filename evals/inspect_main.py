"""Frozen main-study preparation, bounded collection, blinded judging and analysis.

Default operation is offline. Main inference requires --generate and explicit
user authorization; preparing the manifest is not authorization to collect.
"""

import argparse
import asyncio
import fcntl
import json
import os
import random
import re
import xml.etree.ElementTree as ET
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
from grader_power.code_extract import extract_code
from grader_power.consequence_extension import render_consequence_block
from grader_power.inference import load_access_profiles
from grader_power.main_analysis import LABELS, PROFILES, analyze_rows
from grader_power.main_tasks import main_tasks, source_contract, split_tasks
from grader_power.pilot_tasks import PilotTask, require_pilot_verification
from grader_power.style_judge import JudgeInput, parse_judgment, render_judge_input

PLAN = "configs/experiments/main-32.json"


def freeze(path: Path, value: Any) -> None:
    """Create an artifact exclusively, or verify its existing exact JSON value."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Frozen artifact differs: {path.name}")
    else:
        save(path, value)


def build_assignments(
    tasks: tuple[PilotTask, ...], repetitions: int, seed: int
) -> tuple[pilot.Assignment, ...]:
    """Interleave tasks/conditions within each repetition; never insert replicate IDs in prompts."""
    if isinstance(repetitions, bool) or repetitions < 1 or not tasks:
        raise ValueError("Tasks and positive repetitions are required")
    result = []
    rng = random.Random(seed)
    rendered_tasks = [(t, render_consequence_block(t.visible)) for t in tasks]
    for repetition in range(repetitions):
        block = []
        for task, rendered in rendered_tasks:
            for cell in rendered:
                a = pilot.assignment(
                    task,
                    "core",
                    cell.messages[0].content + "\n\n" + pilot.OUTPUT_CONTRACT,
                    cell.condition.model_dump(),
                    variant="status",
                )
                record = dict(a.record, repetition=repetition)
                record["assignment_id"] = digest(
                    "main-v1:"
                    + a.record["assignment_id"]
                    + ":"
                    + record["prompt_sha256"]
                    + ":"
                    + str(repetition)
                )
                block.append(pilot.Assignment(record, task))
        rng.shuffle(block)
        result.extend(block)
    if (
        len({a.record["assignment_id"] for a in result})
        != len(tasks) * 12 * repetitions
    ):
        raise ValueError("Duplicate main assignment identity")
    return tuple(result)


def main_config(root: Path, name: str) -> GenerateConfig:
    """Use the tested GPT-OSS sampling and the unchanged Sonnet provider settings."""
    config = generation_config(load_access_profiles(root / "configs/models.yaml")[name])
    if name == "gpt-oss-120b":
        config.temperature = 0.7
    config.cache = False
    return config


def verify_integration(path: Path) -> None:
    """Reject skipped, failing or incomplete sandbox reports, not merely nonzero exits."""
    cases = ET.parse(path).getroot().findall(".//testcase")
    if len(cases) != 9 or any(list(case) for case in cases):
        raise ValueError("Require nine passing, non-skipped Docker integration cases")


def prepare(
    root: Path, run_id: str, image: str, report: Path
) -> tuple[Path, tuple[pilot.Assignment, ...]]:
    """Freeze source-linked split, 4608 assignments/model, settings, and verification."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple name")
    verify_integration(report)
    oracle_evidence = require_pilot_verification(root, image)
    plan = json.loads((root / PLAN).read_text())
    sampling = json.loads((root / "manifests/sampling_check_run.json").read_text())
    prior = json.loads((root / "manifests/rl_condition_pilot_run.json").read_text())
    if (
        sampling["temperature"] != 0.7
        or not sampling["comprehension"]["passed"]
        or sampling["full_correct"] != 72
        or not prior["models"]["claude-sonnet"]["comprehension"]["passed"]
    ):
        raise ValueError(
            "Strong-model comprehension/sampling readiness is not satisfied"
        )
    if (
        plan["profiles"] != list(PROFILES)
        or plan["task_count"] != 32
        or plan["confirmation_task_count"] != 8
        or plan["repetitions"] != 12
        or plan["core_requests"] != 9216
        or plan["gpt_oss_temperature"] != 0.7
    ):
        raise ValueError("Plan differs from the agreed main design")
    tasks = main_tasks(root)
    split = split_tasks(tasks, plan["split_seed"])
    selected = tuple(
        t for t in tasks if t.visible.source_task_id in split["exploratory"]
    )
    assignments = build_assignments(selected, 12, plan["ordering_seed"])
    profiles = load_access_profiles(root / "configs/models.yaml")
    judge_config = main_config(root, "claude-sonnet")
    judge_config.parallel_tool_calls = False
    judge_config.max_connections = 1
    task_records = [
        {**t.provenance, "split": "exploratory" if t in selected else "confirmation"}
        for t in tasks
    ]
    freeze(root / "manifests/main_selection.json", split)
    task_text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in task_records)
    task_path = root / "manifests/tasks_main.jsonl"
    if task_path.exists():
        if task_path.read_text() != task_text:
            raise ValueError("Frozen main task records changed")
    else:
        with task_path.open("x") as stream:
            stream.write(task_text)
    source_paths = [
        *root.glob("src/grader_power/**/*.py"),
        *root.glob("prompts/**/*.*"),
        *root.glob("fixtures/reference_solutions/*.json"),
        root / PLAN,
        root / "evals/inspect_main.py",
        root / "evals/inspect_pilot.py",
        root / "evals/inspect_mode_a.py",
        root / "scripts/run_main.py",
        root / "configs/models.yaml",
        root / "uv.lock",
        root / "manifests/main_selection.json",
        task_path,
        root / "manifests/reference_execution_20261010.json",
        root / "manifests/sampling_check_run.json",
        root / "manifests/rl_condition_pilot_run.json",
    ]
    definition = {
        "schema_version": 1,
        "purpose": "mode_a_main_exploratory",
        "run_id": run_id,
        "plan": plan,
        "oracle_image": image,
        "verification_sha256": digest(json_text(oracle_evidence)),
        "integration_report": {
            "path": str(report.relative_to(root)),
            "sha256": digest(report.read_bytes()),
        },
        "split": split,
        "tasks": task_records,
        "assignments": [a.record for a in assignments],
        "profiles": {
            name: {
                "model": profiles[name].model,
                "revision": profiles[name].revision,
                "generation": main_config(root, name).model_dump(
                    mode="json", exclude_none=True
                ),
            }
            for name in PROFILES
        },
        "judge": {
            "profile": "claude-sonnet",
            "model": profiles["claude-sonnet"].model,
            "generation": judge_config.model_dump(mode="json", exclude_none=True),
            "system_prompt": (root / "prompts/style_judge/system.txt").read_text(),
        },
        "code_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted(source_paths)
            if p.is_file()
        },
        "authorization": "Preparation only; live phases require separate user authorization and --generate.",
    }
    cohort = root / "runs/main" / run_id
    freeze(cohort / "manifest.json", definition)
    manifest_hash = digest((cohort / "manifest.json").read_bytes())
    for profile in PROFILES:
        if not (cohort / profile / "results.jsonl").exists():
            pilot.export_results(
                cohort / profile,
                assignments,
                profile,
                manifest_hash,
                purpose=definition["purpose"],
            )
    return cohort, assignments


def load_run(
    root: Path, run_id: str
) -> tuple[Path, tuple[pilot.Assignment, ...], dict[str, Any]]:
    """Reject source/settings/task drift before any credential resolution or analysis."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("Invalid run ID")
    cohort = root / "runs/main" / run_id
    definition = json.loads((cohort / "manifest.json").read_text())
    for name, expected in definition["code_sha256"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    report = root / definition["integration_report"]["path"]
    if digest(report.read_bytes()) != definition["integration_report"]["sha256"]:
        raise ValueError("Sandbox verification report changed")
    verify_integration(report)
    evidence = require_pilot_verification(root, definition["oracle_image"])
    if digest(json_text(evidence)) != definition["verification_sha256"]:
        raise ValueError("Oracle verification changed")
    tasks = main_tasks(root)
    split = split_tasks(tasks, definition["plan"]["split_seed"])
    if split != definition["split"]:
        raise ValueError("Main split changed")
    selected = tuple(
        t for t in tasks if t.visible.source_task_id in split["exploratory"]
    )
    assignments = build_assignments(selected, 12, definition["plan"]["ordering_seed"])
    if [a.record for a in assignments] != definition["assignments"]:
        raise ValueError("Main prompts/assignments changed")
    return cohort, assignments, definition


def packet_from_output(raw: dict[str, Any], task: PilotTask) -> JudgeInput | None:
    """Pass complete independent constraints, never experimental metadata, to the judge."""
    output = ModelOutput.model_validate(raw)
    extraction = extract_code(output.completion, task.region.entry_point)
    if extraction.status != "ok" or extraction.code is None:
        return None
    return JudgeInput(
        entry_point=task.region.entry_point,
        transformation=task.region.transformation,
        input_contract=source_contract(task.dataset.tasks[task.visible.source_task_id]),
        code=extraction.code,
    )


async def collect_judgments(
    root: Path,
    cohort: Path,
    assignments: tuple[pilot.Assignment, ...],
    definition: dict[str, Any],
    profile_name: str,
    limit: int,
) -> None:
    """Judge at most limit unstarted responses, retaining raw failures and abstentions."""
    if type(limit) is not int or not 1 <= limit <= 48 or profile_name not in PROFILES:
        raise ValueError(
            "Judge collection requires a known profile and 1 to 48 attempts"
        )
    directory = cohort / profile_name / "judge"
    directory.mkdir(exist_ok=True)
    lock = os.open(directory / ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pending = []
        for a in assignments:
            identity = a.record["assignment_id"]
            path = directory / identity
            raw = cohort / profile_name / "attempts" / identity / "raw.json"
            if (
                raw.exists()
                and not (path / "started.json").exists()
                and not (path / "result.json").exists()
            ):
                pending.append((a, raw, path))
        if not pending:
            return
        judge = load_access_profiles(root / "configs/models.yaml")[
            definition["judge"]["profile"]
        ]
        endpoint, key = judge.resolve_access(
            {**dotenv_values(root / ".env"), **os.environ}
        )
        config = GenerateConfig.model_validate(definition["judge"]["generation"])
        with pilot.zero_retry_clients():
            model = get_model(
                judge.model,
                base_url=endpoint,
                api_key=key,
                config=config,
                memoize=False,
            )
            freeze(directory / "retry-policy.json", pilot.resolved_retry_policy(model))
            for a, raw_path, path in pending[:limit]:
                raw = json.loads(raw_path.read_text())
                output = ModelOutput.model_validate(raw)
                raw_hash = digest(raw_path.read_bytes())
                packet = packet_from_output(raw, a.task)
                if (
                    not output.choices
                    or output.choices[0].stop_reason != "stop"
                    or packet is None
                ):
                    save(
                        path / "result.json",
                        {
                            "status": "invalid_generation",
                            "label": "invalid_or_unparseable",
                            "source_raw_sha256": raw_hash,
                        },
                    )
                    continue
                save(path / "packet.json", packet.model_dump())
                binding = {
                    "source_raw_sha256": raw_hash,
                    "packet_sha256": digest(json_text(packet.model_dump())),
                }
                save(
                    path / "started.json",
                    {"started_at": datetime.now(UTC).isoformat(), **binding},
                )
                result: dict[str, Any] = {
                    **binding,
                    "label": None,
                    "status": "judge_error",
                }
                try:
                    pilot.resolved_retry_policy(model)
                    response = await asyncio.wait_for(
                        model.generate(
                            [
                                ChatMessageSystem(
                                    content=definition["judge"]["system_prompt"]
                                ),
                                ChatMessageUser(content=render_judge_input(packet)),
                            ],
                            tools=[],
                            tool_choice="none",
                            cache=False,
                        ),
                        timeout=200,
                    )
                    payload = response.model_dump(mode="json")
                    if key in json_text(payload):
                        raise ValueError("Credential echo rejected")
                    save(path / "raw.json", payload)
                    if response.choices and response.choices[0].stop_reason == "stop":
                        judgment = parse_judgment(response.completion, packet)
                        result.update(
                            status=judgment.status,
                            label=judgment.label,
                            judgment=judgment.model_dump(mode="json"),
                        )
                    else:
                        result["status"] = "incomplete_judgment"
                except Exception as error:  # noqa: BLE001 -- retain failure classes without provider text/credentials
                    result["error_type"] = type(error).__name__
                save(path / "result.json", result)
    finally:
        os.close(lock)


def measured_rows(
    cohort: Path, assignments: tuple[pilot.Assignment, ...]
) -> list[dict[str, Any]]:
    """Read every assignment; validate judge bindings and optional blinded adjudications."""
    rows = []
    for profile in PROFILES:
        for a in assignments:
            identity = a.record["assignment_id"]
            path = cohort / profile / "attempts" / identity
            raw = pilot.read_optional(path / "raw.json")
            scoring = pilot.read_optional(path / "scoring.json")
            judge_path = cohort / profile / "judge" / identity
            judgment = pilot.read_optional(judge_path / "result.json")
            label = None
            if raw is not None:
                output = ModelOutput.model_validate(raw)
                packet = packet_from_output(raw, a.task)
                if (
                    not output.choices
                    or output.choices[0].stop_reason != "stop"
                    or packet is None
                ):
                    label = "invalid_or_unparseable"
                else:
                    if judgment is not None:
                        if judgment["source_raw_sha256"] != digest(
                            (path / "raw.json").read_bytes()
                        ) or judgment["packet_sha256"] != digest(
                            json_text(packet.model_dump())
                        ):
                            raise ValueError("Judge/source binding changed")
                        if judgment["status"] in {"classified", "abstain"}:
                            judge_raw = ModelOutput.model_validate(
                                json.loads((judge_path / "raw.json").read_text())
                            )
                            parsed = parse_judgment(judge_raw.completion, packet)
                            if parsed.label != judgment["label"]:
                                raise ValueError(
                                    "Saved judge label disagrees with raw judgment"
                                )
                            label = parsed.label
                    adjudication = pilot.read_optional(judge_path / "adjudication.json")
                    if adjudication is not None:
                        if (
                            adjudication.get("packet_sha256")
                            != digest(json_text(packet.model_dump()))
                            or adjudication.get("blinded") is not True
                            or adjudication.get("label") not in LABELS
                            or not adjudication.get("reviewer")
                            or not adjudication.get("rationale")
                        ):
                            raise ValueError(
                                "Adjudication needs blinded review, rationale, and exact packet binding"
                            )
                        label = adjudication["label"]
            rows.append(
                {
                    **{k: v for k, v in a.record.items() if k != "messages"},
                    "profile": profile,
                    "label": label,
                    "full_correct": (scoring or {})
                    .get("correctness", {})
                    .get("full_correct"),
                    "judge_status": (judgment or {}).get("status", "not_judged"),
                }
            )
    return rows


def main() -> None:
    """Prepare offline, or explicitly collect/judge one bounded batch after approval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--phase", choices=["prepare", "collect", "judge", "analyze"], default="prepare"
    )
    parser.add_argument("--oracle-image")
    parser.add_argument("--integration-report", type=Path)
    parser.add_argument("--profile", choices=PROFILES)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-attempts", type=int, default=48)
    parser.add_argument("--analysis-id", default="analysis-v1")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.phase == "prepare":
        if args.oracle_image is None or args.integration_report is None:
            parser.error("Preparation requires --oracle-image and --integration-report")
        cohort, assignments = prepare(
            root, args.run_id, args.oracle_image, args.integration_report.resolve()
        )
        print(
            json.dumps(
                {
                    "cohort": str(cohort.relative_to(root)),
                    "assignments_per_model": len(assignments),
                    "provider_calls": 0,
                }
            )
        )
        return
    cohort, assignments, definition = load_run(root, args.run_id)
    if args.phase == "analyze":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.analysis_id):
            parser.error("analysis-id must be a simple name")
        rows = measured_rows(cohort, assignments)
        result = analyze_rows(rows, definition["split"]["exploratory"], 12)
        result["provenance"] = {
            "manifest_sha256": digest((cohort / "manifest.json").read_bytes()),
            "measured_rows_sha256": digest(json_text(rows)),
            "analysis_source_sha256": digest(
                (root / "src/grader_power/main_analysis.py").read_bytes()
            ),
        }
        save(cohort / args.analysis_id / "measured-rows.json", rows)
        save(cohort / args.analysis_id / "analysis.json", result)
        print(
            json.dumps(
                {
                    "assigned": result["assigned"],
                    "missing_labels": result["missing_labels"],
                }
            )
        )
        return
    if not args.generate or not args.profile:
        parser.error(
            "Live phases require --generate and --profile, after user authorization"
        )
    preflight = json.loads((cohort / "preflight.json").read_text())
    if preflight.get("passed") is not True or preflight.get(
        "manifest_sha256"
    ) != digest((cohort / "manifest.json").read_bytes()):
        parser.error("A passing preflight bound to this manifest is required")
    if not 1 <= args.max_new_attempts <= definition["plan"]["maximum_batch_size"]:
        parser.error("Batch must contain 1 to 48 new attempts")
    if args.phase == "collect":
        pilot.collect_profile(
            root,
            cohort,
            assignments,
            args.profile,
            definition["oracle_image"],
            {**dotenv_values(root / ".env"), **os.environ},
            phase="core",
            resume=args.resume,
            max_new_attempts=args.max_new_attempts,
            generation_override=main_config(root, args.profile),
            export_every=48,
        )
    else:
        asyncio.run(
            collect_judgments(
                root,
                cohort,
                assignments,
                definition,
                args.profile,
                args.max_new_attempts,
            )
        )


if __name__ == "__main__":
    main()
