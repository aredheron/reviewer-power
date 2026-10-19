"""Frozen GPT-only main extension with bounded blocks and restart-safe collection.

The shared adapters own provider calls, zero retries and verified Docker execution.
This controller owns the complete matrix, block ordering, spending and failure
guard. A resumed run only starts assignments and judgments with no prior intent.
"""

import argparse
import asyncio
import json
import os
import random
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

from dotenv import dotenv_values

from evals import inspect_agent
from evals import inspect_extension_checks as checks
from evals import inspect_extension_pilot as extension
from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals.inspect_mode_a import digest, json_text
from grader_power.collection_lock import collection_lock, isolate_collection_process
from grader_power.extension_analysis import (
    SCENARIOS,
    analyze_rows,
    factorial_keys,
    row_key,
)
from grader_power.main_tasks import main_tasks
from grader_power.pilot_tasks import PilotTask, require_pilot_verification
from grader_power.scenario_extension import render_scenario_block
from grader_power.spending import SpendingLedger, SpendingLimit, nanodollars, usage_cost

RUN = "gpt-extension-main-v1"
PROFILE = "gpt-oss-120b"
SEED = 20261013
PLAN = "configs/experiments/extension-gpt-main-draft.json"
OWNER = "gpt-extension-main-v1"


def directory(root: Path) -> Path:
    """Keep this experiment separate from both historical main and pilot records."""
    return root / "runs/extension-main" / RUN


def mode_plan(root: Path, mode: str) -> dict[str, Any]:
    """Reuse exactly the piloted limits and exploration/submission variant."""
    result = json.loads(
        (root / "configs/experiments/extension-mode-b-named-files-v2.json").read_text()
    )
    result.update(mode=mode, owner=OWNER, profiles=[PROFILE])
    if mode == "A":
        result.pop("exploration_variant")
        result["limits"].pop("episode")
    return result


def task_cells(
    root: Path, task: PilotTask, mode: str, repetition: int
) -> tuple[pilot.Assignment, ...]:
    """Render only experimental coding cells; no controls or comprehension probes."""
    plan = mode_plan(root, mode)
    rows = []
    for scenario in extension.NEW_SCENARIOS:
        for cell in render_scenario_block(task.visible, scenario=scenario):
            rows.append(
                extension._record(
                    task,
                    plan,
                    "core",
                    cell.messages[0].content + "\n\n" + pilot.OUTPUT_CONTRACT,
                    cell.condition.model_dump(),
                    [asdict(f) for f in cell.visible_files],
                    cell.coding_request + "\n\n" + pilot.OUTPUT_CONTRACT,
                    repetition_index=repetition,
                    template_sha256=cell.template_sha256,
                )
            )
    bases = extension.render_pilot_block(task.visible, preflight_variant="status")
    for cell in extension.render_consequence_block(task.visible):
        condition = cell.condition.model_dump()
        if condition["grader_consequence"] not in SCENARIOS[mode]:
            continue
        base = next(
            b
            for b in bases
            if b.condition.user_preference == condition["user_preference"]
            and b.condition.grader_preference == condition["grader_preference"]
            and b.condition.grader_consequence == "shadow"
        )
        files = [
            {
                "path": f.path,
                "content": cell.policy
                if f.path == ".ci/selection-policy.yaml"
                else f.content,
            }
            for f in base.visible_files
        ]
        content = cell.messages[0].content
        request = content[: content.index("--- README.md ---")].strip()
        rows.append(
            extension._record(
                task,
                plan,
                "core",
                content + "\n\n" + pilot.OUTPUT_CONTRACT,
                condition,
                files,
                request + "\n\n" + pilot.OUTPUT_CONTRACT,
                repetition_index=repetition,
                template_sha256=cell.template_sha256,
            )
        )
    result = []
    for row in rows:
        record = {**row.record, "workflow": RUN}
        record["assignment_id"] = digest(
            json_text({k: v for k, v in record.items() if k != "assignment_id"})
        )
        assigned = pilot.Assignment(record, row.task)
        if mode == "B":
            inspect_agent.validate_assignment(assigned)
        result.append(assigned)
    return tuple(result)


def blocks(
    root: Path, plan: dict[str, Any]
) -> list[tuple[str, tuple[pilot.Assignment, ...]]]:
    """Freeze balanced task/repetition blocks, alternating paired A/B blocks."""
    selection = json.loads((root / "manifests/main_selection.json").read_text())
    if (
        plan["source_task_ids"] != selection["exploratory"]
        or plan["locked_confirmation_task_ids"] != selection["confirmation"]
        or plan["repetitions_per_cell"] != 12
    ):
        raise ValueError(
            "Main scope must retain the approved 32 tasks, 12 repetitions and locked eight"
        )
    tasks = {t.visible.source_task_id: t for t in main_tasks(root)}
    rng = random.Random(SEED)
    output = []
    for repetition in range(plan["repetitions_per_cell"]):
        task_order = list(plan["source_task_ids"])
        rng.shuffle(task_order)
        for task_id in task_order:
            modes = ["A", "B"]
            rng.shuffle(modes)
            for mode in modes:
                rows = list(task_cells(root, tasks[task_id], mode, repetition))
                rng.shuffle(rows)
                name = f"{len(output):04d}-{mode}-{task_id.lower().replace('/', '-')}-r{repetition:02d}"
                output.append((name, tuple(rows)))
    all_rows = [r.record for _, rows in output for r in rows]
    if (
        len(all_rows) != 12288
        or len({r["assignment_id"] for r in all_rows}) != 12288
        or {row_key(r) for r in all_rows} != factorial_keys(plan["source_task_ids"], 12)
    ):
        raise ValueError("Main matrix does not cover the authorized factorial exactly")
    return output


def sources(root: Path) -> dict[str, str]:
    """Bind public executable inputs; credentials and local endpoints stay excluded."""
    result = extension.source_hashes(root, "B")
    for name in (
        PLAN,
        "evals/inspect_extension_main.py",
        "evals/inspect_extension_checks.py",
        "scripts/run_extension_main.py",
        "src/grader_power/extension_analysis.py",
        "src/grader_power/spending.py",
        "src/grader_power/collection_lock.py",
        "manifests/main_selection.json",
        "manifests/reference_execution_20261010.json",
        "configs/experiments/extension-mode-b-named-files-v2.json",
    ):
        result[name] = digest((root / name).read_bytes())
    return result


def prepare(root: Path, cap_usd: float, reports: list[Path]) -> dict[str, Any]:
    """Freeze runnable manifests after successful offline and sandbox verification."""
    if len(reports) != 3:
        raise ValueError("Require offline, oracle and Mode B integration reports")
    for report in reports:
        cases = ET.parse(report).getroot().findall(".//testcase")
        if not cases or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError("Require passing non-skipped verification reports")
    ready = root / "runs/extension-checks/gpt-oss-readiness-v1/final-verification.json"
    if json.loads(ready.read_text())["issues"]:
        raise ValueError("Readiness verification has unresolved issues")
    plan = json.loads((root / PLAN).read_text())
    plan.update(
        status="authorized_for_collection",
        main_collection_authorized=True,
        api_cap_usd=cap_usd,
        ordering_seed=SEED,
    )
    cap = nanodollars(cap_usd)
    if cap <= 0:
        raise ValueError("An approved positive main spending ceiling is required")
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    schedule = blocks(root, plan)
    code_hashes = sources(root)
    profiles = extension.access_profiles(root)
    judge = extension.generation(root, "claude-sonnet")
    judge.parallel_tool_calls, judge.max_connections = False, 1
    shared = {
        "purpose": "gpt_only_main_extension",
        "oracle_image": image,
        "profiles": {
            PROFILE: {
                "model": profiles[PROFILE].model,
                "generation": extension.generation(root, PROFILE).model_dump(
                    mode="json", exclude_none=True
                ),
            }
        },
        "judge": {
            "profile": "claude-sonnet",
            "model": profiles["claude-sonnet"].model,
            "generation": judge.model_dump(mode="json", exclude_none=True),
            "system_prompt": (root / "prompts/style_judge/system.txt").read_text(),
        },
        "code_sha256": code_hashes,
    }
    index = []
    base = directory(root)
    for name, rows in schedule:
        definition = {
            **shared,
            "plan": mode_plan(root, rows[0].record["mode"]),
            "assignments": [r.record for r in rows],
        }
        definition["plan"].update(
            api_cap_usd=cap_usd,
            purpose="main_extension",
            task_ids=[rows[0].record["source_task_id"]],
        )
        cohort = base / "blocks" / name
        original.freeze(cohort / "manifest.json", definition)
        (cohort / PROFILE).mkdir(exist_ok=True)
        index.append(
            {
                "name": name,
                "mode": rows[0].record["mode"],
                "manifest_sha256": digest(json_text(definition)),
                "assigned": len(rows),
                "maximum_nanodollars": sum(
                    sum(checks.bounds(r).values()) for r in rows
                ),
            }
        )
    frozen = {
        "plan": plan,
        "blocks": index,
        "code_sha256": code_hashes,
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
        "readiness_verification_sha256": digest(ready.read_bytes()),
        "authorization": "User authorized GPT-only main collection; API ceiling separately approved before prepare. Eight confirmation tasks remain locked.",
        "budget_policy": "Reserve each small block's generation and judging worst cases; settle from complete usage; retain unknown charges at their full bound. Global settled block amounts include unresolved reservations, so are conservative committed usage, not an invoice.",
        "error_policy": "No automatic retries. Pause after five failed generation/judge stages in the last 48 stages, any integrity failure, STOP, or insufficient reserved budget.",
    }
    original.freeze(base / "manifest.json", frozen)
    for name, expected in code_hashes.items():
        payload = (root / name).read_bytes()
        if digest(payload) != expected:
            raise ValueError("Source changed during preparation")
        path = base / "source-snapshot" / name
        if path.exists():
            if path.read_bytes() != payload:
                raise ValueError("Source snapshot mismatch")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
    return frozen


def load(root: Path) -> tuple[dict[str, Any], dict[str, tuple[pilot.Assignment, ...]]]:
    """Fail before credentials if sources, task evidence, prompts or settings drift."""
    base = directory(root)
    manifest = json.loads((base / "manifest.json").read_text())
    if sources(root) != manifest["code_sha256"]:
        raise ValueError("Frozen main sources changed")
    schedule = dict(blocks(root, manifest["plan"]))
    for item in manifest["blocks"]:
        frozen = json.loads(
            (base / "blocks" / item["name"] / "manifest.json").read_text()
        )
        if digest(json_text(frozen)) != item["manifest_sha256"] or frozen[
            "assignments"
        ] != [r.record for r in schedule[item["name"]]]:
            raise ValueError("Frozen block/assignment mismatch")
    return manifest, schedule


def measure(cohort: Path, rows: tuple[pilot.Assignment, ...]) -> list[dict[str, Any]]:
    """Validate source-bound judgments and report saved-file exposure and correctness."""
    mode = rows[0].record["mode"]
    judged = cohort / "final-file-evaluation" if mode == "B" else cohort
    with ExitStack() as stack:
        stack.enter_context(patch.object(original, "PROFILES", (PROFILE,)))
        if mode == "B":
            stack.enter_context(
                patch.object(
                    original, "packet_from_output", extension.packet_from_final_file
                )
            )
        measured = original.measured_rows(judged, rows)
    output = []
    for row, outcome in zip(rows, measured, strict=True):
        aid = row.record["assignment_id"]
        path = cohort / PROFILE / "attempts" / aid
        episode = pilot.read_optional(path / "episode.json")
        score = pilot.read_optional(path / "scoring.json")
        if mode == "B" and (judged / PROFILE / "attempts" / aid / "raw.json").exists():
            binding = json.loads(
                (judged / PROFILE / "attempts" / aid / "provenance.json").read_text()
            )
            expected = {
                p: digest((path / p).read_bytes()) if (path / p).exists() else None
                for p in ("final.py", "raw.json", "episode.json")
            }
            if binding["source_sha256"] != expected:
                raise ValueError("Final-file provenance mismatch")
        exposure = (episode or {}).get("exposure_summary", {})
        required = [
            f["path"]
            for f in row.record["visible_files"]
            if f["path"] == "AGENTS.md" or f["path"].startswith(".ci/")
        ]
        correctness = (score or {}).get("correctness", {})
        judge_path = judged / PROFILE / "judge" / aid
        judge_started = (judge_path / "started.json").exists()
        judge_result = (judge_path / "result.json").exists()
        output.append(
            {
                **{
                    k: row.record[k]
                    for k in (
                        "assignment_id",
                        "mode",
                        "scenario",
                        "source_task_id",
                        "repetition_index",
                        "condition",
                    )
                },
                "profile": PROFILE,
                "provider": "fireworks",
                "label": outcome["label"],
                "judge_status": outcome["judge_status"],
                "started": (path / "started.json").exists(),
                "raw_saved": (path / "raw.json").exists(),
                "scored": score is not None,
                "judge_started": judge_started,
                "judge_result": judge_result,
                "pending_judgment": (
                    judged / PROFILE / "attempts" / aid / "raw.json"
                ).exists()
                and not judge_started
                and not judge_result,
                "full_correct": correctness.get("full_correct")
                if correctness.get("status")
                not in ("oracle_error", "infrastructure_error", "not_evaluated")
                else None,
                "episode_status": (episode or {}).get("status"),
                "all_guidance_before_edit": all(
                    exposure.get(p, {}).get("full_before_first_explicit_edit") is True
                    for p in required
                )
                if mode == "B"
                else None,
                "all_guidance_before_final": all(
                    exposure.get(p, {}).get("full_before_finalization") is True
                    for p in required
                )
                if mode == "B"
                else None,
                "exposure": exposure,
            }
        )
    return output


def events(path: Path, record: dict[str, Any]) -> None:
    """Append durable nonsecret operational evidence without overwriting history."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps({"at": datetime.now(UTC).isoformat(), **record}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def guard(base: Path) -> None:
    """Keep request errors cumulative and stop repeated failures before another call."""
    if (base / "operations/STOP").exists():
        raise RuntimeError("User STOP is present")
    path = base / "operations/request-events.jsonl"
    recent = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    if sum(row["failed"] for row in recent[-48:]) >= 5:
        raise RuntimeError("Five failed request stages in the last 48; review required")


def record_stage(base: Path, identity: str, stage: str, failed: bool) -> None:
    """Count a completed/failed stage once, including after crash reconciliation."""
    path = base / "operations/request-events.jsonl"
    key = identity + "/" + stage
    previous = (
        {json.loads(line)["id"] for line in path.read_text().splitlines()}
        if path.exists()
        else set()
    )
    if key not in previous:
        events(path, {"id": key, "failed": failed})


def status(base: Path, manifest: dict[str, Any], state: str, **extra: Any) -> None:
    """Publish cheap aggregate progress from per-block measurements."""
    counts: dict[str, Counter[str]] = {m: Counter() for m in SCENARIOS}
    spent = outstanding = 0
    for item in manifest["blocks"]:
        cohort = base / "blocks" / item["name"]
        measured_path = cohort / "measured.json"
        rows: list[dict[str, Any]] = (
            json.loads(measured_path.read_text()) if measured_path.exists() else []
        )
        for row in rows:
            c = counts[row["mode"]]
            for key in ("started", "raw_saved", "scored"):
                c[key] += bool(row[key])
            c["resolved_labels"] += row["label"] is not None
            c["unresolved_started"] += row["started"] and row["label"] is None
            c["judgments"] += row["judge_result"]
            c["pending_judgments"] += row["pending_judgment"]
            c["unresolved_judgments"] += row["judge_started"] and row["label"] is None
        ledger = pilot.read_optional(cohort / "spending.json")
        if ledger:
            spent += ledger["spent"]
            outstanding += ledger["outstanding"]
    value = {
        "state": state,
        "updated_at": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        "assigned": 12288,
        "counts": {m: dict(v) for m, v in counts.items()},
        "recorded_usage_usd": spent / 1e9,
        "uncertain_or_reserved_usd": outstanding / 1e9,
        "cap_usd": manifest["plan"]["api_cap_usd"],
        **extra,
    }
    pilot.replace_private(base / "operations/status.json", json_text(value))


def collect(root: Path, *, max_blocks: int | None = None) -> None:
    """Collect and judge one reserved block at a time, preserving every started ID."""
    manifest, schedule = load(root)
    base = directory(root)
    ledger = SpendingLedger(
        base / "spending.json",
        cap=nanodollars(manifest["plan"]["api_cap_usd"]),
        scope=digest(json_text(manifest)),
    )
    with collection_lock(root, owner=OWNER, run_id=RUN):
        events(
            base / "operations/events.jsonl",
            {"event": "controller_started", "pid": os.getpid()},
        )
        status(base, manifest, "running")
        completed = 0
        try:
            for item in manifest["blocks"]:
                name = item["name"]
                cohort = base / "blocks" / name
                rows = schedule[name]
                if (cohort / "complete.json").exists():
                    continue
                guard(base)
                if sources(root) != manifest["code_sha256"]:
                    raise ValueError("Frozen sources changed during collection")
                ledger.reserve({name: item["maximum_nanodollars"]})
                block_ledger = SpendingLedger(
                    cohort / "spending.json",
                    cap=item["maximum_nanodollars"],
                    scope=item["manifest_sha256"],
                )
                frozen = json.loads((cohort / "manifest.json").read_text())
                if digest(json_text(frozen)) != item["manifest_sha256"]:
                    raise ValueError("Frozen block changed during collection")
                mode = item["mode"]
                with ExitStack() as stack:
                    stack.enter_context(
                        patch.object(
                            pilot,
                            "load_access_profiles",
                            return_value=extension.access_profiles(root),
                        )
                    )
                    if mode == "B":
                        stack.enter_context(
                            patch.object(pilot, "build_task", inspect_agent.build_task)
                        )
                        stack.enter_context(
                            patch.object(
                                original,
                                "packet_from_output",
                                extension.packet_from_final_file,
                            )
                        )
                    for row in rows:
                        guard(base)
                        aid = row.record["assignment_id"]
                        identity = name + "/" + aid
                        block_ledger.reserve(
                            {
                                aid + "/" + stage: bound
                                for stage, bound in checks.bounds(row).items()
                            }
                        )
                        path = cohort / PROFILE / "attempts" / aid
                        status(
                            base,
                            manifest,
                            "running",
                            block=name,
                            assignment_id=aid,
                            phase="generation",
                        )
                        if not (path / "started.json").exists():
                            pilot.collect_profile(
                                root,
                                cohort,
                                rows,
                                PROFILE,
                                frozen["oracle_image"],
                                {**dotenv_values(root / ".env"), **os.environ},
                                resume=True,
                                max_new_attempts=1,
                                generation_override=extension.generation(root, PROFILE),
                            )
                        if not (path / "started.json").exists():
                            raise RuntimeError(
                                "Generation failed before durable intent; inspect session records"
                            )
                        checks.settle_generation(path, row, block_ledger, aid)
                        if block_ledger.snapshot()["overrun"]:
                            raise SpendingLimit(
                                "Recorded generation exceeded its reservation"
                            )
                        if mode == "B":
                            episode = pilot.read_optional(path / "episode.json")
                            if (
                                episode is None
                                or not episode["cleanup_verified"]
                                or episode["status"]
                                in (
                                    "cleanup_unverified",
                                    "workspace_error",
                                    "sandbox_error",
                                )
                            ):
                                raise RuntimeError(
                                    "Mode B sandbox/episode integrity requires review"
                                )
                        score = pilot.read_optional(path / "scoring.json")
                        failure = pilot.read_optional(path / "failure.json")
                        if (failure or {}).get("stage") == "scoring" or (
                            score or {}
                        ).get("correctness", {}).get("status") in (
                            "oracle_error",
                            "infrastructure_error",
                        ):
                            raise RuntimeError("Scoring infrastructure requires review")
                        record_stage(
                            base,
                            identity,
                            "generation",
                            failure is not None or not (path / "raw.json").exists(),
                        )
                        guard(base)
                        judge_cohort = (
                            extension.prepare_final_file_judging(
                                cohort, (row,), PROFILE
                            )
                            if mode == "B"
                            else cohort
                        )
                        extension.enforce_judge_limit(
                            judge_cohort, (row,), PROFILE, frozen
                        )
                        status(
                            base,
                            manifest,
                            "running",
                            block=name,
                            assignment_id=aid,
                            phase="judging",
                        )
                        asyncio.run(
                            original.collect_judgments(
                                root, judge_cohort, (row,), frozen, PROFILE, 1
                            )
                        )
                        judge_path = judge_cohort / PROFILE / "judge" / aid
                        raw = pilot.read_optional(judge_path / "raw.json")
                        result = pilot.read_optional(judge_path / "result.json")
                        if raw is not None:
                            try:
                                cost = usage_cost(
                                    raw["usage"], input_rate=2, output_rate=10
                                )
                            except (ValueError, KeyError, TypeError):
                                pass  # Unknown billed usage keeps its complete bound.
                            else:
                                block_ledger.settle({aid + "/judge": cost})
                        elif not (judge_path / "started.json").exists():
                            block_ledger.settle({aid + "/judge": 0})
                        if block_ledger.snapshot()["overrun"]:
                            raise SpendingLimit(
                                "Recorded judging exceeded its reservation"
                            )
                        if (judge_path / "started.json").exists():
                            record_stage(
                                base,
                                identity,
                                "judge",
                                result is None
                                or result["status"] not in ("classified", "abstain"),
                            )
                        pilot.replace_private(
                            cohort / "measured.json", json_text(measure(cohort, rows))
                        )
                        status(base, manifest, "running", block=name)
                        guard(base)
                balance = block_ledger.snapshot()
                ledger.settle({name: balance["committed"]})
                original.freeze(
                    cohort / "complete.json",
                    {
                        "all_assignments_attempted": True,
                        "committed_nanodollars": balance["committed"],
                    },
                )
                events(
                    base / "operations/events.jsonl",
                    {"event": "block_completed", "block": name},
                )
                completed += 1
                if max_blocks is not None and completed >= max_blocks:
                    status(base, manifest, "bounded_launch_finished")
                    return
            status(base, manifest, "collection_and_judging_finished")
        except Exception as error:
            # Terminal boundary: report the class only; underlying raw failure
            # records are sanitized by the adapters and never replaced here.
            previous_status = pilot.read_optional(base / "operations/status.json") or {}
            events(
                base / "operations/events.jsonl",
                {"event": "paused", "error_type": type(error).__name__},
            )
            status(
                base,
                manifest,
                "paused",
                error_type=type(error).__name__,
                budget_pause=isinstance(error, SpendingLimit),
                block=previous_status.get("block"),
                assignment_id=previous_status.get("assignment_id"),
                phase=previous_status.get("phase"),
            )
            raise
    analyze(root)


def analyze(root: Path) -> dict[str, Any]:
    """Verify all rows and save additive offline analysis with provider provenance."""
    manifest, schedule = load(root)
    base = directory(root)
    if any(
        not (base / "blocks" / item["name"] / "complete.json").exists()
        for item in manifest["blocks"]
    ):
        raise ValueError(
            "Final analysis requires all blocks to finish; inspect operations/status.json for interim progress"
        )
    rows = [
        row
        for item in manifest["blocks"]
        for row in measure(base / "blocks" / item["name"], schedule[item["name"]])
    ]
    result = analyze_rows(rows, manifest["plan"]["source_task_ids"], 12)
    result["manifest_sha256"] = digest(json_text(manifest))
    result["measured_rows_sha256"] = digest(json_text(rows))
    original.freeze(base / "analysis-v1/measured-rows.json", rows)
    original.freeze(base / "analysis-v1/analysis.json", result)
    status(
        base,
        manifest,
        "finished_with_unresolved" if result["missing_labels"] else "complete",
        unresolved_labels=result["missing_labels"],
    )
    return result


def main() -> None:
    """Preparation/analysis are offline; generation requires an explicit CLI flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze", "verify"))
    parser.add_argument("--cap-usd", type=float)
    parser.add_argument("--reports", nargs="*", type=Path, default=[])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-blocks", type=int)
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "prepare":
        if args.cap_usd is None:
            parser.error("Provide the user-approved main API ceiling")
        frozen = prepare(root, args.cap_usd, [p.resolve() for p in args.reports])
        print(
            json.dumps(
                {
                    "assigned": 12288,
                    "blocks": len(frozen["blocks"]),
                    "cap_usd": args.cap_usd,
                    "paid_calls": 0,
                }
            )
        )
    elif args.action == "verify":
        frozen, _ = load(root)
        print(json.dumps({"verified_blocks": len(frozen["blocks"]), "paid_calls": 0}))
    elif args.action == "run":
        if not args.generate or args.max_blocks is not None and args.max_blocks < 1:
            parser.error("Require --generate; max-blocks must be positive if given")
        isolate_collection_process()
        collect(root, max_blocks=args.max_blocks)
    else:
        result = analyze(root)
        print(
            json.dumps(
                {
                    "assigned": result["assigned"],
                    "missing_labels": result["missing_labels"],
                }
            )
        )


if __name__ == "__main__":
    main()
