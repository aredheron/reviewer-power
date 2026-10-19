"""Portable, sequential replication using frozen packets and existing measurement code.

Planning and analysis are offline. Collection is explicit, bounded by a request
allowance, and never replays a started request without a saved response. New runs
have new identities and do not modify the published observations.
"""

import asyncio
import copy
import fcntl
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)

from grader_power import (
    matched_review,
    matched_syntax,
    preference_judge,
    reviewer_observation,
    style_judge,
)
from grader_power.code_extract import extract_code
from grader_power.datasets.evalplus_loader import load_dataset
from grader_power.datasets.oracle import evaluate_solution
from grader_power.publication import digest, read_json, summarize

CONFIG = "configs/replication/presentation.json"
SANDBOX_FILES = (
    "configs/sandbox/Dockerfile",
    "configs/sandbox/runner.py",
    "src/grader_power/datasets/oracle.py",
    "tests/integration/test_oracle_docker.py",
)


class RequestLimit(RuntimeError):
    """The explicitly requested allowance for this invocation is exhausted."""


def encoded(value: Any) -> str:
    """Serialize stable JSON for identities and append-only artifacts."""
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    )


def save(path: Path, value: Any) -> None:
    """Create a private immutable artifact; never overwrite an attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as stream:
        stream.write(encoded(value))
        stream.flush()
        os.fsync(stream.fileno())


def run_path(root: Path, run_id: str) -> Path:
    """Restrict outputs to ignored local replication storage."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError(
            "run-id must contain only letters, digits, underscores, and hyphens"
        )
    return root / "runs/replications" / run_id


def packets(root: Path, study: str) -> list[dict[str, Any]]:
    """Verify exact published prompt packets before planning any requests."""
    config = read_json(root / CONFIG)
    if study not in config["studies"]:
        raise ValueError("Unknown study")
    spec = config["studies"][study]
    path = root / spec["packets"]
    if (
        not path.resolve().is_relative_to(root.resolve())
        or digest(path) != spec["sha256"]
    ):
        raise ValueError("Frozen packet checksum changed")
    result = read_json(path)
    if len(result) != spec["packet_count"] or len(
        {r["packet_id"] for r in result}
    ) != len(result):
        raise ValueError("Invalid frozen packet coverage")
    for row in result:
        identity = hashlib.sha256(
            encoded({k: v for k, v in row.items() if k != "packet_id"}).encode()
        ).hexdigest()
        if identity != row["packet_id"]:
            raise ValueError("Packet identity changed")
    return result


def scientific_bindings(root: Path) -> dict[str, str]:
    """Bind fresh collection to the exact implementation and blinded judge prompts."""
    names = [
        "src/grader_power/replication.py",
        "src/grader_power/style_judge.py",
        "src/grader_power/preference_judge.py",
        "src/grader_power/code_extract.py",
        "src/grader_power/matched_review.py",
        "src/grader_power/matched_syntax.py",
        "src/grader_power/reviewer_observation.py",
        "src/grader_power/datasets/oracle.py",
        "src/grader_power/datasets/evalplus_loader.py",
        "uv.lock",
    ]
    config = read_json(root / CONFIG)
    names.extend(j["prompt_file"] for j in config["judges"].values())
    return {name: digest(root / name) for name in sorted(set(names))}


def measured(run: Path) -> list[dict[str, Any]]:
    """Retain every planned assignment, including unstarted and unresolved outputs."""
    plan = read_json(run / "plan.json")
    if digest(run / "plan.json") != (run / "plan.sha256").read_text().strip():
        raise ValueError("Frozen replication plan changed")
    result = []
    for row in plan["assignments"]:
        path = run / "attempts" / row["assignment_id"] / "result.json"
        outcome = read_json(path) if path.exists() else {}
        result.append(
            {
                **row,
                "raw_saved": False,
                "label": None,
                "full_correct": None,
                "probe_correct": False,
                **outcome,
            }
        )
    return result


def pilot_gate(rows: list[dict[str, Any]], study: str) -> dict[str, Any]:
    """Apply balanced factual, correctness, and label gates, never adherence gates."""
    if not rows or any(r["phase"] != "pilot" or r["study"] != study for r in rows):
        raise ValueError("Require one complete pilot study")
    profiles = {r["profile"] for r in rows}
    if len(profiles) != 1:
        raise ValueError("A portable run contains one model")
    profile = next(iter(profiles))
    if study == "loops":
        adapted = [{**r, "study": "matched_review"} for r in rows]
        gate = next(
            g
            for g in matched_review.pilot_gate(adapted)["groups"]
            if g["profile"] == profile
        )
    elif study in ("naming", "quotes"):
        gate = next(
            g
            for g in matched_syntax.pilot_gate(rows)["groups"]
            if (g["profile"], g["study"]) == (profile, study)
        )
    else:
        code = [r for r in rows if r["kind"] == "core"]
        facts = [r for r in rows if r["kind"] == "comprehension_probe"]
        groups: dict[str, int] = defaultdict(int)
        for r in facts:
            groups[r["condition"]["arm"]] += r["probe_correct"] is True
        gate = {
            "saved": sum(r["raw_saved"] is True for r in rows),
            "coding_correct": sum(r["full_correct"] is True for r in code),
            "coding_resolved": sum(r["label"] is not None for r in code),
            "facts_per_arm": dict(groups),
        }
        gate["passed"] = (
            len(rows) == 64
            and len(code) == 32
            and len(facts) == 32
            and gate["saved"] == 64
            and gate["coding_correct"] >= 29
            and gate["coding_resolved"] >= 31
            and len(groups) == 4
            and min(groups.values()) >= 7
        )
    return {**gate, "behavior_used_to_gate": False}


def prepare(
    root: Path,
    run_id: str,
    study: str,
    profile: str,
    phase: str,
    pilot_run_id: str | None = None,
) -> Path:
    """Freeze a full balanced pilot or main matrix, with fresh per-run identities."""
    config = read_json(root / CONFIG)
    if profile not in config["profiles"] or phase not in ("pilot", "main"):
        raise ValueError("Unknown profile or phase")
    selected = [r for r in packets(root, study) if r["phase"] == phase]
    bindings = None
    if phase == "main":
        if pilot_run_id is None:
            raise ValueError(
                "Main planning requires --pilot-run-id with a passing pilot"
            )
        parent = run_path(root, pilot_run_id)
        pilot_plan = read_json(parent / "plan.json")
        if (
            pilot_plan["study"],
            pilot_plan["profile"],
            pilot_plan["config_sha256"],
        ) != (study, profile, digest(root / CONFIG)):
            raise ValueError("Pilot study, model, or settings differ")
        if pilot_plan["scientific_bindings"] != scientific_bindings(root):
            raise ValueError("Pilot scientific sources differ from current sources")
        pilot_rows = measured(parent)
        gate = pilot_gate(pilot_rows, study)
        if not gate["passed"]:
            raise ValueError("Pilot admission failed: " + json.dumps(gate))
        bindings = {
            "run_id": pilot_run_id,
            "plan_sha256": digest(parent / "plan.json"),
            "measured_sha256": hashlib.sha256(encoded(pilot_rows).encode()).hexdigest(),
            "gate": gate,
        }
    spec = config["studies"][study]
    repeats = (
        spec["pilot_repetitions"]
        if phase == "pilot"
        else spec["main_repetitions"][profile]
    )
    assignments = []
    for packet in selected:
        for repetition in range(repeats):
            row = {
                **copy.deepcopy(packet),
                "profile": profile,
                "repetition_index": repetition,
            }
            row["assignment_id"] = hashlib.sha256(
                encoded([run_id, profile, packet["packet_id"], repetition]).encode()
            ).hexdigest()
            assignments.append(row)
    random.Random(2026101801).shuffle(assignments)
    run = run_path(root, run_id)
    run.mkdir(parents=True, exist_ok=False)
    save(
        run / "plan.json",
        {
            "study": study,
            "profile": profile,
            "phase": phase,
            "config_sha256": digest(root / CONFIG),
            "scientific_bindings": scientific_bindings(root),
            "pilot": bindings,
            "assignments": assignments,
        },
    )
    (run / "plan.sha256").write_text(digest(run / "plan.json") + "\n")
    return run


def verify_sandbox(root: Path, image: str) -> None:
    """Run the nine isolation tests and bind successful evidence to this image/source."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require an immutable local Docker image ID")
    directory = root / "runs/replication-sandbox"
    directory.mkdir(parents=True, exist_ok=True)
    report = directory / "tests.xml"
    env = {
        **os.environ,
        "GRADER_POWER_DOCKER_TESTS": "1",
        "GRADER_POWER_ORACLE_IMAGE": image,
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/integration/test_oracle_docker.py",
            "-q",
            "--junitxml=" + str(report),
        ],
        cwd=root,
        env=env,
        check=True,
    )
    from evals.inspect_main import verify_integration

    verify_integration(report)
    evidence = {
        "image": image,
        "report_sha256": digest(report),
        "sources": {name: digest(root / name) for name in SANDBOX_FILES},
    }
    (directory / "verified.json").write_text(encoded(evidence))


def require_sandbox(root: Path, image: str) -> None:
    """Reject missing or stale host isolation evidence before any model request."""
    directory = root / "runs/replication-sandbox"
    expected = {
        "image": image,
        "report_sha256": digest(directory / "tests.xml"),
        "sources": {name: digest(root / name) for name in SANDBOX_FILES},
    }
    if read_json(directory / "verified.json") != expected:
        raise ValueError("Sandbox verification is absent or stale; run verify-sandbox")


class Requests:
    """Bound requests and preserve raw provider outputs before parsing or scoring."""

    def __init__(self, maximum: int) -> None:
        if type(maximum) is not int or maximum < 1:
            raise ValueError("max-requests must be a positive integer")
        self.remaining = maximum

    async def call(
        self, directory: Path, model: str, config: dict[str, Any], messages: list[Any]
    ) -> ModelOutput:
        """Never retry started/ambiguous requests; store safe error types only."""
        raw = directory / "raw.json"
        if raw.exists():
            return ModelOutput.model_validate(read_json(raw))
        if (directory / "started.json").exists():
            raise RuntimeError(
                "Started request has no saved response; inspect this attempt before further collection"
            )
        if self.remaining == 0:
            raise RequestLimit(
                "Request allowance exhausted; invoke collect again to continue unstarted work"
            )
        self.remaining -= 1
        resolved_config = {
            **config,
            "max_retries": 0,
            "max_connections": 1,
            "cache": False,
        }
        save(
            directory / "started.json",
            {
                "model": model,
                "generation": resolved_config,
                "started_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        started = time.monotonic()
        try:
            client = get_model(
                model,
                config=GenerateConfig(**resolved_config),
            )
            response = await client.generate(messages)
            save(raw, response.model_dump(mode="json"))
            save(
                directory / "completed.json",
                {
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "elapsed_seconds": time.monotonic() - started,
                    "response_sha256": digest(raw),
                },
            )
            return response
        except Exception as error:  # noqa: BLE001 -- provider boundary records a safe type and stops without retry.
            save(directory / "error.json", {"error_type": type(error).__name__})
            raise RuntimeError(
                "Provider request failed; raw exception text withheld; see error.json"
            ) from None


def judge_packet(
    row: dict[str, Any], code: str
) -> style_judge.JudgeInput | preference_judge.JudgeInput:
    """Expose source and neutral task metadata only to the measurement model."""
    if row["study"] in ("loops", "strong"):
        return style_judge.JudgeInput(
            entry_point=row["entry_point"],
            transformation=row["transformation"],
            input_contract=row["input_contract"],
            code=code,
        )
    return preference_judge.JudgeInput(
        family=row["study"],
        entry_point=row["entry_point"],
        original_parameters=row["original_parameters"],
        code=code,
    )


async def collect(root: Path, run: Path, image: str, max_requests: int) -> None:
    """Collect sequentially with existing blind judges and the isolated full oracle."""
    require_sandbox(root, image)
    plan = read_json(run / "plan.json")
    config = read_json(root / CONFIG)
    if plan["config_sha256"] != digest(root / CONFIG):
        raise ValueError("Replication settings changed after planning")
    packets(root, plan["study"])
    if plan["scientific_bindings"] != scientific_bindings(root):
        raise ValueError("Scientific sources or judge prompts changed after planning")
    rows = measured(run)
    if plan["pilot"] is not None:
        b = plan["pilot"]
        parent = run_path(root, b["run_id"])
        if (
            digest(parent / "plan.json") != b["plan_sha256"]
            or hashlib.sha256(encoded(measured(parent)).encode()).hexdigest()
            != b["measured_sha256"]
        ):
            raise ValueError("Pilot evidence changed after main planning")
    load_dotenv(root / ".env", override=False)
    for name in (
        config["profiles"][plan["profile"]]["api_key_env"],
        "ANTHROPIC_API_KEY",
    ):
        if not os.environ.get(name):
            raise ValueError("Missing credential environment variable: " + name)
    datasets = {
        name: load_dataset(name, root / "data/evalplus")
        for name in ("humaneval", "mbpp")
    }
    requests = Requests(max_requests)
    for row in rows:
        attempt = run / "attempts" / row["assignment_id"]
        if (attempt / "result.json").exists():
            continue
        profile = config["profiles"][row["profile"]]
        output = await requests.call(
            attempt / "generation",
            profile["model"],
            profile["generation"],
            [ChatMessageUser(content=row["messages"][0]["content"])],
        )
        result: dict[str, Any] = {
            "raw_saved": True,
            "label": None,
            "full_correct": False,
            "probe_correct": False,
        }
        stopped = bool(output.choices and output.choices[0].stop_reason == "stop")
        if row["kind"] == "comprehension_probe":
            scorer = (
                matched_review
                if row["study"] == "loops"
                else reviewer_observation
                if row["study"] == "strong"
                else matched_syntax
            )
            probe = scorer.score_probe(output.completion, row)
            result.update(
                probe=probe,
                probe_correct=stopped and probe["fence_only"]["all_correct"],
            )
            save(attempt / "result.json", result)
            continue
        extraction = extract_code(output.completion, row["entry_point"])
        if not (attempt / "correctness.json").exists():
            if extraction.status == "ok" and extraction.code is not None:
                outcome = await asyncio.to_thread(
                    evaluate_solution,
                    datasets[row["dataset"]],
                    row["dataset"],
                    row["source_task_id"],
                    extraction.code,
                    mode="full",
                    image=image,
                )
                correctness = {**asdict(outcome), "full_correct": outcome.full_correct}
            else:
                correctness = {
                    "status": "not_evaluated",
                    "full_correct": False,
                    "reason": "No parseable extracted source",
                }
            save(attempt / "correctness.json", correctness)
        correctness = read_json(attempt / "correctness.json")
        if correctness["status"] in ("infrastructure_error", "oracle_error"):
            raise RuntimeError(
                "Sandbox scoring failed; evidence retained, collection stopped"
            )
        result["full_correct"] = correctness["full_correct"]
        if extraction.status != "ok" or extraction.code is None or not stopped:
            result["label"] = (
                "invalid_or_unparseable"
                if row["study"] in ("loops", "strong")
                else "invalid"
            )
        else:
            packet = judge_packet(row, extraction.code)
            family = "loops" if row["study"] == "strong" else row["study"]
            spec = config["judges"][family]
            prompt = (
                style_judge.render_judge_input(packet)
                if isinstance(packet, style_judge.JudgeInput)
                else preference_judge.render_judge_input(packet)
            )
            judged = await requests.call(
                attempt / "judge",
                spec["model"],
                spec["generation"],
                [
                    ChatMessageSystem(content=(root / spec["prompt_file"]).read_text()),
                    ChatMessageUser(content=prompt),
                ],
            )
            try:
                if not judged.choices or judged.choices[0].stop_reason != "stop":
                    raise ValueError("Incomplete judge response")
                parsed = (
                    style_judge.parse_judgment(judged.completion, packet)
                    if isinstance(packet, style_judge.JudgeInput)
                    else preference_judge.parse_range_judgment(
                        judged.completion, packet, distinguish_no_evidence=True
                    )
                )
                result.update(
                    label=parsed.label, judgment=parsed.model_dump(mode="json")
                )
            except (ValueError, TypeError) as error:
                result.update(label=None, judge_error=type(error).__name__)
        save(attempt / "result.json", result)


def analyze(run: Path) -> dict[str, Any]:
    """Export all assigned rows and a descriptive table or pilot gate, with no inference calls."""
    plan = read_json(run / "plan.json")
    rows = measured(run)
    result = (
        pilot_gate(rows, plan["study"])
        if plan["phase"] == "pilot"
        else {
            "conditions": summarize(rows),
            "interpretation": "Fresh descriptive replication; no historical pooling or new significance claims.",
        }
    )
    (run / "measured-rows.json").write_text(encoded(rows))
    (run / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def locked_collect(root: Path, run: Path, image: str, maximum: int) -> None:
    """Prevent concurrent controllers from sending duplicate requests."""
    with (run / "collection.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        asyncio.run(collect(root, run, image, maximum))
