"""Source-frozen reviewer/reward study with pilots, rolling reservations and pacing."""

import argparse
import asyncio
import fcntl
import json
import os
import random
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from dotenv import dotenv_values

from evals import inspect_extension_checks as checks
from evals import inspect_extension_pilot as extension
from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals.inspect_framing import counts, guard, read_measured, stage_event
from evals.inspect_framing import measure as legacy_measure
from evals.inspect_mode_a import digest, json_text
from grader_power import three_style as study
from grader_power.adaptive_pacing import AdaptivePacer
from grader_power.main_tasks import main_tasks
from grader_power.pacing import WaitClock, append_event, paced_http, write_json
from grader_power.pilot_tasks import load_pilot_tasks, require_pilot_verification
from grader_power.spending import SpendingLedger, nanodollars, usage_cost
from grader_power.three_style_analysis import summarize
from grader_power.three_style_calibration import calibrate, verify_supplement
from grader_power.three_style_judge import parse_judgment
from grader_power.three_style_lock import collection_lock, isolate_collection_process

PROFILE = "gpt-oss-120b"
RUN = study.VERSION
SEED = 2026101602
CAP = study.CAP
BLOCK_SIZE = 12
INITIAL_WORKERS = 16
# The 16 GiB collection host exhausted disk headroom through swap at 28 workers.
# Keep this source-bound operational limit until host capacity is revalidated.
MAX_WORKERS = 16
PILOT_IDS = ("HumanEval/87", "Mbpp/413")
CALIBRATION_AMENDMENTS = {
    "user-approved-calibration-supplement-v1",
    "user-approved-judge-rubric-v2",
    "user-approved-factual-probe-v2",
}
PROBE_AMENDMENT = "user-approved-factual-probe-v2"


def directory(root: Path) -> Path:
    """Return the new study directory, separate from all historical data."""
    return root / "runs/three-style" / RUN


def sources(root: Path) -> dict[str, str]:
    """Bind scientific and operational code, settings and selection evidence."""
    result = extension.source_hashes(root, "A")
    for name in (
        "evals/inspect_three_style.py",
        "evals/inspect_framing.py",
        "src/grader_power/framing.py",
        "src/grader_power/three_style_analysis.py",
        "evals/inspect_extension_checks.py",
        "scripts/run_three_style.py",
        "src/grader_power/three_style.py",
        "src/grader_power/three_style_judge.py",
        "src/grader_power/three_style_calibration.py",
        "src/grader_power/three_style_lock.py",
        "prompts/style_judge/three-style-v1.txt",
        "fixtures/three_style/references.json",
        "fixtures/three_style/calibration.json",
        "scripts/verify_three_style_references.py",
        "src/grader_power/adaptive_pacing.py",
        "src/grader_power/pacing.py",
        "src/grader_power/spending.py",
        "src/grader_power/main_tasks.py",
        "manifests/main_selection.json",
        "manifests/reference_execution_20261010.json",
        "configs/experiments/extension-mode-b-named-files-v2.json",
        "configs/experiments/three-style-v1.json",
        "planning/three_style_protocol.md",
        "tests/test_three_style.py",
        "tests/integration/test_three_style_ownership.py",
    ):
        result[name] = digest((root / name).read_bytes())
    return result


def schedule(root: Path) -> list[tuple[str, tuple[pilot.Assignment, ...]]]:
    """Create deterministic balanced blocks; confirmation tasks never enter them."""
    selection = json.loads((root / "manifests/main_selection.json").read_text())
    rng = random.Random(SEED)
    output = []
    for phase, tasks, repeats in (
        (
            "pilot",
            [
                t
                for t in load_pilot_tasks(root)
                if t.visible.source_task_id in PILOT_IDS
            ],
            1,
        ),
        (
            "main",
            [
                t
                for t in main_tasks(root)
                if t.visible.source_task_id in selection["exploratory"]
            ],
            4,
        ),
    ):
        for repetition in range(repeats):
            rng.shuffle(tasks)
            for task in tasks:
                rows = list(study.assignments(root, task, repetition, phase=phase))
                if phase == "pilot":
                    rows += list(
                        study.assignments(
                            root, task, repetition, probe=True, phase=phase
                        )
                    )
                rng.shuffle(rows)
                for start in range(0, len(rows), BLOCK_SIZE):
                    name = f"{phase}-{len(output):04d}"
                    output.append((name, tuple(rows[start : start + BLOCK_SIZE])))
    main = [r.record for n, rows in output if n.startswith("main-") for r in rows]
    if (
        len(main) != 6912
        or len({r["assignment_id"] for r in main}) != 6912
        or {r["source_task_id"] for r in main} != set(selection["exploratory"])
        or set(selection["exploratory"]) & set(selection["confirmation"])
    ):
        raise ValueError("Main matrix differs from approved design")
    return output


def prepare(root: Path, reports: list[Path]) -> None:
    """Freeze all assignments and passing checks before the first paid request."""
    reports = [(root / report).resolve() for report in reports]
    if any(not report.is_relative_to(root.resolve()) for report in reports):
        raise ValueError("Verification reports must belong to this repository")
    if len(reports) != 2:
        raise ValueError("Require offline and live Docker verification reports")
    for report in reports:
        cases = ET.parse(report).getroot().findall(".//testcase")
        if not cases or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError("Verification must pass without skipped tests")
    base = directory(root)
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    code = sources(root)
    profiles = extension.access_profiles(root)
    judge = extension.generation(root, "claude-sonnet")
    judge.parallel_tool_calls, judge.max_connections = False, 1
    shared = {
        "purpose": RUN,
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
            "system_prompt": (
                root / "prompts/style_judge/three-style-v1.txt"
            ).read_text(),
        },
    }
    reference_report = json.loads((base / "verification/references.json").read_text())
    if reference_report.get("passed") is not True or reference_report.get(
        "references_sha256"
    ) != digest((root / "fixtures/three_style/references.json").read_bytes()):
        raise ValueError("Three-style reference verification must pass and match")
    index = []
    for name, rows in schedule(root):
        frozen = {
            **shared,
            "plan": {
                **study.configuration(root),
                "task_ids": sorted({r.record["source_task_id"] for r in rows}),
                "phase": rows[0].record["phase"],
                "repetitions": 4 if name.startswith("main-") else 1,
            },
            "assignments": [r.record for r in rows],
        }
        cohort = base / "blocks" / name
        original.freeze(cohort / "manifest.json", frozen)
        (cohort / PROFILE).mkdir(exist_ok=True)
        index.append(
            {
                "name": name,
                "assigned": len(rows),
                "manifest_sha256": digest(json_text(frozen)),
                "maximum_nanodollars": sum(
                    sum(checks.bounds(r).values()) for r in rows
                ),
            }
        )
    manifest = {
        "run": RUN,
        "cap_usd": CAP,
        "main_assignments": 6912,
        "pilot_assignments": 216,
        "blocks": index,
        "code_sha256": code,
        "authorization": study.AUTHORIZATION,
        "reference_report_sha256": digest(
            (base / "verification/references.json").read_bytes()
        ),
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
    }
    original.freeze(base / "manifest.json", manifest)
    for name, expected in code.items():
        payload = (root / name).read_bytes()
        if digest(payload) != expected:
            raise ValueError("Source changed during preparation")
        path = base / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Snapshot mismatch")
        path.write_bytes(payload)
    original.freeze(
        base / "operations/recovery-state.json",
        {
            "retry_blocked": False,
            "consecutive_failures": 0,
            "root_cause": None,
            "ladder_position": None,
            "next_check_due_at": None,
            "state": "prepared",
        },
    )


def execution_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Retain original accounting and add only the explicitly approved probe cohort."""
    amendment = current_amendment(directory(root))
    if amendment is None or amendment.get("kind") != PROBE_AMENDMENT:
        return manifest
    supplement = bound_json(directory(root), amendment["probe_supplement"])
    return {
        **manifest,
        "blocks": [*manifest["blocks"], *supplement["blocks"]],
        "active_pilot": "probe2",
        "additional_pilot_phases": ["probe2"],
        "probe2_assignments": 108,
    }


def bound_json(base: Path, binding: dict[str, Any]) -> Any:
    """Read immutable JSON evidence confined to this run and bound by byte hash."""
    path = base / binding["path"]
    if (
        not path.resolve().is_relative_to(base.resolve())
        or digest(path.read_bytes()) != binding["sha256"]
    ):
        raise ValueError("Supplemental evidence binding changed")
    return json.loads(path.read_text())


def revised_probe_record(record: dict[str, Any], request: str) -> dict[str, Any]:
    """Replace only the approved factual question; give the fresh request a new ID."""
    if (
        record["kind"] != "comprehension_probe"
        or record["phase"] != "pilot"
        or record["probe_request"] != study.PROBE_REQUEST
        or len(record["messages"]) != 1
        or not record["messages"][0]["content"].startswith(study.PROBE_REQUEST + "\n\n")
        or not request.startswith(study.PROBE_REQUEST + " ")
    ):
        raise ValueError("Fresh probes must derive from original factual questions")
    messages = [
        {
            **record["messages"][0],
            "content": request
            + record["messages"][0]["content"][len(study.PROBE_REQUEST) :],
        }
    ]
    result = {
        **record,
        "phase": "probe2",
        "probe_contract": "three-style-facts-v2",
        "probe_request": request,
        "messages": messages,
        "prompt_sha256": digest(json_text(messages)),
    }
    result["assignment_id"] = digest(
        json_text({k: v for k, v in result.items() if k != "assignment_id"})
    )
    return result


def revised_probe_blocks(
    root: Path, manifest: dict[str, Any], amendment: dict[str, Any]
) -> list[tuple[str, dict[str, Any]]]:
    """Derive the single approved 108-probe cohort from immutable original blocks."""
    base = directory(root)
    approval = bound_json(
        base,
        {"path": amendment["approval_path"], "sha256": amendment["approval_sha256"]},
    )
    request = approval["proposal"]["proposed_factual_request"]
    output = []
    for item in manifest["blocks"]:
        if not item["name"].startswith("pilot-"):
            continue
        frozen = json.loads(
            (base / "blocks" / item["name"] / "manifest.json").read_text()
        )
        if digest(json_text(frozen)) != item["manifest_sha256"]:
            raise ValueError("Original pilot manifest changed")
        rows = [
            revised_probe_record(r, request)
            for r in frozen["assignments"]
            if r["kind"] == "comprehension_probe"
        ]
        if rows:
            output.append(
                (
                    item["name"].replace("pilot-", "probe2-", 1),
                    {
                        **frozen,
                        "plan": {**frozen["plan"], "phase": "probe2"},
                        "assignments": rows,
                        "judge": {
                            **frozen["judge"],
                            "system_prompt": revised_judge_prompt(
                                root, base, amendment
                            ),
                        },
                    },
                )
            )
    identities = [r["assignment_id"] for _, b in output for r in b["assignments"]]
    if len(identities) != 108 or len(set(identities)) != 108:
        raise ValueError("Supplement must contain exactly 108 fresh factual probes")
    return output


def verify_probe_supplement(
    root: Path, manifest: dict[str, Any], amendment: dict[str, Any]
) -> None:
    """Verify question-only changes, retained pilot evidence and completed calibration."""
    base = directory(root)
    supplement = bound_json(base, amendment["probe_supplement"])
    expected = revised_probe_blocks(root, manifest, amendment)
    if supplement["phase"] != "probe2" or len(supplement["blocks"]) != len(expected):
        raise ValueError("Unapproved factual supplement structure")
    by_task = {t.visible.source_task_id: t for t in load_pilot_tasks(root)}
    for item, (name, frozen) in zip(supplement["blocks"], expected, strict=True):
        actual = json.loads((base / "blocks" / name / "manifest.json").read_text())
        maximum = sum(
            sum(
                checks.bounds(
                    pilot.Assignment(r, by_task[r["source_task_id"]])
                ).values()
            )
            for r in frozen["assignments"]
        )
        if actual != frozen or item != {
            "name": name,
            "assigned": len(frozen["assignments"]),
            "manifest_sha256": digest(json_text(frozen)),
            "maximum_nanodollars": maximum,
        }:
            raise ValueError(
                "Factual supplement changed more than the approved question"
            )
    retained = bound_json(base, amendment["retained_artifacts"])
    for name, expected_hash in retained.items():
        bound_json_or_bytes = base / name
        if (
            not bound_json_or_bytes.resolve().is_relative_to(base.resolve())
            or digest(bound_json_or_bytes.read_bytes()) != expected_hash
        ):
            raise ValueError("Retained pilot/calibration evidence changed")
    gate = bound_json(base, amendment["completed_calibration"])
    if gate.get("passed") is not True or gate.get("assigned") != 24:
        raise ValueError("Previously completed calibration must have passed")


def active_pilot_gate(base: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Use fresh facts plus retained coding, without rescoring the failed old probes."""
    phase = manifest.get("active_pilot", "pilot")
    rows = read_measured(base, manifest, phase)
    if phase == "probe2":
        rows += [
            r for r in read_measured(base, manifest, "pilot") if r["kind"] == "core"
        ]
    return pilot_gate(rows)


def current_amendment(base: Path) -> dict[str, Any] | None:
    """Read the exact recovery-bound amendment, rejecting paths outside the run."""
    recovery = pilot.read_optional(base / "operations/recovery-state.json") or {}
    if not recovery.get("operational_amendment"):
        return None
    path = base / recovery["operational_amendment"]
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Operational amendment must belong to this run")
    if digest(path.read_bytes()) != recovery["operational_amendment_sha256"]:
        raise ValueError("Operational amendment binding changed")
    return json.loads(path.read_text())


def revised_judge_prompt(root: Path, base: Path, amendment: dict[str, Any]) -> str:
    """Allow precisely the human-approved rubric insertion; preserve the old prompt."""
    revision = amendment["judge_revision"]
    path = root / revision["prompt_path"]
    if path.resolve() != (root / "prompts/style_judge/three-style-v2.txt").resolve():
        raise ValueError("Unexpected revised judge prompt path")
    if digest(path.read_bytes()) != revision["prompt_sha256"]:
        raise ValueError("Revised judge prompt binding changed")
    judge_approval = amendment.get(
        "judge_approval",
        {"path": amendment["approval_path"], "sha256": amendment["approval_sha256"]},
    )
    approval_path = base / judge_approval["path"]
    if (
        not approval_path.resolve().is_relative_to(base.resolve())
        or digest(approval_path.read_bytes()) != judge_approval["sha256"]
    ):
        raise ValueError("Judge rubric approval binding changed")
    approval = json.loads(approval_path.read_text())
    addition = approval["proposal"]["proposed_judge_prompt_addition"]
    before = (root / "prompts/style_judge/three-style-v1.txt").read_text()
    expected = before.replace(
        "\nCOMMON TRAPS\n", "\n" + addition + "\n\nCOMMON TRAPS\n"
    )
    if path.read_text() != expected or expected == before:
        raise ValueError("Judge change differs from the approved clarification")
    return expected


def apply_judge_revision(
    root: Path, base: Path, name: str, frozen: dict[str, Any], amendment: dict[str, Any]
) -> dict[str, Any]:
    """Overlay only the audited judge prompt on otherwise identical block manifests."""
    revision = amendment["judge_revision"]
    if digest(json_text(frozen["judge"])) != revision["original_judge_sha256"]:
        raise ValueError("Original judge configuration changed")
    result = {
        **frozen,
        "judge": {
            **frozen["judge"],
            "system_prompt": revised_judge_prompt(root, base, amendment),
        },
    }
    expected = revision["effective_block_manifest_sha256"][name]
    path = base / revision["effective_blocks_path"] / name / "manifest.json"
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError("Amended block outside run")
    effective = json.loads(path.read_text())
    if digest(json_text(result)) != expected or effective != result:
        raise ValueError("Amendment altered more than the approved judge prompt")
    return result


def load(root: Path) -> dict[str, Any]:
    """Fail closed on source, verification-report or ceiling drift."""
    base = directory(root)
    manifest = json.loads((base / "manifest.json").read_text())
    if (
        digest((base / "verification/references.json").read_bytes())
        != manifest["reference_report_sha256"]
    ):
        raise ValueError("Reference verification report changed")
    expected_sources = manifest["code_sha256"]
    amendment = current_amendment(base)
    if amendment is not None:
        if amendment["manifest_sha256"] != digest(json_text(manifest)):
            raise ValueError("Operational amendment belongs to a different manifest")
        amended = amendment["code_sha256"]
        allowed = {
            "evals/inspect_three_style.py",
            "scripts/run_three_style.py",
            "tests/test_three_style.py",
            "tests/integration/test_three_style_ownership.py",
        }
        added_sources: set[str] = set()
        if amendment.get("kind") in CALIBRATION_AMENDMENTS:
            verify_supplement(root, base, amendment["supplemental_calibration"])
            approval = base / amendment["approval_path"]
            if (
                not approval.resolve().is_relative_to(base.resolve())
                or digest(approval.read_bytes()) != amendment["approval_sha256"]
            ):
                raise ValueError("Calibration approval binding changed")
            allowed |= {
                "src/grader_power/three_style_calibration.py",
                "planning/three_style_protocol.md",
            }
            if amendment["kind"] in ("user-approved-judge-rubric-v2", PROBE_AMENDMENT):
                revised_judge_prompt(root, base, amendment)
                added_sources = {"prompts/style_judge/three-style-v2.txt"}
                if (
                    amended.get("prompts/style_judge/three-style-v2.txt")
                    != amendment["judge_revision"]["prompt_sha256"]
                ):
                    raise ValueError("Revised judge prompt source binding changed")
                if set(
                    amendment["judge_revision"]["effective_block_manifest_sha256"]
                ) != {item["name"] for item in manifest["blocks"]}:
                    raise ValueError(
                        "Judge amendment does not cover every frozen block"
                    )
            if amendment["kind"] == PROBE_AMENDMENT:
                prior = bound_json(base, amendment["prior_amendment"])
                if (
                    prior["kind"] != "user-approved-judge-rubric-v2"
                    or prior["judge_revision"] != amendment["judge_revision"]
                    or prior["supplemental_calibration"]
                    != amendment["supplemental_calibration"]
                    or amendment["judge_approval"]
                    != {
                        "path": prior["approval_path"],
                        "sha256": prior["approval_sha256"],
                    }
                ):
                    raise ValueError(
                        "Probe amendment changed the approved judge calibration"
                    )
                verify_probe_supplement(root, manifest, amendment)
        if set(amended) != set(expected_sources) | added_sources or any(
            amended[name] != previous and name not in allowed
            for name, previous in expected_sources.items()
        ):
            raise ValueError("Amendment changed a frozen scientific/shared source")
        for name, expected in amendment["verification_reports"].items():
            if digest((root / name).read_bytes()) != expected:
                raise ValueError("Amendment verification report changed")
        expected_sources = amended
    if (
        manifest["run"] != RUN
        or manifest["cap_usd"] != CAP
        or expected_sources != sources(root)
    ):
        raise ValueError("Source/run/ceiling binding changed")
    for name, expected in manifest["verification_reports"].items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError("Verification report changed")
    return manifest


def block(
    root: Path, manifest: dict[str, Any], name: str
) -> tuple[dict[str, Any], tuple[pilot.Assignment, ...]]:
    """Bind one block and rebuild its trusted task evidence without rendering prompts."""
    item = next(i for i in manifest["blocks"] if i["name"] == name)
    frozen = json.loads(
        (directory(root) / "blocks" / name / "manifest.json").read_text()
    )
    if digest(json_text(frozen)) != item["manifest_sha256"]:
        raise ValueError("Block manifest changed")
    amendment = current_amendment(directory(root))
    if (
        amendment is not None
        and amendment.get("judge_revision")
        and not name.startswith("probe2-")
    ):
        frozen = apply_judge_revision(root, directory(root), name, frozen, amendment)
    tasks = main_tasks(root) if name.startswith("main-") else load_pilot_tasks(root)
    by_id = {t.visible.source_task_id: t for t in tasks}
    return frozen, tuple(
        pilot.Assignment(r, by_id[r["source_task_id"]]) for r in frozen["assignments"]
    )


def child_permission(root: Path, parent: int) -> None:
    """Verify a child belongs to the live supervisor holding the shared lock."""
    coordination = root / "runs/extension-pilot/coordination"
    owner = json.loads((coordination / "owner.json").read_text())
    if (
        parent != os.getppid()
        or parent <= 1
        or os.getpgrp() != parent
        or owner.get("pid") != parent
        or owner.get("run_id") != RUN
        or owner.get("state") != "active"
    ):
        raise RuntimeError("No authorized supervising process")
    with (coordination / "collection.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        raise RuntimeError("Supervisor collection lock is not held")


def scoring_infrastructure_issue(score: dict[str, Any]) -> bool:
    """Recognize top-level and nested oracle failures without treating them as answers."""
    correctness = score.get("correctness", {})
    failures = ("oracle_error", "infrastructure_error")
    return (
        correctness.get("status") in failures
        or correctness.get("adapter", {}).get("status") in failures
    )


def reviewed_scoring_failure(base: Path, identity: str, path: Path) -> bool:
    """Retain an exact audited incomplete outcome after verified sandbox recovery.

    This never repairs or replays the failed assignment. A hash-bound operational
    amendment must name its immutable intent, response and scoring artifacts and
    bind successful cleanup evidence. All other infrastructure errors still stop.
    """
    recovery = pilot.read_optional(base / "operations/recovery-state.json") or {}
    if not recovery.get("operational_amendment"):
        return False
    amendment_path = base / recovery["operational_amendment"]
    if (
        not amendment_path.resolve().is_relative_to(base.resolve())
        or digest(amendment_path.read_bytes())
        != recovery["operational_amendment_sha256"]
    ):
        raise ValueError("Scoring review amendment binding changed")
    amendment = json.loads(amendment_path.read_text())
    evidence = amendment.get("retained_scoring_failures", {}).get(identity)
    if evidence is None:
        return False
    cleanup = base / amendment["sandbox_cleanup"]
    if (
        not cleanup.resolve().is_relative_to(base.resolve())
        or digest(cleanup.read_bytes()) != amendment["sandbox_cleanup_sha256"]
        or json.loads(cleanup.read_text()).get("verified") is not True
        or set(evidence) != {"started.json", "raw.json", "scoring.json"}
    ):
        raise ValueError("Scoring review requires bound verified cleanup")
    for filename, expected in evidence.items():
        if digest((path / filename).read_bytes()) != expected:
            raise ValueError("Reviewed scoring outcome changed")
    if not scoring_infrastructure_issue(
        json.loads((path / "scoring.json").read_text())
    ):
        raise ValueError("Reviewed outcome is not an infrastructure failure")
    return True


def measure(cohort: Path, rows: tuple[pilot.Assignment, ...]) -> list[dict[str, Any]]:
    """Read labels with this study's strict four-mechanism parser."""
    with patch.object(original, "parse_judgment", parse_judgment):
        return legacy_measure(cohort, rows)


def worker(root: Path, name: str, parent: int) -> None:
    """Finish one disjoint block, retaining started IDs and unknown reservations."""
    child_permission(root, parent)
    manifest = execution_manifest(root, load(root))
    if not name.startswith("main-") and not name.startswith(
        manifest.get("active_pilot", "pilot") + "-"
    ):
        raise ValueError(
            "Retired pilot is preserved, not eligible for further collection"
        )
    frozen, rows = block(root, manifest, name)
    base = directory(root)
    cohort = base / "blocks" / name
    item = next(i for i in manifest["blocks"] if i["name"] == name)
    ledger = SpendingLedger(
        cohort / "spending.json",
        cap=item["maximum_nanodollars"],
        scope=item["manifest_sha256"],
    )
    pacer = AdaptivePacer(base / "operations/pacing")
    values = {**dotenv_values(root / ".env"), **os.environ}
    with ExitStack() as stack:
        stack.enter_context(patch.object(original, "parse_judgment", parse_judgment))
        stack.enter_context(
            patch.object(
                pilot,
                "load_access_profiles",
                return_value=extension.access_profiles(root),
            )
        )
        stack.enter_context(
            patch.object(pilot, "score_assigned_probe", study.score_probe)
        )
        for row in rows:
            guard(base)
            aid = row.record["assignment_id"]
            identity = name + "/" + aid
            ledger.reserve({aid + "/" + s: b for s, b in checks.bounds(row).items()})
            path = cohort / PROFILE / "attempts" / aid
            if reviewed_scoring_failure(base, identity, path):
                # Keep raw/scoring/labels and reservations unchanged; no stage replay.
                continue
            for stage in ("generation", "judge"):
                if stage == "judge" and row.record["kind"] == "comprehension_probe":
                    continue
                guard(base)
                clock = WaitClock()
                transport = base / "operations/pacing/attempts" / name / aid / stage
                write_json(
                    base / "operations/workers" / f"{name}.json",
                    {
                        "pid": os.getpid(),
                        "updated_at": time.time(),
                        "identity": identity,
                        "stage": stage,
                    },
                )
                proxy = SimpleNamespace(**{**vars(asyncio), "wait_for": clock.wait_for})
                with (
                    paced_http(pacer, transport, clock, lambda: guard(base)),
                    patch.object(pilot, "asyncio", proxy),
                    patch.object(original, "asyncio", proxy),
                ):
                    if stage == "generation":
                        if not (path / "started.json").exists():
                            pilot.collect_profile(
                                root,
                                cohort,
                                rows,
                                PROFILE,
                                frozen["oracle_image"],
                                values,
                                resume=True,
                                max_new_attempts=1,
                                generation_override=extension.generation(root, PROFILE),
                            )
                        if not (path / "started.json").exists():
                            raise RuntimeError("Generation has no durable intent")
                        raw = pilot.read_optional(path / "raw.json")
                        if raw:
                            pacer.usage("fireworks", raw)
                        checks.settle_generation(path, row, ledger, aid)
                        score = pilot.read_optional(path / "scoring.json") or {}
                        failure = pilot.read_optional(path / "failure.json")
                        if (failure or {}).get(
                            "stage"
                        ) == "scoring" or scoring_infrastructure_issue(score):
                            raise RuntimeError("Scoring infrastructure requires review")
                        stage_event(
                            base,
                            identity,
                            stage,
                            failure is not None or raw is None,
                            (failure or {}).get("error_type"),
                        )
                    else:
                        extension.enforce_judge_limit(cohort, (row,), PROFILE, frozen)
                        asyncio.run(
                            original.collect_judgments(
                                root, cohort, (row,), frozen, PROFILE, 1
                            )
                        )
                        judge = cohort / PROFILE / "judge" / aid
                        raw = pilot.read_optional(judge / "raw.json")
                        outcome = pilot.read_optional(judge / "result.json")
                        if raw:
                            pacer.usage("anthropic", raw)
                            try:
                                amount = usage_cost(
                                    raw["usage"], input_rate=2, output_rate=10
                                )
                            except (KeyError, ValueError, TypeError):
                                pass  # Incomplete usage retains its full reservation.
                            else:
                                ledger.settle({aid + "/judge": amount})
                        elif not (judge / "started.json").exists():
                            ledger.settle({aid + "/judge": 0})
                        if (judge / "started.json").exists():
                            stage_event(
                                base,
                                identity,
                                stage,
                                outcome is None
                                or outcome["status"] not in ("classified", "abstain"),
                                (outcome or {}).get("status"),
                            )
                timing = transport / f"timing-{os.getpid()}-{time.time_ns()}.json"
                write_json(timing, {"transport_wait_seconds": clock.waiting()})
                if ledger.snapshot()["overrun"]:
                    raise RuntimeError("Recorded usage exceeded its reservation")
                guard(base)
            write_json(cohort / "measured.json", measure(cohort, rows))
    original.freeze(cohort / "complete.json", {"all_assignments_attempted": True})


def pilot_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Gate on comprehension/correctness/plumbing, never on treatment preference."""
    probes = [r for r in rows if r["kind"] == "comprehension_probe"]
    core = [r for r in rows if r["kind"] == "core"]
    correct = lambda r: (
        (r["probe"] or {}).get("fence_only", {}).get("all_correct") is True
    )
    facts = {
        p: [r for r in probes if r["condition"]["grader_consequence"] == p]
        for p in study.PURPOSES
    }
    passed = (
        len(probes) == 108
        and len(core) == 108
        and all(r["raw_saved"] for r in rows)
        and all(
            sum(correct(r) for r in group) / len(group) >= 0.85
            for group in facts.values()
        )
        and all(
            sum(correct(r) for r in probes if r["condition"]["variant"] == v)
            / sum(r["condition"]["variant"] == v for r in probes)
            >= 0.80
            for v in study.VARIANTS
        )
        and sum(r["full_correct"] is True for r in core) / len(core) >= 0.9
        and sum(r["label"] is not None for r in core) / len(core) >= 0.95
    )
    return {
        "passed": passed,
        "probe_correct_by_purpose": {
            p: sum(correct(r) for r in group) for p, group in facts.items()
        },
        "probe_correct_by_variant": {
            v: sum(correct(r) for r in probes if r["condition"]["variant"] == v)
            for v in study.VARIANTS
        },
        "correct_core": sum(r["full_correct"] is True for r in core),
        "resolved_core": sum(r["label"] is not None for r in core),
        "counts": counts(rows),
    }


def status(base: Path, manifest: dict[str, Any], state: str, **extra: Any) -> None:
    """Persist conservative costs separately from open request reservations."""
    balances = [
        pilot.read_optional(base / "blocks" / i["name"] / "spending.json") or {}
        for i in manifest["blocks"]
    ]
    for cohort in ("calibration", "calibration-supplement-v1", "calibration-rubric-v2"):
        balances.append(pilot.read_optional(base / cohort / "spending.json") or {})
    write_json(
        base / "operations/status.json",
        {
            "state": state,
            "pid": os.getpid(),
            "updated_at": time.time(),
            "counts": {
                p: {
                    **counts(read_measured(base, manifest, p)),
                    "assigned": manifest[p + "_assignments"],
                }
                for p in ("pilot", *manifest.get("additional_pilot_phases", []), "main")
            },
            "recorded_usd": sum(b.get("spent", 0) for b in balances) / 1e9,
            "outstanding_usd": sum(b.get("outstanding", 0) for b in balances) / 1e9,
            "cap_usd": CAP,
            "pacing": AdaptivePacer(base / "operations/pacing").utilization(),
            **extra,
        },
    )


def collect(root: Path) -> None:
    """Supervise isolated blocks, gate main collection on the prospective pilot."""
    manifest = load(root)
    base = directory(root)
    ledger = SpendingLedger(
        base / "spending.json", cap=nanodollars(CAP), scope=digest(json_text(manifest))
    )
    original_manifest_sha256 = digest(json_text(manifest))
    manifest = execution_manifest(root, manifest)
    active_pilot = manifest.get("active_pilot", "pilot")
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    launched = base / "operations/launches" / tag
    launched.mkdir(parents=True)
    original.freeze(
        launched / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "manifest_sha256": original_manifest_sha256,
            "active_pilot": active_pilot,
            "authorization": manifest["authorization"],
        },
    )
    owner = root / "runs/extension-pilot/coordination/owner.json"
    if owner.exists():
        original.freeze(launched / "previous-owner.json", json.loads(owner.read_text()))
    with collection_lock(root, owner=RUN, run_id=RUN):
        active: dict[str, tuple[subprocess.Popen[bytes], Any]] = {}
        workers = INITIAL_WORKERS
        last_status, last_ramp = 0.0, time.time()
        try:
            frozen_judge, _ = block(root, manifest, manifest["blocks"][0]["name"])
            supplement = None
            amendment = current_amendment(base)
            if (
                amendment is not None
                and amendment.get("kind") in CALIBRATION_AMENDMENTS
            ):
                supplement = amendment["supplemental_calibration"]
            if amendment is not None and amendment.get("kind") == PROBE_AMENDMENT:
                # load() verifies immutable passed calibration and all its evidence.
                # The approved follow-up must never issue another calibration request.
                append_event(
                    base / "operations/events.jsonl",
                    {
                        "event": "reuse_completed_calibration",
                        "launch": tag,
                        "binding": amendment["completed_calibration"],
                    },
                )
            else:
                calibrate(
                    root, base, frozen_judge["judge"], ledger, supplement=supplement
                )
            for item in manifest["blocks"]:
                cohort = base / "blocks" / item["name"]
                if (cohort / "complete.json").exists():
                    balance = json.loads((cohort / "spending.json").read_text())
                    before = ledger.snapshot()["entries"].get(item["name"])
                    if before is None:
                        raise ValueError("Completed block has no global reservation")
                    if before["settled"] is None:
                        append_event(
                            base / "operations/recovery-events.jsonl",
                            {
                                "event": "reconstruct_completed_block_settlement",
                                "name": item["name"],
                                "complete_sha256": digest(
                                    (cohort / "complete.json").read_bytes()
                                ),
                                "block_ledger_sha256": digest(
                                    (cohort / "spending.json").read_bytes()
                                ),
                                "committed_nanodollars": balance["committed"],
                                "retained_unknown_nanodollars": balance["outstanding"],
                            },
                        )
                    ledger.settle({item["name"]: balance["committed"]})
            for phase in (active_pilot, "main"):
                if phase == "main":
                    gate = active_pilot_gate(base, manifest)
                    write_json(base / f"{active_pilot}-gate.json", gate)
                    if not gate["passed"]:
                        raise RuntimeError(
                            "Prospective pilot gate failed; main remains unstarted"
                        )
                queue = [
                    i
                    for i in manifest["blocks"]
                    if i["name"].startswith(phase + "-")
                    and not (base / "blocks" / i["name"] / "complete.json").exists()
                ]
                while queue or active:
                    guard(base)
                    for name, (process, log) in list(active.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        log.close()
                        del active[name]
                        append_event(
                            base / "operations/events.jsonl",
                            {
                                "event": "worker_exited",
                                "name": name,
                                "returncode": code,
                                "launch": tag,
                            },
                        )
                        if code != 0:
                            raise RuntimeError(
                                f"Worker failed: {name}; inspect preserved log"
                            )
                        balance = json.loads(
                            (base / "blocks" / name / "spending.json").read_text()
                        )
                        ledger.settle({name: balance["committed"]})
                        append_event(
                            base / "operations/events.jsonl",
                            {"event": "block_finished", "name": name},
                        )
                    if time.time() - last_status >= 20:
                        if execution_manifest(root, load(root)) != manifest:
                            raise ValueError(
                                "Execution manifest changed while collecting"
                            )
                        status(
                            base,
                            manifest,
                            "running",
                            phase=phase,
                            workers=workers,
                            active=list(active),
                            launch=tag,
                        )
                        last_status = time.time()
                    if time.time() - last_ramp >= 120 and workers < MAX_WORKERS:
                        use = AdaptivePacer(base / "operations/pacing").utilization()
                        if use and all(
                            max(v["utilization"].values()) < 0.65
                            and v["queued"] < workers / 4
                            for v in use.values()
                        ):
                            workers = min(MAX_WORKERS, workers + 4)
                            append_event(
                                base / "operations/events.jsonl",
                                {
                                    "event": "worker_increase",
                                    "workers": workers,
                                    "reason": "API headroom and low pacing queue",
                                },
                            )
                        last_ramp = time.time()
                    while queue and len(active) < workers:
                        item = queue[0]
                        balance = ledger.snapshot()
                        existing = balance["entries"].get(item["name"])
                        if existing is None and balance["committed"] + item[
                            "maximum_nanodollars"
                        ] > nanodollars(CAP):
                            if active:
                                break
                            raise RuntimeError(
                                "Whole-study budget cannot admit the next block"
                            )
                        ledger.reserve({item["name"]: item["maximum_nanodollars"]})
                        queue.pop(0)
                        log = (launched / f"{item['name']}.log").open("xb")
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "scripts/run_three_style.py",
                                "worker",
                                "--block",
                                item["name"],
                                "--parent",
                                str(os.getpid()),
                            ],
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        active[item["name"]] = process, log
                        append_event(
                            base / "operations/events.jsonl",
                            {
                                "event": "worker_started",
                                "name": item["name"],
                                "pid": process.pid,
                                "launch": tag,
                            },
                        )
                    time.sleep(1)
            analyze(root)
            status(base, manifest, "complete", launch=tag)
        except Exception as error:
            # Never interrupt an accepted request. Guards prevent subsequent sends;
            # wait for all existing workers before releasing the collection lock.
            append_event(
                base / "operations/events.jsonl",
                {"event": "paused", "error_type": type(error).__name__, "launch": tag},
            )
            write_json(
                base / "operations/controller-stop.json",
                {"launch": tag, "error_type": type(error).__name__},
            )
            status(
                base,
                manifest,
                "draining_after_pause",
                launch=tag,
                error_type=type(error).__name__,
            )
            for name, (process, log) in active.items():
                code = process.wait()
                append_event(
                    base / "operations/events.jsonl",
                    {
                        "event": "worker_drained",
                        "name": name,
                        "returncode": code,
                        "launch": tag,
                    },
                )
                log.close()
            status(
                base, manifest, "paused", launch=tag, error_type=type(error).__name__
            )
            raise


def analyze(root: Path) -> dict[str, Any]:
    """Verify all assignments and write analysis without any provider calls."""
    manifest = load(root)
    base = directory(root)
    rows = []
    for item in manifest["blocks"]:
        if item["name"].startswith("main-"):
            _, assigned = block(root, manifest, item["name"])
            rows.extend(measure(base / "blocks" / item["name"], assigned))
    if len(rows) != 6912 or len({r["assignment_id"] for r in rows}) != 6912:
        raise ValueError("Analysis denominator mismatch")
    output = summarize(rows)
    output["manifest_sha256"] = digest(json_text(manifest))
    output["measured_rows_sha256"] = digest(json_text(rows))
    amendment = current_amendment(base)
    if amendment is not None:
        output["measurement_amendment_sha256"] = digest(json_text(amendment))
        if amendment.get("judge_revision"):
            output["judge_prompt"] = {
                "path": amendment["judge_revision"]["prompt_path"],
                "sha256": amendment["judge_revision"]["prompt_sha256"],
            }
    original.freeze(base / "analysis-v1/measured-rows.json", rows)
    original.freeze(base / "analysis-v1/analysis.json", output)
    return output


def main() -> None:
    """Expose preparation, supervised collection and read-only offline analysis."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "collect",
            "worker",
            "analyze",
            "status",
        ),
    )
    parser.add_argument("--report", action="append", type=Path, default=[])
    parser.add_argument("--block")
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        prepare(root, args.report)
    elif args.command == "collect":
        isolate_collection_process()
        collect(root)
    elif args.command == "worker":
        if args.block is None or args.parent is None:
            raise ValueError("Worker requires block and parent")
        worker(root, args.block, args.parent)
    elif args.command == "analyze":
        print(json.dumps(analyze(root)))
    else:
        manifest = execution_manifest(root, load(root))
        base = directory(root)
        print(
            json.dumps(
                {
                    p: {
                        **counts(read_measured(base, manifest, p)),
                        "assigned": manifest[p + "_assignments"],
                    }
                    for p in (
                        "pilot",
                        *manifest.get("additional_pilot_phases", []),
                        "main",
                    )
                }
            )
        )
