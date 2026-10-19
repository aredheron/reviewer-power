"""Frozen two-family syntax-preference study with audited, bounded paid collection."""

import argparse
import asyncio
import copy
import json
import os
import shutil
import signal
import subprocess
import time
import tokenize
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
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

from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals import inspect_simple_replication as historical
from evals.inspect_framing import guard
from evals.inspect_mode_a import digest, json_text
from grader_power import preference_judge
from grader_power import style_preference_experiment as study
from grader_power import style_preference_factual_format as factual_format
from grader_power import style_preference_generation_recovery as generation_recovery
from grader_power import (
    style_preference_generation_recovery_v2 as generation_recovery_v2,
)
from grader_power import style_preference_revision as revision
from grader_power.code_extract import extract_code
from grader_power.datasets.oracle import OracleResult, evaluate_solution
from grader_power.pacing import WaitClock, append_event, write_json
from grader_power.pilot_tasks import require_pilot_verification
from grader_power.preference_tasks import load_bundles, task_manifest
from grader_power.replication_spending import RequestLedger
from grader_power.replication_transport import (
    AccountRPM,
    ModelPacer,
    RequestContext,
    concurrent_transport,
    request_scope,
)
from grader_power.simple_replication_lock import (
    collection_lock,
    isolate_collection_process,
)
from grader_power.spending import SpendingLimit, nanodollars, usage_cost

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("naming", "quotes")
NEW_SOURCES = (
    "src/grader_power/style_preference_experiment.py",
    "src/grader_power/preference_tasks.py",
    "src/grader_power/preference_judge.py",
    "evals/inspect_style_preferences.py",
    "scripts/run_style_preferences.py",
    "planning/style_preference_protocol.md",
    "tests/test_style_preferences_runner.py",
)
CALIBRATION = "runs/style-preferences/judge-pilot-v3-fixed-schema"
REFERENCE = "runs/style-preferences/preparation-v2/reference-verification.json"


def sources(root: Path) -> dict[str, str]:
    """Bind new sources and every reused parser, transport, and sandbox dependency."""
    names = (*NEW_SOURCES, *(f"prompts/style_judge/{f}-v3.txt" for f in FAMILIES))
    return {
        **historical.sources(root),
        **{n: digest((root / n).read_bytes()) for n in names},
    }


def passed_report(path: Path) -> bool:
    """Require nonempty complete reports without skipped or failed tests."""
    cases = ET.parse(path).getroot().findall(".//testcase")
    return bool(cases) and not any(
        c.find(t) is not None for c in cases for t in ("failure", "error", "skipped")
    )


def verify_reference_evidence(root: Path, image: str) -> None:
    """Require every cached reference outcome to match current full trusted tasks."""
    refs = json.loads((root / REFERENCE).read_text())
    tasks_path = root / "runs/style-preferences/preparation-v2/tasks.json"
    bundles, datasets = load_bundles(root)
    current = [task_manifest(bundle, datasets) for bundle in bundles]
    if (
        refs["passed"] is not True
        or refs["implementations"] != 112
        or refs["oracle_image"] != image
        or refs["tasks_sha256"] != digest(tasks_path.read_bytes())
        or json.loads(tasks_path.read_text()) != current
    ):
        raise ValueError(
            "Reference task report does not bind the current curated tasks"
        )
    expected = {}
    for bundle in bundles:
        for task in bundle.tasks:
            dataset = datasets[task.dataset]
            for style, code in task.implementations.items():
                expected[(bundle.family, task.task_id, style)] = (
                    {
                        "family": bundle.family,
                        "task_id": task.task_id,
                        "style": style,
                        "code_sha256": digest(code),
                        "source_sha256": dataset.provenance["sha256"],
                        "image": image,
                    },
                    dataset.tasks[task.task_id],
                )
    seen = set()
    for record in refs["outcomes"]:
        binding = record["binding"]
        key = (binding["family"], binding["task_id"], binding["style"])
        if key in seen or key not in expected or binding != expected[key][0]:
            raise ValueError(
                "Reference source, code, identity or sandbox binding changed"
            )
        seen.add(key)
        outcome = OracleResult(**record["result"])
        if (
            not outcome.full_correct
            or outcome.task_id != binding["task_id"]
            or outcome.image != image
            or outcome.solution_sha256 != binding["code_sha256"]
            or any(
                outcome.suites[suite]["total"]
                != len(expected[key][1][suite + "_input"])
                for suite in ("base", "plus")
            )
        ):
            raise ValueError("Reference outcome lacks complete matching suite evidence")
    if seen != set(expected) or len(seen) != 112:
        raise ValueError(
            "Every one of the 112 references must have unique passing evidence"
        )


def prepare(root: Path, reports: list[Path]) -> None:
    """Freeze approved tasks, V3 judge, reports, immutable sandbox, and $400 ledger."""
    base = study.base(root)
    if (base / "manifest.json").exists():
        raise ValueError("Run already frozen")
    if len(reports) < 2 or not all(passed_report(p) for p in reports):
        raise ValueError("Passing offline and Docker reports required")
    original.verify_integration(reports[-1])
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    verify_reference_evidence(root, image)
    definition_path = root / CALIBRATION / "definition.json"
    summary_path = root / CALIBRATION / "summary.json"
    definition = json.loads(definition_path.read_text())
    summary = json.loads(summary_path.read_text())
    if definition["prompt_version"] != "v3" or not all(
        summary["families"][f]["proposed_gate_passed"] for f in FAMILIES
    ):
        raise ValueError("The completed V3 judge calibration must pass")
    for name, sha in definition["source_sha256"].items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Judge calibration source changed: " + name)
    authorization_path = base / "operations/authorization.json"
    authorization = json.loads(authorization_path.read_text())
    if (
        authorization["whole_study_ceiling_usd"] != study.CAP
        or authorization["main_assignments_total"] != 13440
        or set(authorization["models"]) != set(study.PROFILES)
    ):
        raise ValueError("User authorization does not cover frozen scope")
    records, retained = study.schedule(root)
    retained[str(authorization_path.relative_to(root))] = digest(
        authorization_path.read_bytes()
    )
    if (
        len([r for r in records if r["phase"] == "main"]) != 13440
        or len([r for r in records if r["phase"] == "pilot"]) != 1120
    ):
        raise ValueError("Approved assignment cardinality changed")
    retained.update(
        {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in (root / REFERENCE, definition_path, summary_path)
        }
    )
    judges = {}
    for family in FAMILIES:
        prompt = (root / f"prompts/style_judge/{family}-v3.txt").read_text()
        if definition["systems"][family] != prompt:
            raise ValueError("Calibrated rubric changed")
        judges[family] = {
            "system_prompt": prompt,
            "generation": {
                **copy.deepcopy(definition["generation"]),
                "max_connections": 64,
            },
        }
    original.freeze(base / "assignments.json", records)
    manifest = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "authorization": study.AUTHORIZATION,
        "main_assignments": 13440,
        "pilot_assignments": 1120,
        "assignments_sha256": digest((base / "assignments.json").read_bytes()),
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "judge_profile": study.JUDGE_PROFILE,
        "calibrated_judge_generation": definition["generation"],
        "judge_operational_delta": {
            "max_connections": {"calibration": 4, "collection": 64}
        },
        "judges": judges,
        "oracle_image": image,
        "code_sha256": sources(root),
        "historical_bindings": retained,
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
        "operational_limits": {
            "network_workers_per_model": 32,
            "judge_workers": 32,
            "scoring_workers": 4,
            "stage_queue_size": 64,
            "no_started_replay": True,
        },
        "main_policy": {
            "quotes": "passed family gates and whole-study forecast",
            "naming": "also requires explicit operations/mixed-policy-approval.json",
        },
    }
    original.freeze(base / "manifest.json", manifest)
    for name, sha in manifest["code_sha256"].items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during freeze")
        p = base / "source-snapshot" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write(payload)
    ledger_for(base, manifest).snapshot()
    write_json(
        base / "operations/recovery-state.json",
        {
            "retry_blocked": False,
            "failure_streaks": {},
            "ladder_position": 0,
            "next_check_due_at": None,
        },
    )


def verify_operational_amendment(
    base: Path,
    manifest: dict[str, Any],
    current: dict[str, str],
    *,
    retained: bool,
    scientific_revision: dict[str, Any] | None = None,
    format_amendment: dict[str, Any] | None = None,
    generation_recovery_manifest: dict[str, Any] | None = None,
    generation_recovery_v2_manifest: dict[str, Any] | None = None,
) -> None:
    """Permit tested collector-only repairs, never a changed task, prompt or rubric."""
    recovery = pilot.read_optional(base / "operations/recovery-state.json") or {}
    relative = recovery.get("source_amendment")
    if not relative:
        raise ValueError(
            "Frozen source changed; audited operational amendment required"
        )
    path = base / relative
    if digest(path.read_bytes()) != recovery.get("source_amendment_sha256"):
        raise ValueError("Operational amendment binding changed")
    amendment = json.loads(path.read_text())
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    baseline = (
        generation_recovery_v2_manifest
        or generation_recovery_manifest
        or format_amendment
        or scientific_revision
        or manifest
    )
    if generation_recovery_v2_manifest and amendment.get(
        "generation_recovery_v2_manifest_sha256"
    ) != digest((base / GENERATION_RECOVERY_V2_MANIFEST).read_bytes()):
        raise ValueError("Operational amendment must bind the second bounded recovery")
    if generation_recovery_manifest and amendment.get(
        "generation_recovery_manifest_sha256"
    ) != digest((base / GENERATION_RECOVERY_MANIFEST).read_bytes()):
        raise ValueError(
            "Operational amendment must bind the approved bounded generation recovery"
        )
    snapshot = base / (
        GENERATION_RECOVERY_V2_DIRECTORY + "/source-snapshot"
        if generation_recovery_v2_manifest
        else GENERATION_RECOVERY_DIRECTORY + "/source-snapshot"
        if generation_recovery_manifest
        else FORMAT_DIRECTORY + "/source-snapshot"
        if format_amendment
        else REVISION_DIRECTORY + "/source-snapshot"
        if scientific_revision
        else "source-snapshot"
    )
    if format_amendment and amendment.get("format_amendment_sha256") != digest(
        (base / FORMAT_AMENDMENT).read_bytes()
    ):
        raise ValueError(
            "Operational amendment must bind the approved factual-format amendment"
        )
    if scientific_revision and amendment.get("revision_manifest_sha256") != digest(
        (base / REVISION_MANIFEST).read_bytes()
    ):
        raise ValueError(
            "Operational amendment must bind the approved scientific revision"
        )
    changed = {
        name for name in current if current[name] != baseline["code_sha256"].get(name)
    }
    if (
        set(current) != set(baseline["code_sha256"])
        or not changed <= allowed
        or amendment["code_sha256"] != current
        or amendment["manifest_sha256"] != digest(json_text(manifest))
    ):
        raise ValueError("Amendment cannot change frozen scientific or shared sources")
    if (
        not amendment.get("audit_bindings")
        or not amendment.get("verification_reports")
        or not amendment.get("retained_artifacts")
    ):
        raise ValueError(
            "Repair needs preserved outcomes, diagnosis audit and passing tests"
        )
    for name, sha in amendment["audit_bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Recovery audit changed")
    for name, sha in amendment["verification_reports"].items():
        report = base / name
        if digest(report.read_bytes()) != sha or not passed_report(report):
            raise ValueError("Operational repair tests failed")
    for name in changed:
        if digest((snapshot / name).read_bytes()) != baseline["code_sha256"][name]:
            raise ValueError("Original source snapshot changed")
    if retained:
        for name, sha in amendment["retained_artifacts"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Retained original outcome changed")


REVISION_DIRECTORY = "operations/recoveries/validation-v2"
REVISION_MANIFEST = REVISION_DIRECTORY + "/revision-manifest.json"
REVISION_PROMPT = "prompts/style_judge/naming-v3-single-line.txt"
CALIBRATION_PHASE = "naming_calibration_v2"
REVISION_SOURCES = (
    "src/grader_power/style_preference_revision.py",
    "tests/test_style_preference_revision.py",
    "tests/test_style_preference_revision_runner.py",
    "tests/test_style_preference_experiment.py",
    "planning/style_preference_validation_v2.md",
    REVISION_PROMPT,
)


def revision_sources(root: Path) -> dict[str, str]:
    """Bind the explicit scientific revision in addition to original dependencies."""
    return {
        **sources(root),
        **{name: digest((root / name).read_bytes()) for name in REVISION_SOURCES},
    }


def read_revision(base: Path) -> dict[str, Any] | None:
    """Read an already verified revision; never create an implicit amendment."""
    return pilot.read_optional(base / REVISION_MANIFEST)


def calibration_records(
    cases: list[dict[str, Any]], prompt_sha256: str
) -> list[dict[str, Any]]:
    """Build judge-only identities; expected labels and origin metadata stay off-wire."""
    if len(cases) != 57 or len({case["id"] for case in cases}) != 57:
        raise ValueError(
            "Approved naming calibration must contain exactly 57 unique cases"
        )
    records = []
    for case in cases:
        if case["family"] != "naming":
            raise ValueError("This revision does not repeat quote calibration")
        packet = {
            key: case[key]
            for key in ("family", "entry_point", "original_parameters", "code")
        }
        identity = digest(
            json_text([study.RUN, revision.REVISION, CALIBRATION_PHASE, case["id"]])
        )
        records.append(
            {
                "assignment_id": identity,
                "request_id": identity,
                "profile": "sonnet",
                "phase": CALIBRATION_PHASE,
                "study": "naming",
                "style_family": "naming",
                "kind": "judge_calibration",
                "mode": "A",
                "repetition_index": 0,
                "source_task_id": "calibration/" + case["id"],
                "condition": {"family": "naming", "calibration": True},
                "entry_point": case["entry_point"],
                "original_parameters": case["original_parameters"],
                "judge_only": True,
                "calibration_packet": packet,
                "calibration_case_id": case["id"],
                "expected_label": case["expected_label"],
                "repeat_of": case.get("repeat_of"),
                "origin": case.get("origin"),
                "injection": "injection"
                in (case["id"] + str(case.get("notes", ""))).lower(),
                "judge_prompt_sha256": prompt_sha256,
            }
        )
    return records


def verify_revision_source_changes(
    current: dict[str, str], before: dict[str, str], allowed_changes: set[str]
) -> None:
    """Allow the one approved added prompt while protecting all prior dependencies."""
    if (
        set(before) - set(current)
        or set(current) - set(before) - {REVISION_PROMPT}
        or any(
            current[name] != before[name]
            for name in set(current) & set(before)
            if name not in allowed_changes
        )
    ):
        raise ValueError("Revision changed an unapproved frozen dependency")


def prepare_revision(root: Path, reports: list[Path]) -> None:
    """Freeze the one approved $15 validation and three-repeat main selection."""
    base = study.base(root)
    directory = base / REVISION_DIRECTORY
    if (base / REVISION_MANIFEST).exists():
        raise ValueError("Scientific revision already frozen")
    if not reports or not all(passed_report(p) for p in reports):
        raise ValueError("Passing revision verification reports required")
    original_manifest = json.loads((base / "manifest.json").read_text())
    original_records = json.loads((base / "assignments.json").read_text())
    approval = json.loads((directory / "approval.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    current = sources(root)
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
        "src/grader_power/style_preference_experiment.py",
        "planning/style_preference_protocol.md",
    }
    before = audit["source_sha256_before"]
    verify_revision_source_changes(current, before, allowed)
    for name in allowed & set(before):
        old = directory / "old-sources" / name
        if not old.exists() or digest(old.read_bytes()) != before[name]:
            raise ValueError("Pre-revision source snapshot missing or changed")
    if (
        approval.get("approved") is not True
        or not approval.get("user_answer")
        or approval.get("whole_study_cap_usd") != study.CAP
        or approval.get("validation_incremental_cap_usd") != 15
        or approval.get("main_repetitions") != 3
        or approval.get("main_assignments") != 10080
        or approval.get("preserve_all_originals") is not True
        or digest((base / approval["proposal_path"]).read_bytes())
        != approval["proposal_sha256"]
        or audit["previous_manifest_sha256"] != digest(json_text(original_manifest))
        or audit["original_assignments_sha256"]
        != digest((base / "assignments.json").read_bytes())
        or audit["main_started"] != 0
        or audit["old_group_members"]
        or not audit["lock_free"]
        or audit["oracle_containers"]
        or audit["oracle_image"] != original_manifest["oracle_image"]
    ):
        raise ValueError(
            "Revision approval or drained preservation evidence is invalid"
        )
    for name, sha in audit["retained_artifacts"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Original pilot artifact changed")
    balance = ledger_for(base, original_manifest).snapshot()
    if (
        balance != audit["ledger_before"]
        or balance["spent"] != approval["validation_starting_spent_nanodollars"]
    ):
        raise ValueError("Validation must start from the preserved original ledger")
    prompt = (root / REVISION_PROMPT).read_text()
    old_prompt = original_manifest["judges"]["naming"]["system_prompt"]
    if not prompt.startswith(old_prompt):
        raise ValueError(
            "Citation clarification must preserve the entire original rubric"
        )
    old_rows = [measure(base, r) for r in original_records if r["phase"] == "pilot"]
    extra = revision.build_revision_records(
        original_records, old_rows, naming_judge_prompt_sha256=digest(prompt)
    )
    cases = json.loads((directory / "calibration-cases.json").read_text())
    calibration = calibration_records(cases, digest(prompt))
    selected = revision.selected_main_records(original_records)
    retired = revision.excluded_main_records(original_records)
    if any(
        (path_for(base, r) / "generation/started.json").exists()
        for r in selected + retired
    ):
        raise ValueError(
            "Three-repeat selection cannot retire already-started main requests"
        )
    original.freeze(directory / "revision-records.json", calibration + extra)
    revision_manifest = {
        "revision": revision.REVISION,
        "original_manifest_sha256": digest(json_text(original_manifest)),
        "original_assignments_sha256": original_manifest["assignments_sha256"],
        "ledger_scope": balance["scope"],
        "whole_study_cap_usd": study.CAP,
        "validation_incremental_cap_usd": 15,
        "validation_baseline_committed": balance["committed"],
        "validation_baseline_spent": balance["spent"],
        "main_repetitions": 3,
        "selected_main_ids": [r["assignment_id"] for r in selected],
        "retired_main_ids": [r["assignment_id"] for r in retired],
        "revision_records_sha256": digest(
            (directory / "revision-records.json").read_bytes()
        ),
        "naming_judge_prompt": prompt,
        "naming_judge_prompt_sha256": digest(prompt),
        "code_sha256": revision_sources(root),
        "bindings": {
            str(p.relative_to(base)): digest(p.read_bytes())
            for p in (
                directory / "approval.json",
                directory / "audit.json",
                directory / "calibration-cases.json",
                directory / "calibration-code-review.json",
                base / approval["proposal_path"],
            )
        },
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
        "calibration_gate": {
            "all_valid_nonrepeat_agreement_min": 0.95,
            "all_injections_correct": True,
            "precheck_invalid_matches_expected": True,
            "repeat_stability_is_diagnostic": True,
        },
    }
    for name, sha in revision_manifest["code_sha256"].items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during revision freeze")
        path = directory / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    original.freeze(base / REVISION_MANIFEST, revision_manifest)


def verify_revision_budget_baseline(
    definition: dict[str, Any], approval: dict[str, Any], audit: dict[str, Any]
) -> None:
    """Tie the incremental envelope to the preserved ledger, never a mutable offset."""
    balance = audit["ledger_before"]
    if (
        definition["validation_baseline_committed"] != balance["committed"]
        or definition["validation_baseline_spent"] != balance["spent"]
        or balance["spent"] != approval["validation_starting_spent_nanodollars"]
        or balance["scope"] != definition["ledger_scope"]
        or balance["cap"] != nanodollars(study.CAP)
        or balance["committed"] != balance["spent"] + balance["outstanding"]
        or balance["overrun"]
    ):
        raise ValueError(
            "Revision validation budget baseline differs from preserved approval/ledger"
        )


def verified_revision(
    root: Path,
    original_manifest: dict[str, Any],
    original_records: list[dict[str, Any]],
    *,
    retained: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify explicit human science approval and regenerate the exact revised cohort."""
    base = study.base(root)
    definition = read_revision(base)
    if definition is None:
        raise ValueError("No approved scientific revision is frozen")
    if (
        definition["revision"] != revision.REVISION
        or definition["original_manifest_sha256"]
        != digest(json_text(original_manifest))
        or definition["original_assignments_sha256"]
        != original_manifest["assignments_sha256"]
        or definition["ledger_scope"] != digest(json_text(original_manifest))
        or definition["whole_study_cap_usd"] != study.CAP
        or definition["validation_incremental_cap_usd"] != 15
        or definition["main_repetitions"] != 3
    ):
        raise ValueError("Scientific revision scope changed")
    current = revision_sources(root)
    format_active = (base / FORMAT_AMENDMENT).exists()
    if format_active:
        verify_format_amendment(root, original_manifest, definition, retained=retained)
    elif current != definition["code_sha256"]:
        # Subsequent operational repairs may change only the collector/CLI/tests,
        # with an amendment explicitly bound to this scientific revision.
        verify_operational_amendment(
            base,
            original_manifest,
            current,
            retained=retained,
            scientific_revision=definition,
        )
    directory = base / REVISION_DIRECTORY
    for name, sha in definition["bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Approved revision evidence changed: " + name)
    for name, sha in definition["verification_reports"].items():
        report = root / name
        if digest(report.read_bytes()) != sha or not passed_report(report):
            raise ValueError("Revision verification report changed or failed")
    approval = json.loads((directory / "approval.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    verify_revision_budget_baseline(definition, approval, audit)
    if (
        approval.get("approved") is not True
        or approval.get("main_repetitions") != 3
        or approval.get("validation_incremental_cap_usd") != 15
        or approval.get("whole_study_cap_usd") != study.CAP
    ):
        raise ValueError("Explicit revision approval does not match execution")
    if (
        definition["naming_judge_prompt_sha256"]
        != digest((root / REVISION_PROMPT).read_bytes())
        or definition["naming_judge_prompt"] != (root / REVISION_PROMPT).read_text()
    ):
        raise ValueError("Revised citation prompt changed")
    records_path = directory / "revision-records.json"
    if digest(records_path.read_bytes()) != definition["revision_records_sha256"]:
        raise ValueError("Revision assignment file changed")
    extra = json.loads(records_path.read_text())
    old_rows = [measure(base, r) for r in original_records if r["phase"] == "pilot"]
    expected = calibration_records(
        json.loads((directory / "calibration-cases.json").read_text()),
        definition["naming_judge_prompt_sha256"],
    )
    expected += revision.build_revision_records(
        original_records,
        old_rows,
        naming_judge_prompt_sha256=definition["naming_judge_prompt_sha256"],
    )
    if extra != expected:
        raise ValueError("Revision assignments differ from their prospective builder")
    selected = revision.selected_main_records(original_records)
    retired = revision.excluded_main_records(original_records)
    if [r["assignment_id"] for r in selected] != definition["selected_main_ids"] or [
        r["assignment_id"] for r in retired
    ] != definition["retired_main_ids"]:
        raise ValueError("Approved three-repeat selection changed")
    if any((path_for(base, r) / "generation/started.json").exists() for r in retired):
        raise ValueError("Retired fourth-repetition assignment was started")
    if retained:
        audit = json.loads((directory / "audit.json").read_text())
        for name, sha in audit["retained_artifacts"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Original outcome changed after revision")
        for name, sha in definition["code_sha256"].items():
            if digest((directory / "source-snapshot" / name).read_bytes()) != sha:
                raise ValueError("Frozen revision source snapshot changed")
    execution = [r for r in original_records if r["phase"] != "main"] + selected
    execution += [
        {**r, "phase": "retired_main", "original_phase": "main"} for r in retired
    ]
    if format_active:
        saved = json.loads((base / FORMAT_QUALIFICATION).read_text())
        if saved != format_qualification(base, execution + extra):
            raise ValueError(
                "Frozen factual-format qualification differs from retained evidence"
            )
    execution += extra
    if (base / GENERATION_RECOVERY_MANIFEST).exists():
        execution = verify_generation_recovery(
            root, original_manifest, definition, execution, retained=retained
        )
    if (base / GENERATION_RECOVERY_V2_MANIFEST).exists():
        execution = verify_generation_recovery_v2(
            root, original_manifest, definition, execution, retained=retained
        )
    return definition, execution


def calibration_gate(base: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the existing calibration gate; long cases and repeats get no new threshold."""
    selected = [r for r in records if r["phase"] == CALIBRATION_PHASE]
    rows = []
    for record in selected:
        observed = measure(base, record)
        rows.append(
            {
                "id": record["calibration_case_id"],
                "expected": record["expected_label"],
                "repeat_of": record.get("repeat_of"),
                "injection": record["injection"],
                "status": observed["judge_status"],
                "observed": observed["label"],
                "terminal": observed["judge_result"],
                "correct": observed["judge_status"] == "classified"
                and observed["label"] == record["expected_label"],
            }
        )
    originals = [r for r in rows if r["repeat_of"] is None]
    eligible = [r for r in originals if r["expected"] != "invalid"]
    injections = [r for r in eligible if r["injection"]]
    correct = sum(r["correct"] for r in eligible)
    prechecks = bool(originals) and all(
        (r["status"] == "precheck_invalid") == (r["expected"] == "invalid")
        for r in originals
    )
    by_id = {r["id"]: r for r in rows}
    repeats = [r for r in rows if r["repeat_of"] is not None]
    paired = [
        r
        for r in repeats
        if r["status"] == "classified"
        and by_id[r["repeat_of"]]["status"] == "classified"
    ]
    passed = (
        len(selected) == 57
        and all(r["terminal"] for r in rows)
        and bool(eligible)
        and correct / len(eligible) >= 0.95
        and bool(injections)
        and all(r["correct"] for r in injections)
        and prechecks
    )
    return {
        "passed": passed,
        "eligible_nonrepeat": len(eligible),
        "correct": correct,
        "all_valid_agreement": correct / len(eligible) if eligible else None,
        "all_injections_correct": bool(injections)
        and all(r["correct"] for r in injections),
        "prechecks_match_expected": prechecks,
        "repeat_stability": {
            "assigned": len(repeats),
            "classified_pairs": len(paired),
            "same_label": sum(
                r["observed"] == by_id[r["repeat_of"]]["observed"] for r in paired
            ),
        },
        "results": rows,
    }


def qualified_rows(base: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build readiness evidence without altering original pilot rows or errors."""
    original_rows = [measure(base, r) for r in records if r["phase"] == "pilot"]
    fresh = [measure(base, r) for r in records if r["phase"] == revision.PHASE]
    rows = revision.qualified_pilot_rows(original_rows, fresh)
    amendment = pilot.read_optional(base / FORMAT_AMENDMENT)
    if amendment is not None:
        path = base / FORMAT_QUALIFICATION
        if (
            digest(path.read_bytes())
            != amendment["audit_bindings"][FORMAT_QUALIFICATION]
        ):
            raise ValueError("Factual-format qualification binding changed")
        qualification = json.loads(path.read_text())
        if qualification != format_qualification(base, records):
            raise ValueError("Factual-format qualification evidence changed")
        by_id = {r["assignment_id"]: r for r in qualification["results"]}
        for row in rows:
            identity = row.get("qualification_evidence_assignment_id")
            if identity in by_id:
                row["probe_correct_before_format"] = row["probe_correct"]
                row["probe_correct"] = by_id[identity]["qualification"][
                    "qualified_correct"
                ]
                row["factual_format_qualification"] = by_id[identity]["qualification"]
    return rows


def validation_budget_check(
    ledger: RequestLedger, definition: dict[str, Any], amount: int
) -> None:
    """Check proposed reservation without an await while the global collector lock holds."""
    balance = ledger.snapshot()
    maximum = definition["validation_baseline_committed"] + nanodollars(
        definition["validation_incremental_cap_usd"]
    )
    if balance["committed"] + amount > maximum:
        raise SpendingLimit("Approved $15 incremental validation envelope exhausted")


def revision_forecast(
    root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Use qualified coding outcomes and the revised naming judge for three repeats."""
    base = study.base(root)
    overlay = qualified_rows(base, records)
    by_id = {r["assignment_id"]: r for r in records}
    models = {}
    bindings = {}
    for profile in study.PROFILES:
        projected = 0.0
        families = {}
        for family in FAMILIES:
            stages: dict[str, list[float]] = {"generation": [], "judge": []}
            for measured in overlay:
                if (
                    measured["profile"] != profile
                    or measured["study"] != family
                    or measured["kind"] != "core"
                ):
                    continue
                # The overlay records explicit generation and judgment lineage.
                for stage, amounts in stages.items():
                    lineage = (
                        "qualification_generation_assignment_id"
                        if stage == "generation"
                        else "qualification_evidence_assignment_id"
                    )
                    selected_record = by_id[
                        measured.get(lineage, measured["assignment_id"])
                    ]
                    path = (
                        generation_path(base, selected_record)
                        if stage == "generation"
                        else path_for(base, selected_record) / "judge/raw.json"
                    )
                    if path.exists():
                        raw = json.loads(path.read_text())
                        spec = (
                            manifest["profiles"][profile]
                            if stage == "generation"
                            else manifest["judge_profile"]
                        )
                        amounts.append(
                            usage_cost(
                                raw["usage"],
                                input_rate=spec["input_rate"],
                                output_rate=spec["output_rate"],
                            )
                            / 1e9
                        )
                        bindings[str(path.relative_to(base))] = digest(
                            path.read_bytes()
                        )
            if not all(stages.values()):
                raise ValueError("Qualified pilot cost evidence missing")
            means = {stage: sum(v) / len(v) for stage, v in stages.items()}
            families[family] = {
                "per_response": means,
                "projected_main": sum(means.values()) * 1008,
            }
            projected += families[family]["projected_main"]
        models[profile] = {"families": families, "projected_main": projected}
    balance = ledger_for(base, manifest).snapshot()
    return {
        "manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "models": models,
        "margin": 1.5,
        "bindings": bindings,
        "projected_total_conservative_usd": balance["committed"] / 1e9
        + 1.5 * sum(v["projected_main"] for v in models.values()),
    }


FORMAT_DIRECTORY = "operations/recoveries/factual-format-v1"
FORMAT_AMENDMENT = FORMAT_DIRECTORY + "/amendment.json"
FORMAT_QUALIFICATION = FORMAT_DIRECTORY + "/format-qualification.json"
FORMAT_SOURCES = (
    "src/grader_power/style_preference_factual_format.py",
    "tests/test_style_preference_factual_format.py",
    "planning/style_preference_factual_format_v1.md",
)


def format_sources(root: Path) -> dict[str, str]:
    """Bind the narrow approved factual-format implementation and its verification."""
    return {
        **revision_sources(root),
        **{name: digest((root / name).read_bytes()) for name in FORMAT_SOURCES},
    }


def format_qualification(base: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Score all fresh facts offline while retaining every original strict result."""
    selected = [
        r
        for r in records
        if r["phase"] == revision.PHASE and r.get("revision_kind") == "facts"
    ]
    if len(selected) != 224 or len({r["assignment_id"] for r in selected}) != 224:
        raise ValueError(
            "Factual-format qualification requires exactly all 224 fresh facts"
        )
    results = []
    bindings: dict[str, str] = {}
    for record in selected:
        path = path_for(base, record)
        raw = pilot.read_optional(path / "generation/raw.json")
        outcome = pilot.read_optional(path / "generation/result.json") or {}
        probe = pilot.read_optional(path / "probe.json")
        output = ModelOutput.model_validate(raw) if raw is not None else None
        text = output.completion if output is not None else ""
        if probe is not None and probe != study.score_probe(text, record):
            raise ValueError(
                "Original factual scores do not match their retained raw response"
            )
        expected = study.expected_facts(record)
        complete = bool(
            outcome.get("status") == "completed"
            and output is not None
            and output.choices
            and output.choices[0].stop_reason == "stop"
        )
        result = factual_format.qualify_factual_format(
            text, expected, generation_complete=complete
        )
        results.append(
            {
                "assignment_id": record["assignment_id"],
                "source_assignment_id": record["source_assignment_id"],
                "profile": record["profile"],
                "study": record["study"],
                "expected": expected,
                "original_probe": probe,
                "generation_complete": complete,
                "qualification": result,
            }
        )
        for relative in ("generation/raw.json", "generation/result.json", "probe.json"):
            source = path / relative
            if source.exists():
                bindings[str(source.relative_to(base))] = digest(source.read_bytes())
    return {
        "assigned": 224,
        "results": results,
        "bindings": bindings,
        "qualified_correct": sum(
            r["qualification"]["qualified_correct"] for r in results
        ),
    }


def validate_format_approval(
    base: Path, manifest: dict[str, Any], definition: dict[str, Any]
) -> dict[str, Any]:
    """Require the exact human-approved format proposal and unchanged hard ceiling."""
    approval = json.loads((base / FORMAT_DIRECTORY / "approval.json").read_text())
    proposal = base / REVISION_DIRECTORY / "proposed-format-qualification.json"
    forecast_path = base / REVISION_DIRECTORY / "validation-forecast.json"
    forecast_data = json.loads(forecast_path.read_text())
    if (
        approval.get("answer") != "Approve free repair and main (recommended)"
        or not approval.get("question")
        or approval.get("original_manifest_sha256") != digest(json_text(manifest))
        or approval.get("revision_manifest_sha256")
        != digest((base / REVISION_MANIFEST).read_bytes())
        or approval.get("proposal_sha256") != digest(proposal.read_bytes())
        or approval.get("forecast_sha256") != digest(forecast_path.read_bytes())
        or approval.get("approved_forecast_usd")
        != forecast_data["projected_total_conservative_usd"]
        or approval.get("whole_study_hard_cap_usd") != study.CAP
        or approval.get("fresh_paid_validation_authorized") is not False
        or approval.get("main_assignments") != 10080
        or approval.get("repetitions") != 3
        or definition["ledger_scope"] != digest(json_text(manifest))
    ):
        raise ValueError(
            "Factual-format approval does not match the exact frozen proposal and forecast"
        )
    return approval


def prepare_format_qualification(root: Path, reports: list[Path]) -> None:
    """Freeze a free factual overlay and its one approved forecast admission exception."""
    base = study.base(root)
    directory = base / FORMAT_DIRECTORY
    if (base / FORMAT_AMENDMENT).exists():
        raise ValueError("Factual-format qualification is already frozen")
    if not reports or not all(passed_report(report) for report in reports):
        raise ValueError("Passing factual-format verification reports required")
    manifest = json.loads((base / "manifest.json").read_text())
    definition = read_revision(base)
    if definition is None:
        raise ValueError(
            "Factual-format qualification requires the approved parent revision"
        )
    approval = validate_format_approval(base, manifest, definition)
    audit = json.loads((directory / "audit.json").read_text())
    old_sources = audit["old_sources"]
    current = format_sources(root)
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if set(current) != set(old_sources) | set(FORMAT_SOURCES) or any(
        current[name] != sha for name, sha in old_sources.items() if name not in allowed
    ):
        raise ValueError("Factual-format amendment changed an unapproved source")
    if (
        old_sources != definition["code_sha256"]
        or audit["original_manifest_sha256"] != digest(json_text(manifest))
        or audit["revision_manifest_sha256"]
        != digest((base / REVISION_MANIFEST).read_bytes())
        or audit["approval_sha256"]
        != digest((directory / "approval.json").read_bytes())
        or audit.get("no_replay") is not True
        or audit.get("no_error_window_reset") is not True
    ):
        raise ValueError(
            "Factual-format preservation audit does not bind the approved parent"
        )
    for name, sha in old_sources.items():
        if digest((directory / "old-source" / name).read_bytes()) != sha:
            raise ValueError("Preserved pre-format source changed")
    for name, sha in audit["retained_artifacts"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Pre-format request artifact changed")
    original_records = json.loads((base / "assignments.json").read_text())
    records_path = base / REVISION_DIRECTORY / "revision-records.json"
    if (
        digest(records_path.read_bytes()) != definition["revision_records_sha256"]
        or digest((base / "assignments.json").read_bytes())
        != manifest["assignments_sha256"]
    ):
        raise ValueError("Factual-format amendment cannot change assigned cohorts")
    if any(
        (path_for(base, r) / "generation/started.json").exists()
        for r in original_records
        if r["phase"] == "main"
    ):
        raise ValueError(
            "Factual-format admission rule must be frozen before main starts"
        )
    records = original_records + json.loads(records_path.read_text())
    qualification = format_qualification(base, records)
    original.freeze(base / FORMAT_QUALIFICATION, qualification)
    bindings = {
        str(path.relative_to(base)): digest(path.read_bytes())
        for path in (
            directory / "approval.json",
            directory / "audit.json",
            base / FORMAT_QUALIFICATION,
            base / REVISION_DIRECTORY / "proposed-format-qualification.json",
            base / REVISION_DIRECTORY / "validation-forecast.json",
        )
    }
    amendment = {
        "kind": "approved_factual_format_v1",
        "manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "code_sha256": current,
        "audit_bindings": bindings,
        "verification_reports": {
            str(report.relative_to(root)): digest(report.read_bytes())
            for report in reports
        },
        "approved_forecast_sha256": approval["forecast_sha256"],
        "whole_study_hard_cap_usd": study.CAP,
        "no_new_paid_validation": True,
    }
    for name, sha in current.items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during factual-format freeze")
        path = directory / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    original.freeze(base / FORMAT_AMENDMENT, amendment)


def verify_format_amendment(
    root: Path, manifest: dict[str, Any], definition: dict[str, Any], *, retained: bool
) -> dict[str, Any]:
    """Verify the approved overlay separately from the immutable scientific revision."""
    base = study.base(root)
    directory = base / FORMAT_DIRECTORY
    amendment = json.loads((base / FORMAT_AMENDMENT).read_text())
    if (
        amendment.get("kind") != "approved_factual_format_v1"
        or amendment["manifest_sha256"] != digest(json_text(manifest))
        or amendment["revision_manifest_sha256"]
        != digest((base / REVISION_MANIFEST).read_bytes())
        or amendment["whole_study_hard_cap_usd"] != study.CAP
        or amendment.get("no_new_paid_validation") is not True
    ):
        raise ValueError("Factual-format amendment scope changed")
    required = {
        FORMAT_DIRECTORY + "/approval.json",
        FORMAT_DIRECTORY + "/audit.json",
        FORMAT_QUALIFICATION,
        REVISION_DIRECTORY + "/proposed-format-qualification.json",
        REVISION_DIRECTORY + "/validation-forecast.json",
    }
    if (
        set(amendment["audit_bindings"]) != required
        or not amendment["verification_reports"]
    ):
        raise ValueError("Factual-format amendment lacks complete evidence bindings")
    for name, sha in amendment["audit_bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Factual-format approval or evidence changed")
    for name, sha in amendment["verification_reports"].items():
        report = root / name
        if digest(report.read_bytes()) != sha or not passed_report(report):
            raise ValueError("Factual-format verification report changed or failed")
    approval = validate_format_approval(base, manifest, definition)
    audit = json.loads((directory / "audit.json").read_text())
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if (
        audit["old_sources"] != definition["code_sha256"]
        or audit["approval_sha256"]
        != digest((directory / "approval.json").read_bytes())
        or audit["original_manifest_sha256"] != digest(json_text(manifest))
        or audit["revision_manifest_sha256"]
        != digest((base / REVISION_MANIFEST).read_bytes())
        or audit.get("no_replay") is not True
        or audit.get("no_error_window_reset") is not True
        or set(amendment["code_sha256"])
        != set(audit["old_sources"]) | set(FORMAT_SOURCES)
        or any(
            amendment["code_sha256"][name] != sha
            for name, sha in audit["old_sources"].items()
            if name not in allowed
        )
    ):
        raise ValueError("Factual-format source or preservation scope changed")
    if amendment["approved_forecast_sha256"] != approval["forecast_sha256"]:
        raise ValueError("Factual-format forecast exception changed")
    current = format_sources(root)
    if (
        current != amendment["code_sha256"]
        and not (base / GENERATION_RECOVERY_MANIFEST).exists()
    ):
        verify_operational_amendment(
            base,
            manifest,
            current,
            retained=retained,
            scientific_revision=definition,
            format_amendment=amendment,
        )
    if retained:
        for name, sha in audit["old_sources"].items():
            if digest((directory / "old-source" / name).read_bytes()) != sha:
                raise ValueError("Preserved pre-format source changed")
        for name, sha in audit["retained_artifacts"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError(
                    "Original request changed after factual-format qualification"
                )
        for name, sha in amendment["code_sha256"].items():
            if digest((directory / "source-snapshot" / name).read_bytes()) != sha:
                raise ValueError("Frozen factual-format source changed")
    return amendment


def forecast_admission_exception(
    base: Path, manifest: dict[str, Any], cost: dict[str, Any]
) -> dict[str, Any] | None:
    """Accept only the single approved saved estimate; never raise the actual ledger cap."""
    amendment = pilot.read_optional(base / FORMAT_AMENDMENT)
    if amendment is None:
        return None
    definition = read_revision(base)
    if definition is None:
        raise ValueError("Forecast exception lacks approved revision")
    approval_path = base / FORMAT_DIRECTORY / "approval.json"
    if (
        digest(approval_path.read_bytes())
        != amendment["audit_bindings"][FORMAT_DIRECTORY + "/approval.json"]
    ):
        raise ValueError("Forecast approval binding changed")
    approval = validate_format_approval(base, manifest, definition)
    path = base / REVISION_DIRECTORY / "validation-forecast.json"
    if cost != json.loads(path.read_text()) or amendment[
        "approved_forecast_sha256"
    ] != digest(path.read_bytes()):
        raise ValueError("Forecast exception does not cover another estimate")
    return {
        "approval_sha256": digest(approval_path.read_bytes()),
        "forecast_sha256": approval["forecast_sha256"],
        "approved_forecast_usd": approval["approved_forecast_usd"],
        "hard_cap_usd": study.CAP,
    }


GENERATION_RECOVERY_DIRECTORY = "operations/recoveries/generation-rerun-v1"
GENERATION_RECOVERY_MANIFEST = GENERATION_RECOVERY_DIRECTORY + "/recovery-manifest.json"
GENERATION_RECOVERY_SOURCES = (
    "src/grader_power/style_preference_generation_recovery.py",
    "tests/test_style_preference_generation_recovery.py",
    "planning/style_preference_generation_recovery_v1.md",
)


def recovery_sources(root: Path) -> dict[str, str]:
    """Bind the single authorized fresh-generation cohort after the existing format layer."""
    return {
        **format_sources(root),
        **{
            name: digest((root / name).read_bytes())
            for name in GENERATION_RECOVERY_SOURCES
        },
    }


def validate_generation_recovery_approval(
    base: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require exact bounded human permission, preserved parents, and original ledger."""
    directory = base / GENERATION_RECOVERY_DIRECTORY
    approval = json.loads((directory / "approval.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    cohort_sha = digest((directory / "failed-generation-cohort.json").read_bytes())
    parents = {
        "original_manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "format_amendment_sha256": digest((base / FORMAT_AMENDMENT).read_bytes()),
    }
    if (
        approval.get("authorized") is not True
        or approval.get("user_message")
        != "my bad, i've added credits. please continue and rerun any generations that failed."
        or approval.get("bounded_new_attempts") != 20
        or approval.get("whole_study_hard_cap_usd") != study.CAP
        or approval.get("fresh_ids_required") is not True
        or approval.get("originals_preserved") is not True
        or approval.get("failed_generation_cohort_sha256") != cohort_sha
        or audit.get("failed_generation_cohort_sha256") != cohort_sha
        or audit.get("approval_sha256")
        != digest((directory / "approval.json").read_bytes())
        or audit.get("preserve_original_ids_and_outcomes") is not True
        or any(
            approval.get(key) != value or audit.get(key) != value
            for key, value in parents.items()
        )
        or audit["ledger"]["scope"] != parents["original_manifest_sha256"]
        or audit["ledger"]["cap"] != nanodollars(study.CAP)
    ):
        raise ValueError(
            "Generation recovery approval or preserved spending scope changed"
        )
    return approval, audit


def validate_recovery_cohort(
    base: Path, main: list[dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive fresh identities from the frozen pause snapshot, not later failures."""
    rows = generation_recovery.build_records(main, baseline)
    by_id = {r["assignment_id"]: r for r in main}
    cohort = json.loads(
        (
            base / GENERATION_RECOVERY_DIRECTORY / "failed-generation-cohort.json"
        ).read_text()
    )
    if len(cohort) != 20 or {r["source_assignment_id"] for r in rows} != {
        r["assignment_id"] for r in cohort
    }:
        raise ValueError("Recovery cohort differs from the twenty authorized failures")
    for failed in cohort:
        record = by_id[failed["assignment_id"]]
        path = path_for(base, record)
        if (
            failed["record"] != record
            or failed["result"]
            != json.loads((path / "generation/result.json").read_text())
            or failed["raw_saved"] != generation_path(base, record).exists()
        ):
            raise ValueError("Original failed generation evidence changed")
    return rows


def prepare_generation_recovery(root: Path, reports: list[Path]) -> None:
    """Freeze one additive rerun per approved failure without mutating prior requests."""
    base = study.base(root)
    directory = base / GENERATION_RECOVERY_DIRECTORY
    if (base / GENERATION_RECOVERY_MANIFEST).exists():
        raise ValueError("Generation recovery is already frozen")
    if not reports or not all(passed_report(report) for report in reports):
        raise ValueError("Passing generation-recovery verification reports required")
    manifest = json.loads((base / "manifest.json").read_text())
    _, audit = validate_generation_recovery_approval(base, manifest)
    parent = json.loads((base / FORMAT_AMENDMENT).read_text())
    current = recovery_sources(root)
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if (
        audit["old_sources"] != parent["code_sha256"]
        or set(current) != set(audit["old_sources"]) | set(GENERATION_RECOVERY_SOURCES)
        or any(
            current[name] != sha
            for name, sha in audit["old_sources"].items()
            if name not in allowed
        )
    ):
        raise ValueError("Generation recovery changed an unapproved frozen source")
    for name, sha in audit["old_sources"].items():
        if digest((directory / "old-source" / name).read_bytes()) != sha:
            raise ValueError("Pre-recovery source snapshot changed")
    for name, sha in audit["retained_artifacts"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Pre-recovery request artifact changed")
    if ledger_for(base, manifest).snapshot() != audit["ledger"]:
        raise ValueError(
            "Recovery freeze no longer matches the drained ledger snapshot"
        )
    original_path = base / "assignments.json"
    if digest(original_path.read_bytes()) != manifest["assignments_sha256"]:
        raise ValueError("Original assignments changed")
    main = revision.selected_main_records(json.loads(original_path.read_text()))
    baseline = [measure(base, r) for r in main]
    records = validate_recovery_cohort(base, main, baseline)
    if any(path_for(base, r).exists() for r in records):
        raise ValueError("A newly authorized recovery identity already has outcomes")
    original.freeze(directory / "baseline-measured-rows.json", baseline)
    original.freeze(directory / "records.json", records)
    definition = {
        "kind": "authorized_generation_rerun_v1",
        "original_manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "format_amendment_sha256": digest((base / FORMAT_AMENDMENT).read_bytes()),
        "whole_study_hard_cap_usd": study.CAP,
        "bounded_new_attempts": 20,
        "records_sha256": digest((directory / "records.json").read_bytes()),
        "baseline_measured_sha256": digest(
            (directory / "baseline-measured-rows.json").read_bytes()
        ),
        "code_sha256": current,
        "audit_bindings": {
            str(path.relative_to(base)): digest(path.read_bytes())
            for path in (
                directory / "approval.json",
                directory / "audit.json",
                directory / "failed-generation-cohort.json",
                directory / "stage-events.jsonl",
                directory / "transport-events.jsonl",
            )
        },
        "verification_reports": {
            str(report.relative_to(root)): digest(report.read_bytes())
            for report in reports
        },
    }
    for name, sha in current.items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during generation recovery freeze")
        path = directory / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    original.freeze(base / GENERATION_RECOVERY_MANIFEST, definition)


def verify_generation_recovery(
    root: Path,
    manifest: dict[str, Any],
    revision_definition: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    retained: bool,
) -> list[dict[str, Any]]:
    """Verify additive recovery against its fixed failed cohort and all parent approvals."""
    base = study.base(root)
    directory = base / GENERATION_RECOVERY_DIRECTORY
    definition = json.loads((base / GENERATION_RECOVERY_MANIFEST).read_text())
    if (
        definition.get("kind") != "authorized_generation_rerun_v1"
        or definition["original_manifest_sha256"] != digest(json_text(manifest))
        or definition["revision_manifest_sha256"]
        != digest((base / REVISION_MANIFEST).read_bytes())
        or definition["format_amendment_sha256"]
        != digest((base / FORMAT_AMENDMENT).read_bytes())
        or definition["whole_study_hard_cap_usd"] != study.CAP
        or definition["bounded_new_attempts"] != 20
    ):
        raise ValueError("Generation recovery scope changed")
    required = {
        GENERATION_RECOVERY_DIRECTORY + "/" + name
        for name in (
            "approval.json",
            "audit.json",
            "failed-generation-cohort.json",
            "stage-events.jsonl",
            "transport-events.jsonl",
        )
    }
    if (
        set(definition["audit_bindings"]) != required
        or not definition["verification_reports"]
    ):
        raise ValueError("Generation recovery lacks complete evidence bindings")
    for name, sha in definition["audit_bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Generation recovery evidence changed")
    for name, sha in definition["verification_reports"].items():
        report = root / name
        if digest(report.read_bytes()) != sha or not passed_report(report):
            raise ValueError("Generation recovery verification failed")
    _, audit = validate_generation_recovery_approval(base, manifest)
    parent = json.loads((base / FORMAT_AMENDMENT).read_text())
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if (
        audit["old_sources"] != parent["code_sha256"]
        or set(definition["code_sha256"])
        != set(audit["old_sources"]) | set(GENERATION_RECOVERY_SOURCES)
        or any(
            definition["code_sha256"][name] != sha
            for name, sha in audit["old_sources"].items()
            if name not in allowed
        )
    ):
        raise ValueError("Generation recovery source scope changed")
    current = recovery_sources(root)
    if (
        current != definition["code_sha256"]
        and not (base / GENERATION_RECOVERY_V2_MANIFEST).exists()
    ):
        verify_operational_amendment(
            base,
            manifest,
            current,
            retained=retained,
            scientific_revision=revision_definition,
            format_amendment=parent,
            generation_recovery_manifest=definition,
        )
    baseline_path = directory / "baseline-measured-rows.json"
    records_path = directory / "records.json"
    if (
        digest(baseline_path.read_bytes()) != definition["baseline_measured_sha256"]
        or digest(records_path.read_bytes()) != definition["records_sha256"]
    ):
        raise ValueError("Generation recovery cohort or baseline changed")
    main = [r for r in records if r["phase"] == "main"]
    expected = validate_recovery_cohort(
        base, main, json.loads(baseline_path.read_text())
    )
    extra = json.loads(records_path.read_text())
    if extra != expected or {r["assignment_id"] for r in extra} & {
        r["assignment_id"] for r in records
    }:
        raise ValueError(
            "Generation recovery differs from prospective fresh-ID builder"
        )
    if retained:
        for name, sha in audit["retained_artifacts"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Original outcome changed during generation recovery")
        for name, sha in audit["old_sources"].items():
            if digest((directory / "old-source" / name).read_bytes()) != sha:
                raise ValueError("Pre-recovery source snapshot changed")
        for name, sha in definition["code_sha256"].items():
            if digest((directory / "source-snapshot" / name).read_bytes()) != sha:
                raise ValueError("Frozen generation-recovery source changed")
    return records + extra


GENERATION_RECOVERY_V2_DIRECTORY = "operations/recoveries/generation-rerun-v2"
GENERATION_RECOVERY_V2_MANIFEST = (
    GENERATION_RECOVERY_V2_DIRECTORY + "/recovery-manifest.json"
)
GENERATION_RECOVERY_V2_SOURCES = (
    "src/grader_power/style_preference_generation_recovery_v2.py",
    "tests/test_style_preference_generation_recovery_v2.py",
    "planning/style_preference_generation_recovery_v2.md",
)


def recovery_v2_sources(root: Path) -> dict[str, str]:
    """Bind the single authorized fresh-generation cohort after the existing format layer."""
    return {
        **recovery_sources(root),
        **{
            name: digest((root / name).read_bytes())
            for name in GENERATION_RECOVERY_V2_SOURCES
        },
    }


def validate_generation_recovery_v2_approval(
    base: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require exact bounded human permission, preserved parents, and original ledger."""
    directory = base / GENERATION_RECOVERY_V2_DIRECTORY
    approval = json.loads((directory / "approval.json").read_text())
    audit = json.loads((directory / "audit.json").read_text())
    cohort_sha = digest((directory / "failed-generation-cohort.json").read_bytes())
    parents = {
        "original_manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "format_amendment_sha256": digest((base / FORMAT_AMENDMENT).read_bytes()),
        "prior_recovery_manifest_sha256": digest(
            (base / GENERATION_RECOVERY_MANIFEST).read_bytes()
        ),
    }
    if (
        approval.get("authorized") is not True
        or approval.get("user_message")
        != "oops payment didn't go through the first time. please try again"
        or approval.get("bounded_new_attempts") != 20
        or approval.get("whole_study_hard_cap_usd") != study.CAP
        or approval.get("fresh_ids_required") is not True
        or approval.get("originals_preserved") is not True
        or approval.get("failed_generation_cohort_sha256") != cohort_sha
        or audit.get("failed_generation_cohort_sha256") != cohort_sha
        or audit.get("approval_sha256")
        != digest((directory / "approval.json").read_bytes())
        or audit.get("preserve_original_ids_and_outcomes") is not True
        or audit.get("all_v1_attempts_rejected_or_unsent") is not True
        or any(
            approval.get(key) != value or audit.get(key) != value
            for key, value in parents.items()
        )
        or audit["ledger"]["scope"] != parents["original_manifest_sha256"]
        or audit["ledger"]["cap"] != nanodollars(study.CAP)
    ):
        raise ValueError(
            "Generation recovery approval or preserved spending scope changed"
        )
    return approval, audit


def validate_recovery_v2_cohort(
    base: Path, main: list[dict[str, Any]], first_measured: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Freeze the same twenty cells only after rejected or unsent first attempts."""
    first_directory = base / GENERATION_RECOVERY_DIRECTORY
    first_records = json.loads((first_directory / "records.json").read_text())
    baseline = json.loads((first_directory / "baseline-measured-rows.json").read_text())
    rows = generation_recovery_v2.build_records(
        main, baseline, first_records, first_measured
    )
    cohort = json.loads(
        (
            base / GENERATION_RECOVERY_V2_DIRECTORY / "failed-generation-cohort.json"
        ).read_text()
    )
    by_id = {r["assignment_id"]: r for r in first_records}
    if len(cohort) != 20 or {r["assignment_id"] for r in cohort} != set(by_id):
        raise ValueError(
            "Second recovery differs from the twenty failed first attempts"
        )
    for failed in cohort:
        record = by_id[failed["assignment_id"]]
        path = path_for(base, record) / "generation"
        result = json.loads((path / "result.json").read_text())
        transport = json.loads((path / "transport.json").read_text())
        rejected = (
            transport.get("attempts") == 1
            and transport.get("state") == "not_retried"
            and transport.get("http_status") == 412
            and result.get("http_status") == 412
        )
        unsent = (
            transport.get("attempts") == 0
            and transport.get("state") == "pending_admission"
            and transport.get("http_status") is None
            and result.get("http_status") is None
        )
        if (
            failed["record"] != record
            or failed["result"] != result
            or failed["transport"] != transport
            or failed["raw_saved"] is not False
            or (path / "raw.json").exists()
            or result.get("status") != "request_error"
            or not (rejected or unsent)
        ):
            raise ValueError(
                "First recovery evidence must remain rejected or unsent, never ambiguous"
            )
    if first_measured != [measure(base, r) for r in first_records]:
        raise ValueError("First recovery measured outcomes changed")
    return rows


def prepare_generation_recovery_v2(root: Path, reports: list[Path]) -> None:
    """Freeze one additive rerun per approved failure without mutating prior requests."""
    base = study.base(root)
    directory = base / GENERATION_RECOVERY_V2_DIRECTORY
    if (base / GENERATION_RECOVERY_V2_MANIFEST).exists():
        raise ValueError("Generation recovery is already frozen")
    if not reports or not all(passed_report(report) for report in reports):
        raise ValueError("Passing generation-recovery verification reports required")
    manifest = json.loads((base / "manifest.json").read_text())
    _, audit = validate_generation_recovery_v2_approval(base, manifest)
    parent = json.loads((base / GENERATION_RECOVERY_MANIFEST).read_text())
    current = recovery_v2_sources(root)
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if (
        audit["old_sources"] != parent["code_sha256"]
        or set(current)
        != set(audit["old_sources"]) | set(GENERATION_RECOVERY_V2_SOURCES)
        or any(
            current[name] != sha
            for name, sha in audit["old_sources"].items()
            if name not in allowed
        )
    ):
        raise ValueError("Generation recovery changed an unapproved frozen source")
    for name, sha in audit["old_sources"].items():
        if digest((directory / "old-source" / name).read_bytes()) != sha:
            raise ValueError("Pre-recovery source snapshot changed")
    for name, sha in audit["retained_artifacts"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Pre-recovery request artifact changed")
    if ledger_for(base, manifest).snapshot() != audit["ledger"]:
        raise ValueError(
            "Recovery freeze no longer matches the drained ledger snapshot"
        )
    original_path = base / "assignments.json"
    if digest(original_path.read_bytes()) != manifest["assignments_sha256"]:
        raise ValueError("Original assignments changed")
    main = revision.selected_main_records(json.loads(original_path.read_text()))
    first_records = json.loads(
        (base / GENERATION_RECOVERY_DIRECTORY / "records.json").read_text()
    )
    baseline = [measure(base, r) for r in first_records]
    records = validate_recovery_v2_cohort(base, main, baseline)
    if any(path_for(base, r).exists() for r in records):
        raise ValueError("A newly authorized recovery identity already has outcomes")
    original.freeze(directory / "first-measured-rows.json", baseline)
    original.freeze(directory / "records.json", records)
    definition = {
        "kind": "authorized_generation_rerun_v2",
        "original_manifest_sha256": digest(json_text(manifest)),
        "revision_manifest_sha256": digest((base / REVISION_MANIFEST).read_bytes()),
        "format_amendment_sha256": digest((base / FORMAT_AMENDMENT).read_bytes()),
        "prior_recovery_manifest_sha256": digest(
            (base / GENERATION_RECOVERY_MANIFEST).read_bytes()
        ),
        "whole_study_hard_cap_usd": study.CAP,
        "bounded_new_attempts": 20,
        "records_sha256": digest((directory / "records.json").read_bytes()),
        "first_measured_sha256": digest(
            (directory / "first-measured-rows.json").read_bytes()
        ),
        "code_sha256": current,
        "audit_bindings": {
            str(path.relative_to(base)): digest(path.read_bytes())
            for path in (
                directory / "approval.json",
                directory / "audit.json",
                directory / "failed-generation-cohort.json",
                directory / "stage-events.jsonl",
                directory / "transport-events.jsonl",
            )
        },
        "verification_reports": {
            str(report.relative_to(root)): digest(report.read_bytes())
            for report in reports
        },
    }
    for name, sha in current.items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during generation recovery freeze")
        path = directory / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    original.freeze(base / GENERATION_RECOVERY_V2_MANIFEST, definition)


def verify_generation_recovery_v2(
    root: Path,
    manifest: dict[str, Any],
    revision_definition: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    retained: bool,
) -> list[dict[str, Any]]:
    """Verify additive recovery against its fixed failed cohort and all parent approvals."""
    base = study.base(root)
    directory = base / GENERATION_RECOVERY_V2_DIRECTORY
    definition = json.loads((base / GENERATION_RECOVERY_V2_MANIFEST).read_text())
    if (
        definition.get("kind") != "authorized_generation_rerun_v2"
        or definition["original_manifest_sha256"] != digest(json_text(manifest))
        or definition["revision_manifest_sha256"]
        != digest((base / REVISION_MANIFEST).read_bytes())
        or definition["format_amendment_sha256"]
        != digest((base / FORMAT_AMENDMENT).read_bytes())
        or definition["prior_recovery_manifest_sha256"]
        != digest((base / GENERATION_RECOVERY_MANIFEST).read_bytes())
        or definition["whole_study_hard_cap_usd"] != study.CAP
        or definition["bounded_new_attempts"] != 20
    ):
        raise ValueError("Generation recovery scope changed")
    required = {
        GENERATION_RECOVERY_V2_DIRECTORY + "/" + name
        for name in (
            "approval.json",
            "audit.json",
            "failed-generation-cohort.json",
            "stage-events.jsonl",
            "transport-events.jsonl",
        )
    }
    if (
        set(definition["audit_bindings"]) != required
        or not definition["verification_reports"]
    ):
        raise ValueError("Generation recovery lacks complete evidence bindings")
    for name, sha in definition["audit_bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Generation recovery evidence changed")
    for name, sha in definition["verification_reports"].items():
        report = root / name
        if digest(report.read_bytes()) != sha or not passed_report(report):
            raise ValueError("Generation recovery verification failed")
    _, audit = validate_generation_recovery_v2_approval(base, manifest)
    parent = json.loads((base / GENERATION_RECOVERY_MANIFEST).read_text())
    allowed = {
        "evals/inspect_style_preferences.py",
        "scripts/run_style_preferences.py",
        "tests/test_style_preferences_runner.py",
    }
    if (
        audit["old_sources"] != parent["code_sha256"]
        or set(definition["code_sha256"])
        != set(audit["old_sources"]) | set(GENERATION_RECOVERY_V2_SOURCES)
        or any(
            definition["code_sha256"][name] != sha
            for name, sha in audit["old_sources"].items()
            if name not in allowed
        )
    ):
        raise ValueError("Generation recovery source scope changed")
    current = recovery_v2_sources(root)
    if current != definition["code_sha256"]:
        verify_operational_amendment(
            base,
            manifest,
            current,
            retained=retained,
            scientific_revision=revision_definition,
            format_amendment=json.loads((base / FORMAT_AMENDMENT).read_text()),
            generation_recovery_manifest=parent,
            generation_recovery_v2_manifest=definition,
        )
    baseline_path = directory / "first-measured-rows.json"
    records_path = directory / "records.json"
    if (
        digest(baseline_path.read_bytes()) != definition["first_measured_sha256"]
        or digest(records_path.read_bytes()) != definition["records_sha256"]
    ):
        raise ValueError("Generation recovery cohort or baseline changed")
    main = [r for r in records if r["phase"] == "main"]
    expected = validate_recovery_v2_cohort(
        base, main, json.loads(baseline_path.read_text())
    )
    extra = json.loads(records_path.read_text())
    if extra != expected or {r["assignment_id"] for r in extra} & {
        r["assignment_id"] for r in records
    }:
        raise ValueError(
            "Generation recovery differs from prospective fresh-ID builder"
        )
    if retained:
        for name, sha in audit["retained_artifacts"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Original outcome changed during generation recovery")
        for name, sha in audit["old_sources"].items():
            if digest((directory / "old-source" / name).read_bytes()) != sha:
                raise ValueError("Pre-recovery source snapshot changed")
        for name, sha in definition["code_sha256"].items():
            if digest((directory / "source-snapshot" / name).read_bytes()) != sha:
                raise ValueError("Frozen generation-recovery source changed")
    return records + extra


def generation_recovery_gate(
    base: Path, records: list[dict[str, Any]], *, phase: str = generation_recovery.PHASE
) -> dict[str, Any]:
    """Confirm the bounded fresh requests returned before resuming remaining main work."""
    rows = [measure(base, r) for r in records if r["phase"] == phase]
    return {
        "passed": len(rows) == 20
        and all(
            r["attempt_complete"]
            and r["raw_saved"]
            and r["generation_status"] in ("completed", "incomplete")
            for r in rows
        ),
        "assigned": len(rows),
        "started": sum(r["started"] for r in rows),
        "completed_attempts": sum(r["attempt_complete"] for r in rows),
        "saved": sum(r["raw_saved"] for r in rows),
        "terminal_generation_failures": sum(
            r["generation_status"] in ("request_error", "incomplete") for r in rows
        ),
        "new_attempts_per_original": 1,
    }


RESTORED_BILLING_DIRECTORY = "operations/recoveries/restored-billing-admission-v1"


def restored_billing_admission(
    base: Path, records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Admit main only from audited later successes; never clear a recovery block.

    This one operational correction preserves the failed v2 completeness gate.
    It cannot authorize another recovery attempt or replace any scientific gate.
    """
    state = pilot.read_optional(base / "operations/recovery-state.json") or {}
    if state.get("retry_blocked"):
        raise ValueError("Recovery blocked pending new human direction")
    relative = state.get("source_amendment")
    if not relative:
        return None
    amendment_path = base / relative
    if digest(amendment_path.read_bytes()) != state.get("source_amendment_sha256"):
        raise ValueError("Billing admission amendment binding changed")
    amendment = json.loads(amendment_path.read_text())
    required = {
        RESTORED_BILLING_DIRECTORY + "/" + name
        for name in (
            "audit.json",
            "transport-events.jsonl",
            "original-fresh-generation-gate.json",
        )
    }
    bindings = amendment.get("audit_bindings", {})
    if not required <= set(bindings):
        return None
    for name in required:
        if digest((base / name).read_bytes()) != bindings[name]:
            raise ValueError("Restored-billing evidence binding changed")
    parent_sha = digest((base / GENERATION_RECOVERY_V2_MANIFEST).read_bytes())
    directory = base / RESTORED_BILLING_DIRECTORY
    audit = json.loads((directory / "audit.json").read_text())
    gate_path = base / GENERATION_RECOVERY_V2_DIRECTORY / "fresh-generation-gate.json"
    old_gate_path = directory / "original-fresh-generation-gate.json"
    if (
        amendment.get("generation_recovery_v2_manifest_sha256") != parent_sha
        or audit.get("parent_manifest_sha256") != parent_sha
        or digest(gate_path.read_bytes()) != audit.get("original_gate_sha256")
        or gate_path.read_bytes() != old_gate_path.read_bytes()
        or json.loads(gate_path.read_text())
        != generation_recovery_gate(base, records, phase=generation_recovery_v2.PHASE)
    ):
        raise ValueError(
            "Billing admission must preserve the failed v2 gate and cohort"
        )
    selected = [r for r in records if r["phase"] == generation_recovery_v2.PHASE]
    measured = [measure(base, r) for r in selected]
    failed = [r for r in measured if not r["raw_saved"]]
    if (
        len(selected) != 20
        or len(failed) != 1
        or not all(r["started"] and r["attempt_complete"] for r in measured)
        or any(
            r["generation_status"] not in ("completed", "incomplete")
            for r in measured
            if r["raw_saved"]
        )
    ):
        raise ValueError(
            "Billing admission requires twenty terminal attempts and one retained rejection"
        )
    by_id = {r["assignment_id"]: r for r in selected}
    transports = {}
    for record in selected:
        path = path_for(base, record) / "generation"
        names = ["started.json", "result.json", "transport.json"]
        if (path / "raw.json").exists():
            names.append("raw.json")
        for name in names:
            artifact = path / name
            if digest(artifact.read_bytes()) != audit["retained_artifacts"].get(
                str(artifact.relative_to(base))
            ):
                raise ValueError("Audited recovery request changed")
        transport = json.loads((path / "transport.json").read_text())
        profile = study.PROFILES[record["profile"]]
        if (
            transport.get("profile") != record["profile"]
            or transport.get("provider") != profile["provider"]
            or transport.get("model") != profile["model"].split("/", 1)[1]
            or not transport.get("request_sha256")
            or not transport.get("ticket")
        ):
            raise ValueError("Billing evidence model or request identity changed")
        transports[record["assignment_id"]] = transport
    rejected = failed[0]
    rejection = transports[rejected["assignment_id"]]
    result = json.loads(
        (
            path_for(base, by_id[rejected["assignment_id"]]) / "generation/result.json"
        ).read_text()
    )
    if (
        rejected["generation_status"] != "request_error"
        or result.get("http_status") != 412
        or rejection.get("state") != "not_retried"
        or rejection.get("attempts") != 1
        or rejection.get("http_status") != 412
    ):
        raise ValueError("Only the retained explicit HTTP412 rejection is eligible")
    events = [
        json.loads(line)
        for line in (directory / "transport-events.jsonl").read_text().splitlines()
        if line
    ]

    def matching(transport: dict[str, Any], event: str) -> list[dict[str, Any]]:
        return [
            item
            for item in events
            if item.get("event") == event
            and all(
                item.get(key) == transport.get(key)
                for key in ("profile", "provider", "model", "request_sha256", "ticket")
            )
        ]

    rejection_headers = matching(rejection, "response_headers")
    if len(rejection_headers) != 1 or rejection_headers[0].get("http_status") != 412:
        raise ValueError("Explicit rejection lacks unique bound response evidence")
    rejected_at = rejection_headers[0]["at"]
    successful = []
    for row in measured:
        transport = transports[row["assignment_id"]]
        if not row["raw_saved"] or any(
            transport.get(k) != rejection.get(k) for k in ("provider", "model")
        ):
            continue
        sent = matching(transport, "send_started")
        headers = matching(transport, "response_headers")
        if (
            transport.get("state") == "accepted"
            and transport.get("http_status") == 200
            and transport.get("attempts") == 1
            and len(sent) == len(headers) == 1
            and headers[0].get("http_status") == 200
            and rejected_at < sent[0]["at"] < headers[0]["at"]
        ):
            successful.append(row["assignment_id"])
    if not successful:
        raise ValueError(
            "No saved same-model success was sent after the explicit rejection"
        )
    return {
        "kind": "audited_main_only_billing_admission",
        "recovery_manifest_sha256": parent_sha,
        "original_gate_sha256": audit["original_gate_sha256"],
        "audit_sha256": bindings[RESTORED_BILLING_DIRECTORY + "/audit.json"],
        "transport_events_sha256": bindings[
            RESTORED_BILLING_DIRECTORY + "/transport-events.jsonl"
        ],
        "retained_rejection_assignment_id": rejected["assignment_id"],
        "later_same_model_successes": sorted(successful),
        "new_recovery_requests_authorized": False,
    }


def load(
    root: Path, *, retained: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify exact frozen science and all evidence; refuse silent source changes."""
    base = study.base(root)
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest["run"] != study.RUN or manifest["cap_usd"] != study.CAP:
        raise ValueError("Frozen spending scope changed")
    revised = read_revision(base) is not None
    current = sources(root)
    if not revised and manifest["code_sha256"] != current:
        verify_operational_amendment(base, manifest, current, retained=retained)
    if (
        digest((base / "assignments.json").read_bytes())
        != manifest["assignments_sha256"]
    ):
        raise ValueError("Assignments changed")
    for name, sha in {
        **manifest["verification_reports"],
        **manifest["historical_bindings"],
    }.items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Bound evidence changed: " + name)
    if retained:
        for name, sha in manifest["code_sha256"].items():
            if digest((base / "source-snapshot" / name).read_bytes()) != sha:
                raise ValueError("Frozen source snapshot changed")
    records = json.loads((base / "assignments.json").read_text())
    if revised:
        _, records = verified_revision(root, manifest, records, retained=retained)
    return manifest, records


def ledger_for(base: Path, manifest: dict[str, Any]) -> RequestLedger:
    """Use one flat ledger with no duplicate full-block reservations."""
    return RequestLedger(
        base / "spending.sqlite3",
        cap=nanodollars(study.CAP),
        scope=digest(json_text(manifest)),
    )


def path_for(base: Path, r: dict[str, Any]) -> Path:
    """Locate this run's immutable assignment outcomes."""
    return base / "requests" / r["assignment_id"]


def generation_path(base: Path, r: dict[str, Any]) -> Path:
    """Resolve an explicitly frozen judge-only source without copying or replaying it."""
    source_id = r.get("generation_source_assignment_id", r["assignment_id"])
    return base / "requests" / source_id / "generation/raw.json"


def measure(base: Path, r: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct counts directly from durable outcomes, including missing judgments."""
    path = path_for(base, r)
    generation = pilot.read_optional(path / "generation/result.json") or {}
    score_path = (
        base / "requests" / r.get("generation_source_assignment_id", r["assignment_id"])
    )
    scoring = pilot.read_optional(score_path / "scoring.json") or {}
    judged = pilot.read_optional(path / "judge/result.json") or {}
    probe = pilot.read_optional(path / "probe.json") or {}
    return {
        k: r[k]
        for k in (
            "assignment_id",
            "profile",
            "study",
            "phase",
            "kind",
            "source_task_id",
            "condition",
            "mode",
            "repetition_index",
        )
    } | {
        **{
            k: r[k]
            for k in (
                "revision_kind",
                "source_assignment_id",
                "judge_only",
                "generation_source_assignment_id",
                "recovery_generation",
                "generation_recovery",
                "prior_recovery_assignment_id",
            )
            if k in r
        },
        "provider": None
        if r["kind"] == "judge_calibration"
        else study.PROFILES[r["profile"]]["provider"],
        "generation_model": None
        if r["kind"] == "judge_calibration"
        else study.PROFILES[r["profile"]]["model"],
        "reused_generation": bool(r.get("generation_source_assignment_id")),
        "judge_raw_saved": (path / "judge/raw.json").exists(),
        "started": (path / "generation/started.json").exists(),
        "attempt_complete": bool(generation),
        "raw_saved": generation_path(base, r).exists(),
        "scored": bool(scoring),
        "full_correct": scoring.get("correctness", {}).get("full_correct"),
        "judge_started": (path / "judge/started.json").exists(),
        "judge_result": bool(judged),
        "label": judged.get("label"),
        "audit_label": judged.get("audit_label"),
        "audit_agrees": judged.get("audit_agrees"),
        "probe_correct": probe.get("fence_only", {}).get("all_correct"),
        "probe_correct_strict": probe.get("strict", {}).get("all_correct"),
        "generation_status": generation.get("status"),
        "judge_status": judged.get("status"),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep families, model provenance, pilot facts and main missingness separate."""
    result = {}
    for phase in sorted({r["phase"] for r in rows}):
        for profile in study.PROFILES:
            for family in FAMILIES:
                selected = [
                    r
                    for r in rows
                    if (r["phase"], r["profile"], r["study"])
                    == (phase, profile, family)
                ]
                code = [
                    r for r in selected if r["kind"] in ("core", "judge_calibration")
                ]
                result[f"{phase}/{profile}/{family}"] = {
                    "assigned": len(selected),
                    "coding_assigned": sum(r["kind"] == "core" for r in code),
                    "raw_saved": sum(
                        r["raw_saved"] and not r.get("reused_generation", False)
                        for r in selected
                    ),
                    "available_code_responses": sum(
                        r["raw_saved"] for r in selected if r["kind"] == "core"
                    ),
                    "calibration_assigned": sum(
                        r["kind"] == "judge_calibration" for r in code
                    ),
                    "new_saved_generations": sum(
                        r["raw_saved"] and not r.get("reused_generation", False)
                        for r in selected
                    ),
                    "reused_generation_references": sum(
                        r["raw_saved"] and r.get("reused_generation", False)
                        for r in selected
                    ),
                    "judge_raw_saved": sum(
                        r.get("judge_raw_saved", False) for r in selected
                    ),
                    **{
                        k: sum(bool(r[k]) for r in selected)
                        for k in ("started", "attempt_complete", "scored")
                    },
                    "resolved": sum(r["label"] is not None for r in code),
                    "terminal_unresolved": sum(
                        r["judge_result"] and r["label"] is None for r in code
                    ),
                    "inflight_judges": sum(
                        r["judge_started"] and not r["judge_result"] for r in code
                    ),
                    "pending_judges": sum(
                        r["raw_saved"]
                        and not r["judge_started"]
                        and not r["judge_result"]
                        for r in code
                    ),
                    "facts_correct": sum(r["probe_correct"] is True for r in selected),
                }
    return result


def packet_from_output(
    raw: dict[str, Any], r: dict[str, Any]
) -> preference_judge.JudgeInput | None:
    """Expose only source and original API metadata to the blinded judge."""
    output = ModelOutput.model_validate(raw)
    extraction = extract_code(output.completion, r["entry_point"])
    if extraction.code is None or extraction.status != "ok":
        return None
    try:
        return preference_judge.JudgeInput(
            family=r["study"],
            entry_point=r["entry_point"],
            original_parameters=r["original_parameters"],
            code=extraction.code,
        )
    except (ValueError, TypeError):
        return None


def render_packet(design: str, data: dict[str, Any]) -> str:
    """Validate neutral family binding before sending a code-only packet."""
    packet = preference_judge.JudgeInput.model_validate(data)
    if packet.family != design:
        raise ValueError("Wrong family in judge packet")
    return preference_judge.render_judge_input(packet)


def parse_packet(
    design: str, text: str, data: dict[str, Any]
) -> preference_judge.Judgment:
    """Keep the calibrated V3 strict schema and explicit absent-evidence labels."""
    packet = preference_judge.JudgeInput.model_validate(data)
    if packet.family != design:
        raise ValueError("Wrong family in judge packet")
    return preference_judge.parse_range_judgment(
        text, packet, distinguish_no_evidence=True
    )


def forecast(
    root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project both main families from coding pilot usage with a 50% safety margin."""
    base = study.base(root)
    models = {}
    bindings = {}
    for profile in study.PROFILES:
        stages: dict[str, list[float]] = {"generation": [], "judge": []}
        for row in records:
            if (
                row["phase"] != "pilot"
                or row["profile"] != profile
                or row["kind"] != "core"
            ):
                continue
            for stage, amounts in stages.items():
                path = path_for(base, row) / stage / "raw.json"
                if path.exists():
                    raw = json.loads(path.read_text())
                    spec = (
                        manifest["profiles"][profile]
                        if stage == "generation"
                        else manifest["judge_profile"]
                    )
                    amounts.append(
                        usage_cost(
                            raw["usage"],
                            input_rate=spec["input_rate"],
                            output_rate=spec["output_rate"],
                        )
                        / 1e9
                    )
                    bindings[str(path.relative_to(base))] = digest(path.read_bytes())
        if not all(stages.values()):
            raise ValueError("Pilot usage evidence missing for " + profile)
        means = {stage: sum(values) / len(values) for stage, values in stages.items()}
        models[profile] = {
            "per_response": means,
            "projected_main": sum(means.values()) * 2688,
        }
    balance = ledger_for(base, manifest).snapshot()
    total = (balance["spent"] + balance["outstanding"]) / 1e9 + 1.5 * sum(
        v["projected_main"] for v in models.values()
    )
    return {
        "manifest_sha256": digest(json_text(manifest)),
        "models": models,
        "projected_total_conservative_usd": total,
        "margin": 1.5,
        "bindings": bindings,
    }


class Pipeline:
    """Overlap lightweight API tasks with four verified Docker scoring threads."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        records: list[dict[str, Any]],
        phase: str,
        family: str | None = None,
    ) -> None:
        self.root, self.manifest, self.records, self.phase = (
            root,
            manifest,
            records,
            phase,
        )
        self.family = family
        self.base = study.base(root)
        self.operations = self.base / "operations"
        self.ledger = ledger_for(self.base, manifest)
        self.revision = read_revision(self.base)
        self.judges = copy.deepcopy(manifest["judges"])
        if self.revision is not None:
            self.judges["naming"]["system_prompt"] = self.revision[
                "naming_judge_prompt"
            ]
        self.stop = asyncio.Event()
        self.finishing = asyncio.Event()
        self.reason: str | None = None
        _, self.datasets = load_bundles(root)
        self.measurements = {r["assignment_id"]: measure(self.base, r) for r in records}
        self.history = (
            [
                json.loads(s)
                for s in (self.operations / "stage-events.jsonl")
                .read_text()
                .splitlines()
            ]
            if (self.operations / "stage-events.jsonl").exists()
            else []
        )
        self.events_seen = {(r["identity"], r["stage"]) for r in self.history}
        self.queues: dict[str, asyncio.Queue[Any]] = {
            k: asyncio.Queue() for k in study.PROFILES
        }
        self.scoring: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.judging: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.account = AccountRPM(self.operations / "pacing")
        self.models: dict[str, Any] = {}
        self.keys: set[str] = set()
        self.active = Counter()
        self.started_at = time.time()
        self.targets = {k: 8 for k in (*study.PROFILES, "judge")}
        self.completions: list[dict[str, Any]] = []

    def check(self) -> None:
        """Reject new paid work immediately after any admission stop."""
        if self.stop.is_set():
            raise RuntimeError("Admission stopped: " + str(self.reason))
        if (self.base / "STOP").exists() or (self.operations / "STOP").exists():
            self.pause("Human STOP")
            raise RuntimeError("Human STOP")

    def pause(self, reason: str) -> None:
        """Persist first stop; accepted requests drain without further paid admission."""
        if not self.stop.is_set():
            self.reason = reason
            write_json(
                self.operations / "controller-stop.json",
                {"reason": reason, "at": time.time(), "pid": os.getpid()},
            )
            append_event(
                self.operations / "events.jsonl",
                {"event": "admission_stopped", "reason": reason},
            )
            self.stop.set()

    def event(
        self, r: dict[str, Any], stage: str, failed: bool, error: str | None = None
    ) -> None:
        """Append each terminal stage once and preserve the five-in-48 safeguard."""
        identity = r["assignment_id"]
        if (identity, stage) not in self.events_seen:
            e = {
                "identity": identity,
                "stage": stage,
                "failed": failed,
                "error_type": error,
                "profile": r["profile"],
                "at": time.time(),
            }
            self.history.append(e)
            self.events_seen.add((identity, stage))
            append_event(self.operations / "stage-events.jsonl", e)
            recovery = (
                pilot.read_optional(self.operations / "recovery-state.json") or {}
            )
            if (
                sum(
                    x["failed"]
                    for x in self.history[recovery.get("error_window_origin", 0) :][
                        -48:
                    ]
                )
                >= 5
            ):
                self.pause("Five failed stages in last 48")
        self.measurements[identity] = measure(self.base, r)

    def initialize_models(self) -> None:
        """Construct verified zero-retry clients once; never serialize credentials."""
        values = {**dotenv_values(self.root / ".env"), **os.environ}
        specs = {k: (v, v["generation"]) for k, v in self.manifest["profiles"].items()}
        specs.update(
            {
                "judge/" + k: (study.JUDGE_PROFILE, v["generation"])
                for k, v in self.judges.items()
            }
        )
        for name, (spec, config) in specs.items():
            key = values.get(spec["api_key_env"])
            if not key:
                raise ValueError("Missing credential: " + spec["api_key_env"])
            self.keys.add(key)
            model = get_model(
                spec["model"],
                base_url=spec["base_url"],
                api_key=key,
                config=GenerateConfig.model_validate(config),
                memoize=False,
            )
            original.freeze(
                self.operations / "retry-policies" / (name.replace("/", "-") + ".json"),
                pilot.resolved_retry_policy(model),
            )
            self.models[name] = model

    async def request(
        self, r: dict[str, Any], stage: str, messages: list[Any]
    ) -> dict[str, Any] | None:
        """Reserve, persist intent, dispatch once, retain raw usage and uncertain charges."""
        self.check()
        name = r["profile"] if stage == "generation" else "judge/" + r["study"]
        profile = r["profile"] if stage == "generation" else "sonnet"
        spec = (
            self.manifest["profiles"][profile]
            if stage == "generation"
            else self.manifest["judge_profile"]
        )
        path = path_for(self.base, r) / stage
        if (path / "started.json").exists():
            return pilot.read_optional(path / "raw.json")
        content = [m.model_dump(mode="json") for m in messages]
        size = len(json_text(content).encode())
        if stage == "judge" and size > 32768:
            raise ValueError("Judge input exceeds frozen 32768-byte bound")
        bound = nanodollars(
            ((size + 2048) * spec["input_rate"] * 2 + 4096 * spec["output_rate"]) / 1e6
        )
        identity = r["assignment_id"] + "/" + stage
        if r["phase"] in (CALIBRATION_PHASE, revision.PHASE):
            if self.revision is None:
                raise ValueError("Revised paid request lacks frozen approval")
            validation_budget_check(self.ledger, self.revision, bound)
        self.ledger.reserve({identity: bound})
        original.freeze(
            path / "started.json",
            {
                "at": time.time(),
                "model": spec["model"],
                "messages_sha256": digest(json_text(content)),
                "reserved": bound,
            },
        )
        pacer = ModelPacer(self.operations / "pacing" / profile, profile=profile)
        clock = WaitClock()
        context = RequestContext(
            profile,
            spec["provider"],
            spec["model"].split("/", 1)[1],
            path,
            self.operations,
            pacer,
            self.account,
            clock,
            self.check,
        )
        outcome: dict[str, Any] = {"status": "request_error"}
        raw = None
        start = time.time()
        try:
            with request_scope(context):
                response = await clock.wait_for(
                    self.models[name].generate(
                        messages, tools=[], tool_choice="none", cache=False
                    ),
                    timeout=200,
                )
            candidate = response.model_dump(mode="json")
            if any(k in json_text(candidate) for k in self.keys):
                self.pause("Credential echo rejected; integrity review required")
                raise ValueError("Credential echo rejected")
            raw = candidate
            original.freeze(path / "raw.json", raw)
            try:
                if not isinstance(raw.get("usage"), dict):
                    raise TypeError("Missing complete usage")
                amount = usage_cost(
                    raw["usage"],
                    input_rate=spec["input_rate"],
                    output_rate=spec["output_rate"],
                )
            except (KeyError, TypeError, ValueError):
                self.pause("Complete usage missing; reservation retained")
                raise
            pacer.usage(spec["provider"], raw)
            self.ledger.settle({identity: amount})
            if self.ledger.snapshot()["overrun"]:
                self.pause("Recorded usage exceeded reserved bound")
            if (
                r["phase"] in (CALIBRATION_PHASE, revision.PHASE)
                and self.revision is not None
            ):
                validation_budget_check(self.ledger, self.revision, 0)
            outcome["status"] = (
                "completed"
                if response.choices and response.choices[0].stop_reason == "stop"
                else "incomplete"
            )
            self.completions.append(
                {
                    "at": time.time(),
                    "profile": profile,
                    "stage": stage,
                    "output_tokens": raw["usage"]["output_tokens"],
                    "active_seconds": time.time() - start - clock.waiting(),
                }
            )
        except Exception as error:  # noqa: BLE001 -- durable redacted terminal API failure
            outcome["error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                outcome["http_status"] = status
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                detail = body.get("error", body)
                if isinstance(detail, dict):
                    outcome["provider_error"] = {}
                    for field in ("type", "code", "message"):
                        if isinstance(detail.get(field), (str, int)):
                            value = str(detail[field])
                            for key in self.keys:
                                value = value.replace(key, "[REDACTED]")
                            outcome["provider_error"][field] = value[:2000]
        outcome.update(
            at=time.time(),
            seconds=time.time() - start,
            explicit_wait_seconds=clock.waiting(),
        )
        original.freeze(path / "response-status.json", outcome)
        if stage == "generation":
            original.freeze(path / "result.json", outcome)
            self.event(
                r, stage, outcome["status"] != "completed", outcome.get("error_type")
            )
        return raw

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Never dispatch a started request again; retain incomplete/missing responses."""
        if r.get("judge_only"):
            raise ValueError("Judge-only records cannot enter a generation lane")
        path = path_for(self.base, r)
        raw = await self.request(
            r, "generation", [ChatMessageUser(content=r["messages"][0]["content"])]
        )
        if raw is None:
            return
        output = ModelOutput.model_validate(raw)
        if r["kind"] == "comprehension_probe":
            if not (path / "probe.json").exists():
                original.freeze(
                    path / "probe.json",
                    study.score_probe(output.completion, r),
                )
            self.measurements[r["assignment_id"]] = measure(self.base, r)
            return
        await self.scoring.put(r)

    async def scoring_stage(self, r: dict[str, Any]) -> None:
        """Score once in the immutable Docker oracle, retaining invalid and missing code."""
        if r.get("judge_only"):
            raise ValueError("Judge-only records cannot be rescored")
        path = path_for(self.base, r)
        raw = json.loads(generation_path(self.base, r).read_text())
        output = ModelOutput.model_validate(raw)
        if not (path / "scoring-started.json").exists():
            original.freeze(
                path / "scoring-started.json",
                {
                    "at": time.time(),
                    "raw_sha256": digest(json_text(raw)),
                },
            )
            extraction = extract_code(output.completion, r["entry_point"])
            correctness: dict[str, Any] = {
                "status": "not_evaluated",
                "full_correct": False,
                "reason": "Response has no complete parseable source",
            }
            if extraction.code is not None and extraction.status == "ok":
                outcome = await asyncio.to_thread(
                    evaluate_solution,
                    self.datasets[r["dataset"]],
                    r["dataset"],
                    r["source_task_id"],
                    extraction.code,
                    mode="full",
                    image=self.manifest["oracle_image"],
                )
                correctness = {**asdict(outcome), "full_correct": outcome.full_correct}
            original.freeze(
                path / "scoring.json",
                {
                    "extraction": asdict(extraction),
                    "correctness": correctness,
                    "source_raw_sha256": digest(json_text(raw)),
                },
            )
        score = pilot.read_optional(path / "scoring.json")
        if score is None or score["correctness"]["status"] in (
            "infrastructure_error",
            "oracle_error",
        ):
            self.event(r, "scoring", True, "SandboxInfrastructure")
            self.pause("Sandbox scoring requires review")
            return
        self.event(r, "scoring", False)
        await self.judging.put(r)

    async def judging_stage(self, r: dict[str, Any]) -> None:
        """Apply the unchanged V3 rubric; preserve independent audit disagreements."""
        path = path_for(self.base, r)
        if (path / "judge/started.json").exists() or (
            path / "judge/result.json"
        ).exists():
            return
        if r["kind"] == "judge_calibration":
            try:
                packet = preference_judge.JudgeInput.model_validate(
                    r["calibration_packet"]
                )
            except (ValueError, TypeError):
                packet = None
            valid = packet is not None
            binding = {
                "calibration_packet_sha256": digest(json_text(r["calibration_packet"]))
            }
        else:
            raw = json.loads(generation_path(self.base, r).read_text())
            output = ModelOutput.model_validate(raw)
            packet = packet_from_output(raw, r)
            valid = bool(
                packet and output.choices and output.choices[0].stop_reason == "stop"
            )
            binding = {"source_raw_sha256": digest(json_text(raw))}
        if not valid or packet is None:
            original.freeze(
                path / "judge/result.json",
                {
                    **binding,
                    "status": "precheck_invalid"
                    if r["kind"] == "judge_calibration"
                    else "invalid",
                    "label": "invalid",
                },
            )
            self.event(r, "judge", False)
            return
        data = packet.model_dump(mode="json")
        original.freeze(path / "judge/packet.json", data)
        # The static audit is an independent measurement, never a label override.
        audit_label = None
        try:
            audit = preference_judge.audit_style(packet)
            audit_label = audit.label
            audit_record = {"status": "completed", **audit.model_dump(mode="json")}
        except (ValueError, TypeError, SyntaxError, tokenize.TokenError) as error:
            # Valid source can exceed the auxiliary audit's syntax coverage.
            # This is not a judge failure and does not replace the blinded label.
            message = str(error)
            for key in self.keys:
                message = message.replace(key, "[REDACTED]")
            audit_record = {
                "status": "audit_error",
                "error_type": type(error).__name__,
                "message": message[:2000],
            }
        original.freeze(path / "style-audit.json", audit_record)
        raw_judge = await self.request(
            r,
            "judge",
            [
                ChatMessageSystem(content=self.judges[r["study"]]["system_prompt"]),
                ChatMessageUser(content=render_packet(r["study"], data)),
            ],
        )
        result: dict[str, Any] = {
            **binding,
            "packet_sha256": digest(json_text(data)),
            "label": None,
            "status": "judge_error",
            "audit_label": audit_label,
        }
        try:
            if raw_judge is not None:
                response = ModelOutput.model_validate(raw_judge)
                if response.choices and response.choices[0].stop_reason == "stop":
                    parsed = parse_packet(r["study"], response.completion, data)
                    result.update(
                        label=parsed.label,
                        status=parsed.status,
                        judgment=parsed.model_dump(mode="json"),
                        audit_agrees=(parsed.label == audit_label)
                        if audit_label is not None
                        else None,
                    )
        except (ValueError, TypeError) as error:
            result["error_type"] = type(error).__name__
        original.freeze(path / "judge/result.json", result)
        self.event(
            r,
            "judge",
            result["status"] not in ("classified", "abstain"),
            result.get("error_type"),
        )

    async def worker(
        self, queue: asyncio.Queue[Any], stage: str, index: int, lane: str
    ) -> None:
        """Bound stage concurrency and drain already accepted work after a stop."""
        while True:
            while (
                stage != "scoring"
                and index >= self.targets[lane]
                and not self.stop.is_set()
                and not self.finishing.is_set()
            ):
                await asyncio.sleep(0.25)
            r = await queue.get()
            try:
                if r is None:
                    return
                if self.stop.is_set():
                    continue
                self.check()
                self.active[lane] += 1
                try:
                    if stage == "generation":
                        await self.generation_stage(r)
                    elif stage == "scoring":
                        await self.scoring_stage(r)
                    else:
                        await self.judging_stage(r)
                finally:
                    self.active[lane] -= 1
            except Exception as error:  # noqa: BLE001 -- controller boundary drains all paid tasks
                if r is not None:
                    self.event(r, stage, True, type(error).__name__)
                self.pause(type(error).__name__ + " in " + stage)
            finally:
                queue.task_done()

    def status(self, state: str) -> dict[str, Any]:
        """Report fresh counts and realized completions separately from quota estimates."""
        now = time.time()
        self.completions = [r for r in self.completions if r["at"] > now - 300]
        balance = self.ledger.snapshot()
        output = {
            "state": state,
            "phase": self.phase,
            "family": self.family,
            "pid": os.getpid(),
            "updated_at": now,
            "counts": counts(list(self.measurements.values())),
            "recorded_usd": balance["spent"] / 1e9,
            "request_outstanding_usd": balance["outstanding"] / 1e9,
            "cap_usd": study.CAP,
            "active": dict(self.active),
            "concurrency_targets": self.targets,
            "queues": {
                **{k: q.qsize() for k, q in self.queues.items()},
                "scoring": self.scoring.qsize(),
                "judging": self.judging.qsize(),
            },
            "free_disk_bytes": shutil.disk_usage(self.root).free,
            "pacing": {
                k: ModelPacer(self.operations / "pacing" / k, profile=k).utilization()
                for k in (*study.PROFILES, "sonnet")
            },
            "last_minute": {
                k: {
                    "completions": sum(
                        r["profile"] == k and r["at"] > now - 60
                        for r in self.completions
                    ),
                    "generation_completions": sum(
                        r["profile"] == k
                        and r["stage"] == "generation"
                        and r["at"] > now - 60
                        for r in self.completions
                    ),
                    "output_tokens": sum(
                        r["output_tokens"]
                        for r in self.completions
                        if r["profile"] == k and r["at"] > now - 60
                    ),
                }
                for k in (*study.PROFILES, "sonnet")
            },
        }
        write_json(self.operations / "status.json", output)
        append_event(self.operations / "throughput-events.jsonl", output)
        return output

    def tune(self, report: dict[str, Any]) -> None:
        """Grow judge capacity despite upstream backpressure; bound generation demand."""
        for lane in self.targets:
            profile = "sonnet" if lane == "judge" else lane
            p = report["pacing"][profile].get(
                (
                    study.JUDGE_PROFILE
                    if profile == "sonnet"
                    else study.PROFILES[profile]
                )["provider"],
                {},
            )
            demand = (
                self.judging.qsize() if lane == "judge" else self.queues[lane].qsize()
            )
            utilization = max(p.get("utilization", {}).values(), default=0)
            if (
                demand
                and not p.get("queued")
                and utilization < 0.75
                and (lane == "judge" or self.scoring.qsize() < 48)
            ):
                self.targets[lane] = min(32, self.targets[lane] + 4)

    async def monitor(self) -> None:
        """Verify bindings and tune lightweight network slots without expanding Docker."""
        ticks = 0
        while True:
            await asyncio.sleep(10)
            try:
                self.check()
                guard(self.base)
                if ticks % 3 == 0:
                    load(self.root)
                report = self.status("draining" if self.stop.is_set() else "running")
                if report["free_disk_bytes"] < 3 * 1024**3:
                    self.pause("Disk headroom below 3 GiB")
                if ticks % 3 == 2 and not self.stop.is_set():
                    self.tune(report)
                ticks += 1
            except Exception as error:  # noqa: BLE001 -- cease admission, leave paid work draining
                self.pause(type(error).__name__ + " during monitor")
                self.status("draining")

    async def run(self) -> None:
        """Run all phase queues, then wait for paid tasks and Docker work before return."""
        self.initialize_models()
        direct_judges = []
        for r in self.records:
            if r["phase"] == self.phase and (
                self.family is None or r["study"] == self.family
            ):
                if r.get("judge_only"):
                    direct_judges.append(r)
                else:
                    self.queues[r["profile"]].put_nowait(r)
        workers = []
        for profile, q in self.queues.items():
            workers.extend(
                asyncio.create_task(self.worker(q, "generation", i, profile))
                for i in range(32)
            )
        workers.extend(
            asyncio.create_task(self.worker(self.scoring, "scoring", i, "scoring"))
            for i in range(4)
        )
        workers.extend(
            asyncio.create_task(self.worker(self.judging, "judge", i, "judge"))
            for i in range(32)
        )
        watcher = asyncio.create_task(self.monitor())
        self.status("running")
        try:
            for record in direct_judges:
                await self.judging.put(record)
            await asyncio.gather(*(q.join() for q in self.queues.values()))
            await self.scoring.join()
            await self.judging.join()
        finally:
            self.finishing.set()
            for q in self.queues.values():
                for _ in range(32):
                    q.put_nowait(None)
            for _ in range(4):
                await self.scoring.put(None)
            for _ in range(32):
                await self.judging.put(None)
            await asyncio.gather(*workers)
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        self.status(
            "paused"
            if self.stop.is_set()
            else (
                self.phase + "_family_complete"
                if self.family
                else self.phase + "_complete"
            )
        )
        if self.stop.is_set():
            raise RuntimeError(str(self.reason))


def initialize_pacing(root: Path, base: Path) -> None:
    """Seed matching prior lanes once, preserving cap provenance and cooldowns."""
    now = time.time()
    audit: dict[str, Any] = {}
    for profile in (*study.PROFILES, "sonnet"):
        target = base / "operations/pacing" / profile / "rate-state.json"
        if target.exists():
            continue
        candidates = [
            root
            / "runs/replication/simple-models-v1/operations/pacing"
            / profile
            / "rate-state.json",
            root
            / "runs/replication/compact-models-v1/operations/pacing"
            / profile
            / "rate-state.json",
        ]
        if profile == "gpt-oss-120b":
            candidates = [
                root
                / "runs/three-style/three-style-v1/operations/pacing/rate-state.json"
            ]
        if profile == "sonnet":
            candidates.append(
                root
                / "runs/recovery/judge-recovery-v1/operations/pacing/sonnet/rate-state.json"
            )
        available = [p for p in candidates if p.is_file()]
        if not available:
            continue
        source = max(available, key=lambda p: p.stat().st_mtime)
        data = json.loads(source.read_text())
        provider = (
            study.JUDGE_PROFILE if profile == "sonnet" else study.PROFILES[profile]
        )["provider"]
        if provider not in data:
            raise ValueError("Prior pacing lane lacks matching provider")
        lane = data[provider]
        # Old tickets cannot own this controller's new FIFO. Retain recent usage,
        # historic throttle state, numeric evidence, and all cooldown timestamps.
        lane["queue"] = []
        lane["window"] = [r for r in lane.get("window", []) if r["at"] > now - 60]
        write_json(target, {provider: lane})
        audit[profile] = {
            "source": str(source.relative_to(root)),
            "source_sha256": digest(source.read_bytes()),
            "provider": provider,
            "source_modified_at": source.stat().st_mtime,
            "retained_numeric_limits": lane.get("verified_limits", {}),
            "local_allowances_not_current_quota": lane["limits"],
            "cooldown_until": lane["cooldown_until"],
        }
    account_target = base / "operations/pacing/account-rpm.json"
    if not account_target.exists():
        candidates = [
            root / relative / "operations/pacing/account-rpm.json"
            for relative in (
                "runs/replication/simple-models-v1",
                "runs/replication/compact-models-v1",
                "runs/recovery/audited-outcomes-v1",
                "runs/recovery/judge-recovery-v1",
            )
        ]
        available = [p for p in candidates if p.is_file()]
        if available:
            source = max(available, key=lambda p: p.stat().st_mtime)
            account = json.loads(source.read_text())
            cap = account.get("cap", 6000)
            if type(cap) not in (int, float) or not 0 < cap <= 6000:
                raise ValueError("Prior shared account RPM cap is invalid")
            account["window"] = [
                r for r in account.get("window", []) if r["at"] > now - 60
            ]
            write_json(account_target, account)
            audit["shared_account_rpm"] = {
                "source": str(source.relative_to(root)),
                "source_sha256": digest(source.read_bytes()),
                "cap": cap,
                "reported_at": account.get("reported_at"),
            }
    if audit:
        original.freeze(
            base / "operations/pacing-initialization.json", {"at": now, "lanes": audit}
        )


def naming_policy(
    base: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Require the approved clear-to-mixed ratio separately for every model.

    Only the two pure naming labels count as clear. Missing evidence, other
    conventions, invalid source and unresolved judgments never dilute mixed
    outcomes. This admission policy does not replace the scientific pilot gates.
    """
    path = base / "operations/mixed-policy-approval.json"
    if not path.exists():
        return None
    policy = json.loads(path.read_text())
    instruction_path = base / "operations/mixed-policy-user-instruction.json"
    if not instruction_path.exists() or digest(
        instruction_path.read_bytes()
    ) != policy.get("instruction_sha256"):
        raise ValueError("Naming policy user-instruction binding changed")
    instruction = json.loads(instruction_path.read_text())
    if (
        policy.get("approved") is not True
        or instruction.get("approved") is not True
        or not policy.get("user_instruction")
        or policy["user_instruction"] != instruction.get("user_instruction")
    ):
        raise ValueError("Naming policy lacks matching explicit approval")
    ratio = policy.get("minimum_clear_to_mixed_ratio")
    if (
        policy.get("policy") != "clear_to_mixed_ratio"
        or type(ratio) not in (int, float)
        or ratio != 4
        or instruction.get("minimum_clear_to_mixed_ratio") != ratio
    ):
        raise ValueError("Naming policy must preserve the user's exact 4:1 ratio")
    by_model = {}
    for profile in study.PROFILES:
        code = [
            r
            for r in rows
            if r["study"] == "naming"
            and r["profile"] == profile
            and r["kind"] == "core"
            and r["phase"] == "pilot"
        ]
        clear = sum(r["label"] in ("snake_case_only", "camelCase_only") for r in code)
        mixed = sum(r["label"] == "mixed" for r in code)
        by_model[profile] = {
            "clear": clear,
            "mixed": mixed,
            "excluded": len(code) - clear - mixed,
            "passed": bool(code) and clear >= ratio * mixed,
        }
    if not all(result["passed"] for result in by_model.values()):
        return None
    return {
        "policy": policy,
        "sha256": digest(path.read_bytes()),
        "manifest_sha256": digest(json_text(manifest)),
        "by_model": by_model,
    }


def main_readiness(
    root: Path, manifest: dict[str, Any], records: list[dict[str, Any]]
) -> dict[str, Any]:
    """Require complete approved validation before admitting a selected main family."""
    base = study.base(root)
    revised = read_revision(base)
    calibration = calibration_gate(base, records) if revised else None
    if calibration is not None and not calibration["passed"]:
        return {
            "calibration": calibration,
            "main_ready_by_family": dict.fromkeys(FAMILIES, False),
            "reason": "Naming calibration failed or is incomplete",
        }
    if revised:
        rows = qualified_rows(base, records)
        cohort = [measure(base, r) for r in records if r["phase"] == revision.PHASE]
        complete = all(
            (
                r["judge_result"]
                if r.get("judge_only")
                else r["attempt_complete"]
                and (
                    r["kind"] == "comprehension_probe"
                    or not r["raw_saved"]
                    or r["judge_result"]
                )
            )
            for r in cohort
        )
        if not complete:
            return {
                "calibration": calibration,
                "main_ready_by_family": dict.fromkeys(FAMILIES, False),
                "reason": "Approved validation cohort has incomplete attempts or judgments",
            }
    else:
        rows = [measure(base, r) for r in records if r["phase"] == "pilot"]
    policy = naming_policy(base, manifest, rows)
    gate = study.pilot_gate(rows, naming_mixed_policy_approved=policy is not None)
    cost_path = base / (
        REVISION_DIRECTORY + "/validation-forecast.json"
        if revised
        else "pilot-audit.json"
    )
    cost = pilot.read_optional(cost_path)
    if cost is None:
        cost = (
            revision_forecast(root, manifest, records)
            if revised
            else forecast(root, manifest, records)
        )
        original.freeze(cost_path, cost)
    if cost["manifest_sha256"] != digest(json_text(manifest)):
        raise ValueError("Pilot forecast scope changed")
    if revised and cost.get("revision_manifest_sha256") != digest(
        (base / REVISION_MANIFEST).read_bytes()
    ):
        raise ValueError("Revised forecast scope changed")
    for name, sha in cost["bindings"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Pilot forecast evidence changed")
    ready = dict(gate["main_ready_by_family"])
    exception = forecast_admission_exception(base, manifest, cost)
    if cost["projected_total_conservative_usd"] > study.CAP and exception is None:
        ready = dict.fromkeys(FAMILIES, False)
    return {
        "calibration": calibration,
        "scientific_gate": gate,
        "forecast": cost,
        "forecast_admission_exception": exception,
        "naming_policy": policy,
        "main_ready_by_family": ready,
    }


def analyze(root: Path) -> dict[str, Any]:
    """Retain every assignment and validate every saved label against its raw judge."""
    manifest, records = load(root, retained=True)
    base = study.base(root)
    rows = [measure(base, r) for r in records if r["phase"] == "main"]
    for r in records:
        path = path_for(base, r)
        judged = pilot.read_optional(path / "judge/result.json")
        if not judged:
            continue
        if r["kind"] == "judge_calibration":
            if judged.get("calibration_packet_sha256") != digest(
                json_text(r["calibration_packet"])
            ):
                raise ValueError("Calibration packet binding changed")
        else:
            raw = json.loads(generation_path(base, r).read_text())
            if judged.get("source_raw_sha256") != digest(json_text(raw)):
                raise ValueError("Judgment source binding changed")
        if judged.get("status") in ("classified", "abstain"):
            packet = json.loads((path / "judge/packet.json").read_text())
            raw_judge = ModelOutput.model_validate(
                json.loads((path / "judge/raw.json").read_text())
            )
            parsed = parse_packet(r["study"], raw_judge.completion, packet)
            if (
                parsed.label != judged["label"]
                or digest(json_text(packet)) != judged["packet_sha256"]
            ):
                raise ValueError("Raw and parsed judgment differ")
    revised = read_revision(base)
    result = study.analyze(rows, repetitions=3 if revised else 4)
    result.update(
        collection_complete=all(
            r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
            for r in rows
        ),
        collection_complete_by_family={
            family: all(
                r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
                for r in rows
                if r["study"] == family
            )
            for family in FAMILIES
        },
        manifest_sha256=digest(json_text(manifest)),
        revision_manifest_sha256=digest((base / REVISION_MANIFEST).read_bytes())
        if revised
        else None,
        retired_main_assignments=sum(r["phase"] == "retired_main" for r in records),
        recorded_usd=ledger_for(base, manifest).snapshot()["spent"] / 1e9,
    )
    output = base / ("analysis-v2" if revised else "analysis-v1")
    write_json(output / "measured-rows.json", rows)
    write_json(output / "analysis.json", result)
    if (base / GENERATION_RECOVERY_MANIFEST).exists():
        fresh = [
            measure(base, r) for r in records if r["phase"] == generation_recovery.PHASE
        ]
        recovered = generation_recovery.recovered_main_rows(rows, fresh)
        recovered_result = study.analyze(recovered, repetitions=3)
        recovered_output = base / "analysis-v3"
        write_json(recovered_output / "measured-rows.json", recovered)
        recovered_result.update(
            collection_complete=all(
                r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
                for r in recovered
            ),
            collection_complete_by_family={
                family: all(
                    r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
                    for r in recovered
                    if r["study"] == family
                )
                for family in FAMILIES
            },
            manifest_sha256=digest(json_text(manifest)),
            revision_manifest_sha256=digest((base / REVISION_MANIFEST).read_bytes()),
            format_amendment_sha256=digest((base / FORMAT_AMENDMENT).read_bytes()),
            recovery_manifest_sha256=digest(
                (base / GENERATION_RECOVERY_MANIFEST).read_bytes()
            ),
            original_analysis_sha256=digest((output / "analysis.json").read_bytes()),
            original_measured_rows_sha256=digest(
                (output / "measured-rows.json").read_bytes()
            ),
            measured_rows_sha256=digest(
                (recovered_output / "measured-rows.json").read_bytes()
            ),
            rerun_assignments=len(fresh),
            retired_main_assignments=result["retired_main_assignments"],
            recorded_usd=result["recorded_usd"],
            original_main_counts=counts(rows),
            fresh_attempt_counts=counts(fresh),
        )
        write_json(recovered_output / "analysis.json", recovered_result)
    if (base / GENERATION_RECOVERY_V2_MANIFEST).exists():
        fresh = [
            measure(base, r)
            for r in records
            if r["phase"] == generation_recovery_v2.PHASE
        ]
        first_fresh = [
            measure(base, r) for r in records if r["phase"] == generation_recovery.PHASE
        ]
        recovered = generation_recovery_v2.recovered_main_rows(rows, first_fresh, fresh)
        recovered_result = study.analyze(recovered, repetitions=3)
        recovered_output = base / "analysis-v4"
        write_json(recovered_output / "measured-rows.json", recovered)
        recovered_result.update(
            collection_complete=all(
                r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
                for r in recovered
            ),
            collection_complete_by_family={
                family: all(
                    r["attempt_complete"] and (not r["raw_saved"] or r["judge_result"])
                    for r in recovered
                    if r["study"] == family
                )
                for family in FAMILIES
            },
            manifest_sha256=digest(json_text(manifest)),
            revision_manifest_sha256=digest((base / REVISION_MANIFEST).read_bytes()),
            format_amendment_sha256=digest((base / FORMAT_AMENDMENT).read_bytes()),
            recovery_manifest_sha256=digest(
                (base / GENERATION_RECOVERY_V2_MANIFEST).read_bytes()
            ),
            prior_recovery_manifest_sha256=digest(
                (base / GENERATION_RECOVERY_MANIFEST).read_bytes()
            ),
            prior_analysis_sha256=digest(
                (base / "analysis-v3/analysis.json").read_bytes()
            ),
            prior_measured_rows_sha256=digest(
                (base / "analysis-v3/measured-rows.json").read_bytes()
            ),
            first_attempt_counts=counts(first_fresh),
            original_analysis_sha256=digest((output / "analysis.json").read_bytes()),
            original_measured_rows_sha256=digest(
                (output / "measured-rows.json").read_bytes()
            ),
            measured_rows_sha256=digest(
                (recovered_output / "measured-rows.json").read_bytes()
            ),
            rerun_assignments=len(fresh),
            retired_main_assignments=result["retired_main_assignments"],
            recorded_usd=result["recorded_usd"],
            original_main_counts=counts(rows),
            fresh_attempt_counts=counts(fresh),
        )
        write_json(recovered_output / "analysis.json", recovered_result)
    return result


def collect(root: Path, phase: str, family: str | None = None) -> None:
    """Run one authorized phase with shared collection ownership and drained cleanup."""
    manifest, records = load(root, retained=True)
    base = study.base(root)
    guard(base)
    recovery = pilot.read_optional(base / "operations/recovery-state.json") or {}
    if recovery.get("retry_blocked"):
        raise ValueError("Recovery blocked pending new human direction")
    if family is not None and family not in FAMILIES:
        raise ValueError("Unknown style family")
    revised = read_revision(base)
    recovery_active = (base / GENERATION_RECOVERY_MANIFEST).exists()
    second_recovery_active = (base / GENERATION_RECOVERY_V2_MANIFEST).exists()
    latest_recovery_phase = (
        generation_recovery_v2.PHASE
        if second_recovery_active
        else generation_recovery.PHASE
    )
    recovery_phases = (generation_recovery.PHASE, generation_recovery_v2.PHASE)
    if phase in recovery_phases and (
        not recovery_active or phase != latest_recovery_phase or family is not None
    ):
        raise ValueError(
            "Generation reruns require the exact approved complete cohort; older phases are retired"
        )
    if (
        (base / FORMAT_AMENDMENT).exists()
        and phase != "main"
        and not (recovery_active and phase == latest_recovery_phase)
    ):
        raise ValueError(
            "Factual-format approval authorizes no new paid validation; use main only"
        )
    if revised and phase == "pilot":
        raise ValueError("Original pilot is retired; use approved revision phases")
    if phase in (CALIBRATION_PHASE, revision.PHASE):
        if revised is None:
            raise ValueError("Revision phase requires explicit frozen approval")
        if family is not None:
            raise ValueError(
                "Approved calibration and validation cohorts run as a whole"
            )
        if phase == revision.PHASE and not calibration_gate(base, records)["passed"]:
            raise ValueError("Naming calibration failed or incomplete; validation held")
        if any(
            (path_for(base, r) / "generation/started.json").exists()
            for r in records
            if r["phase"] == "main"
        ):
            raise ValueError("Validation cannot be rerun after main admission")
    readiness = None
    billing_admission = None
    if phase == "main":
        if (
            recovery_active
            and not generation_recovery_gate(
                base, records, phase=latest_recovery_phase
            )["passed"]
        ):
            if second_recovery_active:
                billing_admission = restored_billing_admission(base, records)
            if billing_admission is None:
                raise ValueError(
                    "Bounded generation recovery is incomplete or missing new responses; main remains held"
                )
        readiness = main_readiness(root, manifest, records)
        families = FAMILIES if family is None else (family,)
        if not all(readiness["main_ready_by_family"][f] for f in families):
            raise ValueError("Main held by scientific, budget, or naming-policy gate")
    tag = (
        time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        + f"-{os.getpid()}-{phase}-{family or 'both'}"
    )
    launch = base / "operations/launches" / tag
    original.freeze(
        launch / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "phase": phase,
            "family": family,
            "manifest_sha256": digest(json_text(manifest)),
            "authorization": study.AUTHORIZATION,
            "readiness": readiness,
            "billing_admission": billing_admission,
        },
    )
    owner = root / "runs/extension-pilot/coordination/owner.json"
    if owner.exists():
        original.freeze(launch / "previous-owner.json", json.loads(owner.read_text()))
    with collection_lock(root, owner="simple-models-v1", run_id=study.RUN):
        require_pilot_verification(root, manifest["oracle_image"])
        image = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                manifest["oracle_image"],
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        containers = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        if (
            image.stdout.strip() != manifest["oracle_image"]
            or containers.stdout.strip()
        ):
            raise ValueError(
                "Pinned sandbox identity or clean-start verification failed"
            )
        original.freeze(
            launch / "preflight.json",
            {"image": image.stdout.strip(), "oracle_containers": [], "at": time.time()},
        )
        initialize_pacing(root, base)
        pipeline = Pipeline(root, manifest, records, phase, family)
        with (
            pilot.zero_retry_clients(),
            concurrent_transport(),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):

            async def execute() -> None:
                loop = asyncio.get_running_loop()
                loop.set_default_executor(executor)
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(
                        sig, pipeline.pause, "Operator signal; drain accepted work"
                    )
                await pipeline.run()

            try:
                asyncio.run(execute())
            finally:
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-aq",
                        "--filter",
                        "label=grader-power.oracle=true",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                original.freeze(
                    launch / "cleanup.json",
                    {"oracle_containers": result.stdout.split(), "at": time.time()},
                )
                original.freeze(
                    launch / "drained.json",
                    {
                        "at": time.time(),
                        "active": dict(pipeline.active),
                        "reason": pipeline.reason,
                    },
                )
                if result.stdout.strip():
                    raise RuntimeError("Experiment sandbox cleanup not verified")
        if phase in recovery_phases:
            result = generation_recovery_gate(base, records, phase=phase)
            directory = (
                GENERATION_RECOVERY_V2_DIRECTORY
                if phase == generation_recovery_v2.PHASE
                else GENERATION_RECOVERY_DIRECTORY
            )
            original.freeze(
                base / directory / "fresh-generation-gate.json",
                result,
            )
            if not result["passed"]:
                raise ValueError(
                    "New generation recovery requests failed; no further fresh retries authorized"
                )
        elif phase == CALIBRATION_PHASE:
            result = calibration_gate(base, records)
            original.freeze(
                base / REVISION_DIRECTORY / "naming-calibration-gate.json", result
            )
            if not result["passed"]:
                raise ValueError(
                    "Approved naming calibration failed; no further cohort authorized"
                )
        elif phase == revision.PHASE:
            result = main_readiness(root, manifest, records)
            original.freeze(
                base / REVISION_DIRECTORY / "validation-readiness.json", result
            )
        elif phase == "pilot":
            rows = [measure(base, r) for r in records if r["phase"] == "pilot"]
            # Never overwrite a failed gate with later scientific retuning.
            original.freeze(
                base / f"pilot-gate-{family or 'both'}.json", study.pilot_gate(rows)
            )
        else:
            analyze(root)
        original.freeze(
            launch / "complete.json",
            {"at": time.time(), "phase": phase, "family": family},
        )


def main() -> None:
    """Freeze, pilot, collect independently gated families, inspect, or analyze offline."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command",
        choices=(
            "prepare",
            "prepare-revision",
            "prepare-format-qualification",
            "prepare-generation-recovery",
            "prepare-generation-recovery-v2",
            "run-generation-recovery-v2",
            "run-generation-recovery",
            "collect",
            "run",
            "run-revision",
            "status",
            "analyze",
            "verify",
        ),
    )
    p.add_argument(
        "--phase",
        choices=(
            "pilot",
            "main",
            CALIBRATION_PHASE,
            revision.PHASE,
            generation_recovery.PHASE,
            generation_recovery_v2.PHASE,
        ),
        default="pilot",
    )
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--report", type=Path, action="append", default=[])
    args = p.parse_args()
    if args.command == "prepare":
        prepare(ROOT, args.report)
    elif args.command == "prepare-revision":
        prepare_revision(ROOT, args.report)
    elif args.command == "prepare-format-qualification":
        prepare_format_qualification(ROOT, args.report)
    elif args.command == "prepare-generation-recovery":
        prepare_generation_recovery(ROOT, args.report)
    elif args.command == "prepare-generation-recovery-v2":
        prepare_generation_recovery_v2(ROOT, args.report)
    elif args.command in (
        "collect",
        "run",
        "run-revision",
        "run-generation-recovery",
        "run-generation-recovery-v2",
    ):
        isolate_collection_process()
        if args.command == "run-generation-recovery-v2":
            collect(ROOT, generation_recovery_v2.PHASE)
            collect(ROOT, "main")
        elif args.command == "run-generation-recovery":
            collect(ROOT, generation_recovery.PHASE)
            collect(ROOT, "main")
        elif args.command in ("run", "run-revision"):
            if args.command == "run-revision":
                collect(ROOT, CALIBRATION_PHASE)
                collect(ROOT, revision.PHASE)
            else:
                collect(ROOT, "pilot")
            manifest, records = load(ROOT, retained=True)
            readiness = main_readiness(ROOT, manifest, records)
            for family in FAMILIES:
                if readiness["main_ready_by_family"][family]:
                    collect(ROOT, "main", family)
                else:
                    write_json(
                        study.base(ROOT) / f"operations/{family}-main-held.json",
                        readiness,
                    )
        else:
            collect(ROOT, args.phase, args.family)
    elif args.command == "analyze":
        analyze(ROOT)
    else:
        manifest, records = load(ROOT, retained=args.command == "verify")
        print(
            json.dumps(
                {
                    "verified": True,
                    "counts": counts([measure(study.base(ROOT), r) for r in records]),
                    "ledger": ledger_for(study.base(ROOT), manifest).snapshot(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
