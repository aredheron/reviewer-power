"""Separate, bounded hosted-model readiness check on already exposed pilot tasks."""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from evals import inspect_pilot as pilot
from evals.inspect_checkpoint_diagnostics import comprehension_gate
from evals.inspect_mode_a import digest, generation_config, json_text, save
from grader_power.inference import load_access_profiles
from grader_power.pilot_tasks import load_pilot_tasks, require_pilot_verification

PLAN = "configs/experiments/readiness-v2.json"
OLD_SCORE_USE = "score_use: logged_for_diagnostic_analysis"
NEW_SCORE_USE = "score_use: recorded_and_used_only_as_specified_by_selection_policy"
KINDS = {"comprehension_probe", "direct_style", "no_grader"}


def build_assignments(root: Path) -> tuple[pilot.Assignment, ...]:
    """Retain balanced pilot diagnostics; change only one conflicting metadata line.

    Do not edit shared templates or include any reserved task. New prompt-derived
    IDs distinguish this cohort from original pilot observations.
    """
    result = []
    for original in pilot.build_assignments(load_pilot_tasks(root)):
        if original.record["kind"] not in KINDS:
            continue
        record = json.loads(json_text(original.record))
        content = record["messages"][0]["content"]
        expected = int(record["kind"] == "comprehension_probe")
        if content.count(OLD_SCORE_USE) != expected:
            raise ValueError("Unexpected readiness prompt metadata")
        record["messages"][0]["content"] = content.replace(OLD_SCORE_USE, NEW_SCORE_USE)
        record["prompt_sha256"] = digest(json_text(record["messages"]))
        record["assignment_id"] = digest(
            "readiness-v2:" + original.record["assignment_id"] + record["prompt_sha256"]
        )
        result.append(pilot.Assignment(record, original.task))
    if Counter(a.record["kind"] for a in result) != {
        "comprehension_probe": 16,
        "direct_style": 4,
        "no_grader": 4,
    } or {a.record["source_task_id"] for a in result} != {"HumanEval/21", "Mbpp/778"}:
        raise ValueError("Readiness diagnostics must use exactly the two pilot tasks")
    return tuple(result)


def prepare_run(
    root: Path, run_id: str, image: str
) -> tuple[Path, tuple[pilot.Assignment, ...]]:
    """Freeze prompts, source hashes, settings, and a bounded cost estimate offline."""
    import re

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("run-id must be a simple name")
    evidence = require_pilot_verification(root, image)
    plan = json.loads((root / PLAN).read_text())
    assignments = build_assignments(root)
    profiles = load_access_profiles(root / "configs/models.yaml")
    if (
        plan["profiles"] != ["gpt-oss-120b", "claude-sonnet"]
        or plan["maximum_requests"] != 48
    ):
        raise ValueError("Unexpected readiness model panel or request bound")
    costs = {}
    for name in plan["profiles"]:
        prices = plan["planning_usd_per_million_tokens"][name]
        # Bytes plus framing are a conservative token-planning proxy, not billing.
        costs[name] = (
            sum(
                (len(json_text(a.record["messages"]).encode()) + 2048) * prices["input"]
                + profiles[name].max_tokens * prices["output"]
                for a in assignments
            )
            / 1_000_000
        )
    if sum(costs.values()) > plan["planning_ceiling_usd"]:
        raise ValueError("Readiness estimate exceeds its fixed planning ceiling")
    files = [
        *root.glob("src/grader_power/**/*.py"),
        *root.glob("prompts/**/*.*"),
        root / PLAN,
        root / "evals/inspect_pilot.py",
        root / "evals/inspect_readiness.py",
        root / "evals/inspect_mode_a.py",
        root / "evals/inspect_checkpoint_diagnostics.py",
        root / "configs/models.yaml",
        root / "manifests/tasks_pilot.jsonl",
        root / "uv.lock",
    ]
    definition = {
        "schema_version": 1,
        "purpose": "mode_a_readiness_v2",
        "mode": "A",
        "run_id": run_id,
        "oracle_image": image,
        "plan": plan,
        "verification_sha256": digest(json_text(evidence)),
        "profiles": {
            name: {
                "model": profiles[name].model,
                "revision": profiles[name].revision,
                "generation": generation_config(profiles[name]).model_dump(
                    mode="json", exclude_none=True
                ),
            }
            for name in plan["profiles"]
        },
        "assignments": [a.record for a in assignments],
        "cost_estimate_usd": costs,
        "code_sha256": {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in sorted(files)
            if p.is_file()
        },
    }
    cohort = root / "runs/readiness" / run_id
    manifest = cohort / "manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != definition:
            raise ValueError(
                "Readiness manifest is stale; preserve it and use a new run ID"
            )
    else:
        save(manifest, definition)
    for name in plan["profiles"]:
        pilot.export_results(
            cohort / name,
            assignments,
            name,
            digest(manifest.read_bytes()),
            purpose=definition["purpose"],
        )
    return cohort, assignments


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep probe comprehension, correctness, and automatic style coverage separate."""
    controls = [r for r in rows if r["kind"] != "comprehension_probe"]
    return {
        "comprehension_gate": comprehension_gate(rows),
        "strict_probe_both_correct": sum(
            bool((r.get("probe") or {}).get("both_correct")) for r in rows
        ),
        "controls_assigned": len(controls),
        "controls_full_correct": sum(
            bool((r.get("scoring") or {}).get("correctness", {}).get("full_correct"))
            for r in controls
        ),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "style_readiness": "Requires blinded semantic audit of all eight controls; no automatic overall pass.",
    }


def main() -> None:
    """Prepare without credentials by default; explicitly opt in to at most 24 calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-image", required=True)
    parser.add_argument("--profile", choices=["gpt-oss-120b", "claude-sonnet"])
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cohort, assignments = prepare_run(root, args.run_id, args.oracle_image)
    if args.generate:
        if not args.profile:
            parser.error("--generate requires --profile")
        values = {**dotenv_values(root / ".env"), **os.environ}
        rows = pilot.collect_profile(
            root,
            cohort,
            assignments,
            args.profile,
            args.oracle_image,
            values,
            resume=args.resume,
        )
        summary = summarize(rows)
        pilot.replace_private(
            cohort / args.profile / "summary.json", json_text(summary)
        )
        print(json.dumps(summary, indent=2))
        if any(r["status"] != "scored" for r in rows):
            raise SystemExit(1)
    else:
        print(f"Prepared {len(assignments)} assignments per model; no requests made.")


if __name__ == "__main__":
    main()
