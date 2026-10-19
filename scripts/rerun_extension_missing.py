"""One explicitly authorized extra attempt for the ten missing extension outputs."""

import argparse
import asyncio
import json
import sys
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import inspect_extension_main as runner
from evals import inspect_paced_extension as paced
from grader_power.collection_lock import collection_lock, isolate_collection_process
from grader_power.extension_analysis import analyze_rows
from grader_power.pacing import (
    PacingDeferred,
    SharedPacer,
    WaitClock,
    locked,
    paced_http,
    write_json,
)
from grader_power.pilot_tasks import require_pilot_verification
from grader_power.spending import SpendingLedger

BATCH = "generation-reruns-v1"
AUTHORIZED_IDS = frozenset(
    (
        "45b3fc443a0d7cb1bea1bee2bf04d9826627a675fb42bb50073f522f6667a2dd",
        "6ae08712ba80071826f335a82190ffb90c430eac40c4ef17add2068b5734c46e",
        "23e6ad5361911060802e596051afaec7b08312035cd3648aaaeb778de4f85a2e",
        "2034325dc32ec32823b23ca1cbed3e9a07d1f8efffaa103e58e5507ca05ccee3",
        "09f49526c822722f107cb1b3a98b6a469dbe07ed1c220d0fd980f9a63cefc59e",
        "66d6ca53a835ba9de3e17c4df7793b244fd5a5de9bd66ee900b07655f58b0b3b",
        "dc17c85d596d73b2945d12abdde092b51dcbc66d8d4bbbf31dc7e4accf996a40",
        "4b51c4a153d61124d6f5e31650de00aaf3253ba3e9fa95074a07efc73c262814",
        "aae0bf97be67f8b979ae9c6a4f6b87f4f5886a230205c9d0b755043decf26748",
        "5e993176e1ea579bf4422ddc36cb87c23e129c17e751437a317899da7100e9c4",
    )
)
AUTHORIZATION = (
    'User on 2026-10-14: "cool. can you rerun all 10? thx". Authorizes one '
    "additional generation attempt for these ten missing Mode A outputs, with "
    "the existing blinded judge, providers, frozen inputs and $250 whole-run cap. "
    "Original failures are preserved; this batch never replays its own started IDs."
)
IDENTITY_FIELDS = (
    "assignment_id",
    "condition",
    "mode",
    "profile",
    "provider",
    "repetition_index",
    "scenario",
    "source_task_id",
)


def select_missing(rows: list[dict[str, Any]]) -> set[str]:
    """Reject a changed denominator or a request extending beyond the ten IDs."""
    if len(rows) != 12288 or len({r["assignment_id"] for r in rows}) != 12288:
        raise ValueError("Original analysis must have 12,288 unique assignments")
    missing = {r["assignment_id"] for r in rows if not r["raw_saved"]}
    if missing != AUTHORIZED_IDS or any(
        r["mode"] != "A" for r in rows if r["assignment_id"] in missing
    ):
        raise ValueError("Missing generation scope differs from the authorized ten")
    return missing


def overlay(
    original: list[dict[str, Any]], replacements: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace only authorized outcomes, retaining order and experimental identity."""
    targets = select_missing(original)
    updates = {r["assignment_id"]: r for r in replacements}
    if set(updates) != targets or len(replacements) != len(targets):
        raise ValueError("Overlay must contain each authorized assignment once")
    for row in original:
        updated = updates.get(row["assignment_id"])
        if updated is not None and any(
            row[key] != updated[key] for key in IDENTITY_FIELDS
        ):
            raise ValueError("Rerun changed experimental identity or provider")
    return [updates.get(r["assignment_id"], r) for r in original]


def prepare(
    root: Path, amendment_path: Path
) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, tuple[runner.pilot.Assignment, ...]]
]:
    """Freeze a separate, source-bound subset without changing original evidence."""
    amendment, manifest, schedule = paced.verify(root, amendment_path)
    base = runner.directory(root)
    batch = base / "reruns" / BATCH
    old_rows = json.loads((base / "analysis-v1/measured-rows.json").read_text())
    targets = select_missing(old_rows)
    evidence = {
        str(p.relative_to(base)): runner.digest(p.read_bytes())
        for p in (base / "analysis-v1").glob("*.json")
    }
    selected: dict[str, tuple[runner.pilot.Assignment, ...]] = {}
    index = []
    for item in manifest["blocks"]:
        old_name = item["name"]
        rows = tuple(
            r for r in schedule[old_name] if r.record["assignment_id"] in targets
        )
        if not rows:
            continue
        old_cohort = base / "blocks" / old_name
        frozen = json.loads((old_cohort / "manifest.json").read_text())
        require_pilot_verification(root, frozen["oracle_image"])
        for row in rows:
            attempt = (
                old_cohort / runner.PROFILE / "attempts" / row.record["assignment_id"]
            )
            if (attempt / "raw.json").exists() or not (
                attempt / "failure.json"
            ).exists():
                raise ValueError("Original missing response or failure record changed")
            for path in attempt.rglob("*"):
                if path.is_file():
                    evidence[str(path.relative_to(base))] = runner.digest(
                        path.read_bytes()
                    )
        name = BATCH + "-" + old_name
        selected[name] = rows
        definition = {**frozen, "assignments": [r.record for r in rows]}
        cohort = batch / "blocks" / name
        runner.original.freeze(cohort / "manifest.json", definition)
        (cohort / runner.PROFILE).mkdir(exist_ok=True)
        index.append(
            {
                "name": name,
                "original_block": old_name,
                "mode": "A",
                "assigned": len(rows),
                "manifest_sha256": runner.digest(runner.json_text(definition)),
                "maximum_nanodollars": sum(
                    sum(runner.checks.bounds(r).values()) for r in rows
                ),
            }
        )
    if {
        r.record["assignment_id"] for rows in selected.values() for r in rows
    } != targets:
        raise ValueError("Frozen schedule does not cover all authorized targets")
    scoped = {**manifest, "blocks": index, "authorization": AUTHORIZATION}
    runner.original.freeze(batch / "manifest.json", scoped)
    runner.original.freeze(
        batch / "authorization.json",
        {
            "authorization": AUTHORIZATION,
            "assignment_ids": sorted(targets),
            "source_sha256": runner.digest(Path(__file__).read_bytes()),
            "main_manifest_sha256": runner.digest(runner.json_text(manifest)),
            "pacing_amendment_sha256": runner.digest(amendment_path.read_bytes()),
            "original_evidence_sha256": evidence,
            "maximum_nanodollars": sum(i["maximum_nanodollars"] for i in index),
        },
    )
    return amendment, scoped, selected


def analyze(
    root: Path,
    scoped: dict[str, Any],
    schedule: dict[str, tuple[runner.pilot.Assignment, ...]],
) -> dict[str, Any]:
    """Write an additive full-denominator analysis and explicit rerun lineage."""
    base = runner.directory(root)
    batch = base / "reruns" / BATCH
    authorization = json.loads((batch / "authorization.json").read_text())
    for name, expected in authorization["original_evidence_sha256"].items():
        if runner.digest((base / name).read_bytes()) != expected:
            raise ValueError("Original evidence changed during reruns")
    replacements = []
    lineage = []
    for item in scoped["blocks"]:
        cohort = batch / "blocks" / item["name"]
        if not (cohort / "complete.json").exists():
            raise ValueError("Rerun collection and judging are not finished")
        replacements.extend(runner.measure(cohort, schedule[item["name"]]))
        for row in schedule[item["name"]]:
            aid = row.record["assignment_id"]
            lineage.append(
                {
                    "assignment_id": aid,
                    "original_cohort": "blocks/" + item["original_block"],
                    "rerun_cohort": str(cohort.relative_to(base)),
                    "selected_attempt": 2,
                }
            )
    rows = overlay(
        json.loads((base / "analysis-v1/measured-rows.json").read_text()), replacements
    )
    result = analyze_rows(rows, scoped["plan"]["source_task_ids"], 12)
    result.update(
        manifest_sha256=authorization["main_manifest_sha256"],
        measured_rows_sha256=runner.digest(runner.json_text(rows)),
        rerun_authorization_sha256=runner.digest(
            (batch / "authorization.json").read_bytes()
        ),
    )
    destination = base / "analysis-generation-reruns-v1"
    for name, value in (
        ("measured-rows", rows),
        ("analysis", result),
        ("rerun-lineage", lineage),
    ):
        runner.original.freeze(destination / (name + ".json"), value)
    return result


def run(root: Path, amendment_path: Path) -> None:
    """Reuse the frozen collector under the original lock, pacing and global cap."""
    amendment, scoped, schedule = prepare(root, amendment_path)
    base = runner.directory(root)
    batch = base / "reruns" / BATCH
    authorization = json.loads((batch / "authorization.json").read_text())
    main_ledger = SpendingLedger(
        base / "spending.json",
        cap=runner.nanodollars(250),
        scope=authorization["main_manifest_sha256"],
    )
    pacer = SharedPacer(base / "operations" / paced.VERSION)
    real_status, real_record = runner.status, runner.record_stage
    stage = ExitStack()
    stage_path: Path | None = None

    def check(_: Path | None = None) -> None:
        paced.guard(base, amendment)
        if (batch / "operations/STOP").exists():
            raise RuntimeError("Rerun STOP is present")
        if stage_path is not None and (stage_path / "pacing-pending.json").exists():
            raise PacingDeferred("Current rerun stage has a deferred request")

    @contextmanager
    def acquire(_: Path, **kwargs: Any) -> Iterator[None]:
        owner = root / "runs/extension-pilot/coordination/owner.json"
        backup = batch / "previous-owner.json"
        if owner.exists() and not backup.exists():
            runner.original.freeze(backup, json.loads(owner.read_text()))
        with collection_lock(root, owner=runner.OWNER, run_id=BATCH):
            check()
            main_ledger.reserve(
                {i["name"]: i["maximum_nanodollars"] for i in scoped["blocks"]}
            )
            yield

    def ledger(path: Path, *, cap: int, scope: str) -> SpendingLedger:
        if path == batch / "spending.json":
            if cap != runner.nanodollars(250):
                raise ValueError("Whole-experiment cap changed")
            return main_ledger
        return SpendingLedger(path, cap=cap, scope=scope)

    def record(_: Path, identity: str, phase: str, failed: bool) -> None:
        with locked(base / "operations/stage-events.lock"):
            real_record(base, identity, phase, failed)
        real_record(batch, identity, phase, failed)

    def status(path: Path, manifest: dict[str, Any], state: str, **extra: Any) -> None:
        nonlocal stage_path
        stage.close()
        stage_path = None
        phase = extra.get("phase")
        if state == "running" and phase in ("generation", "judging"):
            clock = WaitClock()
            current_path = (
                base
                / "operations"
                / paced.VERSION
                / "attempts"
                / extra["block"]
                / extra["assignment_id"]
                / phase
            )
            stage_path = current_path
            stage.enter_context(paced_http(pacer, current_path, clock, check))
            proxy = SimpleNamespace(**{**vars(asyncio), "wait_for": clock.wait_for})
            stage.enter_context(patch.object(runner.pilot, "asyncio", proxy))
            stage.enter_context(patch.object(runner.original, "asyncio", proxy))
            stage.callback(paced.save_timing, current_path, clock)
        real_status(path, manifest, state, **extra)
        snapshot = json.loads((path / "operations/status.json").read_text())
        snapshot.update(assigned=10, batch=BATCH)
        write_json(path / "operations/status.json", snapshot)

    isolate_collection_process()
    with ExitStack() as stack:
        stack.enter_context(
            patch.object(runner, "load", return_value=(scoped, schedule))
        )
        stack.enter_context(patch.object(runner, "directory", return_value=batch))
        stack.enter_context(patch.object(runner, "collection_lock", acquire))
        stack.enter_context(patch.object(runner, "SpendingLedger", ledger))
        stack.enter_context(patch.object(runner, "guard", check))
        stack.enter_context(patch.object(runner, "record_stage", record))
        stack.enter_context(patch.object(runner, "status", status))
        stack.enter_context(patch.object(runner, "analyze", lambda _: None))
        try:
            runner.collect(root)
        finally:
            stage.close()
    result = analyze(root, scoped, schedule)
    print(
        json.dumps(
            {
                "assigned": result["assigned"],
                "missing_labels": result["missing_labels"],
                "batch": BATCH,
            }
        ),
        flush=True,
    )


def main() -> None:
    """Require an explicit paid flag; preparation and additive analysis are offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run", "analyze"))
    parser.add_argument("--amendment", required=True, type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    root = Path.cwd()
    if args.action == "run":
        if not args.generate:
            parser.error("Paid reruns require --generate")
        run(root, args.amendment.resolve())
    else:
        _, scoped, schedule = prepare(root, args.amendment.resolve())
        if args.action == "analyze":
            result = analyze(root, scoped, schedule)
            print(
                json.dumps(
                    {
                        "assigned": result["assigned"],
                        "missing_labels": result["missing_labels"],
                    }
                )
            )
        else:
            print(
                json.dumps(
                    {
                        "assignments": sum(len(r) for r in schedule.values()),
                        "maximum_usd": sum(
                            i["maximum_nanodollars"] for i in scoped["blocks"]
                        )
                        / 1e9,
                        "paid_calls": 0,
                    }
                )
            )


if __name__ == "__main__":
    main()
