"""Authorized pacing amendment around the unchanged frozen main collector.

The supervisor owns the original collection lock and whole-run spending scope.
Each child runs exactly one frozen block in an isolated process, reusing original
generation, judging, settlement, integrity checks and started-request handling.
"""

import argparse
import asyncio
import fcntl
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from evals import inspect_agent
from evals import inspect_extension_main as runner
from grader_power.agent_sandbox import EpisodeLimits
from grader_power.pacing import (
    WAIT_ALLOWANCE,
    PacingDeferred,
    SharedPacer,
    WaitClock,
    append_event,
    locked,
    paced_http,
    write_json,
)
from grader_power.spending import SpendingLedger
from scripts.resume_extension_main import recovery_guard

VERSION = "pacing-v1"
REVISION = "fifo-v4"
INITIAL_WORKERS = 12
MAX_WORKERS = 12
RAMP_INTERVAL_SECONDS = 900
FILES = (
    "src/grader_power/pacing.py",
    "evals/inspect_paced_extension.py",
    "scripts/run_paced_extension.py",
    "scripts/resume_extension_main.py",
)


def save_timing(path: Path, clock: WaitClock) -> None:
    """Keep the first stage timing intact when recovery revisits a started ID."""
    target = path / "timing.json"
    if target.exists():
        target = path / f"timing-revisit-{os.getpid()}-{time.time_ns()}.json"
    write_json(
        target,
        {
            "transport_wait_seconds": clock.waiting(),
            "wait_allowance_seconds": WAIT_ALLOWANCE,
        },
    )


def verify(
    root: Path, amendment_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Bind the amendment and all original inputs before any provider access."""
    base = runner.directory(root)
    if not amendment_path.resolve().is_relative_to(
        (base / "operations" / VERSION).resolve()
    ):
        raise ValueError("Amendment must belong to this main run")
    amendment = json.loads(amendment_path.read_text())
    manifest, schedule = runner.load(root)
    if (
        amendment.get("version") != VERSION
        or amendment.get("revision") != REVISION
        or amendment.get("authorized") is not True
        or not amendment.get("authorization")
        or amendment.get("manifest_sha256")
        != runner.digest((base / "manifest.json").read_bytes())
        or amendment.get("source_sha256")
        != {name: runner.digest((root / name).read_bytes()) for name in FILES}
        or manifest["plan"]["api_cap_usd"] != 250
        or amendment.get("initial_workers") != INITIAL_WORKERS
        or amendment.get("max_workers") != MAX_WORKERS
        or amendment.get("ramp_interval_seconds") != RAMP_INTERVAL_SECONDS
        or amendment.get("wait_allowance_seconds") != WAIT_ALLOWANCE
    ):
        raise ValueError("Pacing authorization/source/ceiling binding mismatch")
    recovery_path = base / amendment["recovery_record"]
    if not recovery_path.resolve().is_relative_to(
        (base / "operations/recoveries").resolve()
    ):
        raise ValueError("Invalid prior recovery location")
    if runner.digest(recovery_path.read_bytes()) != amendment["recovery_sha256"]:
        raise ValueError("Prior recovery amendment changed")
    amendment["recovery"] = json.loads(recovery_path.read_text())
    reports = amendment.get("verification_reports", {})
    if set(reports) != {"offline", "integration"}:
        raise ValueError("Require offline and live isolation verification reports")
    for kind, report in reports.items():
        path = base / report["path"]
        if (
            not path.resolve().is_relative_to((base / "operations" / VERSION).resolve())
            or runner.digest(path.read_bytes()) != report["sha256"]
        ):
            raise ValueError("Verification report binding changed")
        cases = ET.parse(path).getroot().findall(".//testcase")
        if len(cases) < (20 if kind == "offline" else 2) or any(
            case.find(tag) is not None
            for case in cases
            for tag in ("failure", "error", "skipped")
        ):
            raise ValueError("Pacing verification must pass without skipped cases")
    return amendment, manifest, schedule


def pending(directory: Path) -> list[Path]:
    """Refuse unattended restarts of transport-deferred, partly executed episodes."""
    return list(directory.glob("deferred/*.json"))


def guard(base: Path, amendment: dict[str, Any]) -> None:
    """Retain the reviewed failure-window origin; this change does not reset it."""
    with locked(base / "operations/stage-events.lock"):
        recovery_guard(base, amendment["recovery"])
    if pending(base / "operations" / VERSION):
        raise PacingDeferred(
            "A deferred request needs recovery; no episode replay permitted"
        )


@contextmanager
def child_permission(root: Path, parent: int) -> Iterator[None]:
    """Require the live supervising group and its held collection lock, fail closed."""
    directory = root / "runs/extension-pilot/coordination"
    owner = json.loads((directory / "owner.json").read_text())
    if (
        parent <= 1
        or parent != os.getppid()
        or os.getpgid(parent) != parent
        or os.getpgrp() != parent
        or owner.get("pid") != parent
        or owner.get("state") != "active"
        or owner.get("run_id") != runner.RUN
    ):
        raise RuntimeError("Worker has no live authorized supervisor")
    with (directory / "collection.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise RuntimeError("Supervisor is not holding collection lock")
    yield


def worker(root: Path, amendment_path: Path, name: str, parent: int) -> None:
    """Run one disjoint block with unchanged adapters and shared operational guards."""
    amendment, manifest, schedule = verify(root, amendment_path)
    base = runner.directory(root)
    pacing = base / "operations" / VERSION
    item = next(item for item in manifest["blocks"] if item["name"] == name)
    cohort = base / "blocks" / name
    scoped = {**manifest, "blocks": [item]}
    original_stage = runner.record_stage
    original_save = inspect_agent.save
    pacer = SharedPacer(pacing)
    stage = ExitStack()
    clock = WaitClock()
    stage_path: Path | None = None
    worker_status = pacing / "workers" / f"{name}.json"

    def check(_: Path | None = None) -> None:
        guard(base, amendment)
        if stage_path is not None and (stage_path / "pacing-pending.json").exists():
            raise PacingDeferred("Current stage has a deferred request")

    def record_stage(path: Path, identity: str, phase: str, failed: bool) -> None:
        check()
        with locked(base / "operations/stage-events.lock"):
            original_stage(path, identity, phase, failed)

    def ledger(path: Path, *, cap: int, scope: str) -> SpendingLedger:
        # Original collect() hashes its manifest argument. Its scoped block view
        # must still charge the original, immutable whole-run ledger identity.
        if path == base / "spending.json":
            if cap != runner.nanodollars(manifest["plan"]["api_cap_usd"]):
                raise ValueError("Global cap changed")
            scope = runner.digest(runner.json_text(manifest))
        return SpendingLedger(path, cap=cap, scope=scope)

    def save(path: Path, value: Any) -> None:
        if path.name == "episode.json":
            value = {
                **value,
                "active_work_seconds": value["wall_seconds"],
                "wall_seconds": value["wall_seconds"] + clock.waiting(),
                "transport_wait_seconds": clock.waiting(),
                "pacing_amendment": VERSION,
                "pacing_revision": REVISION,
            }
        original_save(path, value)

    def status(_: Path, __: dict[str, Any], state: str, **extra: Any) -> None:
        nonlocal clock, stage_path
        phase = extra.get("phase")
        if phase in ("generation", "judging"):
            stage.close()
            clock = WaitClock()
            current_path = pacing / "attempts" / name / extra["assignment_id"] / phase
            stage_path = current_path
            stage.enter_context(paced_http(pacer, current_path, clock, check))
            proxy = SimpleNamespace(**{**vars(asyncio), "wait_for": clock.wait_for})
            stage.enter_context(patch.object(inspect_agent, "asyncio", proxy))
            stage.enter_context(patch.object(runner.pilot, "asyncio", proxy))
            stage.enter_context(patch.object(runner.original, "asyncio", proxy))
            stage.callback(save_timing, current_path, clock)
            stage.enter_context(
                patch.object(
                    inspect_agent, "time", SimpleNamespace(monotonic=clock.monotonic)
                )
            )
        else:
            stage.close()
        write_json(
            worker_status,
            {
                "pid": os.getpid(),
                "state": state,
                "updated_at": time.time(),
                "block": name,
                **extra,
            },
        )

    with (
        child_permission(root, parent),
        locked(pacing / "workers" / f"{name}.lock"),
        ExitStack() as stack,
    ):
        check()
        stack.enter_context(
            patch.object(runner, "load", return_value=(scoped, schedule))
        )
        stack.enter_context(
            patch.object(
                runner,
                "collection_lock",
                lambda *a, **kw: child_permission(root, parent),
            )
        )
        stack.enter_context(patch.object(runner, "SpendingLedger", ledger))
        stack.enter_context(patch.object(runner, "guard", check))
        stack.enter_context(patch.object(runner, "record_stage", record_stage))
        stack.enter_context(patch.object(runner, "status", status))
        stack.enter_context(patch.object(runner, "analyze", lambda _: None))
        stack.enter_context(patch.object(inspect_agent, "save", save))
        lifetime = EpisodeLimits.container_lifetime_seconds.fget
        if lifetime is None:
            raise ValueError("Missing bounded container lifetime")
        stack.enter_context(
            patch.object(
                EpisodeLimits,
                "container_lifetime_seconds",
                property(lambda limits: lifetime(limits) + int(WAIT_ALLOWANCE)),
            )
        )
        try:
            runner.collect(root, max_blocks=1)
        finally:
            stage.close()
            # Scheduling estimates do not affect the scientific/scoring records.
            for row in schedule[name]:
                aid = row.record["assignment_id"]
                attempt = cohort / runner.PROFILE / "attempts" / aid
                for raw in attempt.glob("provider/*.raw.json"):
                    pacer.usage("fireworks", json.loads(raw.read_text()))
                if item["mode"] == "A" and (attempt / "raw.json").exists():
                    pacer.usage(
                        "fireworks", json.loads((attempt / "raw.json").read_text())
                    )
                judged = (
                    cohort / "final-file-evaluation" if item["mode"] == "B" else cohort
                )
                raw = judged / runner.PROFILE / "judge" / aid / "raw.json"
                if raw.exists():
                    pacer.usage("anthropic", json.loads(raw.read_text()))


def ramp(
    state: dict[str, Any], *, now: float, throttles: int, completed: int, errors: int
) -> bool:
    """Grow after 15 minutes of progress; halve only on new confirmed throttling.

    Failed stages remain visible in the audit counter and original pause guard.
    Invalid judgments provide no evidence of rate pressure and do not reset the
    ramp timer. Reobserving the same throttle must not repeatedly halve workers.
    """
    changed = False
    if throttles > state["throttles"]:
        state["workers"] = max(1, state["workers"] // 2)
        state["last_change"] = now
        state["completed_at_change"] = completed
        changed = True
    elif (
        now - state["last_change"] >= RAMP_INTERVAL_SECONDS
        and completed > state["completed_at_change"]
        and state["workers"] < MAX_WORKERS
    ):
        state["workers"] += 1
        state["last_change"] = now
        state["completed_at_change"] = completed
        changed = True
    state.update(throttles=throttles, errors=errors)
    return changed


def supervise(root: Path, amendment_path: Path) -> None:
    """Drain workers on pause, reserve globally before spawning, and analyze at end."""
    amendment, manifest, _ = verify(root, amendment_path)
    base = runner.directory(root)
    pacing = base / "operations" / VERSION
    runner.isolate_collection_process()
    ledger = SpendingLedger(
        base / "spending.json",
        cap=runner.nanodollars(250),
        scope=runner.digest(runner.json_text(manifest)),
    )
    live: dict[str, subprocess.Popen[bytes]] = {}
    queue = [
        item
        for item in manifest["blocks"]
        if not (base / "blocks" / item["name"] / "complete.json").exists()
    ]
    state = {
        "workers": INITIAL_WORKERS,
        "last_change": time.time(),
        "completed_at_change": 0,
        "throttles": 0,
        "errors": 0,
    }
    completed = 0
    pause: str | None = None
    with runner.collection_lock(root, owner=runner.OWNER, run_id=runner.RUN):
        guard(base, amendment)
        if (pacing / "rate-state.json").exists():
            state["throttles"] = sum(
                v["throttles"]
                for v in json.loads((pacing / "rate-state.json").read_text()).values()
            )
        history = base / "operations/request-events.jsonl"
        state["errors"] = sum(
            json.loads(line)["failed"] for line in history.read_text().splitlines()
        )
        append_event(
            pacing / "controller-events.jsonl",
            {
                "event": "started",
                "pid": os.getpid(),
                "workers": INITIAL_WORKERS,
                "revision": REVISION,
            },
        )
        try:
            while queue or live:
                for name, process in list(live.items()):
                    code = process.poll()
                    if code is not None:
                        del live[name]
                        if (
                            code
                            or not (base / "blocks" / name / "complete.json").exists()
                        ):
                            pause = "worker_failed"
                        else:
                            completed += 1
                if pause is None:
                    try:
                        guard(base, amendment)
                        if runner.sources(root) != manifest["code_sha256"]:
                            raise ValueError("Frozen sources changed")
                        if {
                            name: runner.digest((root / name).read_bytes())
                            for name in FILES
                        } != amendment["source_sha256"]:
                            raise ValueError("Amended sources changed")
                    except Exception as error:  # noqa: BLE001 -- drain children on a terminal guard failure
                        pause = type(error).__name__
                rates = (
                    json.loads((pacing / "rate-state.json").read_text())
                    if (pacing / "rate-state.json").exists()
                    else {}
                )
                throttles = sum(v["throttles"] for v in rates.values())
                with locked(base / "operations/stage-events.lock"):
                    errors = sum(
                        json.loads(line)["failed"]
                        for line in history.read_text().splitlines()
                    )
                if ramp(
                    state,
                    now=time.time(),
                    throttles=throttles,
                    completed=completed,
                    errors=errors,
                ):
                    append_event(
                        pacing / "controller-events.jsonl",
                        {"event": "concurrency_changed", **state},
                    )
                while pause is None and queue and len(live) < state["workers"]:
                    item = queue[0]
                    try:
                        ledger.reserve({item["name"]: item["maximum_nanodollars"]})
                    except runner.SpendingLimit:
                        # A draining block may release enough reserved allowance.
                        if not live:
                            pause = "spending_guard"
                        break
                    queue.pop(0)
                    name = item["name"]
                    log_path = pacing / "logs" / f"{name}-{time.time_ns()}.log"
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with log_path.open("xb") as log:
                        process = subprocess.Popen(
                            [
                                sys.executable,
                                "scripts/run_paced_extension.py",
                                "worker",
                                "--amendment",
                                str(amendment_path),
                                "--block",
                                name,
                                "--parent",
                                str(os.getpid()),
                                "--generate",
                            ],
                            cwd=root,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                        )
                    live[name] = process
                    append_event(
                        pacing / "controller-events.jsonl",
                        {
                            "event": "worker_started",
                            "block": name,
                            "pid": process.pid,
                            "log": str(log_path.relative_to(base)),
                        },
                    )
                runner.status(
                    base,
                    manifest,
                    "draining_for_pause" if pause else "running",
                    pacing=VERSION,
                    pacing_revision=REVISION,
                    active_workers={name: p.pid for name, p in live.items()},
                    concurrency_target=state["workers"],
                    pause_reason=pause,
                )
                write_json(pacing / "ramp-state.json", state)
                if pause and not live:
                    break
                time.sleep(2)
        except Exception as error:  # noqa: BLE001 -- terminal boundary records state and drains active children
            pause = type(error).__name__
            runner.status(
                base,
                manifest,
                "draining_for_pause",
                pacing=VERSION,
                pacing_revision=REVISION,
                pause_reason=pause,
            )
            append_event(
                pacing / "controller-events.jsonl",
                {"event": "controller_error", "error_type": pause},
            )
        finally:
            # Do not kill active requests or abandon a child's collection lock.
            for process in live.values():
                process.wait()
        if pause:
            runner.status(
                base,
                manifest,
                "paused",
                pacing=VERSION,
                pacing_revision=REVISION,
                pause_reason=pause,
            )
            append_event(
                pacing / "controller-events.jsonl", {"event": "paused", "reason": pause}
            )
            return
    runner.analyze(root)


def main() -> None:
    """Require the authorized source-bound amendment and an explicit paid flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "worker", "verify"))
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--block")
    parser.add_argument("--parent", type=int)
    args = parser.parse_args()
    if args.action == "verify":
        _, manifest, _ = verify(Path.cwd(), args.amendment)
        print(json.dumps({"verified_blocks": len(manifest["blocks"]), "paid_calls": 0}))
        return
    if not args.generate:
        parser.error("Paid collection requires --generate")
    if args.action == "worker":
        if args.block is None or args.parent is None:
            parser.error("Worker requires its block and supervisor")
        worker(Path.cwd(), args.amendment, args.block, args.parent)
    else:
        supervise(Path.cwd(), args.amendment)


if __name__ == "__main__":
    main()
