"""One fail-closed local collection lock for extension pilots, including judging.

The CLI must enter a dedicated process group before acquisition. A crash leaves
an active owner record; a later process checks that group's surviving children
before starting. No command-line arguments, environment, or credentials are read.
"""

import fcntl
import json
import os
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def process_group_members(group: int) -> set[int]:
    """Read process IDs only, failing closed if process inspection is unavailable."""
    with subprocess.Popen(
        ["ps", "-axo", "pid=,pgid="],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            stdout, _ = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()
            raise RuntimeError("Process inspection timed out") from None
        if process.returncode != 0:
            raise RuntimeError("Process inspection failed")
        return {
            int(fields[0])
            for line in stdout.splitlines()
            if len(fields := line.split()) == 2
            and int(fields[1]) == group
            and int(fields[0]) != process.pid
        }


def isolate_collection_process() -> None:
    """Give a command-line worker a dedicated group for orphan detection."""
    if os.getpgrp() != os.getpid():
        os.setsid()
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Collection requires a dedicated process group")


def _write_owner(path: Path, record: dict[str, Any]) -> None:
    """Atomically replace nonsecret lock metadata while the OS lock is held."""
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def collection_lock(root: Path, *, owner: str, run_id: str) -> Iterator[None]:
    """Hold one atomic lock across all paid generation and judging in a batch.

    A stale active owner is reusable only when its whole process group is gone.
    Surviving children on exit keep the record active and require investigation;
    this helper never kills processes or clears ambiguous ownership automatically.
    """
    if owner not in (
        "auto1",
        "auto5",
        "auto4",
        "exploration-v1",
        "named-files-v1",
        "named-files-v2",
        "development-six-v1",
        "gpt-readiness-v1",
        "gpt-extension-main-v1",
        "framing-names-v1",
        "reward-conflict-v1",
        "three-style-v1",
        "reversal-v1",
        "compact-models-v1",
        "simple-models-v1",
        "reviewer-observation-v1",
        "strong-preferences-v1",
        "expanded-preferences-v1",
    ):
        raise ValueError("Unknown extension collection owner")
    if os.getpgrp() != os.getpid():
        raise RuntimeError(
            "Call isolate_collection_process in the CLI before collecting"
        )
    directory = root / "runs/extension-pilot/coordination"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(directory / "collection.lock", os.O_CREAT | os.O_RDWR, 0o600)
    path = directory / "owner.json"
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another extension collection/judge worker holds the lock"
            ) from None
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["state"] == "active" and process_group_members(
                previous["pgid"]
            ):
                raise RuntimeError(
                    "Previous collection process group still has surviving processes"
                )
        previous_owner = {
            "auto5": "auto1",
            "auto4": "auto5",
            "exploration-v1": "auto4",
            "named-files-v1": "exploration-v1",
            "named-files-v2": "named-files-v1",
            "development-six-v1": "named-files-v2",
            "gpt-readiness-v1": "development-six-v1",
        }.get(owner)
        if previous_owner is not None:
            release = directory / f"{previous_owner}-complete.json"
            if (
                not release.exists()
                or json.loads(release.read_text()).get("complete") is not True
            ):
                raise RuntimeError(
                    f"Paid collection order requires {previous_owner}'s completion handoff"
                )
        record = {
            "owner": owner,
            "run_id": run_id,
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "state": "active",
            "started_at": datetime.now(UTC).isoformat(),
        }
        _write_owner(path, record)
        try:
            yield
        finally:
            survivors = process_group_members(os.getpgrp()) - {os.getpid()}
            if survivors:
                _write_owner(
                    path, {**record, "surviving_child_pids": sorted(survivors)}
                )
                raise RuntimeError(
                    "Collection has surviving child processes; owner remains active"
                )
            _write_owner(
                path,
                {
                    **record,
                    "state": "released",
                    "finished_at": datetime.now(UTC).isoformat(),
                },
            )
    finally:
        os.close(descriptor)
