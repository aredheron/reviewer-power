"""Independent process-level lock checks; no model calls or solution execution."""

import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from grader_power.collection_lock import process_group_members

# Fixed test harness code, never model-provided code. Each worker owns its group.
WORKER = """
import json
import os
import subprocess
import sys
from pathlib import Path
from grader_power.collection_lock import collection_lock, isolate_collection_process

isolate_collection_process()
try:
    with collection_lock(Path(sys.argv[1]), owner='auto1', run_id='independent-test'):
        if sys.argv[2] == 'orphan':
            child = subprocess.Popen(
                [sys.executable, '-c', 'import time; time.sleep(30)'],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(json.dumps({'entered': True, 'child': child.pid, 'group': os.getpgrp()}), flush=True)
            os._exit(0)
        print(json.dumps({'entered': True}), flush=True)
        if sys.argv[2] == 'hold':
            sys.stdin.readline()
except RuntimeError as error:
    print(json.dumps({'entered': False, 'reason': str(error)}), flush=True)
"""


def attempt(root: Path) -> dict[str, Any]:
    """Attempt a real lock from a fresh process without inheriting the test group."""
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(root), "once"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


def test_real_processes_cannot_overlap_collection(tmp_path: Path) -> None:
    """The lock spans work after entry and excludes a separate CLI process."""
    with subprocess.Popen(
        [sys.executable, "-c", WORKER, str(tmp_path), "hold"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as holder:
        try:
            assert holder.stdout is not None
            ready, _, _ = select.select([holder.stdout], [], [], 10)
            assert ready, "Lock holder failed to report readiness"
            assert json.loads(holder.stdout.readline()) == {"entered": True}
            blocked = attempt(tmp_path)
            assert blocked["entered"] is False
            assert "holds the lock" in blocked["reason"]
            holder.communicate("finish\n", timeout=10)
            assert holder.returncode == 0
        finally:
            if holder.poll() is None:
                holder.kill()
                holder.communicate(timeout=5)
    assert attempt(tmp_path) == {"entered": True}


def test_orphan_process_prevents_restart(tmp_path: Path) -> None:
    """A crashed parent releasing flock does not permit surviving work to overlap."""
    result = subprocess.run(
        [sys.executable, "-c", WORKER, str(tmp_path), "orphan"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    record = json.loads(result.stdout)
    child = record["child"]
    group = record["group"]
    try:
        assert child in process_group_members(group)
        blocked = attempt(tmp_path)
        assert blocked["entered"] is False
        assert "surviving processes" in blocked["reason"]
    finally:
        # This PID is our own inert test child, not a collection worker.
        os.kill(child, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while process_group_members(group) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not process_group_members(group), "Test child did not terminate"
    assert attempt(tmp_path) == {"entered": True}
