"""Resume the frozen main extension after an explicitly authorized guard reset.

Only the operational error-window origin changes. The complete event history is
retained and its reviewed prefix is hash-bound. Source, budget, sandbox, STOP and
started-request protections remain in the original collector.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals import inspect_extension_main as runner


def recovery_guard(base: Path, recovery: dict[str, Any]) -> None:
    """Exclude only the authorized prefix; never erase history or weaken limits."""
    if (base / "operations/STOP").exists():
        raise RuntimeError("User STOP is present")
    payload = (base / "operations/request-events.jsonl").read_bytes()
    cutoff = recovery["reviewed_prefix_bytes"]
    if type(cutoff) is not int or cutoff < 0 or cutoff > len(payload):
        raise ValueError("Invalid recovery event offset")
    prefix = payload[:cutoff]
    if (
        runner.digest(prefix) != recovery["reviewed_prefix_sha256"]
        or prefix
        and not prefix.endswith(b"\n")
    ):
        raise ValueError("Reviewed request history changed")
    new_events = [json.loads(line) for line in payload[cutoff:].splitlines()]
    if any(type(row.get("failed")) is not bool for row in new_events):
        raise ValueError("Malformed request-event history")
    if sum(row["failed"] for row in new_events[-48:]) >= 5:
        raise RuntimeError("Five failed request stages in the last 48; review required")


def run(root: Path, recovery_file: Path) -> None:
    """Verify the auditable amendment, then use the original locked collector."""
    base = runner.directory(root)
    if not recovery_file.resolve().is_relative_to(
        (base / "operations/recoveries").resolve()
    ):
        raise ValueError("Recovery record must belong to this run")
    recovery = json.loads(recovery_file.read_text())
    if (
        recovery.get("run_id") != runner.RUN
        or recovery.get("authorized_resume") is not True
        or not recovery.get("authorization")
        or not recovery.get("diagnosis")
        or recovery.get("manifest_sha256")
        != runner.digest((base / "manifest.json").read_bytes())
        or recovery.get("resume_script_sha256")
        != runner.digest(Path(__file__).read_bytes())
    ):
        raise ValueError("Recovery authorization/source binding mismatch")
    recovery_guard(base, recovery)
    runner.isolate_collection_process()
    with patch.object(runner, "guard", lambda path: recovery_guard(path, recovery)):
        runner.collect(root)


def main() -> None:
    """Require a reviewed recovery artifact and explicit paid-collection flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery", required=True, type=Path)
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()
    if not args.generate:
        parser.error("Resumption requires --generate and user authorization")
    run(Path.cwd(), args.recovery)


if __name__ == "__main__":
    main()
