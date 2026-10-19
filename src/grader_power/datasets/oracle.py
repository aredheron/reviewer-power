"""Full EvalPlus adapter. All executable code stays in disposable Docker containers.

Inputs must come from the trusted loader, never from model-provided pickle files.
Only one task is staged outside the repository; no repository/cache is mounted.
"""

import hashlib
import json
import math
import pickle
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from grader_power.datasets.evalplus_loader import DatasetName, LoadedDataset

Status = Literal[
    "pass",
    "wrong_answer",
    "syntax_error",
    "timeout",
    "crash",
    "infrastructure_error",
    "oracle_error",
]
STATUSES = {"pass", "wrong_answer", "syntax_error", "timeout", "crash", "oracle_error"}
MEMORY_BYTES = 512 * 1024 * 1024
PIDS_LIMIT = 64
NANO_CPUS = 1_000_000_000


@dataclass(frozen=True)
class OracleResult:
    """Attempt outcome; full correctness requires complete passing suite evidence."""

    status: Status
    task_id: str
    image: str
    solution_sha256: str
    message: str
    suites: dict[str, Any] = field(default_factory=dict)
    mode: Literal["full"] = "full"

    @property
    def full_correct(self) -> bool:
        """Return true only for validated success from both complete suites."""
        return self.status == "pass" and all(
            name in self.suites
            and self.suites[name]["status"] == "pass"
            and len(self.suites[name]["details"]) == self.suites[name]["total"]
            and all(self.suites[name]["details"])
            for name in ("base", "plus")
        )


def _docker(*args: str, timeout: float = 15) -> str:
    """Call Docker without a shell; never return CLI stderr containing host details."""
    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout, check=True
    )
    return result.stdout.strip()


def docker_available() -> bool:
    """Check daemon reachability without starting Docker or running a container."""
    try:
        return bool(_docker("version", "--format", "{{.Server.Version}}"))
    except (OSError, subprocess.SubprocessError):
        return False


def _verify_container(info: dict[str, Any], image: str, staged: Path) -> None:
    """Fail closed unless Docker reports every required isolation control."""
    host = info["HostConfig"]
    config = info["Config"]
    mounts = sorted(info["Mounts"], key=lambda mount: mount["Destination"])
    valid = (
        info["Image"] == image
        and host["NetworkMode"] == "none"
        and host["NanoCpus"] == NANO_CPUS
        and host["Memory"] == MEMORY_BYTES
        and host["MemorySwap"] == MEMORY_BYTES
        and host["PidsLimit"] == PIDS_LIMIT
        and host["ReadonlyRootfs"] is True
        and host["Privileged"] is False
        and host["CapDrop"] == ["ALL"]
        and host["SecurityOpt"] == ["no-new-privileges"]
        and host["PidMode"] == ""
        and host["IpcMode"] == "private"
        and not host.get("Devices")
        and not host.get("DeviceRequests")
        and config["User"] == "65534:65534"
        and config["Entrypoint"] == ["python", "-I", "/opt/oracle/runner.py"]
        and len(mounts) in (1, 2)
        and mounts[0]["Type"] == "bind"
        and Path(mounts[0]["Source"]).resolve() == staged.resolve()
        and mounts[0]["Destination"] == "/oracle-input"
        and mounts[0]["RW"] is False
        and all(
            mount["Type"] == "tmpfs" and mount["Destination"] == "/tmp"
            for mount in mounts[1:]
        )
        and host["Tmpfs"] == {"/tmp": "rw,noexec,nosuid,size=128m,mode=1777"}
        and host["LogConfig"]
        == {"Type": "json-file", "Config": {"max-size": "1m", "max-file": "1"}}
    )
    if not valid:
        raise ValueError("Container isolation verification failed")


def _parse_result(output: str, task: dict[str, Any]) -> dict[str, Any]:
    """Reject incomplete, contradictory, or malformed success reports."""
    result = json.loads(output)
    if not isinstance(result, dict) or result.get("status") not in STATUSES:
        raise ValueError("Invalid worker result")
    suites = result.get("suites")
    if not isinstance(suites, dict):
        raise TypeError("Missing suite results")
    if result["status"] in {"syntax_error", "oracle_error"} and not suites:
        return result
    if set(suites) != {"base", "plus"}:
        raise ValueError("Both suites are required")
    for name, suite in suites.items():
        if not isinstance(suite, dict) or suite.get("status") not in {
            "pass",
            "wrong_answer",
            "timeout",
            "crash",
            "oracle_error",
        }:
            raise ValueError("Invalid suite")
        total = suite.get("total")
        details = suite.get("details")
        if (
            type(total) is not int
            or total != len(task[f"{name}_input"])
            or not isinstance(details, list)
            or len(details) > total
            or any(type(item) is not bool for item in details)
        ):
            raise ValueError("Invalid test counts or details")
        if suite["status"] == "pass" and (len(details) != total or not all(details)):
            raise ValueError("Incomplete passing suite")
    aggregate = next(
        status
        for status in ("oracle_error", "timeout", "crash", "wrong_answer", "pass")
        if any(suite["status"] == status for suite in suites.values())
    )
    if result["status"] != aggregate:
        raise ValueError("Contradictory result")
    return result


def evaluate_solution(
    dataset: LoadedDataset,
    name: DatasetName,
    task_id: str,
    solution: str,
    *,
    mode: Literal["full"],
    image: str,
    wall_seconds: float = 180,
    verify_upstream: bool = False,
) -> OracleResult:
    """Evaluate an exact complete function/file implementation, never a completion.

    Require explicit ``mode='full'`` and a local immutable ``sha256:...`` image
    built from configs/sandbox. No quick mode, sanitizer, retries, or host fallback
    exists. Native normalized inputs are transported with pickle, preserving
    tuples/complex values. Invalid interface arguments raise ValueError/TypeError;
    malformed source records yield oracle_error. Docker/cleanup failures yield
    infrastructure_error, including a missing daemon. A wall timeout is an
    incomplete timeout outcome, never successful verification.

    With verify_upstream=True, independently run EvalPlus's official checker in
    the same sandbox and require identical full-suite pass/detail evidence.
    """
    if mode != "full":
        raise ValueError("Final correctness requires mode='full'; quick is unsupported")
    if type(verify_upstream) is not bool:
        raise TypeError("verify_upstream must be a boolean")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("image must be an immutable local sha256 image ID")
    if name not in {"humaneval", "mbpp"}:
        raise ValueError("Unsupported dataset")
    if not isinstance(solution, str):
        raise TypeError("solution must be the complete, unmodified Python source")
    if not math.isfinite(wall_seconds) or not 1 <= wall_seconds <= 600:
        raise ValueError("wall_seconds must be finite and between 1 and 600")
    digest = hashlib.sha256(solution.encode()).hexdigest()

    def outcome(status: Status, message: str, **kwargs: Any) -> OracleResult:
        """Attach identity to every failure and successful result."""
        return OracleResult(status, task_id, image, digest, message, **kwargs)

    task = dataset.tasks.get(task_id)
    prefix = "HumanEval/" if name == "humaneval" else "Mbpp/"
    if (
        not task
        or task.get("task_id") != task_id
        or not task_id.startswith(prefix)
        or any(
            not isinstance(task.get(key), str) or not task[key].strip()
            for key in ("entry_point", "prompt", "canonical_solution")
        )
        or any(
            not isinstance(task.get(f"{suite}_input"), list)
            or not task[f"{suite}_input"]
            for suite in ("base", "plus")
        )
        or type(task.get("atol")) not in (int, float)
        or not math.isfinite(task["atol"])
        or task["atol"] < 0
    ):
        return outcome("oracle_error", "Missing, malformed, or empty full source suite")
    if dataset.provenance.get("variant") != "full":
        return outcome(
            "oracle_error", "Source provenance must identify the full corpus"
        )
    if not docker_available():
        return outcome("infrastructure_error", "A usable Docker daemon is required")

    container = f"grader-power-oracle-{uuid.uuid4().hex}"
    result = outcome("infrastructure_error", "Container did not complete")
    # /tmp is intentional: never stage inside a repository or model workspace.
    with tempfile.TemporaryDirectory(prefix="grader-power-oracle-", dir="/tmp") as temp:
        staged = Path(temp)
        staged.chmod(0o755)
        request = staged / "request.pkl"
        request.write_bytes(
            pickle.dumps(
                {
                    "dataset": name,
                    "task": task,
                    "solution": solution,
                    "verify_upstream": verify_upstream,
                }
            )
        )
        request.chmod(0o444)
        waiting_for_evaluation = False
        try:
            _docker(
                "create",
                "--name",
                container,
                "--pull=never",
                "--label",
                "grader-power.oracle=true",
                "--network=none",
                "--cpus=1",
                "--memory=512m",
                "--memory-swap=512m",
                "--pids-limit=64",
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
                "--ipc=private",
                "--user=65534:65534",
                "--env=HTTP_PROXY=",
                "--env=HTTPS_PROXY=",
                "--env=ALL_PROXY=",
                "--env=NO_PROXY=",
                "--env=http_proxy=",
                "--env=https_proxy=",
                "--env=all_proxy=",
                "--env=no_proxy=",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=128m,mode=1777",
                "--log-driver=json-file",
                "--log-opt=max-size=1m",
                "--log-opt=max-file=1",
                "--mount",
                f"type=bind,src={staged},dst=/oracle-input,readonly",
                image,
            )
            info = json.loads(_docker("inspect", container))[0]
            _verify_container(info, image, staged)
            _docker("start", container)
            waiting_for_evaluation = True
            exit_code = _docker("wait", container, timeout=wall_seconds)
            waiting_for_evaluation = False
            state = json.loads(_docker("inspect", container))[0]["State"]
            if state.get("OOMKilled"):
                result = outcome("crash", "Container exceeded memory limit")
            elif exit_code != "0" or state.get("ExitCode") != 0:
                result = outcome("oracle_error", "Evaluator process exited abnormally")
            else:
                parsed = _parse_result(_docker("logs", container), task)
                if (
                    verify_upstream
                    and parsed["status"] != "oracle_error"
                    and (
                        any(
                            suite.get("upstream_parity", {}).get("status") != "matched"
                            for suite in parsed["suites"].values()
                        )
                        or set(parsed["suites"]) != {"base", "plus"}
                    )
                ):
                    raise ValueError("Missing upstream parity evidence")
                result = outcome(
                    parsed["status"],
                    "Full evaluation requested",
                    suites=parsed["suites"],
                )
        except subprocess.TimeoutExpired:
            result = outcome(
                "timeout" if waiting_for_evaluation else "infrastructure_error",
                "Evaluation wall deadline exceeded"
                if waiting_for_evaluation
                else "Docker control operation timed out",
            )
        except (
            OSError,
            subprocess.SubprocessError,
            ValueError,
            KeyError,
            TypeError,
            IndexError,
        ):
            result = outcome(
                "infrastructure_error", "Docker, isolation, or result protocol failed"
            )
        finally:
            try:
                _docker("rm", "--force", container)
            except (OSError, subprocess.SubprocessError):
                result = outcome(
                    "infrastructure_error",
                    f"Cleanup unverified for {container}; remove it before continuing",
                )
    return result
