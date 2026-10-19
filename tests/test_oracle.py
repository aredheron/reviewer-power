"""Offline contract tests; Docker and every executable oracle operation are mocked."""

import copy
import json
import pickle
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from grader_power.datasets import oracle
from grader_power.datasets.evalplus_loader import LoadedDataset

IMAGE = "sha256:" + "a" * 64
SOLUTION = "def transform(items):\n    return [x * 2 for x in items]\n"


def source() -> LoadedDataset:
    """Build a synthetic full-source record without executing its solution."""
    return LoadedDataset(
        {
            "HumanEval/999": {
                "task_id": "HumanEval/999",
                "entry_point": "transform",
                "prompt": "def transform(items):\n",
                "canonical_solution": "    return [x * 2 for x in items]\n",
                "base_input": [([1, 2],)],
                "plus_input": [([],), ([3],)],
                "atol": 0,
            }
        },
        {"variant": "full"},
    )


def passing() -> dict[str, Any]:
    """Return complete base/plus evidence for the synthetic source."""
    return {
        "status": "pass",
        "suites": {
            "base": {"status": "pass", "total": 1, "details": [True]},
            "plus": {"status": "pass", "total": 2, "details": [True, True]},
        },
    }


class FakeDocker:
    """Stateful Docker protocol fake that inspects staged inputs without execution."""

    def __init__(self) -> None:
        """Initialize a complete successful evaluation transcript."""
        self.calls: list[tuple[str, ...]] = []
        self.output = passing()
        self.staged: Path | None = None
        self.payload: dict[str, Any] = {}
        self.info: dict[str, Any] = {}
        self.fail_at: str | None = None
        self.timeout_at: str | None = None
        self.isolation_fault: tuple[str, Any] | None = None
        self.cleanup_failure = False
        self.exit_code = "0"

    def __call__(self, *args: str, timeout: float = 15) -> str:
        """Respond to protocol operations, optionally failing one operation."""
        self.calls.append(args)
        if args[0] == self.timeout_at:
            raise subprocess.TimeoutExpired(args, timeout)
        if args[0] == self.fail_at or (args[0] == "rm" and self.cleanup_failure):
            raise subprocess.CalledProcessError(1, args)
        if args[0] == "version":
            return "28.0"
        if args[0] == "create":
            mount = args[args.index("--mount") + 1]
            self.staged = Path(mount.split("src=", 1)[1].split(",dst=", 1)[0])
            self.payload = pickle.loads((self.staged / "request.pkl").read_bytes())
            self.info = {
                "Image": IMAGE,
                "HostConfig": {
                    "NetworkMode": "none",
                    "NanoCpus": oracle.NANO_CPUS,
                    "Memory": oracle.MEMORY_BYTES,
                    "MemorySwap": oracle.MEMORY_BYTES,
                    "PidsLimit": 64,
                    "ReadonlyRootfs": True,
                    "Privileged": False,
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges"],
                    "PidMode": "",
                    "IpcMode": "private",
                    "Tmpfs": {"/tmp": "rw,noexec,nosuid,size=128m,mode=1777"},
                    "LogConfig": {
                        "Type": "json-file",
                        "Config": {"max-size": "1m", "max-file": "1"},
                    },
                },
                "Config": {
                    "User": "65534:65534",
                    "Entrypoint": ["python", "-I", "/opt/oracle/runner.py"],
                },
                "Mounts": [
                    {
                        "Type": "bind",
                        "Source": str(self.staged),
                        "Destination": "/oracle-input",
                        "RW": False,
                    }
                ],
                "State": {"ExitCode": int(self.exit_code), "OOMKilled": False},
            }
            if self.isolation_fault:
                key, value = self.isolation_fault
                self.info["HostConfig"][key] = value
            return "container-id"
        if args[0] == "inspect":
            return json.dumps([self.info])
        if args[0] == "wait":
            return self.exit_code
        if args[0] == "logs":
            return json.dumps(self.output)
        return ""


@pytest.fixture
def docker(monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    """Replace all Docker calls, guaranteeing unit tests cannot execute code."""
    fake = FakeDocker()
    monkeypatch.setattr(oracle, "_docker", fake)
    return fake


def evaluate(dataset: LoadedDataset | None = None) -> oracle.OracleResult:
    """Exercise the explicit complete-source/full-evaluation interface."""
    return oracle.evaluate_solution(
        dataset or source(),
        "humaneval",
        "HumanEval/999",
        SOLUTION,
        mode="full",
        image=IMAGE,
    )


def test_full_success_preserves_native_inputs_and_source(docker: FakeDocker) -> None:
    """No JSON coercion, code repair, or mutation; temporary payload is removed."""
    dataset = source()
    dataset.tasks["HumanEval/999"]["plus_input"].append(((1 + 2j,),))
    docker.output["suites"]["plus"].update(total=3, details=[True] * 3)
    original = copy.deepcopy(dataset)
    result = evaluate(dataset)
    assert result.full_correct
    assert result.mode == "full"
    assert docker.payload["solution"] == SOLUTION
    assert docker.payload["task"] == original.tasks["HumanEval/999"]
    assert dataset == original
    assert docker.staged and not docker.staged.exists()
    assert docker.calls[-1][:2] == ("rm", "--force")
    create = next(call for call in docker.calls if call[0] == "create")
    assert "--pull=never" in create
    assert "--env-file" not in create
    assert create.count("--mount") == 1
    assert "--env=HTTP_PROXY=" in create


@pytest.mark.parametrize("matched", [False, True])
def test_upstream_parity_requires_explicit_evidence(
    docker: FakeDocker, matched: bool
) -> None:
    """An ordinary pass cannot satisfy an explicitly requested parity check."""
    if matched:
        for suite in docker.output["suites"].values():
            suite["upstream_parity"] = {"status": "matched"}
    result = oracle.evaluate_solution(
        source(),
        "humaneval",
        "HumanEval/999",
        SOLUTION,
        mode="full",
        image=IMAGE,
        verify_upstream=True,
    )
    assert docker.payload["verify_upstream"] is True
    assert result.full_correct is matched
    assert result.status == ("pass" if matched else "infrastructure_error")


def test_missing_docker_is_unmet_prerequisite(docker: FakeDocker) -> None:
    """Missing daemon produces infrastructure failure without host fallback."""
    docker.fail_at = "version"
    result = evaluate()
    assert result.status == "infrastructure_error"
    assert "Docker daemon" in result.message
    assert [call[0] for call in docker.calls] == ["version"]


@pytest.mark.parametrize(
    "status", ["wrong_answer", "syntax_error", "timeout", "crash", "oracle_error"]
)
def test_failure_never_becomes_correct(docker: FakeDocker, status: str) -> None:
    """Failure categories survive transport and all attempted containers are removed."""
    docker.output["status"] = status
    if status in {"syntax_error", "oracle_error"}:
        docker.output["suites"] = {}
    else:
        docker.output["suites"]["plus"].update(status=status, details=[False])
    result = evaluate()
    assert result.status == status
    assert not result.full_correct
    assert docker.calls[-1][0] == "rm"


@pytest.mark.parametrize(
    "key,value",
    [
        ("NetworkMode", "bridge"),
        ("Memory", 0),
        ("MemorySwap", -1),
        ("NanoCpus", 0),
        ("PidsLimit", -1),
        ("ReadonlyRootfs", False),
        ("Privileged", True),
        ("CapDrop", []),
        ("SecurityOpt", ["no-new-privileges", "seccomp=unconfined"]),
        ("PidMode", "host"),
        ("IpcMode", "host"),
        ("Tmpfs", {}),
    ],
)
def test_isolation_must_be_verified_before_execution(
    docker: FakeDocker, key: str, value: Any
) -> None:
    """A daemon ignoring any required control cannot execute the submission."""
    docker.isolation_fault = (key, value)
    assert evaluate().status == "infrastructure_error"
    assert not any(call[0] == "start" for call in docker.calls)
    assert docker.calls[-1][0] == "rm"


@pytest.mark.parametrize("operation", ["create", "inspect", "start", "wait", "logs"])
def test_docker_failures_always_attempt_cleanup(
    docker: FakeDocker, operation: str
) -> None:
    """Even uncertain create/start outcomes are followed by forced cleanup."""
    docker.fail_at = operation
    assert evaluate().status == "infrastructure_error"
    assert docker.calls[-1][0] == "rm"
    assert docker.staged is None or not docker.staged.exists()


def test_wall_timeout_and_cleanup_failure(docker: FakeDocker) -> None:
    """Timeout cannot pass; cleanup failure overrides even a valid pass report."""
    docker.timeout_at = "wait"
    assert evaluate().status == "timeout"
    assert docker.calls[-1][0] == "rm"
    docker.timeout_at = None
    docker.cleanup_failure = True
    result = evaluate()
    assert result.status == "infrastructure_error"
    assert "Cleanup unverified" in result.message
    assert not result.full_correct


def test_control_timeout_is_infrastructure_failure(docker: FakeDocker) -> None:
    """A stuck Docker command is distinct from a running evaluation's deadline."""
    docker.timeout_at = "inspect"
    assert evaluate().status == "infrastructure_error"
    assert docker.calls[-1][0] == "rm"


@pytest.mark.parametrize(
    "fault", ["missing_plus", "partial", "false", "count", "contradiction", "not_bool"]
)
def test_invalid_success_evidence_is_rejected(docker: FakeDocker, fault: str) -> None:
    """Success needs consistent counts and literal true for every base/plus input."""
    plus = docker.output["suites"]["plus"]
    if fault == "missing_plus":
        del docker.output["suites"]["plus"]
    elif fault == "partial":
        plus["details"] = [True]
    elif fault == "false":
        plus["details"] = [True, False]
    elif fault == "count":
        plus["total"] = 1
    elif fault == "contradiction":
        plus.update(status="timeout", details=[])
    else:
        plus["details"] = [1, 1]
    assert evaluate().status == "infrastructure_error"


def test_nonzero_exit_rejects_even_passing_output(docker: FakeDocker) -> None:
    """A crashed worker's output cannot rescue a failed process."""
    docker.exit_code = "1"
    assert evaluate().status == "oracle_error"


@pytest.mark.parametrize(
    "field,value", [("plus_input", {}), ("base_input", []), ("atol", -1)]
)
def test_malformed_source_is_not_repaired(
    docker: FakeDocker, field: str, value: Any
) -> None:
    """Invalid/empty source suites fail before Docker; original records remain intact."""
    dataset = source()
    dataset.tasks["HumanEval/999"][field] = value
    assert evaluate(dataset).status == "oracle_error"
    assert dataset.tasks["HumanEval/999"][field] == value
    assert not docker.calls


def test_explicit_full_and_immutable_image_required(docker: FakeDocker) -> None:
    """No quick mode or mutable image tag can masquerade as final verification."""
    kwargs: dict[str, Any] = {"mode": "quick", "image": IMAGE}
    with pytest.raises(ValueError, match="mode='full'"):
        oracle.evaluate_solution(
            source(), "humaneval", "HumanEval/999", SOLUTION, **kwargs
        )
    kwargs.update(mode="full", image="oracle:latest")
    with pytest.raises(ValueError, match="immutable"):
        oracle.evaluate_solution(
            source(), "humaneval", "HumanEval/999", SOLUTION, **kwargs
        )
    missing_mode: dict[str, Any] = {"image": IMAGE}
    with pytest.raises(TypeError):
        oracle.evaluate_solution(
            source(), "humaneval", "HumanEval/999", SOLUTION, **missing_mode
        )
    assert not docker.calls


def test_worker_contract_without_executing_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect worker orchestration using mock canonical results and mock checks."""
    worker = runpy.run_path(
        str(Path(__file__).parents[1] / "configs/sandbox/runner.py")
    )
    run = worker["evaluate"]
    namespace = run.__globals__
    monkeypatch.setattr(namespace["platform"], "python_version", lambda: "3.12.12")
    monkeypatch.setitem(namespace, "version", lambda _: "0.3.1")
    seen: list[tuple[str, Any]] = []

    def canonical(
        code: str, inputs: list[Any], entry: str, **kwargs: Any
    ) -> tuple[list[Any], list[float]]:
        """Record canonical source and native inputs; never evaluate them."""
        seen.append(("canonical", (code, copy.deepcopy(inputs), entry, kwargs)))
        return [None] * len(inputs), [0.001] * len(inputs)

    def check(
        dataset: str,
        task: dict[str, Any],
        solution: str,
        inputs: list[Any],
        expected: list[Any],
        times: list[float],
        verify_upstream: bool = False,
    ) -> dict[str, Any]:
        """Record exact submission bytes and return full mocked evidence."""
        assert verify_upstream is False
        seen.append(("submission", solution))
        return {"status": "pass", "total": len(inputs), "details": [True] * len(inputs)}

    monkeypatch.setitem(namespace, "trusted_exec", canonical)
    monkeypatch.setitem(namespace, "check_suite", check)
    task = source().tasks["HumanEval/999"]
    assert (
        run({"task": task, "dataset": "humaneval", "solution": SOLUTION}) == passing()
    )
    assert len(seen) == 4
    assert seen[0][1][0] == task["prompt"] + task["canonical_solution"]
    assert seen[1] == ("submission", SOLUTION)
    seen.clear()
    assert (
        run({"task": task, "dataset": "humaneval", "solution": "def x(:"})["status"]
        == "syntax_error"
    )
    assert seen == []


@pytest.mark.parametrize(
    "values,progress,stat,fault,exitcode,alive,status",
    [
        ([1, 1], 2, 0, 0, 0, [False, False], "pass"),
        ([1, 0], 2, 0, 0, 0, [False, False], "wrong_answer"),
        ([0, 0], 2, 0, 2, 0, [False, False], "timeout"),
        ([0, 0], 2, 0, 1, 0, [False, False], "crash"),
        ([0, 0], 0, 3, 0, 7, [False, False], "crash"),
        ([0, 0], 0, 3, 0, 0, [False, False], "crash"),
        ([1, 0], 1, 0, 0, 0, [False, False], "oracle_error"),
        ([0, 0], 0, 3, 0, -15, [True, False], "timeout"),
        ([0, 0], 0, 3, 0, -9, [True, True], "timeout"),
    ],
)
def test_worker_supervision_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    values: list[int],
    progress: int,
    stat: int,
    fault: int,
    exitcode: int,
    alive: list[bool],
    status: str,
) -> None:
    """Simulate child outcomes without starting a process or executing any source."""
    worker = runpy.run_path(
        str(Path(__file__).parents[1] / "configs/sandbox/runner.py")
    )
    process = Mock()
    process.exitcode = exitcode
    process.is_alive.side_effect = alive
    context = Mock()
    context.Value.side_effect = [
        SimpleNamespace(value=stat),
        SimpleNamespace(value=progress),
        SimpleNamespace(value=fault),
    ]
    context.Array.return_value = values
    context.Process.return_value = process
    get_context = Mock(return_value=context)
    monkeypatch.setattr(worker["mp"], "get_context", get_context)
    task = source().tasks["HumanEval/999"]
    result = worker["check_suite"](
        "humaneval", task, SOLUTION, task["plus_input"], [[], [6]], [0.1, 0.5]
    )
    assert result == {
        "status": status,
        "details": [bool(v) for v in values[:progress]],
        "total": 2,
    }
    get_context.assert_called_once_with("fork")
    process.start.assert_called_once()
    assert process.join.call_args_list[0].args == (6.0,)
    assert process.terminate.called == alive[0]
    assert process.kill.called == alive[1]
    args = context.Process.call_args.kwargs["args"]
    assert args[2] == SOLUTION
    assert args[5] == [1.0, 2.0]


def test_observer_preserves_function_global_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The diagnostic wrapper must not replace recursive/global user lookups."""
    worker = runpy.run_path(
        str(Path(__file__).parents[1] / "configs/sandbox/runner.py")
    )
    module = worker["evaluation"]
    task = source().tasks["HumanEval/999"]
    namespace: dict[str, Any] = {}
    replacement = object()
    original = Mock()

    def mock_call(*args: Any) -> None:
        """Check original binding on first call and retain deliberate later mutation."""
        assert namespace["transform"] is (
            original if original.call_count == 1 else replacement
        )
        namespace["transform"] = replacement

    original.side_effect = mock_call

    def mock_exec(code: str, globals_dict: dict[str, Any]) -> None:
        """Inject a callable mock without compiling or executing source."""
        assert code == SOLUTION
        assert globals_dict is namespace
        namespace["transform"] = original

    def mock_engine(*args: Any) -> None:
        """Model EvalPlus's single lookup of the callable followed by repeated calls."""
        module.exec(args[2], namespace)
        target = namespace["transform"]
        target([1])
        target([2])

    with monkeypatch.context() as patch:
        patch.setattr(module, "exec", None, raising=False)
        patch.setattr(module, "unsafe_execute", mock_engine)
        patch.setattr(worker["builtins"], "exec", mock_exec)
        worker["execute_observed"](
            "humaneval",
            task,
            SOLUTION,
            [],
            [],
            [],
            SimpleNamespace(value=3),
            [],
            SimpleNamespace(value=0),
            SimpleNamespace(value=0),
        )
    assert original.call_count == 2
