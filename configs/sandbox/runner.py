"""Container-only EvalPlus worker; never execute this module on the host.

The comparison engine is EvalPlus 0.3.1's unsafe_execute, unchanged. Its process
supervision is reproduced here to retain exit codes and exception diagnostics
that untrusted_check discards. No submission text is rewritten or appended.
"""

import builtins
import json
import multiprocessing as mp
import pickle
import platform
from functools import wraps
from importlib.metadata import version
from pathlib import Path
from typing import Any

import evalplus.eval as evaluation
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.eval.utils import TimeoutException
from evalplus.gen.util import trusted_exec


def execute_observed(
    dataset: str,
    task: dict[str, Any],
    solution: str,
    inputs: list[Any],
    expected: list[Any],
    limits: list[float],
    stat: Any,
    details: Any,
    progress: Any,
    fault: Any,
) -> None:
    """Observe exceptions escaping submitted code without replacing comparisons."""

    def observed_exec(code: str, namespace: dict[str, Any]) -> None:
        """Execute the exact source and wrap only the callable's exception boundary."""
        try:
            # Intentional execution boundary: this worker runs only inside Docker.
            builtins.exec(code, namespace)  # noqa: S102
            original = namespace[task["entry_point"]]
            first_call = True

            @wraps(original)
            def observed_call(*args: Any, **kwargs: Any) -> Any:
                """Record escaping exceptions; EvalPlus still handles their outcome."""
                nonlocal first_call
                if first_call:
                    # unsafe_execute has now captured this wrapper as its local
                    # fn. Restore the original global binding before user code
                    # runs so recursion/identity retain their normal semantics.
                    namespace[task["entry_point"]] = original
                    first_call = False
                try:
                    return original(*args, **kwargs)
                except TimeoutException:
                    fault.value = 2
                    raise
                except BaseException:
                    if fault.value != 2:
                        fault.value = 1
                    raise

            namespace[task["entry_point"]] = observed_call
        except BaseException:
            fault.value = 1
            raise

    # Module-local exec lookup, only in the disposable child process.
    evaluation.__dict__["exec"] = observed_exec
    evaluation.unsafe_execute(
        dataset,
        task["entry_point"],
        solution,
        inputs,
        expected,
        limits,
        task["atol"],
        False,
        stat,
        details,
        progress,
    )


def check_suite(
    dataset: str,
    task: dict[str, Any],
    solution: str,
    inputs: list[Any],
    expected: list[Any],
    times: list[float],
    verify_upstream: bool = False,
) -> dict[str, Any]:
    """Run every input with official default deadlines and comparison semantics."""
    limits = [max(1.0, 4.0 * elapsed) for elapsed in times]
    context = mp.get_context("fork")
    stat = context.Value("i", 3)
    details = context.Array("b", [False] * len(inputs))
    progress = context.Value("i", 0)
    fault = context.Value("i", 0)
    process = context.Process(
        target=execute_observed,
        args=(
            dataset,
            task,
            solution,
            inputs,
            expected,
            limits,
            stat,
            details,
            progress,
            fault,
        ),
    )
    process.start()
    # EvalPlus full-detail mode: min(60, sum(limits)) + 2, then join(timeout + 1).
    process.join(min(60, sum(limits)) + 3)
    deadline = process.is_alive()
    if deadline:
        process.terminate()
        process.join(0.2)
    if process.is_alive():
        process.kill()
        process.join()
    passed = [bool(value) for value in details[: progress.value]]
    if deadline or fault.value == 2:
        status = "timeout"
    elif process.exitcode != 0 or fault.value == 1 or stat.value != 0:
        status = "crash"
    elif len(passed) != len(inputs):
        status = "oracle_error"
    elif not all(passed):
        status = "wrong_answer"
    else:
        status = "pass"
    result: dict[str, Any] = {"status": status, "details": passed, "total": len(inputs)}
    if verify_upstream:
        official_status, official_details = evaluation.untrusted_check(
            dataset,
            solution,
            inputs,
            task["entry_point"],
            expected,
            task["atol"],
            times,
            fast_check=False,
        )
        official_passed = [bool(value) for value in official_details]
        if (official_status == "pass") != (
            status == "pass"
        ) or official_passed != passed:
            raise ValueError("Oracle differs from the official full-detail checker")
        result["upstream_parity"] = {
            "status": "matched",
            "official_status": official_status,
            "official_details": official_passed,
        }
    return result


def evaluate(request: dict[str, Any]) -> dict[str, Any]:
    """Evaluate full base and plus inputs; canonical execution also stays isolated."""
    if platform.python_version() != "3.12.12" or version("evalplus") != "0.3.1":
        raise RuntimeError("Unsupported Python/EvalPlus execution version")
    task = request["task"]
    solution = request["solution"]
    try:
        compile(solution, "submission.py", "exec")
    except (SyntaxError, ValueError):
        return {"status": "syntax_error", "suites": {}}

    suites = {}
    for suite in ("base", "plus"):
        inputs = task[f"{suite}_input"]
        # Same code construction and output-not-None rules as get_groundtruth.
        expected, times = trusted_exec(
            task["prompt"] + task["canonical_solution"],
            inputs,
            task["entry_point"],
            record_time=True,
            output_not_none=(
                request["dataset"] == "mbpp"
                and task["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
            ),
        )
        if not isinstance(expected, list) or not isinstance(times, list):
            raise TypeError("Invalid EvalPlus ground-truth result")
        suites[suite] = check_suite(
            request["dataset"],
            task,
            solution,
            inputs,
            expected,
            times,
            verify_upstream=request.get("verify_upstream", False),
        )
    statuses = {suite["status"] for suite in suites.values()}
    status = next(
        name
        for name in ("oracle_error", "timeout", "crash", "wrong_answer", "pass")
        if name in statuses
    )
    return {"status": status, "suites": suites}


def main() -> None:
    """Read trusted host-created input; emit bounded metadata, never hidden values."""
    if not Path("/.dockerenv").exists():
        raise RuntimeError("The oracle worker must only run inside Docker")
    try:
        with Path("/oracle-input/request.pkl").open("rb") as stream:
            request = pickle.load(stream)
        result = evaluate(request)
    except BaseException:  # noqa: BLE001 -- canonical SystemExit must be reported too
        # This is the worker reporting boundary, including broken canonical code.
        # Do not print exception messages: these can contain hidden inputs.
        result = {"status": "oracle_error", "suites": {}}
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
