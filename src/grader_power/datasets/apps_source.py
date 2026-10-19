"""Pinned APPS data and strict callable-list checks; task code runs only in Docker."""

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from grader_power.agent_sandbox import DockerWorkspace, EpisodeLimits
from grader_power.datasets.mbpp_full import digest

REVISION = "21e74ddf8de1a21436da12e3e653065c5213e9d1"
CACHE = Path("data/apps") / REVISION
MANIFEST = Path("manifests/apps_source.json")
SPLITS = {
    "train": (
        107101272,
        "45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae",
    ),
    "test": (
        1292436853,
        "5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c",
    ),
}


def file_digest(path: Path) -> str:
    """Hash a large source without loading the entire dataset into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_url(name: str) -> str:
    """Return a revision-pinned URL; no upstream Python loader is executed."""
    return f"https://huggingface.co/datasets/codeparrot/apps/resolve/{REVISION}/{name}"


def records(path: Path) -> Iterator[dict[str, Any]]:
    """Stream source records, rejecting missing fields, duplicate IDs and bad types."""
    seen: set[int] = set()
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not isinstance(row, dict) or type(row.get("id")) is not int:
                raise ValueError("APPS record requires an integer id")
            if row["id"] < 0 or row["id"] in seen:
                raise ValueError("Invalid or duplicate APPS id")
            seen.add(row["id"])
            for key in (
                "question",
                "solutions",
                "input_output",
                "difficulty",
                "url",
                "starter_code",
            ):
                if not isinstance(row.get(key), str):
                    raise TypeError(f"APPS/{row['id']}: {key} must be text")
            yield row


def inspect_record(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Classify interface and test shape, retaining unsupported nested JSON explicitly."""
    result: dict[str, Any] = {
        "task_id": f"APPS/{split}/{row['id']}",
        "split": split,
        "source_id": row["id"],
        "difficulty": row["difficulty"],
        "source_record_sha256": digest(json.dumps(row, sort_keys=True).encode()),
    }
    try:
        io = json.loads(row["input_output"])
    except ValueError as error:
        result.update(
            interface="unparsed",
            screening="nested_json_unsupported",
            error_type=type(error).__name__,
        )
        return result
    if not isinstance(io, dict):
        result.update(interface="unparsed", screening="invalid_io_structure")
        return result
    outputs = io.get("outputs", [])
    inputs = io.get("inputs", [])
    callable_task = isinstance(io.get("fn_name"), str) and bool(io["fn_name"])
    wrapped = (
        callable_task
        and isinstance(outputs, list)
        and bool(outputs)
        and isinstance(inputs, list)
        and len(inputs) == len(outputs)
        and all(
            isinstance(x, list) and len(x) == 1 and isinstance(x[0], list)
            for x in outputs
        )
    )
    result.update(
        interface="callable" if callable_task else "stdin_stdout",
        entry_point=io.get("fn_name"),
        supplied_cases=len(inputs) if isinstance(inputs, list) else None,
        screening="wrapped_list_candidate"
        if wrapped
        else "outside_initial_function_list_screen",
    )
    return result


def load_selected(root: Path, split: str, ids: set[int]) -> dict[int, dict[str, Any]]:
    """Verify both raw split bindings, then load only requested records from one split."""
    manifest = json.loads((root / MANIFEST).read_text())
    if manifest["revision"] != REVISION or split not in SPLITS:
        raise ValueError("APPS revision or split mismatch")
    for name, (size, expected) in SPLITS.items():
        path = root / CACHE / f"{name}.jsonl"
        if path.stat().st_size != size or file_digest(path) != expected:
            raise ValueError(f"Changed APPS {name} bytes")
        if manifest["splits"][name]["sha256"] != expected:
            raise ValueError("APPS manifest binding mismatch")
    selected = {
        row["id"]: row
        for row in records(root / CACHE / f"{split}.jsonl")
        if row["id"] in ids
    }
    if set(selected) != ids:
        raise ValueError("Requested APPS IDs are missing")
    return selected


def callable_cases(source: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Decode the shortlisted singleton-output wrapper exactly once, never heuristically.

    This subset requires positional-argument lists and list-valued returns. It
    intentionally does not emulate APPS's permissive alternate equality checks,
    tuple coercions, string parsing, stdin tasks or Solution-class adapters.
    """
    io = json.loads(source["input_output"])
    if (
        not isinstance(io, dict)
        or not isinstance(io.get("fn_name"), str)
        or not io["fn_name"].isidentifier()
    ):
        raise ValueError("Expected a callable entry point")
    inputs, outputs = io.get("inputs"), io.get("outputs")
    if (
        not isinstance(inputs, list)
        or not inputs
        or not isinstance(outputs, list)
        or len(inputs) != len(outputs)
    ):
        raise ValueError("Missing or unmatched APPS cases")
    cases = []
    for args, wrapped in zip(inputs, outputs, strict=True):
        if (
            not isinstance(args, list)
            or not isinstance(wrapped, list)
            or len(wrapped) != 1
            or not isinstance(wrapped[0], list)
        ):
            raise ValueError("Unsupported APPS argument/output encoding")
        cases.append({"args": args, "expected": wrapped[0]})
    return io["fn_name"], cases


CALLABLE_WORKER = """import contextlib
import io
import json
import pathlib

root = pathlib.Path("/workspace")
request = json.loads((root / "request.json").read_text())
results = []
for index in range(request["case_count"]):
    case = json.loads((root / f"case-{index:04d}.json").read_text())
    namespace = {"__name__": "__apps_candidate__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(request["code"], "candidate.py", "exec"), namespace)
            actual = namespace[request["entry_point"]](*case["args"])
            assert isinstance(actual, list), "List return required"
            assert actual == case["expected"], "Wrong return value"
        results.append({"passed": True, "error_type": None})
    except BaseException as error:
        results.append({"passed": False, "error_type": type(error).__name__})
print(json.dumps({"results": results}))
"""


def validate_details(parsed: Any, count: int) -> list[dict[str, Any]]:
    """Reject truncated or success-shaped worker evidence."""
    if not isinstance(parsed, dict) or set(parsed) != {"results"}:
        raise ValueError("Invalid APPS worker report")
    details = parsed["results"]
    if (
        not isinstance(details, list)
        or len(details) != count
        or any(
            not isinstance(row, dict)
            or set(row) != {"passed", "error_type"}
            or type(row["passed"]) is not bool
            or (row["passed"] and row["error_type"] is not None)
            or (not row["passed"] and not isinstance(row["error_type"], str))
            for row in details
        )
    ):
        raise ValueError("Incomplete or inconsistent APPS worker evidence")
    return details


def evaluate(source: dict[str, Any], code: str, image: str) -> dict[str, Any]:
    """Check every supplied case in the existing verified, resource-limited Docker boundary.

    Separate case files respect the existing per-file limit without relaxing it.
    Each case has fresh module globals and freshly decoded inputs. Infrastructure,
    timeout and cleanup failures propagate rather than masquerading as test passes.
    """
    entry_point, cases = callable_cases(source)
    files = {
        "request.json": json.dumps(
            {"code": code, "entry_point": entry_point, "case_count": len(cases)}
        ),
        "check.py": CALLABLE_WORKER,
        **{f"case-{i:04d}.json": json.dumps(case) for i, case in enumerate(cases)},
    }
    limits = EpisodeLimits()
    if any(len(value.encode()) > limits.max_file_bytes for value in files.values()):
        raise ValueError("APPS source or case exceeds the unchanged sandbox file limit")
    workspace = DockerWorkspace(image, files, limits)
    try:
        workspace.start()
        result = workspace.command("python -I /workspace/check.py")
        if result["exit_code"] != 0:
            raise RuntimeError("APPS worker exited abnormally")
        details = validate_details(json.loads(result["output"]), len(cases))
    finally:
        workspace.close()
    if not workspace.cleanup_verified:
        raise RuntimeError("APPS sandbox cleanup unverified")
    return {
        "source_id": source["id"],
        "source_record_sha256": digest(json.dumps(source, sort_keys=True).encode()),
        "code_sha256": digest(code.encode()),
        "image": image,
        "supplied_cases": len(cases),
        "details": details,
        "passed": all(row["passed"] for row in details),
        "cleanup_verified": True,
        "decoder": "singleton_wrapped_list",
    }
