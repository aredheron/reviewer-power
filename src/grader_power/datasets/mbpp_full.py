"""Pinned original MBPP import and assertion checks, separate from MBPP+.

Source, setup and test strings are inert on the host. Execution is available
only through the existing verified Docker workspace; no model API is used.
"""

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from grader_power.agent_sandbox import DockerWorkspace, EpisodeLimits

REVISION = "f82046ba5aabbbb427dbfd38a254d26bff08b533"
SOURCE_URL = f"https://raw.githubusercontent.com/google-research/google-research/{REVISION}/mbpp/mbpp.jsonl"
SOURCE_SHA256 = "ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f"
CACHE = Path("data/mbpp-full") / REVISION / "mbpp.jsonl"
MANIFEST = Path("manifests/mbpp_full_source.json")


def digest(data: bytes) -> str:
    """Hash original bytes without newline or source normalization."""
    return hashlib.sha256(data).hexdigest()


def parse_records(raw: bytes) -> dict[int, dict[str, Any]]:
    """Validate record structure without importing or evaluating source strings."""
    rows: dict[int, dict[str, Any]] = {}
    for line in raw.decode("utf-8").splitlines():
        item = json.loads(line)
        if not isinstance(item, dict) or type(item.get("task_id")) is not int:
            raise ValueError("MBPP record requires an integer task_id")
        task_id = item["task_id"]
        if task_id < 1 or task_id in rows:
            raise ValueError(f"Invalid or duplicate MBPP task_id: {task_id}")
        for key in ("text", "code", "test_setup_code"):
            if not isinstance(item.get(key), str):
                raise TypeError(f"MBPP/{task_id}: {key} must be text")
        if not item["text"].strip() or not item["code"].strip():
            raise ValueError(f"MBPP/{task_id}: empty source")
        for key in ("test_list", "challenge_test_list"):
            tests = item.get(key)
            if not isinstance(tests, list) or any(
                not isinstance(test, str) or not test.strip() for test in tests
            ):
                raise TypeError(f"MBPP/{task_id}: invalid {key}")
        if len(item["test_list"]) != 3:
            raise ValueError(f"MBPP/{task_id}: expected three original tests")
        rows[task_id] = item
    return rows


def load(root: Path) -> dict[int, dict[str, Any]]:
    """Load the exact pinned 974-task import, failing closed on changed bytes."""
    manifest = json.loads((root / MANIFEST).read_text())
    raw = (root / CACHE).read_bytes()
    if (
        manifest["revision"] != REVISION
        or manifest["sha256"] != SOURCE_SHA256
        or digest(raw) != SOURCE_SHA256
    ):
        raise ValueError("Original MBPP source binding changed")
    rows = parse_records(raw)
    if set(rows) != set(range(1, 975)):
        raise ValueError("Expected all 974 original MBPP IDs")
    return rows


def write_once(path: Path, raw: bytes) -> None:
    """Allow identical reuse but never overwrite a different source or artifact."""
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace existing artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def signature(code: str, entry_point: str) -> str:
    """Compare function interfaces statically, without executing default values."""
    targets = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == entry_point
    ]
    if len(targets) != 1:
        raise ValueError("Expected one unambiguous top-level target function")
    target = targets[0]
    return ast.dump(target.args) + (
        ast.dump(target.returns) if target.returns is not None else ""
    )


# This program is staged as data and runs only inside the verified container.
# Every assertion gets fresh source/setup globals. No tests are rewritten.
ASSERTION_WORKER = """import contextlib
import io
import json
import pathlib

request = json.loads(pathlib.Path("/workspace/request.json").read_text())
results = []
for test in request["tests"]:
    namespace = {"__name__": "__mbpp_candidate__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(request["code"], "candidate.py", "exec"), namespace)
            exec(compile(request["setup"], "setup.py", "exec"), namespace)
            exec(compile(test, "assertion.py", "exec"), namespace)
        results.append({"passed": True, "error_type": None})
    except BaseException as error:
        results.append({"passed": False, "error_type": type(error).__name__})
print(json.dumps({"results": results}))
"""


def evaluate_assertions(
    source: dict[str, Any], code: str, image: str
) -> dict[str, Any]:
    """Run all original and challenge assertions in an isolated Docker workspace.

    The whole command has a ten-second deadline. Sandbox, timeout, cleanup and
    malformed-report errors propagate; they are never converted to test passes.
    This is original MBPP test evidence, not EvalPlus base/plus coverage.
    """
    tests = source["test_list"] + source["challenge_test_list"]
    if not tests:
        raise ValueError("Empty assertion suite")
    # A source test must actually assert something, rather than merely run code.
    for test in tests:
        if not any(isinstance(node, ast.Assert) for node in ast.walk(ast.parse(test))):
            raise ValueError("Source test has no assertion")
    request = {"code": code, "setup": source["test_setup_code"], "tests": tests}
    workspace = DockerWorkspace(
        image,
        {
            "request.json": json.dumps(request),
            "check.py": ASSERTION_WORKER,
        },
        EpisodeLimits(),
    )
    try:
        workspace.start()
        result = workspace.command("python -I /workspace/check.py")
        if result["exit_code"] != 0:
            raise RuntimeError("MBPP assertion worker exited abnormally")
        parsed = json.loads(result["output"])
        if not isinstance(parsed, dict):
            raise TypeError("Invalid MBPP assertion evidence")
        details = parsed.get("results")
        if (
            not isinstance(details, list)
            or len(details) != len(tests)
            or any(
                not isinstance(row, dict)
                or set(row) != {"passed", "error_type"}
                or type(row.get("passed")) is not bool
                or (row["passed"] and row.get("error_type") is not None)
                or (not row["passed"] and not isinstance(row.get("error_type"), str))
                for row in details
            )
        ):
            raise ValueError("Incomplete or invalid MBPP assertion evidence")
    finally:
        workspace.close()
    if not workspace.cleanup_verified:
        raise RuntimeError("MBPP sandbox cleanup unverified")
    return {
        "source_id": source["task_id"],
        "code_sha256": digest(code.encode()),
        "source_record_sha256": digest(json.dumps(source, sort_keys=True).encode()),
        "image": image,
        "supplied_tests": len(source["test_list"]),
        "challenge_tests": len(source["challenge_test_list"]),
        "details": details,
        "passed": all(row["passed"] for row in details),
        "coverage": "all supplied original MBPP assertions; no augmented tests",
        "cleanup_verified": True,
    }
