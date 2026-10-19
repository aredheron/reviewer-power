"""Independent final-file checks through native Inspect; model/oracle calls are mocked."""

import ast
import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import test_inspect_agent as harness

from grader_power.agent_sandbox import EpisodeLimits, SandboxError, WorkspaceFileError
from grader_power.datasets.oracle import OracleResult


@pytest.mark.parametrize(
    "saved",
    [
        "def transform(items):\r\n    return [2*x for x in items]  \r\n",
        "```python\ndef transform(items):\n    return [2*x for x in items]\n```\n",
        "def transform(:\n",
    ],
    ids=["exact-crlf", "fenced-file-is-invalid", "syntax-failure"],
)
def test_saved_file_reaches_full_oracle_without_chat_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, saved: str
) -> None:
    """A malformed saved file cannot become a repaired, successful submission."""
    received: list[tuple[str, str]] = []

    def oracle(
        dataset: Any,
        dataset_name: str,
        task_id: str,
        solution: str,
        *,
        mode: str,
        image: str,
    ) -> OracleResult:
        received.append((solution, mode))
        try:
            ast.parse(solution)
        except SyntaxError:
            status = "syntax_error"
            suites = {}
        else:
            status = "pass"
            suites = {
                suite: {"status": "pass", "details": [True], "total": 1}
                for suite in ("base", "plus")
            }
        return OracleResult(
            status=status,
            task_id=task_id,
            image=image,
            solution_sha256=hashlib.sha256(solution.encode()).hexdigest(),
            message="Synthetic oracle: syntax only, no execution",
            suites=suites,
        )

    monkeypatch.setattr(harness.agent, "DockerWorkspace", harness.FakeWorkspace)
    monkeypatch.setattr("inspect_ai._eval.context.init_dotenv", lambda: None)
    monkeypatch.setattr("inspect_ai._eval.task.log.git_context", lambda: None)
    monkeypatch.setattr("grader_power.scoring.evaluate_solution", oracle)
    item = harness.assigned()
    harness.run(
        tmp_path,
        item,
        [
            harness.response("write_file", path="src/task.py", content=saved),
            harness.response(),
        ],
    )
    attempt = tmp_path / "attempts" / item.record["assignment_id"]
    assert (attempt / "final.py").read_bytes() == saved.encode()
    assert received == [(saved, "full")]
    result = json.loads((attempt / "scoring.json").read_text())
    assert (
        result["correctness"]["solution_sha256"]
        == hashlib.sha256(saved.encode()).hexdigest()
    )
    if saved.startswith("```") or saved == "def transform(:\n":
        assert result["correctness"]["status"] == "syntax_error"
        assert result["correctness"]["full_correct"] is False


def test_exposure_separates_listing_read_attempt_and_returned_content(
    tmp_path: Path,
) -> None:
    """Tool arguments and paths do not prove model-visible content; truncation matters."""
    files = {
        "AGENTS.md": "Unique preference evidence\n",
        ".ci/selection-policy.yaml": "Unique policy " + "x" * 300,
    }
    limits = EpisodeLimits(max_tool_output_bytes=128)
    workspace = harness.FakeWorkspace(harness.IMAGE, files, limits)
    workspace.start()
    episode = harness.agent.Episode(tmp_path, workspace, limits)

    async def exercise() -> None:
        await episode.operate("list_files", {})
        await episode.operate("run_command", {"command": "cat AGENTS.md >/dev/null"})
        await episode.operate("read_file", {"path": "AGENTS.md"})
        await episode.operate("read_file", {"path": ".ci/selection-policy.yaml"})
        await episode.operate("read_file", {"path": "missing.py"})

    asyncio.run(exercise())
    listing, command, read, truncated, missing = episode.events
    assert listing["exposure"] == command["exposure"] == {}
    assert listing["explicit_read_path"] is None
    assert command["explicit_read_path"] is None
    assert read["exposure"] == {"AGENTS.md": "full_content_observed"}
    assert read["explicit_read_path"] == "AGENTS.md"
    assert truncated["explicit_read_path"] == ".ci/selection-policy.yaml"
    assert truncated["truncated"] is True
    assert len(truncated["output"].encode()) <= limits.max_tool_output_bytes
    assert truncated["exposure"] == {}
    assert missing["explicit_read_path"] == "missing.py"
    assert "error" in json.loads(missing["output"])
    assert missing["exposure"] == {}
    for event in episode.events:
        assert event["completed_at"] >= event["started_at"]
        assert (tmp_path / "tools" / f"{event['index']:04d}.result.json").exists()


@pytest.mark.parametrize("failure", ["file_missing", "file_invalid", "sandbox_error"])
def test_final_capture_failures_are_distinct_and_never_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """Missing source, invalid file bytes and Docker errors preserve distinct evidence."""

    class CaptureFailure(harness.FakeWorkspace):
        def file_operation(
            self, action: str, path: str = ".", content: str = ""
        ) -> dict[str, Any]:
            if action == "read":
                if failure == "sandbox_error":
                    raise SandboxError("Synthetic control failure")
                raise WorkspaceFileError(failure)
            return super().file_operation(action, path, content)

    monkeypatch.setattr(harness.agent, "DockerWorkspace", CaptureFailure)
    monkeypatch.setattr("inspect_ai._eval.context.init_dotenv", lambda: None)
    monkeypatch.setattr("inspect_ai._eval.task.log.git_context", lambda: None)
    item = harness.assigned()
    harness.run(tmp_path, item, [harness.response()])
    attempt = tmp_path / "attempts" / item.record["assignment_id"]
    episode = json.loads((attempt / "episode.json").read_text())
    assert episode["final_file_status"] == failure
    assert episode["status"] != "completed"
    assert episode["cleanup_verified"] is True
    assert not (attempt / "final.py").exists()
    assert not (attempt / "scoring.json").exists()
    assert (attempt / "failure.json").exists()
