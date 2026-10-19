"""Opt-in live isolation tests; synthetic commands execute only after Docker inspection."""

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from grader_power.agent_sandbox import (
    DockerWorkspace,
    EpisodeLimits,
    SandboxError,
    SandboxLimit,
    verify_container,
)


@pytest.fixture
def workspace() -> Iterator[DockerWorkspace]:
    """Require explicit verification opt-in and the existing local immutable image."""
    if os.environ.get("GRADER_POWER_MODE_B_DOCKER_TESTS") != "1":
        pytest.skip("Mode B Docker verification requires explicit opt-in")
    root = Path(__file__).resolve().parents[2]
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    workspace = DockerWorkspace(
        image,
        {
            "README.md": "# Synthetic verification\n",
            "AGENTS.md": "Read-only guidance\n",
            ".ci/style-review.yaml": "style: loops\n",
            ".ci/selection-policy.yaml": "policy: shadow\n",
            "src/task.py": "print('synthetic initial')\n",
        },
        EpisodeLimits(command_seconds=1),
    )
    try:
        workspace.start()
        yield workspace
    finally:
        workspace.close()
    assert workspace.cleanup_verified


def test_workspace_permissions_and_exact_file(workspace: DockerWorkspace) -> None:
    """Keep mutable code separate from immutable instructions and host/oracle data."""
    text = "print('synthetic replacement')\r\n"
    workspace.file_operation("write", "src/task.py", text)
    assert workspace.file_operation("read", "src/task.py")["text"] == text
    assert (
        workspace.command("python src/task.py")["output"] == "synthetic replacement\n"
    )
    assert workspace.command("printf changed > AGENTS.md")["exit_code"] != 0
    assert workspace.command("rm .ci/style-review.yaml")["exit_code"] != 0
    assert (
        workspace.file_operation("read", "AGENTS.md")["text"] == "Read-only guidance\n"
    )
    result = workspace.command(
        "id -u; test ! -e /oracle-input; test ! -e /var/run/docker.sock; test ! -e /workspace/.env"
    )
    assert result == {"exit_code": 0, "output": "65534\n"}


def test_command_timeout_removes_container_and_descendants(
    workspace: DockerWorkspace,
) -> None:
    """A killed client alone is insufficient; destroy the whole command workspace."""
    with pytest.raises(SandboxLimit, match="command_timeout"):
        workspace.command("(sleep 3; echo late > src/late) & sleep 5")
    assert workspace.cleanup_verified
    assert not workspace.ready
    code, _ = workspace._call(["inspect", workspace.name])
    assert code != 0


def test_output_bomb_is_bounded_and_removes_container(
    workspace: DockerWorkspace,
) -> None:
    """Limit host buffering while untrusted output is still streaming."""
    with pytest.raises(SandboxLimit) as error:
        workspace.command("python -c 'print(\"x\" * 100000)' ")
    assert error.value.kind == "tool_output_limit"
    assert len(error.value.output.encode()) <= 8192
    assert workspace.cleanup_verified


def test_direct_control_stages_only_its_frozen_files(
    workspace: DockerWorkspace,
) -> None:
    """Real Docker must not inject guidance into an explicit source-only control."""
    files = {"README.md": "Synthetic control\n", "src/task.py": "# Implement task\n"}
    control = DockerWorkspace(workspace.image, files, EpisodeLimits())
    try:
        control.start()
        assert control.file_operation("list")["paths"] == [
            "README.md",
            "src",
            "src/task.py",
        ]
        control.file_operation("write", "src/task.py", "print('synthetic control')\n")
        assert control.command("python src/task.py") == {
            "exit_code": 0,
            "output": "synthetic control\n",
        }
    finally:
        control.close()
    assert control.cleanup_verified


def test_expanded_episode_has_verified_bounded_container_lifetime(
    workspace: DockerWorkspace,
) -> None:
    """A 360-second episode gets 420 seconds of container life with isolation intact."""
    limits = EpisodeLimits(
        max_model_turns=16,
        max_tool_calls=48,
        episode_seconds=360,
        max_model_input_bytes=98304,
    )
    expanded = DockerWorkspace(workspace.image, workspace.files, limits)
    try:
        expanded.start()
        info = json.loads(expanded._checked(["inspect", expanded.name]))[0]
        assert info["Config"]["Cmd"] == ["-I", "-c", "import time; time.sleep(420)"]
        verify_container(info, workspace.image, limits)
        with pytest.raises(SandboxError):
            verify_container(info, workspace.image)
        with pytest.raises(ValueError):
            expanded.file_operation("write", "task.py", "# wrong path\n")
        expanded.file_operation("write", "src/task.py", "# saved source\n")
        assert (
            expanded.file_operation("read", "src/task.py")["text"] == "# saved source\n"
        )
    finally:
        expanded.close()
    assert expanded.cleanup_verified
