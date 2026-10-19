"""Real process ownership and opt-in bounded Docker checks for paced collection."""

import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest

from grader_power.agent_sandbox import DockerWorkspace, EpisodeLimits, verify_container
from grader_power.pacing import WAIT_ALLOWANCE


@pytest.mark.parametrize("worker_count", [2, 12])
def test_worker_requires_supervisor_holding_collection_lock(
    tmp_path: Path, worker_count: int
) -> None:
    """Real children share only their supervisor's lock; outsiders are denied."""
    child = """
import json,sys
from pathlib import Path
from evals.inspect_paced_extension import child_permission
with child_permission(Path(sys.argv[1]),int(sys.argv[2])):
    print(json.dumps({'entered': True}))
"""
    parent = """
import json,os,subprocess,sys
from pathlib import Path
from grader_power.collection_lock import collection_lock,isolate_collection_process
isolate_collection_process()
with collection_lock(Path(sys.argv[1]),owner='gpt-extension-main-v1',run_id='gpt-extension-main-v1'):
    children=[subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1],str(os.getpid())],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(int(sys.argv[3]))]
    output=[]
    for child in children:
        stdout,stderr=child.communicate(timeout=30)
        if child.returncode:
            raise RuntimeError(stderr)
        output.append(json.loads(stdout))
    print(json.dumps(output))
"""
    result = subprocess.run(
        [sys.executable, "-c", parent, str(tmp_path), child, str(worker_count)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert json.loads(result.stdout) == [{"entered": True}] * worker_count
    outsider = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), "123"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert outsider.returncode != 0
    assert (
        json.loads(
            (tmp_path / "runs/extension-pilot/coordination/owner.json").read_text()
        )["state"]
        == "released"
    )


@pytest.mark.parametrize("worker_count", [2, 12])
def test_paced_sandboxes_remain_isolated_and_bounded(worker_count: int) -> None:
    if os.environ.get("GRADER_POWER_MODE_B_DOCKER_TESTS") != "1":
        pytest.skip("Requires explicit verified Docker opt-in")
    root = Path(__file__).resolve().parents[2]
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    lifetime = EpisodeLimits.container_lifetime_seconds.fget
    assert lifetime is not None
    limits = EpisodeLimits(episode_seconds=360, command_seconds=1)
    files = {"README.md": "Synthetic pacing verification", "src/task.py": "# empty\n"}
    with patch.object(
        EpisodeLimits,
        "container_lifetime_seconds",
        property(lambda value: lifetime(value) + int(WAIT_ALLOWANCE)),
    ):
        workspaces = [
            DockerWorkspace(image, files, limits) for _ in range(worker_count)
        ]
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                list(pool.map(lambda w: w.start(), workspaces))
            for w in workspaces:
                info = json.loads(w._checked(["inspect", w.name]))[0]
                verify_container(info, image, limits)
                assert info["Config"]["Cmd"] == [
                    "-I",
                    "-c",
                    "import time; time.sleep(1020)",
                ]
            for index, workspace in enumerate(workspaces):
                workspace.file_operation(
                    "write", "src/task.py", f"# isolated {index}\n"
                )
            for index, workspace in enumerate(workspaces):
                assert (
                    workspace.file_operation("read", "src/task.py")["text"]
                    == f"# isolated {index}\n"
                )
                assert workspace.command("test ! -e /workspace/.env")["exit_code"] == 0
        finally:
            for w in workspaces:
                w.close()
        assert all(w.cleanup_verified for w in workspaces)
