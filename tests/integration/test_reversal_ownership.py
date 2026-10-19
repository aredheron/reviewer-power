"""Verify the new supervisor/child boundary using real isolated processes."""

import json
import subprocess
import sys
from pathlib import Path


def test_reversal_children_require_live_shared_lock(tmp_path: Path) -> None:
    child = """
import sys
from pathlib import Path
from evals.inspect_reversal import child_permission
child_permission(Path(sys.argv[1]), int(sys.argv[2]))
print('authorized')
"""
    parent = """
import sys,os,subprocess
from pathlib import Path
from grader_power.reversal_lock import isolate_collection_process,collection_lock
isolate_collection_process()
with collection_lock(Path(sys.argv[1]),owner='reversal-v1',run_id='reversal-v1'):
    children=[subprocess.Popen([sys.executable,'-c',sys.argv[2],sys.argv[1],str(os.getpid())],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for _ in range(4)]
    for child in children:
        out,err=child.communicate(timeout=45)
        assert child.returncode == 0,err
        assert out.strip() == 'authorized'
"""
    subprocess.run(
        [sys.executable, "-c", parent, str(tmp_path), child],
        check=True,
        timeout=60,
        capture_output=True,
    )
    owner = json.loads(
        (tmp_path / "runs/extension-pilot/coordination/owner.json").read_text()
    )
    assert owner["state"] == "released"
    denied = subprocess.run(
        [sys.executable, "-c", child, str(tmp_path), "123"],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert denied.returncode != 0
