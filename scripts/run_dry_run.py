"""Run the offline Inspect demo in a worker with no inherited credentials."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


def main() -> None:
    """Forward demo options with an explicit environment; never open .env.

    Inspect receives a temporary HOME and no provider/Runpod/cloud variables.
    The worker additionally disables Inspect's automatic dotenv/Git-origin reads.
    """
    root = Path(__file__).resolve().parents[1]
    docker = shutil.which("docker")
    worker_path = os.defpath
    if docker is not None:
        worker_path += os.pathsep + str(Path(docker).parent)
    with TemporaryDirectory(prefix="grader-power-offline-") as directory:
        result = subprocess.run(
            [sys.executable, str(root / "evals" / "inspect_oneshot.py"), *sys.argv[1:]],
            cwd=directory,
            env={
                "PATH": worker_path,
                "HOME": directory,
                "TMPDIR": directory,
                "LANG": "en_US.UTF-8",
                "PYTHON_DOTENV_DISABLED": "1",
            },
            check=False,
        )
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
