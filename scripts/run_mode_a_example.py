"""Run the bounded real-task Mode A example; no inference without --generate."""

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "evals/inspect_mode_a.py"),
        run_name="__main__",
    )
