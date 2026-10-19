"""Prepare, collect, judge or analyze the frozen 32-task Mode A main cohort."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "evals/inspect_main.py"), run_name="__main__")
