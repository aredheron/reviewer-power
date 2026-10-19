"""Prepare, collect, judge, or summarize the GPT-OSS temperature-0.7 pilot check."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "evals/inspect_sampling_check.py"), run_name="__main__")
