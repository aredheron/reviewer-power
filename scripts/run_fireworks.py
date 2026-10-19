"""Run the source-bound Fireworks pilot and explicit main-provider continuation."""

import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    runpy.run_path(str(root / "evals/inspect_fireworks.py"), run_name="__main__")
