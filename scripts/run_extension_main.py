"""Prepare, verify, collect or analyze the frozen GPT-only main extension."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.inspect_extension_main import main

if __name__ == "__main__":
    main()
