"""Collect the authorized additional strong-preference repetitions."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_strong_preferences import main

if __name__ == "__main__":
    main()
