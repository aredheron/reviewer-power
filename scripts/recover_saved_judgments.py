"""Run the separately approved saved-code judging recovery and offline analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_judge_recovery import main

if __name__ == "__main__":
    main()
