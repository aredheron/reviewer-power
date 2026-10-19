"""Run the approved reviewer-observation study under its $375 ceiling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_reviewer_observation import main

if __name__ == "__main__":
    main()
