"""Run the authorized reviewer/reward disagreement study and its prerequisite pilot."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_three_style import main

if __name__ == "__main__":
    main()
