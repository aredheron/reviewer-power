"""CLI for the source-bound, user-authorized main pacing amendment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.inspect_paced_extension import main

if __name__ == "__main__":
    main()
