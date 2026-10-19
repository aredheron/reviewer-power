"""Collect the additive same-limit matched-review recovery after syntax collection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_matched_gaps import main

if __name__ == "__main__":
    main()
