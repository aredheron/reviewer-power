"""Audit the historical reversal, then run the gated, fixed prompt ablation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_reversal import main

if __name__ == "__main__":
    main()
