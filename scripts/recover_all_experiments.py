"""Run the authorized additive main-study recovery without changing original runs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_all_recovery import main

if __name__ == "__main__":
    main()
