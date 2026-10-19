"""Run the separately authorized audited outcome recovery; no automatic retries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.inspect_outcome_recovery import main

if __name__ == "__main__":
    main()
