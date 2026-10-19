"""Run the approved two-design, four-model replication with one $400 ledger."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_compact_replication import main

if __name__ == "__main__":
    main()
