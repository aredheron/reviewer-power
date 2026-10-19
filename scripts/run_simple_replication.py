"""Run the approved simpler three-model replication under its $100 ceiling."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_simple_replication import main

if __name__ == "__main__":
    main()
