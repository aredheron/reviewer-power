"""Run the authorized Mode A framing/name pilot and main study."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_framing import main

if __name__ == "__main__":
    main()
