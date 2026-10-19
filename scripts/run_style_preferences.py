"""Run the approved naming and quote-preference studies under their shared $400 cap."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_style_preferences import main

if __name__ == "__main__":
    main()
