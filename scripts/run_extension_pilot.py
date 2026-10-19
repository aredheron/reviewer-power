"""Run the shared extension-pilot preparation, collection and summaries."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evals.inspect_extension_pilot import main

if __name__ == "__main__":
    main()
