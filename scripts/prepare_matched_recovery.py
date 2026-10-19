"""Freeze free repairs and designated recovery inputs without issuing paid requests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_main import freeze
from grader_power import matched_recovery as recovery


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    b = recovery.base(root)
    freeze(b / "free-repairs-v1.json", recovery.free_repairs(root))
    freeze(b / "designated-records.json", recovery.recovery_records(root))


if __name__ == "__main__":
    main()
