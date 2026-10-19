"""Verify packaged evidence and reproduce the presentation's result tables offline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power.publication import write_tables


def main() -> None:
    """Write tables, optionally checking the original paired statistical analyses."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=root / "runs/reproduced-tables")
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Recompute the original 10,000-draw paired analyses",
    )
    args = parser.parse_args()
    summary = write_tables(root, args.output, inference=args.inference)
    print(
        f"Verified {summary['artifacts_verified']} artifacts; wrote five tables to {args.output}"
    )


if __name__ == "__main__":
    main()
