"""Build the personal offline experiment explorer from existing local results."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power.results_dashboard import write_dashboard


def main() -> None:
    """Write a standalone page; perform no inference or experiment execution."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=root / "runs/dashboard/index.html"
    )
    args = parser.parse_args()
    data = write_dashboard(root, args.output)
    print(f"Wrote {args.output.resolve()}")
    print(
        f"{data['assigned']:,} observations; {len(data['conditions'])} conditions; {len(data['groups'])} comparison panels"
    )
    for path in data["missing"]:
        print(f"Unavailable source: {path}")


if __name__ == "__main__":
    main()
