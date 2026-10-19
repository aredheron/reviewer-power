"""Freeze 11 approved tasks or verify both references in the established Docker sandbox."""

import argparse
from pathlib import Path

from grader_power.pilot_tasks import freeze_pilot_tasks, verify_pilot_tasks


def main() -> None:
    """Require explicit selection or execution; never make model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--oracle-image")
    parser.add_argument("--integration-report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.freeze:
        if args.oracle_image or args.integration_report:
            parser.error("--oracle-image and --integration-report require --verify")
        print("Frozen:", ", ".join(freeze_pilot_tasks(root)))
    else:
        if not args.oracle_image or not args.integration_report:
            parser.error("--verify requires --oracle-image and --integration-report")
        evidence = verify_pilot_tasks(root, args.oracle_image, args.integration_report)
        print(f"Verified {len(evidence['checks'])} full reference implementations.")


if __name__ == "__main__":
    main()
