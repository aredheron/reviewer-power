"""Audit all provisional reference pairs; optionally use a verified Docker oracle."""

import argparse
import json
import re
from pathlib import Path

from grader_power.datasets.reference_solutions import (
    MANIFEST_PATH,
    REPORT_PATH,
    audit_references,
    render_reference_review,
    run_reference_oracles,
)
from grader_power.datasets.task_filter import write_text_atomic


def main() -> None:
    """Default to static inspection; require explicit image/run ID for execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--oracle-image",
        help="Immutable image ID after the documented sandbox verification gate",
    )
    parser.add_argument(
        "--run-id", help="New run name under ignored runs/reference-validation/"
    )
    parser.add_argument(
        "--verify-upstream",
        action="store_true",
        help="Compare every full-suite outcome with the official checker inside Docker",
    )
    args = parser.parse_args()
    if bool(args.oracle_image) != bool(args.run_id):
        parser.error("--oracle-image and --run-id must be supplied together")
    if args.verify_upstream and not args.oracle_image:
        parser.error("--verify-upstream requires --oracle-image and --run-id")
    if args.run_id and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("--run-id must be a simple name without path separators")
    root = Path(__file__).resolve().parents[1]
    pairs, datasets, records = audit_references(root)
    write_text_atomic(
        root / MANIFEST_PATH,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records),
    )
    write_text_atomic(root / REPORT_PATH, render_reference_review(records))
    passed = sum(row["static_status"] == "pass" for row in records)
    print(
        f"Audited {len(pairs)} pairs: {passed} static passes, {len(pairs) - passed} needing resolution."
    )
    print("All remain provisional; static evidence does not verify correctness.")
    if args.oracle_image:
        attempts = run_reference_oracles(
            pairs,
            datasets,
            records,
            image=args.oracle_image,
            output_dir=root / "runs/reference-validation" / args.run_id,
            verify_upstream=args.verify_upstream,
        )
        correct = sum(attempt["full_correct"] for attempt in attempts)
        print(
            f"Full oracle: {correct}/{len(attempts)} implementations passed; human QA still required."
        )
        if correct != len(attempts):
            raise SystemExit(1)
    if passed != len(pairs):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
