"""Analyze a frozen Mode A pilot without model calls or submission execution."""

import argparse
import hashlib
import json
from pathlib import Path

from grader_power.analysis import analyze_cohort, render_report


def main() -> None:
    """Write a fresh report directory; preserve all original records and analyses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261010)
    args = parser.parse_args()
    analysis = analyze_cohort(
        args.run,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    analysis["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report = render_report(analysis)
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (args.output / "analysis.json").write_text(
        json.dumps(analysis, sort_keys=True, indent=2) + "\n"
    )
    (args.output / "report.md").write_text(report)
    print(
        json.dumps(
            {"run_id": analysis["run_id"], "report": str(args.output / "report.md")}
        )
    )


if __name__ == "__main__":
    main()
