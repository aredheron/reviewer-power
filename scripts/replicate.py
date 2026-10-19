"""Plan, verify, collect, and analyze a fresh replication of the presented experiments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power import replication


def main() -> None:
    """Keep all paid work behind the explicit collect command and request allowance."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser(
        "plan", help="Offline: freeze exact prompts and complete assignment matrix"
    )
    plan.add_argument(
        "--study", choices=("loops", "naming", "quotes", "strong"), required=True
    )
    plan.add_argument(
        "--profile", choices=("gpt-oss-120b", "deepseek", "glm"), required=True
    )
    plan.add_argument("--phase", choices=("pilot", "main"), required=True)
    plan.add_argument("--run-id", required=True)
    plan.add_argument("--pilot-run-id")
    sandbox = sub.add_parser(
        "verify-sandbox", help="Run Docker isolation checks; no model calls"
    )
    sandbox.add_argument("--oracle-image", required=True)
    collect = sub.add_parser(
        "collect",
        help="PAID: make at most the specified number of new provider requests",
    )
    collect.add_argument("--run-id", required=True)
    collect.add_argument("--oracle-image", required=True)
    collect.add_argument(
        "--max-requests",
        type=int,
        required=True,
        help="Allowance for this invocation, including generator and judge requests",
    )
    analysis = sub.add_parser(
        "analyze", help="Offline: report every assigned outcome and pilot admission"
    )
    analysis.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.command == "plan":
        run = replication.prepare(
            root, args.run_id, args.study, args.profile, args.phase, args.pilot_run_id
        )
        print(run)
    elif args.command == "verify-sandbox":
        replication.verify_sandbox(root, args.oracle_image)
    elif args.command == "collect":
        try:
            replication.locked_collect(
                root,
                replication.run_path(root, args.run_id),
                args.oracle_image,
                args.max_requests,
            )
        except replication.RequestLimit as error:
            print(str(error))
        finally:
            replication.analyze(replication.run_path(root, args.run_id))
    else:
        print(
            json.dumps(
                replication.analyze(replication.run_path(root, args.run_id)), indent=2
            )
        )


if __name__ == "__main__":
    main()
