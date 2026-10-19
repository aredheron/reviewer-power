"""Print a complete synthetic draft block without model calls or task execution."""

import argparse
import json
from dataclasses import asdict
from typing import cast

from grader_power.render import render_pilot_block
from grader_power.schemas import PREFLIGHT_VARIANTS, PreflightVariant, RenderTask


def main() -> None:
    """Render the explicitly chosen draft phrasing, optionally shuffling by seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", choices=PREFLIGHT_VARIANTS, required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    task = RenderTask(
        source_task_id="synthetic/double-values",
        derived_task_id="synthetic-double-values",
        family="synthetic_mapping_demo",
        stub_path="src/values.py",
        stub=(
            "def double_values(values: list[int]) -> list[int]:\n"
            '    """Return twice each input value, in input order; leave values unchanged."""\n'
            "    # TODO\n"
            "    raise NotImplementedError\n"
        ),
        readme="# Value utilities\n\nSmall utilities for integer sequences.\n",
    )
    block = render_pilot_block(
        task, preflight_variant=cast(PreflightVariant, args.preflight), seed=args.seed
    )
    print(json.dumps([asdict(row) for row in block], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
