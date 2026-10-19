"""Rebuild static candidate metadata and its review packet from verified caches."""

import hashlib
import json
from pathlib import Path

from grader_power.datasets.task_filter import (
    load_screening_pool,
    render_review,
    screen_pool,
    write_text_atomic,
)


def main() -> None:
    """Screen all 542 source IDs, without downloads, execution, or model calls."""
    root = Path(__file__).resolve().parents[1]
    source = root / "manifests" / "evalplus_source.json"
    datasets = load_screening_pool(root / "data" / "evalplus", source)
    records = screen_pool(datasets, hashlib.sha256(source.read_bytes()).hexdigest())
    if len(records) != 542:
        raise ValueError(f"Expected all 542 source tasks, got {len(records)}")
    manifest = "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    review = render_review(records)
    write_text_atomic(root / "manifests" / "task_candidates.jsonl", manifest)
    write_text_atomic(root / "planning" / "task_screening_review.md", review)
    print(
        f"Screened {len(records)} original IDs; all remain provisional and unreviewed."
    )
    print("Candidate metadata: manifests/task_candidates.jsonl")
    print("Review packet: planning/task_screening_review.md")


if __name__ == "__main__":
    main()
