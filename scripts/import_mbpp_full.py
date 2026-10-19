"""Import exact original MBPP bytes and provenance without executing task code."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from grader_power.datasets import mbpp_full as mbpp


def main() -> None:
    """Download the pinned dataset or import an already downloaded exact file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.source_file is None:
        with urlopen(mbpp.SOURCE_URL, timeout=60) as response:
            raw = response.read(4 * 1024 * 1024)
    else:
        raw = args.source_file.read_bytes()
    if mbpp.digest(raw) != mbpp.SOURCE_SHA256:
        raise ValueError("Download does not match the pinned MBPP source")
    records = mbpp.parse_records(raw)
    if set(records) != set(range(1, 975)):
        raise ValueError("Expected all 974 original task IDs")
    manifest = {
        "name": "Original MBPP (full, unsanitized)",
        "revision": mbpp.REVISION,
        "download_url": mbpp.SOURCE_URL,
        "sha256": mbpp.SOURCE_SHA256,
        "cache_path": str(mbpp.CACHE),
        "task_count": len(records),
        "original_task_ids": sorted(records),
        "supplied_test_count": sum(len(r["test_list"]) for r in records.values()),
        "challenge_test_count": sum(
            len(r["challenge_test_list"]) for r in records.values()
        ),
        "source_code_executed_during_import": False,
        "upstream_readme": f"https://github.com/google-research/google-research/blob/{mbpp.REVISION}/mbpp/README.md",
        "license_declarations": [
            {
                "scope": "original MBPP dataset card",
                "url": "https://huggingface.co/datasets/google-research-datasets/mbpp",
                "spdx": "CC-BY-4.0",
            },
            {
                "scope": "source repository LICENSE; separate from dataset-card declaration",
                "url": f"https://github.com/google-research/google-research/blob/{mbpp.REVISION}/LICENSE",
                "spdx": "Apache-2.0",
            },
        ],
        "coverage_approval": {
            "user_request": "Can you import MBPP to the project and search for more candidates there? I'll elect to allow the test case shortage for now.",
            "scope": "Candidate audit using supplied tests; no paid experiments or semantic changes",
        },
    }
    mbpp.write_once(root / mbpp.CACHE, raw)
    mbpp.write_once(
        root / mbpp.MANIFEST,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(f"Imported {len(mbpp.load(root))} exact original MBPP records")


if __name__ == "__main__":
    main()
