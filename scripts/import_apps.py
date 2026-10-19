"""Import revision-pinned APPS JSONL bytes and inventory interfaces without running code."""

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

from grader_power.datasets import apps_source as apps
from grader_power.datasets.mbpp_full import write_once


def main() -> None:
    """Download both exact splits, or import supplied local copies, without a dataset SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    index = []
    split_metadata = {}
    with tempfile.TemporaryDirectory(prefix="apps-import-") as directory:
        for split, (size, expected) in apps.SPLITS.items():
            source = getattr(args, f"{split}_file")
            if source is None:
                source = Path(directory) / f"{split}.jsonl"
                with (
                    urlopen(apps.source_url(f"{split}.jsonl"), timeout=120) as response,
                    source.open("xb") as output,
                ):
                    shutil.copyfileobj(response, output)
            if source.stat().st_size != size or apps.file_digest(source) != expected:
                raise ValueError(f"Downloaded {split} does not match pinned bytes")
            seen = set()
            for row in apps.records(source):
                seen.add(row["id"])
                index.append(apps.inspect_record(row, split))
            if seen != set(range(5000)):
                raise ValueError(f"Expected all 5000 APPS {split} IDs")
            target = root / apps.CACHE / f"{split}.jsonl"
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if (
                    target.stat().st_size != size
                    or apps.file_digest(target) != expected
                ):
                    raise ValueError("Refusing to replace existing APPS source")
            else:
                with source.open("rb") as incoming, target.open("xb") as output:
                    shutil.copyfileobj(incoming, output)
                if apps.file_digest(target) != expected:
                    raise ValueError("Copied APPS source failed verification")
            split_metadata[split] = {
                "bytes": size,
                "sha256": expected,
                "count": 5000,
                "url": apps.source_url(f"{split}.jsonl"),
                "path": str(target.relative_to(root)),
            }
    index_bytes = (
        "\n".join(json.dumps(row, sort_keys=True) for row in index) + "\n"
    ).encode()
    write_once(root / apps.CACHE / "index.jsonl", index_bytes)
    manifest = {
        "name": "APPS, original train and test splits",
        "revision": apps.REVISION,
        "splits": split_metadata,
        "total_records": len(index),
        "inventory_sha256": apps.digest(index_bytes),
        "interfaces": dict(Counter(row["interface"] for row in index)),
        "screening": dict(Counter(row["screening"] for row in index)),
        "upstream_dataset_card": apps.source_url("README.md"),
        "dataset_card_license": "MIT",
        "upstream_repository": "https://github.com/hendrycks/apps",
        "source_code_executed_during_import": False,
        "unsupported_nested_records_preserved": [
            row["task_id"] for row in index if row["interface"] == "unparsed"
        ],
        "coverage": "Original supplied tests; not augmented tests or a correctness proof. User accepted limited supplied coverage.",
    }
    write_once(
        root / apps.MANIFEST,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
