"""Cache both complete source datasets and write their provenance manifest."""

from pathlib import Path

from grader_power.datasets.evalplus_loader import load_dataset, write_source_manifest


def main() -> None:
    """Load full releases under data/ and publish metadata only after both succeed."""
    root = Path(__file__).resolve().parents[1]
    cache_dir = root / "data" / "evalplus"
    datasets = [load_dataset(name, cache_dir) for name in ("humaneval", "mbpp")]
    write_source_manifest(datasets, root / "manifests" / "evalplus_source.json")
    for dataset in datasets:
        info = dataset.provenance
        print(
            f"{info['name']} {info['release_version']}: {info['task_count']} tasks, "
            f"{info['base_input_count']} base inputs, {info['plus_input_count']} plus inputs"
        )
    print("Source metadata: manifests/evalplus_source.json")


if __name__ == "__main__":
    main()
