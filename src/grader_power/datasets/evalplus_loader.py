"""Load full EvalPlus releases without executing benchmark code.

Only provenance belongs in tracked manifests. Loaded records contain hidden test
inputs and canonical solutions and must never be passed wholesale to a model.
"""

import hashlib
import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

from evalplus.data import humaneval, mbpp, utils

DatasetName = Literal["humaneval", "mbpp"]
EVALPLUS_VERSION = "0.3.1"


@dataclass(frozen=True)
class SourceSpec:
    """Pinned release identity and expected full-dataset size."""

    name: str
    release_name: str
    release_version: str
    task_prefix: str
    task_count: int
    dataset_card: str
    original_license: str
    original_license_url: str


SOURCES: dict[DatasetName, SourceSpec] = {
    "humaneval": SourceSpec(
        "HumanEval+",
        "HumanEvalPlus",
        "v0.1.10",
        "HumanEval",
        164,
        "https://huggingface.co/datasets/evalplus/humanevalplus",
        "MIT",
        "https://github.com/openai/human-eval/blob/master/LICENSE",
    ),
    "mbpp": SourceSpec(
        "MBPP+",
        "MbppPlus",
        "v0.2.0",
        "Mbpp",
        378,
        "https://huggingface.co/datasets/evalplus/mbppplus",
        "CC-BY-4.0",
        "https://huggingface.co/datasets/google-research-datasets/mbpp",
    ),
}


@dataclass
class LoadedDataset:
    """Normalized records plus metadata safe to include in source manifests."""

    tasks: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def load_dataset(name: DatasetName, cache_dir: Path) -> LoadedDataset:
    """Download or reuse a full release through the pinned EvalPlus loaders.

    Preserves EvalPlus's native input types, including MBPP tuples and complex
    numbers. Raises ValueError for unsupported names, overrides, or malformed
    records, TypeError for invalid test-input types, and RuntimeError for
    incompatible package/release versions. The known empty-dict plus_input in
    Mbpp/793 is retained unchanged and flagged in provenance for later screening.
    Download and filesystem failures propagate; no task code is executed.

    EvalPlus 0.3.1 has no cache-directory argument, so its cache global is changed
    temporarily and restored even on failure. Call this function serially, without
    concurrent direct EvalPlus calls in the same process.
    """
    if name not in SOURCES:
        raise ValueError(f"Unsupported dataset: {name!r}")
    if version("evalplus") != EVALPLUS_VERSION:
        raise RuntimeError(f"This loader requires evalplus=={EVALPLUS_VERSION}")
    for variable, active in (
        ("HUMANEVAL_OVERRIDE_PATH", humaneval.HUMANEVAL_OVERRIDE_PATH),
        ("MBPP_OVERRIDE_PATH", mbpp.MBPP_OVERRIDE_PATH),
    ):
        if active or os.environ.get(variable):
            raise ValueError(f"Unset {variable} to load the official source releases")

    spec = SOURCES[name]
    upstream_version = (
        humaneval.HUMANEVAL_PLUS_VERSION
        if name == "humaneval"
        else mbpp.MBPP_PLUS_VERSION
    )
    if upstream_version != spec.release_version:
        raise RuntimeError(f"Unexpected upstream release for {spec.name}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    previous_cache = utils.CACHE_DIR
    try:
        utils.CACHE_DIR = str(cache_dir.resolve())
        # Use the package defaults: the 0.3.1 HumanEval loader ignores its version
        # argument. The package and default release are checked explicitly above.
        if name == "humaneval":
            tasks = humaneval.get_human_eval_plus()
            upstream_hash = humaneval.get_human_eval_plus_hash()
        else:
            tasks = mbpp.get_mbpp_plus()
            upstream_hash = mbpp.get_mbpp_plus_hash()
        url, cache_path = utils.get_dataset_metadata(
            spec.release_name, spec.release_version, mini=False, noextreme=False
        )
    finally:
        utils.CACHE_DIR = previous_cache

    if len(tasks) != spec.task_count:
        raise ValueError(
            f"{spec.name}: expected {spec.task_count} tasks, got {len(tasks)}; "
            "check for an incomplete or modified cache"
        )
    source_issues: list[dict[str, str]] = []
    for task_id, task in tasks.items():
        if (
            not task_id.startswith(f"{spec.task_prefix}/")
            or task.get("task_id") != task_id
        ):
            raise ValueError(f"{spec.name}: inconsistent task ID {task_id!r}")
        for field in ("task_id", "entry_point", "prompt", "canonical_solution"):
            if not isinstance(task.get(field), str) or not task[field].strip():
                raise ValueError(f"{task_id}: missing or invalid {field}")
        for field in ("base_input", "plus_input"):
            # This anomaly exists in the full v0.2.0 release. Preserve it as
            # source evidence; task screening must decide whether to exclude it.
            if (
                name == "mbpp"
                and task_id == "Mbpp/793"
                and field == "plus_input"
                and task.get(field) == {}
            ):
                source_issues.append(
                    {
                        "task_id": task_id,
                        "field": field,
                        "issue": "empty object instead of input list; retained unchanged",
                    }
                )
                continue
            if not isinstance(task.get(field), list):
                raise TypeError(
                    f"{task_id}: missing or invalid {field}; expected a list"
                )

    with Path(cache_path).open("rb") as stream:
        sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    return LoadedDataset(
        tasks,
        {
            "name": spec.name,
            "release_version": spec.release_version,
            "download_url": url,
            "cache_file": Path(cache_path).name,
            "variant": "full",
            "hash_scope": "uncompressed upstream JSONL bytes, before input normalization",
            "evalplus_md5": upstream_hash,
            "sha256": sha256,
            "task_count": len(tasks),
            "original_task_ids": sorted(tasks),
            "base_input_count": sum(len(task["base_input"]) for task in tasks.values()),
            "plus_input_count": sum(len(task["plus_input"]) for task in tasks.values()),
            "screening_status": "not_screened",
            "source_issues": source_issues,
            "licenses": [
                {
                    "scope": "EvalPlus dataset card declaration",
                    "spdx": "Apache-2.0",
                    "source_url": spec.dataset_card,
                },
                {
                    "scope": "original benchmark",
                    "spdx": spec.original_license,
                    "source_url": spec.original_license_url,
                },
            ],
        },
    )


def write_source_manifest(datasets: list[LoadedDataset], path: Path) -> None:
    """Atomically write provenance only, leaving an existing manifest on failure.

    Original task IDs describe the source pool, not accepted pilot/main tasks.
    License identifiers record upstream declarations, not a relicensing decision.
    """
    manifest = {
        "schema_version": 1,
        "evalplus_version": version("evalplus"),
        "python_version": platform.python_version(),
        "evalplus_package_license": {
            "spdx": "Apache-2.0",
            "source_url": "https://github.com/evalplus/evalplus/blob/v0.3.1/LICENSE",
            "note": "evalplus/eval additionally carries upstream HumanEval MIT terms",
        },
        "datasets": [dataset.provenance for dataset in datasets],
    }
    content = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
