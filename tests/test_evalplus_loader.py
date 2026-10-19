"""Offline ingestion tests using small synthetic records and the real loaders."""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from grader_power.datasets import evalplus_loader as loader


@pytest.fixture(autouse=True)
def isolate_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent network access and dependence on the developer's override settings."""
    monkeypatch.delenv("HUMANEVAL_OVERRIDE_PATH", raising=False)
    monkeypatch.delenv("MBPP_OVERRIDE_PATH", raising=False)
    monkeypatch.setattr(loader.humaneval, "HUMANEVAL_OVERRIDE_PATH", None)
    monkeypatch.setattr(loader.mbpp, "MBPP_OVERRIDE_PATH", None)

    def reject_download(*args: Any, **kwargs: Any) -> None:
        """Fail immediately if an offline test tries to fetch data."""
        raise OSError("network disabled in tests")

    monkeypatch.setattr(loader.utils.wget, "download", reject_download)


def make_cache(directory: Path, name: loader.DatasetName) -> Path:
    """Write complete synthetic source records, including a normalization case."""
    spec = loader.SOURCES[name]
    records: list[dict[str, Any]] = []
    for index in range(spec.task_count):
        # Use neutral IDs for MBPP except its documented complex-number case.
        number = index if name == "humaneval" else index + 1000
        records.append(
            {
                "task_id": f"{spec.task_prefix}/{number}",
                "entry_point": "example",
                "prompt": "def example(value):\n",
                "canonical_solution": "    raise RuntimeError('must not execute')\n",
                "contract": "",
                "atol": 0,
                "base_input": [[1]],
                "plus_input": [[2]],
            }
        )
    if name == "mbpp":
        records[0].update(
            task_id="Mbpp/124", base_input=[["1.5", "2j"]], plus_input=[["3.5", "4j"]]
        )
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.release_name}-{spec.release_version}.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    return path


@pytest.mark.parametrize("name", ["humaneval", "mbpp"])
def test_load_full_cache_and_provenance(
    tmp_path: Path, name: loader.DatasetName
) -> None:
    """Loading preserves all IDs and reports hashes of the original cached bytes."""
    path = make_cache(tmp_path, name)
    previous_cache = loader.utils.CACHE_DIR
    result = loader.load_dataset(name, tmp_path)
    assert loader.utils.CACHE_DIR == previous_cache
    assert len(result.tasks) == loader.SOURCES[name].task_count
    assert result.provenance["original_task_ids"] == sorted(result.tasks)
    assert result.provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        result.provenance["evalplus_md5"] == hashlib.md5(path.read_bytes()).hexdigest()
    )
    assert result.provenance["screening_status"] == "not_screened"
    assert result.provenance["variant"] == "full"
    if name == "mbpp":
        assert result.tasks["Mbpp/124"]["base_input"] == [(1.5, 2j)]
        assert result.tasks["Mbpp/124"]["plus_input"] == [(3.5, 4j)]
        assert '"2j"' in path.read_text()


def test_manifest_excludes_solutions_and_inputs(tmp_path: Path) -> None:
    """The tracked manifest contains provenance but no oracle content."""
    make_cache(tmp_path / "cache", "humaneval")
    result = loader.load_dataset("humaneval", tmp_path / "cache")
    manifest_path = tmp_path / "manifests" / "evalplus_source.json"
    loader.write_source_manifest([result], manifest_path)
    text = manifest_path.read_text()
    manifest = json.loads(text)
    assert manifest["evalplus_version"] == "0.3.1"
    assert manifest["datasets"][0]["licenses"][1]["spdx"] == "MIT"
    assert "canonical_solution" not in text
    assert '"base_input"' not in text
    assert '"plus_input"' not in text
    assert "must not execute" not in text
    loader.write_source_manifest([result], manifest_path)
    assert manifest_path.read_text() == text


def test_partial_cache_is_rejected(tmp_path: Path) -> None:
    """A truncated but valid JSONL file must not become a successful source pool."""
    path = make_cache(tmp_path, "humaneval")
    path.write_text(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="expected 164 tasks, got 1"):
        loader.load_dataset("humaneval", tmp_path)


@pytest.mark.parametrize("field,value", [("entry_point", None), ("base_input", "bad")])
def test_invalid_record_is_rejected(tmp_path: Path, field: str, value: Any) -> None:
    """Malformed required fields produce actionable validation errors."""
    path = make_cache(tmp_path, "humaneval")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0][field] = value
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    expected_error = TypeError if field == "base_input" else ValueError
    with pytest.raises(expected_error, match=field):
        loader.load_dataset("humaneval", tmp_path)


def test_known_mbpp_anomaly_is_preserved_and_reported(tmp_path: Path) -> None:
    """Retain the release's empty-object input without silently repairing it."""
    path = make_cache(tmp_path, "mbpp")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1].update(task_id="Mbpp/793", plus_input={})
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = loader.load_dataset("mbpp", tmp_path)
    assert result.tasks["Mbpp/793"]["plus_input"] == {}
    assert result.provenance["source_issues"] == [
        {
            "task_id": "Mbpp/793",
            "field": "plus_input",
            "issue": "empty object instead of input list; retained unchanged",
        }
    ]


def test_download_failure_restores_cache_setting(tmp_path: Path) -> None:
    """Download failures propagate and leave EvalPlus's process settings intact."""
    previous_cache = loader.utils.CACHE_DIR
    with pytest.raises(OSError, match="network disabled"):
        loader.load_dataset("humaneval", tmp_path)
    assert loader.utils.CACHE_DIR == previous_cache


def test_source_override_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Custom sources must not be mislabeled as the official release."""
    monkeypatch.setenv("MBPP_OVERRIDE_PATH", "custom.jsonl")
    with pytest.raises(ValueError, match="Unset MBPP_OVERRIDE_PATH"):
        loader.load_dataset("mbpp", tmp_path)


def test_manifest_failure_preserves_existing_file(tmp_path: Path) -> None:
    """A serialization failure must not truncate a previous valid manifest."""
    path = tmp_path / "manifest.json"
    path.write_text('{"existing": true}\n')
    with pytest.raises(TypeError):
        loader.write_source_manifest([loader.LoadedDataset({}, {"bad": {1, 2}})], path)
    assert path.read_text() == '{"existing": true}\n'
