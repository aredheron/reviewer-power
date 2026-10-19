"""Validate preference preparation using synthetic, never-executed code strings."""

import ast
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from grader_power import preference_tasks as preferences
from grader_power.datasets.evalplus_loader import DatasetName, LoadedDataset


def source_data(dataset: DatasetName = "humaneval") -> dict[str, Any]:
    """Separate public scaffolding from recognizable private test/body markers."""
    prefix = "HumanEval" if dataset == "humaneval" else "Mbpp"
    signature = "def transform(items):\n"
    return {
        "task_id": f"{prefix}/999",
        "entry_point": "transform",
        "prompt": signature if dataset == "humaneval" else '"""PUBLIC_TASK_TEXT"""\n',
        "canonical_solution": ("" if dataset == "humaneval" else signature)
        + "    return 'PRIVATE_CANONICAL_BODY'\n",
        "base_input": [["PRIVATE_BASE_INPUT"]],
        "plus_input": [["PRIVATE_PLUS_INPUT"]],
        "contract": "PUBLIC_INPUT_CONTRACT",
        "atol": 0,
    }


def task_data(family: preferences.Family = "naming") -> dict[str, Any]:
    """Provide two inert implementations that differ only in the chosen style."""
    implementations = (
        {
            "snake_case": (
                "def transform(items):\n"
                "    total_sum = sum(items)\n"
                "    return total_sum\n"
            ),
            "camelCase": (
                "def transform(items):\n"
                "    totalSum = sum(items)\n"
                "    return totalSum\n"
            ),
        }
        if family == "naming"
        else {
            "single_quotes": "def transform(items):\n    return 'result'\n",
            "double_quotes": 'def transform(items):\n    return "result"\n',
        }
    )
    return {
        "task_id": "HumanEval/999",
        "dataset": "humaneval",
        "split": "study",
        "entry_point": "transform",
        "selection_rationale": "Synthetic observable style opportunity.",
        "style_opportunities": ["Result accumulator or return literal"],
        "implementations": implementations,
    }


def bundle_data(family: preferences.Family = "naming") -> dict[str, Any]:
    """Create the smallest valid family bundle for boundary checks."""
    return {
        "family": family,
        "styles": list(preferences.STYLES[family]),
        "tasks": [task_data(family)],
    }


@pytest.mark.parametrize("family", ["naming", "quotes"])
def test_style_only_pairs_are_accepted_without_execution(
    family: preferences.Family,
) -> None:
    """Even matching top-level execution traps remain inert during validation."""
    data = task_data(family)
    trap = "raise AssertionError('REFERENCE_CODE_MUST_NEVER_RUN')\n"
    data["implementations"] = {
        style: trap + code for style, code in data["implementations"].items()
    }
    preferences.validate_pair(
        preferences.PreferenceTask.model_validate(data), family, source_data()
    )


@pytest.mark.parametrize(
    "replacement",
    [
        "def transform(items):\n    totalSum = max(items)\n    return totalSum\n",
        "def transform(items):\n    totalSum = sum(items) + 1\n    return totalSum\n",
        "def transform(items):\n    totalSum = sum(items)\n    return -totalSum\n",
    ],
)
def test_naming_pair_rejects_algorithm_changes(replacement: str) -> None:
    """Equivalent spelling cannot authorize changed operations or calls."""
    data = task_data()
    data["implementations"]["camelCase"] = replacement
    with pytest.raises(ValueError, match="more than the requested style"):
        preferences.validate_pair(
            preferences.PreferenceTask.model_validate(data), "naming", source_data()
        )


@pytest.mark.parametrize(
    "code",
    [
        'def transform(items):\n    return "different"\n',
        'def transform(items):\n    return "result" + str(items)\n',
    ],
)
def test_quote_pair_rejects_literal_and_algorithm_changes(code: str) -> None:
    """Quote style pairs must preserve decoded strings and computational syntax."""
    data = task_data("quotes")
    data["implementations"]["double_quotes"] = code
    with pytest.raises(ValueError, match="more than the requested style"):
        preferences.validate_pair(
            preferences.PreferenceTask.model_validate(data), "quotes", source_data()
        )


@pytest.mark.parametrize(
    "signature,reason",
    [
        ("def transform(values):", "parameter signature"),
        ("def transform(items, extra=0):", "parameter signature"),
        ("def transform(items, /):", "parameter signature"),
        ("def transform(*, items):", "parameter signature"),
        ("def transform(items: list):", "parameter signature"),
        ("def transform(items) -> list:", "return annotation"),
        ("def other(items):", "entry point is missing or ambiguous"),
    ],
)
def test_pair_rejects_changed_upstream_interface(signature: str, reason: str) -> None:
    """Task APIs remain fixed even if both paired implementations change them."""
    data = task_data()
    data["implementations"] = {
        style: code.replace("def transform(items):", signature)
        for style, code in data["implementations"].items()
    }
    with pytest.raises(ValueError, match=reason):
        preferences.validate_pair(
            preferences.PreferenceTask.model_validate(data), "naming", source_data()
        )


@pytest.mark.parametrize(
    "code",
    [
        "def transform(items):\n    totalSum = sum(items)\n    return total_sum\n",
        "def transform(items):\n    totalSum = Sum(items)\n    return totalSum\n",
    ],
)
def test_naming_pair_rejects_inconsistent_or_global_renames(code: str) -> None:
    """A stylistic rename must preserve binding identity and external calls."""
    data = task_data()
    data["implementations"]["camelCase"] = code
    with pytest.raises(ValueError):
        preferences.validate_pair(
            preferences.PreferenceTask.model_validate(data), "naming", source_data()
        )


def test_identical_pair_is_not_two_style_references() -> None:
    """Two labels on identical code do not create a paired style contrast."""
    data = task_data()
    data["implementations"]["camelCase"] = data["implementations"]["snake_case"]
    with pytest.raises(ValueError, match="identical"):
        preferences.validate_pair(
            preferences.PreferenceTask.model_validate(data), "naming", source_data()
        )


@pytest.mark.parametrize("failure", ["duplicate", "prefix", "styles", "missing"])
def test_bundle_membership_errors_are_rejected(failure: str) -> None:
    """Reject duplicate IDs, source mismatches, and incomplete style membership."""
    data = bundle_data()
    if failure == "duplicate":
        data["tasks"].append(deepcopy(data["tasks"][0]))
    elif failure == "prefix":
        data["tasks"][0]["dataset"] = "mbpp"
    elif failure == "styles":
        data["styles"] = ["single_quotes", "double_quotes"]
    else:
        del data["tasks"][0]["implementations"]["camelCase"]
    with pytest.raises(ValidationError):
        preferences.PreferenceBundle.model_validate(data)


def local_bundle_root(tmp_path: Path, reserved: list[str]) -> Path:
    """Create placeholder caches; a patched loader supplies synthetic data only."""
    (tmp_path / "data/evalplus").mkdir(parents=True)
    for filename in ("HumanEvalPlus-v0.1.10.jsonl", "MbppPlus-v0.2.0.jsonl"):
        (tmp_path / "data/evalplus" / filename).write_text("SYNTHETIC_CACHE\n")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/main_selection.json").write_text(
        json.dumps({"confirmation": reserved})
    )
    (tmp_path / "fixtures/style_preferences").mkdir(parents=True)
    for family in ("naming", "quotes"):
        (tmp_path / f"fixtures/style_preferences/{family}.json").write_text(
            json.dumps(bundle_data(family))
        )
    return tmp_path


def synthetic_loader(name: DatasetName, cache_dir: Path) -> LoadedDataset:
    """Return metadata without opening caches, downloading, or executing code."""
    assert cache_dir.is_dir()
    source = source_data(name)
    return LoadedDataset(
        tasks={source["task_id"]: source},
        provenance={"name": name, "variant": "full"},
    )


def test_reserved_original_confirmation_task_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reserved source IDs cannot silently enter either new preference family."""
    root = local_bundle_root(tmp_path, ["HumanEval/999"])
    monkeypatch.setattr(preferences, "load_dataset", synthetic_loader)
    with pytest.raises(ValueError, match="confirmation task must remain unused"):
        preferences.load_bundles(root)


def test_missing_cache_fails_before_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation must not turn absent local datasets into an implicit download."""

    def forbidden_loader(name: DatasetName, cache_dir: Path) -> LoadedDataset:
        raise AssertionError("Dataset loader must not run without both caches")

    monkeypatch.setattr(preferences, "load_dataset", forbidden_loader)
    with pytest.raises(FileNotFoundError, match="no download is attempted"):
        preferences.load_bundles(tmp_path)


def test_cross_family_pilot_study_overlap_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A task used to develop one judge cannot also be reserved for study use."""
    root = local_bundle_root(tmp_path, [])
    path = root / "fixtures/style_preferences/quotes.json"
    data = json.loads(path.read_text())
    data["tasks"][0]["split"] = "pilot"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(preferences, "load_dataset", synthetic_loader)
    with pytest.raises(ValueError, match="disjoint across both"):
        preferences.load_bundles(root)


@pytest.mark.parametrize("dataset", ["humaneval", "mbpp"])
def test_visible_task_excludes_private_bodies_and_test_values(
    dataset: DatasetName,
) -> None:
    """Whitelisted packets expose only public scaffolding, APIs, and contracts."""
    source = source_data(dataset)
    data = task_data()
    data["task_id"] = source["task_id"]
    data["dataset"] = dataset
    packet = preferences.visible_task(
        preferences.PreferenceTask.model_validate(data), source
    )
    rendered = json.dumps(packet)
    assert "PRIVATE_" not in rendered
    assert "total_sum" not in rendered
    assert "totalSum" not in rendered
    assert "PUBLIC_INPUT_CONTRACT" in rendered
    assert packet["original_parameters"] == ["items"]
    assert "raise NotImplementedError" in packet["stub"]
    ast.parse(packet["stub"])
    if dataset == "humaneval":
        assert packet["stub"].startswith(source["prompt"])
    else:
        assert "PUBLIC_TASK_TEXT" in packet["stub"]
        assert "def transform(items):" in packet["stub"]


def test_all_parameter_kinds_are_preserved_in_visible_mbpp_stub() -> None:
    """Public API extraction includes positional-only and variadic parameters."""
    source = source_data("mbpp")
    source["canonical_solution"] = (
        "def transform(items, /, *extras, limit: int = 2, **options) -> list:\n"
        "    return ['PRIVATE_CANONICAL_BODY']\n"
    )
    data = task_data()
    data.update(task_id=source["task_id"], dataset="mbpp")
    packet = preferences.visible_task(
        preferences.PreferenceTask.model_validate(data), source
    )
    assert packet["original_parameters"] == ["items", "limit", "extras", "options"]
    stub_target = preferences.target_function(packet["stub"], "transform")
    source_target = preferences.target_function(
        source["canonical_solution"], "transform"
    )
    assert ast.dump(stub_target.args) == ast.dump(source_target.args)
    assert stub_target.returns is not None
    assert source_target.returns is not None
    assert ast.dump(stub_target.returns) == ast.dump(source_target.returns)
