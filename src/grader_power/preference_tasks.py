"""Prepare style-preference tasks without executing code or exposing hidden tests."""

import ast
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from grader_power.code_extract import find_target
from grader_power.config import Contract, Text
from grader_power.datasets.evalplus_loader import (
    DatasetName,
    LoadedDataset,
    load_dataset,
)
from grader_power.pilot_tasks import digest

Family = Literal["naming", "quotes"]
STYLES = {
    "naming": ("snake_case", "camelCase"),
    "quotes": ("single_quotes", "double_quotes"),
}


class PreferenceTask(Contract):
    """Authored references and outcome-independent selection rationale."""

    task_id: Text
    dataset: DatasetName
    split: Literal["pilot", "study"]
    entry_point: Text
    selection_rationale: Text
    style_opportunities: list[Text] = Field(min_length=1)
    notes: list[Text] = Field(default_factory=list)
    implementations: dict[str, Text]


class PreferenceBundle(Contract):
    """One preference family with unique tasks and exactly two reference styles."""

    family: Family
    styles: list[Text]
    tasks: list[PreferenceTask] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_membership(self) -> "PreferenceBundle":
        """Reject duplicate tasks and style labels inconsistent with the family."""
        if tuple(self.styles) != STYLES[self.family]:
            raise ValueError("Unexpected styles for this preference family")
        if len({task.task_id for task in self.tasks}) != len(self.tasks):
            raise ValueError("Task IDs must be unique within a family")
        for task in self.tasks:
            prefix = "HumanEval/" if task.dataset == "humaneval" else "Mbpp/"
            if not task.task_id.startswith(prefix):
                raise ValueError("Task ID and dataset disagree")
            if set(task.implementations) != set(self.styles):
                raise ValueError("Each task needs exactly both reference styles")
        return self


def target_function(code: str, entry_point: str) -> ast.FunctionDef:
    """Find one unconditional, unambiguously bound synchronous entry point."""
    target, reason = find_target(ast.parse(code), entry_point)
    if target is None:
        raise ValueError(f"Reference entry point is missing or ambiguous: {reason}")
    if not isinstance(target, ast.FunctionDef):
        raise TypeError("Reference entry point must be synchronous")
    return target


def parameter_names(target: ast.FunctionDef) -> list[str]:
    """List all API parameters, including positional-only, varargs, and keywords."""
    args = target.args
    values = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    values += [arg for arg in (args.vararg, args.kwarg) if arg is not None]
    return [arg.arg for arg in values]


def _equivalent(
    left: Any,
    right: Any,
    *,
    allowed_names: set[str],
    renames: dict[str, str],
    reverse: dict[str, str],
) -> bool:
    """Compare syntax trees allowing only underscore/case changes to bindings.

    Attribute names, literals, operators, imports, defaults, and algorithmic
    structure must remain identical. Signature preservation is checked separately.
    No candidate code is compiled, imported, or executed by this comparison.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, ast.AST):
        for field, value in ast.iter_fields(left):
            other = getattr(right, field)
            allowed = (
                (isinstance(left, ast.Name) and field == "id")
                or (isinstance(left, ast.arg) and field == "arg")
                or (isinstance(left, ast.FunctionDef) and field == "name")
            )
            if allowed and value in allowed_names:
                if (
                    value.replace("_", "").lower() != other.replace("_", "").lower()
                    or renames.get(value, other) != other
                    or reverse.get(other, value) != value
                ):
                    return False
                renames[value] = other
                reverse[other] = value
            elif not _equivalent(
                value,
                other,
                allowed_names=allowed_names,
                renames=renames,
                reverse=reverse,
            ):
                return False
        return True
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _equivalent(
                a, b, allowed_names=allowed_names, renames=renames, reverse=reverse
            )
            for a, b in zip(left, right, strict=True)
        )
    return left == right


def validate_pair(task: PreferenceTask, family: Family, source: dict[str, Any]) -> None:
    """Require unchanged upstream interfaces and paired, style-only implementations."""
    if source["entry_point"] != task.entry_point:
        raise ValueError(f"Source entry point changed: {task.task_id}")
    canonical = target_function(
        source["prompt"] + source["canonical_solution"], task.entry_point
    )
    codes = [task.implementations[style] for style in STYLES[family]]
    if codes[0] == codes[1]:
        raise ValueError(f"Style references are identical: {task.task_id}")
    for code in codes:
        target = target_function(code, task.entry_point)
        if ast.dump(target.args) != ast.dump(canonical.args):
            raise ValueError(f"Original parameter signature changed: {task.task_id}")
        if (ast.dump(target.returns) if target.returns is not None else None) != (
            ast.dump(canonical.returns) if canonical.returns is not None else None
        ):
            # Return annotations are not part of naming style and should be kept.
            raise ValueError(f"Original return annotation changed: {task.task_id}")
    first = ast.parse(codes[0])
    allowed_names: set[str] = set()
    if family == "naming":
        for function in (
            node for node in ast.walk(first) if isinstance(node, ast.FunctionDef)
        ):
            allowed_names.update(
                node.id
                for node in ast.walk(function)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            )
            if function.name != task.entry_point:
                allowed_names.add(function.name)
                allowed_names.update(parameter_names(function))
        allowed_names.difference_update(parameter_names(canonical))
        allowed_names.discard(task.entry_point)
    if not _equivalent(
        first, ast.parse(codes[1]), allowed_names=allowed_names, renames={}, reverse={}
    ):
        raise ValueError(
            f"Reference pair changes more than the requested style: {task.task_id}"
        )


def load_bundles(
    root: Path,
) -> tuple[list[PreferenceBundle], dict[DatasetName, LoadedDataset]]:
    """Read pinned local sources and exclude the original study's reserved tasks."""
    for filename in ("HumanEvalPlus-v0.1.10.jsonl", "MbppPlus-v0.2.0.jsonl"):
        if not (root / "data/evalplus" / filename).is_file():
            raise FileNotFoundError(
                "Existing full EvalPlus caches are required; no download is attempted"
            )
    names: tuple[DatasetName, ...] = ("humaneval", "mbpp")
    datasets: dict[DatasetName, LoadedDataset] = {
        name: load_dataset(name, root / "data/evalplus") for name in names
    }
    reserved = set(
        json.loads((root / "manifests/main_selection.json").read_text())["confirmation"]
    )
    bundles: list[PreferenceBundle] = []
    for family in STYLES:
        bundle = PreferenceBundle.model_validate_json(
            (root / f"fixtures/style_preferences/{family}.json").read_text()
        )
        if bundle.family != family:
            raise ValueError("Bundle filename disagrees with family")
        for task in bundle.tasks:
            if task.task_id in reserved:
                raise ValueError(
                    f"Original confirmation task must remain unused: {task.task_id}"
                )
            validate_pair(
                task, bundle.family, datasets[task.dataset].tasks[task.task_id]
            )
        bundles.append(bundle)
    pilots = {
        task.task_id
        for bundle in bundles
        for task in bundle.tasks
        if task.split == "pilot"
    }
    study = {
        task.task_id
        for bundle in bundles
        for task in bundle.tasks
        if task.split == "study"
    }
    if pilots & study:
        raise ValueError(
            "Pilot and study tasks must be disjoint across both preference families"
        )
    return bundles, datasets


def visible_task(task: PreferenceTask, source: dict[str, Any]) -> dict[str, Any]:
    """Project only upstream public text, API, and independently supplied contract.

    MBPP's canonical function signature is public scaffolding, not its body.
    HumanEval's original prompt is retained verbatim. Reference implementations,
    canonical bodies, private inputs, and oracle outcomes never enter this packet.
    """
    canonical = target_function(
        source["prompt"] + source["canonical_solution"], task.entry_point
    )
    stub = source["prompt"]
    if task.dataset == "mbpp":
        returns = (
            "" if canonical.returns is None else " -> " + ast.unparse(canonical.returns)
        )
        stub += f"\n\ndef {task.entry_point}({ast.unparse(canonical.args)}){returns}:\n"
    stub += "    # TODO\n    raise NotImplementedError\n"
    return {
        "task_id": task.task_id,
        "entry_point": task.entry_point,
        "original_parameters": parameter_names(canonical),
        "stub": stub,
        "input_contract": source.get("contract", ""),
    }


def task_manifest(
    bundle: PreferenceBundle, datasets: dict[DatasetName, LoadedDataset]
) -> dict[str, Any]:
    """Attach exact source/reference hashes and provenance, with no hidden inputs."""
    rows = []
    for task in bundle.tasks:
        dataset = datasets[task.dataset]
        source = dataset.tasks[task.task_id]
        rows.append(
            {
                "task_id": task.task_id,
                "dataset": task.dataset,
                "split": task.split,
                "selection_rationale": task.selection_rationale,
                "style_opportunities": task.style_opportunities,
                "notes": task.notes,
                "visible": visible_task(task, source),
                "source_prompt_sha256": digest(source["prompt"]),
                "reference_sha256": {
                    style: digest(code) for style, code in task.implementations.items()
                },
                "suite_counts": {
                    suite: len(source[f"{suite}_input"]) for suite in ("base", "plus")
                },
            }
        )
    return {
        "family": bundle.family,
        "styles": bundle.styles,
        "sources": {name: dataset.provenance for name, dataset in datasets.items()},
        "tasks": rows,
    }
