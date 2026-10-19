"""Source-faithful task packets for the 72-task list-style expansion.

Dataset text and reference code are data. This module never executes them.
Public packets contain original statements and signature-only stubs; solutions
and hidden cases remain in the separate private scoring records.
"""

import ast
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evals.inspect_mode_a import digest, json_text
from grader_power.datasets import apps_source, mbpp_full
from grader_power.datasets.task_filter import load_screening_pool
from grader_power.main_tasks import source_contract

BANKS = {
    "evalplus": ("task_expansion_v1", "runs/task-bank-expansion/20261017-v1"),
    "mbpp_full": ("mbpp_full_v1", "runs/mbpp-full-audit/20261017-v1-offline"),
    "apps": ("apps_v1", "runs/apps-audit/20261017-v1"),
}
README = "# Sequence utilities\n\nUtilities for working with sequences.\n"
MARKERS = ("--- README.md ---", "--- src/task.py ---", "--- AGENTS.md ---")


def target_function(code: str, entry: str) -> ast.FunctionDef:
    """Find exactly one top-level function without evaluating annotations/defaults."""
    targets = [
        node
        for node in ast.parse(code).body
        if isinstance(node, ast.FunctionDef) and node.name == entry
    ]
    if len(targets) != 1:
        raise ValueError(f"Expected one target function: {entry}")
    return targets[0]


def signature_stub(code: str, entry: str) -> str:
    """Retain the complete interface without implementation code or imports."""
    target = target_function(code, entry)
    if target.decorator_list:
        raise ValueError("Decorated source interface needs explicit review")
    signature = f"def {target.name}({ast.unparse(target.args)})"
    if target.returns is not None:
        signature += " -> " + ast.unparse(target.returns)
    # The admitted MBPP/APPS signatures have no annotation/default dependencies.
    # Refuse future additions rather than copying implementation imports/hints.
    if any(
        isinstance(n, ast.Name)
        for a in (
            *target.args.defaults,
            *[v for v in target.args.kw_defaults if v is not None],
        )
        for n in ast.walk(a)
    ):
        raise ValueError("Nonliteral default requires explicit interface review")
    stub = signature + ":\n    # TODO\n    raise NotImplementedError\n"
    if mbpp_full.signature(stub, entry) != mbpp_full.signature(code, entry):
        raise ValueError("Generated stub changed the source function interface")
    return stub


def reference_evidence(
    root: Path, bank: str, reference: dict[str, Any]
) -> dict[str, str]:
    """Verify both saved reference executions against their original audit hashes."""
    folder, run = BANKS[bank]
    verification = json.loads(
        (root / "fixtures" / folder / "verification.json").read_text()
    )
    bindings = verification["bindings"]
    ref_path = f"fixtures/{folder}/references.json"
    if digest((root / ref_path).read_bytes()) != bindings[ref_path]:
        raise ValueError("Reference bank changed after its verification")
    result = {}
    for style, code in reference["implementations"].items():
        stem = (
            reference["task_id"].lower().replace("/", "-")
            if bank == "evalplus"
            else str(reference["source_id"])
        )
        name = f"{run}/outcomes/{stem}-{style}.json"
        payload = (root / name).read_bytes()
        evidence = json.loads(payload)
        expected = bindings[name]
        if digest(payload) != expected:
            raise ValueError(f"Reference execution changed: {name}")
        code_hash = evidence.get("solution_sha256", evidence.get("code_sha256"))
        passed = evidence.get("full_correct", evidence.get("passed"))
        if code_hash != digest(code) or passed is not True:
            raise ValueError(
                f"Reference execution does not verify current code: {name}"
            )
        result[name] = expected
    result[ref_path] = bindings[ref_path]
    return result


def load_tasks(root: Path) -> list[dict[str, Any]]:
    """Build the exact 14 EvalPlus, 24 MBPP and 34 APPS admitted task packets."""
    references = {
        bank: [
            r
            for r in json.loads(
                (root / "fixtures" / folder / "references.json").read_text()
            )
            if r["hold"] is None
        ]
        for bank, (folder, _) in BANKS.items()
    }
    if {bank: len(rows) for bank, rows in references.items()} != {
        "evalplus": 14,
        "mbpp_full": 24,
        "apps": 34,
    }:
        raise ValueError("The approved 72-task composition changed")
    datasets = load_screening_pool(
        root / "data/evalplus", root / "manifests/evalplus_source.json"
    )
    evalplus = {
        key: value for dataset in datasets for key, value in dataset.tasks.items()
    }
    mbpp = mbpp_full.load(root)
    apps = apps_source.load_selected(
        root, "train", {r["source_id"] for r in references["apps"]}
    )
    tasks = []
    for bank, rows in references.items():
        for ref in rows:
            entry = ref["entry_point"]
            if bank == "evalplus":
                source = evalplus[ref["task_id"]]
                statement = source["prompt"]
                if ref["task_id"].startswith("HumanEval/"):
                    stub = statement + "    # TODO\n    raise NotImplementedError\n"
                    canonical = statement + source["canonical_solution"]
                else:
                    canonical = source["canonical_solution"]
                    stub = statement + "\n\n" + signature_stub(canonical, entry)
                readme = README
                contract = source_contract(source)
            else:
                source = (
                    mbpp[ref["source_id"]]
                    if bank == "mbpp_full"
                    else apps[ref["source_id"]]
                )
                statement = (
                    source["text"] if bank == "mbpp_full" else source["question"]
                )
                if statement != ref["original_prompt"]:
                    raise ValueError(
                        "Saved original problem text differs from pinned source"
                    )
                if (
                    digest(json.dumps(source, sort_keys=True))
                    != ref["source_record_sha256"]
                ):
                    raise ValueError(
                        "Selected source record differs from the audited record"
                    )
                canonical = (
                    source["code"]
                    if bank == "mbpp_full"
                    else json.loads(source["solutions"])[
                        ref["canonical_solution_index"]
                    ]
                )
                interface = canonical
                if bank == "apps":
                    if source["starter_code"] != ref["original_starter_code"]:
                        raise ValueError("APPS starter code drifted")
                    interface = source["starter_code"].rstrip() + "\n    pass\n"
                    # Upstream answers sometimes rename positional parameters.
                    # Preserve the public starter, which both audited variants use.
                    if any(
                        mbpp_full.signature(interface, entry)
                        != mbpp_full.signature(code, entry)
                        for code in ref["implementations"].values()
                    ):
                        raise ValueError(
                            "APPS starter and audited variant signatures differ"
                        )
                    canonical = interface
                stub = signature_stub(interface, entry)
                readme = README + "\n## Task\n\n" + statement
                if bank == "mbpp_full":
                    # Match the original MBPP+ presentation: one source example
                    # disambiguates return shape and parameter meaning. Remaining
                    # supplied assertions are withheld from the prompt.
                    readme += (
                        "\n\n## Example\n\n```python\n"
                        + source["test_list"][0]
                        + "\n```\n"
                    )
                contract = (
                    readme[len(README) :]
                    + "\n\nPython function interface:\n"
                    + stub.split("    # TODO", 1)[0]
                )
            if mbpp_full.signature(stub, entry) != mbpp_full.signature(
                canonical, entry
            ):
                raise ValueError(f"Public signature changed: {ref['task_id']}")
            if any(marker in statement or marker in stub for marker in MARKERS):
                raise ValueError("Source text collides with prompt section boundaries")
            target = target_function(stub, entry)
            if any(
                isinstance(n, (ast.For, ast.ListComp, ast.While))
                for n in ast.walk(target)
            ):
                raise ValueError("Implementation accidentally included in public stub")
            tasks.append(
                {
                    "task_id": ref["task_id"],
                    "dataset": bank,
                    "entry_point": entry,
                    "family": ref["family"],
                    "transformation": ref.get(
                        "transformation",
                        "Construct the returned list as specified in the supplied problem statement.",
                    ),
                    "original_statement": statement,
                    "readme": readme,
                    "stub": stub,
                    "input_contract": contract,
                    "source": source,
                    "source_record_sha256": digest(json.dumps(source, sort_keys=True)),
                    "implementations": ref["implementations"],
                    "reference_evidence": reference_evidence(root, bank, ref),
                    "qualifications": ref.get("notes", ref.get("review_notes", [])),
                    "public_test_indices": [0] if bank == "mbpp_full" else [],
                }
            )
    ids = [r["task_id"] for r in tasks]
    selection = json.loads((root / "manifests/main_selection.json").read_text())
    if len(set(ids)) != 72 or set(ids) & set(
        selection["exploratory"] + selection["confirmation"]
    ):
        raise ValueError("New tasks overlap an original or locked task")
    return sorted(tasks, key=lambda t: t["task_id"])


def replace_task(
    messages: list[dict[str, str]], task: dict[str, Any]
) -> list[dict[str, str]]:
    """Replace only task text; retain the exact historical instruction/policy suffix."""
    if len(messages) != 1 or messages[0]["role"] != "user":
        raise ValueError("Unexpected parent prompt structure")
    text = messages[0]["content"]
    if any(text.count(marker) != 1 for marker in MARKERS):
        raise ValueError("Parent prompt section markers changed")
    prefix = text.split(MARKERS[0], 1)[0]
    suffix = text.split(MARKERS[2], 1)[1]
    result = copy.deepcopy(messages)
    result[0]["content"] = (
        prefix
        + MARKERS[0]
        + "\n"
        + task["readme"]
        + "\n\n"
        + MARKERS[1]
        + "\n"
        + task["stub"]
        + "\n\n"
        + MARKERS[2]
        + suffix
    )
    return result


def public_inventory(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize source coverage and every visible packet without leaking solutions."""
    return {
        "tasks": len(tasks),
        "by_dataset": dict(Counter(t["dataset"] for t in tasks)),
        "records": [
            {
                k: t[k]
                for k in (
                    "task_id",
                    "dataset",
                    "entry_point",
                    "family",
                    "original_statement",
                    "readme",
                    "stub",
                    "public_test_indices",
                    "source_record_sha256",
                    "qualifications",
                )
            }
            | {
                "public_packet_sha256": digest(
                    json_text({k: t[k] for k in ("readme", "stub")})
                )
            }
            for t in tasks
        ],
    }
