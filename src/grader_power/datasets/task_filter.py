"""Conservative, static EvalPlus triage; never execute or approve a task.

Rules inspect contracts and AST shapes, not grader outcomes. Source syntax is
reported separately and never used as a positive selection feature. A priority
is a review order, not an eligibility decision or a semantic guarantee.
"""

import ast
import hashlib
import json
import re
import warnings
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from grader_power.datasets.evalplus_loader import (
    EVALPLUS_VERSION,
    SOURCES,
    LoadedDataset,
    load_dataset,
)

SCREEN_VERSION = 1
SUPPORTED_IMPORTS = frozenset(
    [
        "math",
        "itertools",
        "collections",
        "functools",
        "operator",
        "re",
        "string",
        "bisect",
        "heapq",
        "typing",
        "statistics",
        "fractions",
        "decimal",
        "copy",
        "sys",
    ]
)
NONSTDLIB_IMPORTS = frozenset({"numpy", "scipy", "pandas", "sympy"})
RISK_IMPORTS = frozenset(
    [
        "random",
        "secrets",
        "time",
        "datetime",
        "os",
        "pathlib",
        "io",
        "socket",
        "requests",
        "subprocess",
        "urllib",
    ]
)
# These names permit structural inspection, not a claim about runtime purity.
INSPECTABLE_CALLS = frozenset(
    [
        "abs",
        "all",
        "any",
        "bool",
        "chr",
        "dict",
        "enumerate",
        "filter",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "ord",
        "range",
        "reversed",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
        "type",
    ]
)
INSPECTABLE_METHODS = frozenset(
    [
        "append",
        "extend",
        "count",
        "index",
        "split",
        "join",
        "strip",
        "lower",
        "upper",
        "startswith",
        "endswith",
        "isdigit",
        "isalpha",
        "islower",
        "isupper",
        "replace",
        "keys",
        "values",
        "items",
        "get",
    ]
)
STYLE_PATTERNS = {
    "comprehension_wording": r"\bcomprehensions?\b",
    "loop_wording": r"\b(?:for[- ]loops?|while[- ]loops?|loops?|looping)\b",
    "functional_wording": r"\b(?:map|filter)\s*\(|\blambda\b",
    "construction_example": r"\bfor\s+\w+\s+in\b|\.append\s*\(",
}
FAMILY_PATTERNS = {
    "filtering": r"\b(?:filter|remove|retain|discard|select|keep|extract)\w*\b",
    "elementwise_mapping": r"\b(?:square|cube|multiply|double|convert|uppercase|lowercase|increment|decrement)\w*\b|\b(?:absolute difference|linear transform)\b",
    "conditional_transformation": r"\b(?:replace|substitute)\w*\b|\bif\b.*\botherwise\b",
    "record_extraction": r"\b(?:nth|kth|first|second|last|rear)\b.*\b(?:each|tuples)\b|\bextract\b.*\b(?:field|column)\w*\b",
    "indexing": r"\b(?:indices|indexes|enumerate)\b",
    "one_level_flattening": r"\bflatten\w*\b",
}
REASON_DESCRIPTIONS = {
    "list_output_evidence": "All explicit return shapes resolve to lists in the limited AST analysis.",
    "non_list_output": "All resolved return shapes are non-list; this is static triage only.",
    "static_unknown": "Unsupported syntax, unresolved values/calls, or control/data flow needs review.",
    "ambiguous_target": "Mixed return shapes or multiple possible output constructions need review.",
    "contract_output_conflict": "The prose appears to name a non-list result while source return shapes are lists.",
    "style_implied": "Statement wording or visible code may prime an implementation style.",
    "requires_nonstdlib": "A known third-party import occurs in source code.",
    "unsupported_import": "An import is outside the explicitly supported static allowlist.",
    "side_effects": "Potential external effects or input mutation occur; intrinsic necessity is unverified.",
    "determinism_review": "Ordering or external-state operations need determinism review.",
    "complex_control_flow": "Recursion, state updates, or complex control flow need dual-style review.",
    "family_hint": "Contract wording suggests a provisional transformation family.",
    "family_unknown": "No sufficiently specific family hint was found in the statement.",
    "oracle_review": "Recorded upstream input anomaly requires oracle review; no repair performed.",
    "parse_error": "The source cannot be parsed using the source-specific assembly rule.",
    "parse_warning": "The Python parser reported a source warning; inspect the unchanged source.",
    "invalid_entry_point": "The entry point is missing, duplicated, async, or otherwise unsupported.",
}
REQUIRED_REVIEW = [
    "Verify the original behavioral contract, output type, ordering, and edge cases.",
    "Verify determinism, boundedness, supported dependencies, and externally visible effects.",
    "Verify the unchanged base and plus oracle in an isolated Docker sandbox.",
    "Independently implement both styles and verify equivalent behavior on the full oracle.",
    "Human review must designate one scoring region and judge both styles idiomatic.",
    "Candidate membership and any prompt edits remain provisional pending human review.",
]


def _nodes(function: ast.FunctionDef) -> list[ast.AST]:
    """Return target-scope nodes without conflating nested helper return values."""
    result: list[ast.AST] = []

    def visit(node: ast.AST) -> None:
        """Collect target nodes while keeping nested definition bodies opaque."""
        result.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    for statement in function.body:
        visit(statement)
    return result


def _call_name(node: ast.expr) -> str:
    """Describe a call structurally, without evaluating attribute access."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_call_name(node.value)}.{node.attr}"
    return "<dynamic>"


def _contract_prose(prompt: str, entry: str) -> str:
    """Extract prose without using identifiers or doctests as family evidence."""
    try:
        with warnings.catch_warnings(record=True):
            tree = ast.parse(prompt)
    except (SyntaxError, ValueError):
        prose = prompt
    else:
        target = next(
            (
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == entry
            ),
            None,
        )
        prose = ast.get_docstring(target or tree) or ""
    return prose.split(">>>")[0].split("assert ")[0]


def _contract_excerpt(prose: str, matches: dict[str, re.Match[str]]) -> str:
    """Bound the visible prose excerpt, centering long statements on family cues."""
    start = max(0, min((match.start() for match in matches.values()), default=0) - 100)
    text = " ".join(prose[start:].split())
    if start:
        text = "… " + text.partition(" ")[2]
    return text if len(text) <= 300 else text[:300].rsplit(" ", 1)[0] + " …"


def _return_shapes(
    function: ast.FunctionDef, nodes: list[ast.AST]
) -> tuple[set[str], list[str]]:
    """Infer only coarse list/non-list shapes, retaining unresolved writes.

    This is deliberately flow-insensitive: conflicting assignments yield unknown
    or mixed shapes. It is evidence about source syntax, not a type proof.
    """
    bindings: dict[str, list[ast.expr | None]] = {}
    for arg in [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]:
        bindings[arg.arg] = [None]
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    bindings.setdefault(target.id, []).append(node.value)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for child in ast.walk(target):
                        if isinstance(child, ast.Name):
                            bindings.setdefault(child.id, []).append(None)
        elif isinstance(node, (ast.For, ast.AugAssign)):
            for child in ast.walk(node.target):
                if isinstance(child, ast.Name):
                    bindings.setdefault(child.id, []).append(None)

    def shape(value: ast.expr | None, seen: frozenset[str] = frozenset()) -> set[str]:
        """Infer coarse syntax shapes, retaining unknown or cyclic local bindings."""
        if value is None:
            return {"non_list"}
        if isinstance(value, (ast.List, ast.ListComp)):
            return {"list"}
        if isinstance(
            value,
            (
                ast.Constant,
                ast.Tuple,
                ast.Dict,
                ast.Set,
                ast.Compare,
                ast.JoinedStr,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
            ),
        ):
            return {"non_list"}
        if isinstance(value, ast.Name):
            if value.id in seen or value.id not in bindings:
                return {"unknown"}
            return set().union(
                *(
                    {"unknown"} if item is None else shape(item, seen | {value.id})
                    for item in bindings[value.id]
                )
            )
        if isinstance(value, ast.IfExp):
            return shape(value.body, seen) | shape(value.orelse, seen)
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            if value.func.id in bindings:
                return {"unknown"}
            if value.func.id in {"list", "sorted"}:
                return {"list"}
            if value.func.id in {
                "tuple",
                "set",
                "dict",
                "str",
                "int",
                "float",
                "bool",
                "len",
                "any",
                "all",
            }:
                return {"non_list"}
        if isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            left, right = shape(value.left, seen), shape(value.right, seen)
            if left == right == {"list"}:
                return {"list"}
        return {"unknown"}

    returns = [node for node in nodes if isinstance(node, ast.Return)]
    shapes = set().union(*(shape(node.value) for node in returns))
    # A trailing return is the only fall-through pattern this small screen handles.
    if not function.body or not isinstance(function.body[-1], (ast.Return, ast.Raise)):
        shapes.add("unknown")
    targets = sorted(
        {
            node.value.id
            if isinstance(node.value, ast.Name)
            else f"expression at line {node.lineno}"
            for node in returns
            if shape(node.value) != {"non_list"}
            and not (isinstance(node.value, ast.List) and not node.value.elts)
        }
    )
    return shapes or {"unknown"}, targets


def screen_task(task: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """Screen one loaded record without execution, mutation, or oracle disclosure.

    Evidence contains rule summaries and AST line numbers, never solution text or
    test values. Every task receives a record, including parse/unsupported cases.
    Human review, oracle execution, and dual-style feasibility remain unverified.
    """
    reasons: dict[str, list[str]] = {}

    def flag(code: str, evidence: str) -> None:
        """Accumulate review evidence without replacing earlier reasons for a flag."""
        reasons.setdefault(code, []).append(evidence)

    prompt = task["prompt"]
    solution = task["canonical_solution"]
    entry = task["entry_point"]
    prose = _contract_prose(prompt, entry)
    family_matches = {
        name: match
        for name, pattern in FAMILY_PATTERNS.items()
        if (match := re.search(pattern, prose, re.IGNORECASE | re.DOTALL))
    }
    family_hints = sorted(family_matches)
    if "record_extraction" in family_hints and "filtering" in family_hints:
        family_hints.remove("filtering")
    if "one_level_flattening" in family_hints and not re.search(
        r"one[- ]level|list of lists", prose, re.IGNORECASE
    ):
        family_hints.remove("one_level_flattening")
        flag(
            "static_unknown",
            "Flattening depth is unspecified by the contract keyword hint.",
        )
    for name, pattern in STYLE_PATTERNS.items():
        if re.search(pattern, prompt, re.IGNORECASE):
            flag(
                "style_implied",
                f"Prompt matches {name}; wording needs human inspection.",
            )
    if family_hints:
        for family in family_hints:
            match = family_matches[family]
            flag(
                "family_hint",
                f"Contract prose characters {match.start()}:{match.end()} suggest {family}.",
            )
    else:
        flag(
            "family_unknown",
            "No supported transformation-family keyword in contract prose.",
        )
    issues = [
        issue
        for issue in provenance["source_issues"]
        if issue["task_id"] == task["task_id"]
    ]
    for issue in issues:
        flag("oracle_review", f"{issue['field']}: {issue['issue']}")
    if task["task_id"] == "Mbpp/793" and task.get("plus_input") == {} and not issues:
        flag(
            "oracle_review",
            "plus_input: empty object instead of input list; retained unchanged",
        )

    # HumanEval provides a completion body; MBPP provides a complete module.
    code = prompt + solution if provenance["name"] == "HumanEval+" else solution
    source_style: dict[str, int] = {}
    shapes: set[str] = {"unknown"}
    try:
        with warnings.catch_warnings(record=True) as parser_warnings:
            warnings.simplefilter("always")
            tree = ast.parse(code)
    except (SyntaxError, ValueError):
        flag("parse_error", "AST parsing failed; source left unchanged.")
    else:
        if parser_warnings:
            flag("parse_warning", "Python parser warning; source left unchanged.")
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == entry
        ]
        if not entry.isidentifier() or len(functions) != 1:
            flag(
                "invalid_entry_point",
                "Expected exactly one top-level synchronous entry function.",
            )
        else:
            function = functions[0]
            nodes = _nodes(function)
            source_style = {
                "list_comprehensions": sum(isinstance(n, ast.ListComp) for n in nodes),
                "for_loops": sum(isinstance(n, ast.For) for n in nodes),
                "append_calls": sum(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "append"
                    for n in nodes
                ),
                "alternative_comprehensions": sum(
                    isinstance(n, (ast.SetComp, ast.DictComp, ast.GeneratorExp))
                    for n in nodes
                ),
                "map_filter_calls": sum(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id in {"map", "filter"}
                    for n in nodes
                ),
            }
            shapes, targets = _return_shapes(function, nodes)
            if shapes == {"list"}:
                flag(
                    "list_output_evidence",
                    "Return shapes resolve to list constructors/literals or their local bindings.",
                )
                if re.search(
                    r"\b(?:returns?|returning|to)\s+(?:(?:a|an|the|nested)\s+)*(?:dictionary|dict|tuple|set|boolean|integer|string)\b",
                    prose,
                    re.IGNORECASE,
                ):
                    flag(
                        "contract_output_conflict",
                        "Prose names a non-list result after return/to; inspect the original contract and oracle.",
                    )
                if re.search(
                    r"list of (?:two|three|\d+) lists|first list.*second",
                    prose,
                    re.IGNORECASE | re.DOTALL,
                ):
                    flag(
                        "ambiguous_target",
                        "Contract describes multiple output lists; a single target region is unresolved.",
                    )
            elif shapes == {"non_list"}:
                flag(
                    "non_list_output",
                    "Return shapes resolve exclusively to non-list syntax/builtins.",
                )
            else:
                flag("static_unknown", "Return shapes: " + ", ".join(sorted(shapes)))
            if len(shapes) > 1 or len(targets) > 1:
                flag(
                    "ambiguous_target",
                    "Multiple return shapes or candidate constructions: "
                    + ", ".join(targets),
                )
            collections = {
                target.id
                for node in nodes
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
                if isinstance(node.value, (ast.List, ast.ListComp))
                or (
                    isinstance(node.value, ast.Call)
                    and _call_name(node.value.func) in {"list", "sorted"}
                )
            }
            if len(collections) > 1:
                flag(
                    "ambiguous_target",
                    "Multiple locally constructed list bindings; identify the output-building region.",
                )
            if any(
                isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Store)
                and node.id in INSPECTABLE_CALLS
                for node in ast.walk(tree)
            ) or any(
                arg.arg in INSPECTABLE_CALLS
                for arg in ast.walk(function)
                if isinstance(arg, ast.arg)
            ):
                flag(
                    "static_unknown",
                    "Builtin name is rebound; call/return inference is only syntactic.",
                )
            if function.decorator_list:
                flag(
                    "static_unknown",
                    "Decorated entry point; runtime wrapper behavior is unverified.",
                )
            if any(
                isinstance(node, ast.Call)
                for value in [*function.args.defaults, *function.args.kw_defaults]
                if value is not None
                for node in ast.walk(value)
            ):
                flag(
                    "static_unknown",
                    "Call in entry-point default expression; definition-time behavior is unverified.",
                )
            for node in nodes:
                location = f"line {getattr(node, 'lineno', 0)}"
                if isinstance(
                    node,
                    (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef),
                ):
                    flag("static_unknown", f"Nested helper at {location}.")
                if isinstance(
                    node,
                    (
                        ast.While,
                        ast.Try,
                        ast.TryStar,
                        ast.With,
                        ast.Raise,
                        ast.Break,
                        ast.AugAssign,
                    ),
                ):
                    flag(
                        "complex_control_flow", f"{type(node).__name__} at {location}."
                    )
                if isinstance(node, (ast.Yield, ast.YieldFrom, ast.Await, ast.Match)):
                    flag(
                        "static_unknown",
                        f"Unsupported {type(node).__name__} at {location}.",
                    )
                if isinstance(node, (ast.Global, ast.Nonlocal, ast.Delete)):
                    flag("side_effects", f"{type(node).__name__} at {location}.")
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    targets = (
                        node.targets if isinstance(node, ast.Assign) else [node.target]
                    )
                    if any(
                        isinstance(target, (ast.Subscript, ast.Attribute))
                        for target in targets
                    ):
                        flag(
                            "side_effects",
                            f"Subscript/attribute write at {location}; check ownership and aliasing.",
                        )
                if isinstance(node, (ast.Set, ast.SetComp)):
                    flag(
                        "determinism_review",
                        f"Set construction at {location}; inspect output ordering.",
                    )
                if isinstance(node, ast.Call):
                    name = _call_name(node.func)
                    if name not in INSPECTABLE_CALLS and not (
                        isinstance(node.func, ast.Attribute)
                        and node.func.attr in INSPECTABLE_METHODS
                    ):
                        flag(
                            "static_unknown",
                            f"Call outside the static allowlist at {location}.",
                        )
                    if name == entry:
                        flag(
                            "complex_control_flow", f"Possible recursion at {location}."
                        )
                    if name in {
                        "eval",
                        "exec",
                        "open",
                        "print",
                        "input",
                        "__import__",
                        "setattr",
                        "delattr",
                    } or name.endswith(
                        (".sort", ".reverse", ".pop", ".remove", ".update", ".write")
                    ):
                        flag(
                            "side_effects",
                            f"Potential effectful call at {location}; inspect ownership and aliasing.",
                        )
                    if name == "set" or name.startswith(
                        ("random.", "time.", "secrets.", "datetime.")
                    ):
                        flag(
                            "determinism_review",
                            f"Ordering/external-state call at {location}.",
                        )
                    if isinstance(node.func, ast.Attribute) and node.func.attr in {
                        "append",
                        "extend",
                        "clear",
                        "insert",
                    }:
                        args = {
                            arg.arg
                            for arg in [
                                *function.args.posonlyargs,
                                *function.args.args,
                                *function.args.kwonlyargs,
                            ]
                        }
                        if (
                            isinstance(node.func.value, ast.Name)
                            and node.func.value.id in args
                        ):
                            flag(
                                "side_effects",
                                f"Input-parameter mutation at {location}.",
                            )
                        elif not (
                            isinstance(node.func.value, ast.Name)
                            and node.func.value.id in collections
                        ):
                            flag(
                                "side_effects",
                                f"Mutation of unresolved receiver at {location}; inspect aliasing.",
                            )
            # Helpers and module-level statements can affect behavior beyond the target.
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node is not function:
                    flag(
                        "static_unknown",
                        f"Additional module helper at line {node.lineno}.",
                    )
                elif not isinstance(
                    node, (ast.FunctionDef, ast.Import, ast.ImportFrom)
                ) and not (
                    isinstance(node, ast.Expr)
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)
                ):
                    flag(
                        "static_unknown",
                        f"Module-level {type(node).__name__} at line {node.lineno}.",
                    )
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                modules = (
                    [a.name.split(".")[0] for a in node.names]
                    if isinstance(node, ast.Import)
                    else [(node.module or "<relative>").split(".")[0]]
                )
                for module in modules:
                    if module in NONSTDLIB_IMPORTS:
                        flag(
                            "requires_nonstdlib",
                            f"Import {module} at line {node.lineno}.",
                        )
                    elif module not in SUPPORTED_IMPORTS or (
                        isinstance(node, ast.ImportFrom) and node.level
                    ):
                        flag(
                            "unsupported_import",
                            f"Import {module} at line {node.lineno} is outside the static allowlist.",
                        )
                    if module in RISK_IMPORTS:
                        flag(
                            "determinism_review",
                            f"Import {module}; external state/effects require review.",
                        )

    allowed = {"list_output_evidence", "family_hint"}
    if "oracle_review" in reasons:
        status = "manual_review"
    elif set(reasons) & {
        "non_list_output",
        "requires_nonstdlib",
        "parse_error",
        "invalid_entry_point",
    }:
        status = "deprioritized"
    elif shapes == {"list"} and family_hints and set(reasons) <= allowed:
        status = "priority_review"
    else:
        status = "manual_review"
    return {
        "schema_version": 1,
        "screen_version": SCREEN_VERSION,
        "task_id": task["task_id"],
        "entry_point": entry,
        "contract_excerpt": _contract_excerpt(prose, family_matches),
        "source": {
            key: provenance[key]
            for key in ("name", "release_version", "sha256", "evalplus_md5")
        },
        "screening_status": status,
        "reason_codes": sorted(reasons),
        "evidence": {key: sorted(set(reasons[key])) for key in sorted(reasons)},
        "provisional_families": family_hints if "list" in shapes else [],
        "source_style": source_style,
        "uncertainties": [
            *REQUIRED_REVIEW,
            *[REASON_DESCRIPTIONS[code] for code in sorted(set(reasons) - allowed)],
        ],
        "human_review_status": "not_reviewed",
        "oracle_status": "not_run",
        "dual_style_status": "not_verified",
    }


def load_screening_pool(cache_dir: Path, source_manifest: Path) -> list[LoadedDataset]:
    """Reuse the pinned loader only after verifying existing caches and provenance.

    Missing caches or changed bytes/IDs fail closed before screening. This command
    never downloads, modifies a source cache, or regenerates source provenance.
    """
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest["evalplus_version"] != EVALPLUS_VERSION:
        raise ValueError(
            "Source manifest EvalPlus version does not match the pinned loader"
        )
    expected = {item["name"]: item for item in manifest["datasets"]}
    if len(manifest["datasets"]) != len(SOURCES) or set(expected) != {
        s.name for s in SOURCES.values()
    }:
        raise ValueError(
            "Source manifest must contain exactly both full source releases"
        )
    for spec in SOURCES.values():
        info = expected[spec.name]
        filename = f"{spec.release_name}-{spec.release_version}.jsonl"
        if (
            info["release_version"] != spec.release_version
            or info["cache_file"] != filename
            or info["variant"] != "full"
        ):
            raise ValueError(f"{spec.name}: source release identity mismatch")
        path = cache_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path}; run scripts/load_evalplus.py separately"
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != info["sha256"]:
            raise ValueError(
                f"{spec.name}: cache SHA-256 differs from source manifest; do not repair automatically"
            )
    datasets = [load_dataset(name, cache_dir) for name in SOURCES]
    for dataset in datasets:
        if dataset.provenance != expected[dataset.provenance["name"]]:
            raise ValueError(
                f"{dataset.provenance['name']}: loaded provenance differs from source manifest"
            )
    return datasets


def screen_pool(
    datasets: list[LoadedDataset], source_manifest_sha256: str
) -> list[dict[str, Any]]:
    """Produce one stable record per original ID; reject duplicate/missing linkage."""
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for dataset in datasets:
        if sorted(dataset.tasks) != dataset.provenance["original_task_ids"]:
            raise ValueError("Loaded task IDs differ from source provenance")
        for task_id, task in sorted(dataset.tasks.items()):
            if task_id in seen or task["task_id"] != task_id:
                raise ValueError(f"Duplicate or inconsistent task ID: {task_id}")
            seen.add(task_id)
            record = screen_task(task, dataset.provenance)
            record["source_manifest_sha256"] = source_manifest_sha256
            record["evalplus_version"] = EVALPLUS_VERSION
            records.append(record)
    return sorted(records, key=lambda record: record["task_id"])


def render_review(records: list[dict[str, Any]]) -> str:
    """Render a compact, reproducible review queue with counts and explicit limits."""
    lines = [
        "# Provisional EvalPlus screening review",
        "",
        "Generated by `uv run --locked python scripts/screen_evalplus.py`. Do not edit this generated packet; change documented screening rules and regenerate.",
        "",
        f"Screen version: {SCREEN_VERSION}. Source manifest SHA-256: `{records[0]['source_manifest_sha256']}`."
        if records
        else "Empty source pool.",
        "",
        "Static triage only. No benchmark code, models, or oracles were executed. No task is human reviewed, accepted, or assigned to a pilot/main split. No grader effects were consulted; no prompt edits were made.",
        "",
        "## Method and interpretation",
        "",
        "The complete source pool is hash-checked against existing provenance before the pinned loader is used. HumanEval completion bodies are parsed with their original prompts; MBPP solutions are parsed as complete modules. AST return shapes and contract keyword hints prioritize review. Only simple list-shaped returns with a family hint and no additional flags enter priority_review. Everything else remains visible as manual_review or deprioritized; these are triage statuses, not eligibility decisions.",
        "",
        "Return inference is conservative and flow-insensitive. Dynamic dispatch, aliases, helper semantics, implicit returns, annotations, external state, and control-dependent types are not verified. Missing flags never establish safety or determinism. Family labels are provisional keyword hints and can be wrong. AST locations refer to the assembled source, without copying code or hidden inputs into the manifest.",
        "",
        "Canonical list comprehensions, loops, appends, and alternative comprehensions are descriptive counts only. No count contributes to positive prioritization. Source implementations can still influence negative hazard flags; a flagged canonical algorithm need not be intrinsically required by the contract. Review such cases independently before excluding them.",
        "",
        "## Source coverage",
        "",
        "| Source | Total | Priority review | Manual review | Deprioritized |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for source in sorted({r["source"]["name"] for r in records}):
        subset = [r for r in records if r["source"]["name"] == source]
        counts = Counter(r["screening_status"] for r in subset)
        lines.append(
            f"| {source} | {len(subset)} | {counts['priority_review']} | {counts['manual_review']} | {counts['deprioritized']} |"
        )
    lines += [
        "",
        f"Total: {len(records)} records / {len({r['task_id'] for r in records})} unique original IDs. No quota is imposed.",
        "",
        "## Counts by screening reason",
        "",
        "Counts overlap: a record may carry several reasons.",
        "",
        "| Reason | Tasks | Meaning |",
        "| --- | ---: | --- |",
    ]
    counts = Counter(code for r in records for code in r["reason_codes"])
    for code, description in sorted(REASON_DESCRIPTIONS.items()):
        lines.append(f"| `{code}` | {counts[code]} | {description} |")
    lines += [
        "",
        "`dual_style_failed`, `oracle_failed`, and `manual_reject` are reserved for later verified stages and are never inferred here.",
        "",
        "## Strongest static candidates",
        "",
        "All priority_review records are listed in original-ID order, without ranking by source style or forcing a quota. These are a review shortlist, not a pilot set.",
        "",
        "| Original ID / entry point | Why it appears suitable | What still needs checking |",
        "| --- | --- | --- |",
    ]
    for record in records:
        if record["screening_status"] == "priority_review":
            excerpt = record["contract_excerpt"].replace("|", "\\|").replace("`", "'")
            lines.append(
                f"| {record['task_id']} / `{record['entry_point']}` | {excerpt} — List-shaped returns; contract hints: {', '.join(record['provisional_families'])}. | Confirm family and one target; contract/order/edge cases, effects, and both idiomatic styles on full base + plus oracle; human QA. |"
            )
    lines += [
        "",
        "## Representative borderline cases",
        "",
        "Deterministic examples: first non-priority record with each flag in each source, plus Mbpp/793. Selection is for audit coverage, not a scientific sample.",
        "",
        "| Original ID / entry point | Evidence for concern | Review action |",
        "| --- | --- | --- |",
    ]
    selected: dict[str, dict[str, Any]] = {}
    for source in sorted({r["source"]["name"] for r in records}):
        for code in (
            "ambiguous_target",
            "contract_output_conflict",
            "static_unknown",
            "style_implied",
            "side_effects",
            "complex_control_flow",
            "non_list_output",
            "requires_nonstdlib",
            "unsupported_import",
            "determinism_review",
            "family_unknown",
            "oracle_review",
        ):
            match = next(
                (
                    r
                    for r in records
                    if r["source"]["name"] == source and code in r["reason_codes"]
                ),
                None,
            )
            if match:
                selected[match["task_id"]] = match
    for record in sorted(selected.values(), key=lambda r: r["task_id"]):
        flags = [
            code
            for code in record["reason_codes"]
            if code not in {"family_hint", "list_output_evidence", "family_unknown"}
        ]
        details = " ".join(record["evidence"][code][0] for code in flags)
        lines.append(
            f"| {record['task_id']} / `{record['entry_point']}` | {details} | Inspect flagged source evidence and contract; do not repair or approve automatically. |"
        )
    lines += [
        "",
        "## Source-style composition (descriptive only)",
        "",
        "Counts overlap and refer to the entry function, excluding nested helpers. They do not label the eventual scoring region.",
        "",
        "| Review status | Records | With ListComp | With For | With append | With map/filter |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for status in ("priority_review", "manual_review", "deprioritized"):
        subset = [r for r in records if r["screening_status"] == status]
        counts_style = [
            sum(r["source_style"].get(key, 0) > 0 for r in subset)
            for key in (
                "list_comprehensions",
                "for_loops",
                "append_calls",
                "map_filter_calls",
            )
        ]
        lines.append(
            f"| {status} | {len(subset)} | {' | '.join(map(str, counts_style))} |"
        )
    lines += [
        "",
        "## Required human decisions and prerequisites",
        "",
        "- Mbpp/793 retains the upstream empty-object plus_input anomaly unchanged and carries oracle_review, even if its output is deprioritized by another rule. Establish upstream intent and a valid full oracle before considering eligibility; no oracle failure or repair is claimed.",
        "- Inspect contracts and exact style-priming text. Decide whether a minimal semantics-preserving statement edit is justified; record exact edits separately. Current wording and candidate membership remain provisional.",
        "- A verified, pinned, isolated Docker runner is required for any future benchmark execution. This pass supplies no runtime correctness, determinism, or dual-style evidence.",
        "- Verify two independent reference implementations against unchanged full base and plus inputs, then designate the scoring region and perform human QA before freezing any task set.",
        "- Review upstream licenses and attribution before publishing adapted fixtures. Full solutions and hidden inputs remain only in ignored source caches.",
        "",
    ]
    return "\n".join(lines)


def write_text_atomic(path: Path, content: str) -> None:
    """Replace a generated text artifact atomically after successful serialization."""
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
