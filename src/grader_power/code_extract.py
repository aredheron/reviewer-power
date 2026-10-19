"""Deterministic extraction of complete Python files/functions; never execution.

Accept raw Python verbatim, or exactly one column-zero triple-backtick fence
tagged `python`, `py`, or untagged. Prose may surround that fence. Multiple
fences (even identical or non-Python ones), malformed fences, patches, JSON
wrappers and body-only completions are not combined or repaired. Fence contents
retain their original whitespace and line endings. Raw text is always retained.
"""

import ast
import re
from collections.abc import Iterator
from typing import Literal

from grader_power.schemas import ExtractionResult, validate_identifier


def scope_nodes(node: ast.AST) -> Iterator[ast.AST]:
    """Walk possible outer-scope bindings without entering nested definition bodies.

    Defaults, decorators, bases and annotations may bind in the enclosing scope.
    Comprehension iteration variables are local, but their assignment expressions
    can bind outside the comprehension. This is conservative syntax inspection,
    not evaluation of annotations, generator timing or conditional execution.
    """
    yield node
    children: list[ast.AST]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        children = [node.args, *node.decorator_list]
        if node.returns is not None:
            children.append(node.returns)
    elif isinstance(node, ast.Lambda):
        children = [node.args]
    elif isinstance(node, ast.ClassDef):
        children = [*node.bases, *node.keywords, *node.decorator_list]
    elif isinstance(node, ast.arguments):
        children = [*node.defaults]
        children.extend(value for value in node.kw_defaults if value is not None)
        parameters = [*node.posonlyargs, *node.args, *node.kwonlyargs]
        if node.vararg is not None:
            parameters.append(node.vararg)
        if node.kwarg is not None:
            parameters.append(node.kwarg)
        children.extend(
            parameter.annotation
            for parameter in parameters
            if parameter.annotation is not None
        )
    elif isinstance(node, ast.comprehension):
        # Retain the comprehension node (including is_async), but not its local
        # iteration-variable bindings. Target index expressions can still bind.
        yield from _comprehension_target_nodes(node.target)
        children = [node.iter, *node.ifs]
    else:
        children = list(ast.iter_child_nodes(node))
    for child in children:
        yield from scope_nodes(child)


def _comprehension_target_nodes(target: ast.expr) -> Iterator[ast.AST]:
    """Inspect target expressions while excluding local iteration-variable stores."""
    if isinstance(target, ast.Name):
        return
    if isinstance(target, (ast.Tuple, ast.List)):
        for element in target.elts:
            yield from _comprehension_target_nodes(element)
    elif isinstance(target, ast.Starred):
        yield from _comprehension_target_nodes(target.value)
    else:
        yield from scope_nodes(target)


def binds_name(node: ast.AST, name: str) -> bool:
    """Recognize explicit name bindings/deletions, including non-Name AST fields."""
    if isinstance(node, ast.Name):
        return node.id == name and isinstance(node.ctx, (ast.Store, ast.Del))
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.alias):
        return (node.asname or node.name.split(".")[0]) in (name, "*")
    if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
        return node.name == name
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    return isinstance(node, ast.arg) and node.arg == name


def find_target(
    tree: ast.Module, entry_point: str
) -> tuple[ast.FunctionDef | ast.AsyncFunctionDef | None, str]:
    """Require one unconditional module-level definition and no static rebinding.

    Dynamic namespace manipulation cannot be resolved by this static check.
    Nested methods/functions do not satisfy the module entry-point contract.
    """
    nodes = list(scope_nodes(tree))
    definitions = [
        node
        for node in nodes
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == entry_point
    ]
    if not definitions:
        return None, "target_missing"
    rebound = any(
        binds_name(node, entry_point) for node in nodes if node not in definitions
    )
    if len(definitions) != 1 or definitions[0] not in tree.body or rebound:
        return None, "target_ambiguous"
    return definitions[0], "ok"


def extract_code(raw_response: str, entry_point: str) -> ExtractionResult:
    """Select source and validate target identity using AST parsing only.

    Raises TypeError/ValueError for invalid API inputs. Response failures are
    returned as explicit statuses. No sanitizer, compile check, or execution is
    performed; parse success alone does not imply compile validity/correctness.
    """
    if not isinstance(raw_response, str):
        raise TypeError("raw_response must be a string")
    validate_identifier(entry_point)

    lines = raw_response.splitlines(keepends=True)
    fences = [i for i, line in enumerate(lines) if line.lstrip().startswith("```")]
    # A valid Python file can contain fence-looking lines in a string literal.
    # Prefer that complete file over interpreting its contents as Markdown.
    try:
        ast.parse(raw_response)
    except (SyntaxError, ValueError, RecursionError):
        pass
    else:
        fences = []
    code = raw_response
    output_format: Literal["raw_python", "fenced_python"] = "raw_python"
    if fences:
        if len(fences) > 2:
            return ExtractionResult(
                raw_response,
                entry_point,
                None,
                None,
                "ambiguous_output",
                "Multiple code fences; no selection made",
            )
        if (
            len(fences) != 2
            or not re.fullmatch(r"```(?:python|py)?[ \t]*(?:\r?\n)?", lines[fences[0]])
            or not re.fullmatch(r"```[ \t]*(?:\r?\n)?", lines[fences[1]])
        ):
            return ExtractionResult(
                raw_response,
                entry_point,
                None,
                None,
                "format_error",
                "Unsupported or unclosed code fence",
            )
        code = "".join(lines[fences[0] + 1 : fences[1]])
        output_format = "fenced_python"
        outside = "".join(lines[: fences[0]] + lines[fences[1] + 1 :])
        if re.search(r"(?m)^\s*(?:async\s+)?def\s+", outside):
            return ExtractionResult(
                raw_response,
                entry_point,
                None,
                None,
                "ambiguous_output",
                "Function definition outside code fence",
            )
    if not code.strip():
        return ExtractionResult(
            raw_response,
            entry_point,
            code,
            output_format,
            "format_error",
            "Empty source",
        )
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError) as error:
        return ExtractionResult(
            raw_response,
            entry_point,
            code,
            output_format,
            "parse_error",
            f"AST parsing failed: {type(error).__name__}",
        )
    target, status = find_target(tree, entry_point)
    if target is None:
        return ExtractionResult(
            raw_response,
            entry_point,
            code,
            output_format,
            "target_missing" if status == "target_missing" else "target_ambiguous",
            "Require one unambiguous module-level target function",
        )
    return ExtractionResult(
        raw_response,
        entry_point,
        code,
        output_format,
        "ok",
        "Exact source extracted; target found",
    )
