"""Conservative syntactic scoring of an explicitly contracted return construction.

This is not semantic/data-flow analysis. Unsupported shapes fail closed with a
separate region status. No model code, defaults, decorators or imports execute.
"""

import ast

from grader_power.code_extract import binds_name, find_target, scope_nodes
from grader_power.schemas import (
    ExtractionResult,
    FeatureKind,
    ScoreStatus,
    ScoringRegion,
    SourceSpan,
    StyleFeature,
    StyleLabel,
    StyleScore,
)


class _RegionFailure(Exception):
    """Internal control flow for a region that cannot be classified reliably."""

    def __init__(self, status: ScoreStatus, reason: str) -> None:
        """Carry a diagnostic status without inventing a style label."""
        super().__init__(reason)
        self.status: ScoreStatus = status


def _span(node: ast.expr | ast.stmt) -> SourceSpan:
    """Return a parsed node's source extent, never an inferred line range."""
    if node.end_lineno is None or node.end_col_offset is None:
        raise ValueError("AST node has no source extent")
    return SourceSpan(
        node.lineno, node.col_offset, node.end_lineno, node.end_col_offset
    )


def _feature(kind: FeatureKind, node: ast.expr | ast.stmt) -> StyleFeature:
    """Attach exact source evidence to a style feature."""
    return StyleFeature(kind, _span(node))


def _expression_features(expression: ast.expr) -> list[StyleFeature]:
    """Inspect output constructors, excluding iteration inputs and predicates.

    Calls in element expressions are opaque scalar computations. Unknown calls
    constructing the returned collection itself require adjudication. Syntactic
    map/filter names are recorded without claiming they resolve to builtins.
    """
    if isinstance(expression, ast.ListComp):
        return [_feature("list_comprehension", expression)] + _element_features(
            expression.elt
        )
    alternatives: dict[type[ast.expr], FeatureKind] = {
        ast.SetComp: "set_comprehension",
        ast.DictComp: "dict_comprehension",
        ast.GeneratorExp: "generator_expression",
    }
    if type(expression) in alternatives:
        return [_feature(alternatives[type(expression)], expression)]
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return [
            feature
            for element in expression.elts
            for feature in _element_features(element)
        ]
    if isinstance(expression, (ast.Constant, ast.Dict)):
        return []
    if isinstance(expression, ast.BinOp) and isinstance(expression.op, ast.Add):
        if not isinstance(
            expression.left, (ast.List, ast.ListComp, ast.BinOp)
        ) or not isinstance(expression.right, (ast.List, ast.ListComp, ast.BinOp)):
            raise _RegionFailure(
                "region_unsupported", "Cannot establish list concatenation operands"
            )
        return (
            [_feature("list_concatenation", expression)]
            + _expression_features(expression.left)
            + _expression_features(expression.right)
        )
    if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name):
        name = expression.func.id
        if name in ("map", "filter"):
            return [_feature(name, expression)]
        if name == "list" and not expression.keywords and len(expression.args) <= 1:
            return _expression_features(expression.args[0]) if expression.args else []
    raise _RegionFailure(
        "region_unsupported", "Opaque or unsupported output expression"
    )


def _element_features(expression: ast.expr) -> list[StyleFeature]:
    """Record explicit nested constructors, without following scalar/helper calls."""
    if isinstance(
        expression,
        (
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.List,
            ast.Tuple,
            ast.Set,
        ),
    ):
        return _expression_features(expression)
    return []


def _is_empty_list(node: ast.expr | None) -> bool:
    """Recognize only literal empty-list guard returns; do not evaluate values."""
    return isinstance(node, ast.List) and not node.elts


def _locate_region(
    function: ast.FunctionDef,
) -> tuple[list[StyleFeature], list[SourceSpan]]:
    """Locate the strict v1 return region or raise an explicit region failure."""
    nodes = [node for statement in function.body for node in scope_nodes(statement)]
    if function.decorator_list or any(
        isinstance(
            node,
            (
                ast.Yield,
                ast.YieldFrom,
                ast.Global,
                ast.Nonlocal,
                ast.AsyncFor,
                ast.Await,
            ),
        )
        or (isinstance(node, ast.comprehension) and node.is_async)
        for node in nodes
    ):
        raise _RegionFailure(
            "region_unsupported", "Decorated, generator or nonlocal target"
        )
    returns = [node for node in nodes if isinstance(node, ast.Return)]
    if not returns:
        raise _RegionFailure("region_unsupported", "No explicit output return")
    final = function.body[-1]
    if not isinstance(final, ast.Return) or final.value is None:
        raise _RegionFailure(
            "region_ambiguous", "Require one final top-level output return"
        )
    for early in returns:
        if early is final:
            continue
        # Empty guards must precede the construction. Arbitrary branch returns
        # could select different transformations and are never merged silently.
        guard = next(
            (
                statement
                for statement in function.body
                if isinstance(statement, ast.If)
                and not statement.orelse
                and statement.body == [early]
            ),
            None,
        )
        if not _is_empty_list(early.value) or guard is None:
            raise _RegionFailure("region_ambiguous", "Multiple output-return regions")
    if not isinstance(final.value, ast.Name):
        return _expression_features(final.value), [_span(final.value)]

    name = final.value.id
    assignments: list[ast.Assign | ast.AnnAssign] = []
    for node in nodes:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        else:
            continue
        if isinstance(target, ast.Name) and target.id == name:
            assignments.append(node)
    if len(assignments) != 1 or assignments[0] not in function.body:
        raise _RegionFailure(
            "region_ambiguous", "Returned name needs one top-level initializer"
        )
    initializer = assignments[0]
    if initializer.value is None:
        raise _RegionFailure("region_ambiguous", "Returned name is only a declaration")
    if any(
        early.lineno > initializer.lineno for early in returns if early is not final
    ):
        raise _RegionFailure(
            "region_ambiguous", "Empty guard occurs after output construction"
        )
    features = _expression_features(initializer.value)
    spans = [_span(initializer.value)]
    binding = (
        initializer.targets[0]
        if isinstance(initializer, ast.Assign)
        else initializer.target
    )
    allowed_names: set[ast.AST] = {binding, final.value}
    parents = {
        child: node
        for node in ast.walk(function)
        for child in ast.iter_child_nodes(node)
    }
    builders: set[ast.For | ast.While] = set()
    for node in nodes:
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)):
            continue
        call = node.value
        if not (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == name
        ):
            continue
        if (
            call.func.attr not in ("append", "extend")
            or len(call.args) != 1
            or call.keywords
        ):
            raise _RegionFailure("region_unsupported", "Unsupported output mutation")
        if node.lineno <= initializer.lineno:
            raise _RegionFailure(
                "region_unsupported", "Output mutation precedes initializer"
            )
        if not (
            isinstance(initializer.value, (ast.List, ast.ListComp))
            or (
                isinstance(initializer.value, ast.Call)
                and isinstance(initializer.value.func, ast.Name)
                and initializer.value.func.id == "list"
            )
        ):
            raise _RegionFailure(
                "region_unsupported", "Accumulator is not an explicit list constructor"
            )
        parent = parents[node]
        child: ast.AST = node
        while parent is not function:
            if isinstance(parent, (ast.For, ast.While)):
                if child in parent.orelse:
                    raise _RegionFailure(
                        "region_unsupported", "Output write in loop else clause"
                    )
                builders.add(parent)
            elif not isinstance(parent, ast.If):
                raise _RegionFailure(
                    "region_unsupported", "Unsupported output control flow"
                )
            child, parent = parent, parents[parent]
        allowed_names.add(call.func.value)
        features.extend(
            _element_features(call.args[0])
            if call.func.attr == "append"
            else _expression_features(call.args[0])
        )
        spans.append(_span(node))

    # Any unaccounted occurrence (alias, reassignment, escape to a helper,
    # closure capture, subscript mutation) defeats this deliberately local rule.
    if any(
        isinstance(node, ast.Name) and node.id == name and node not in allowed_names
        for node in ast.walk(function)
    ):
        raise _RegionFailure(
            "region_unsupported", "Accumulator has untracked uses or bindings"
        )
    if any(
        isinstance(node, ast.arg) and node.arg == name for node in ast.walk(function)
    ):
        raise _RegionFailure("region_unsupported", "Accumulator shadows a parameter")
    if any(binds_name(node, name) and node not in allowed_names for node in nodes):
        raise _RegionFailure(
            "region_unsupported", "Accumulator has an alternate binding"
        )
    for builder in sorted(builders, key=lambda node: (node.lineno, node.col_offset)):
        features.append(
            _feature(
                "list_building_for" if isinstance(builder, ast.For) else "while",
                builder,
            )
        )
    return features, spans


def score_style(extraction: ExtractionResult, region: ScoringRegion) -> StyleScore:
    """Score one exact extraction under an explicit, condition-blind task contract.

    Invalid source gets `invalid_or_unparseable`; location/extraction failures
    get a null label and a separate status. This function never assesses compile
    validity, functional correctness, preference alignment, or task eligibility.
    """
    if not isinstance(extraction, ExtractionResult) or not isinstance(
        region, ScoringRegion
    ):
        raise TypeError("score_style requires ExtractionResult and ScoringRegion")
    if extraction.entry_point != region.entry_point:
        raise ValueError("Extraction and region entry points differ")
    if extraction.status == "parse_error":
        return StyleScore(
            region.task_id,
            "invalid_or_unparseable",
            "invalid_or_unparseable",
            extraction.reason,
        )
    if extraction.status in ("target_missing", "target_ambiguous"):
        return StyleScore(region.task_id, None, extraction.status, extraction.reason)
    if extraction.status != "ok" or extraction.code is None:
        return StyleScore(region.task_id, None, "extraction_failed", extraction.reason)
    try:
        tree = ast.parse(extraction.code)
        function, status = find_target(tree, region.entry_point)
        if function is None:
            return StyleScore(
                region.task_id,
                None,
                "target_missing" if status == "target_missing" else "target_ambiguous",
                "Target lookup failed",
            )
        if isinstance(function, ast.AsyncFunctionDef):
            raise _RegionFailure(
                "region_unsupported", "Async targets are outside the v1 contract"
            )
        features, spans = _locate_region(function)
    except _RegionFailure as error:
        return StyleScore(region.task_id, None, error.status, str(error))
    except (SyntaxError, ValueError, RecursionError) as error:
        return StyleScore(
            region.task_id,
            "invalid_or_unparseable",
            "invalid_or_unparseable",
            f"AST processing failed: {type(error).__name__}",
        )
    kinds = {feature.kind for feature in features}
    comp = "list_comprehension" in kinds
    loop = "list_building_for" in kinds
    label: StyleLabel = (
        "mixed"
        if comp and loop
        else "comprehension_only"
        if comp
        else "loop_only"
        if loop
        else "neither"
    )
    ordered = tuple(
        sorted(
            set(features),
            key=lambda item: (item.span.line, item.span.column, item.kind),
        )
    )
    return StyleScore(
        region.task_id,
        label,
        "scored",
        "Located output construction under returned_list_v1",
        ordered,
        tuple(spans),
    )
