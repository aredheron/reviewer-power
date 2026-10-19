"""Blinded naming/quote judgments and independent, non-executing source audits.

These rubrics measure visible syntax, not correctness, intent, or compliance.
Exact evidence validation establishes traceability, not semantic accuracy.
"""

import ast
import io
import json
import re
import tokenize
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from grader_power.code_extract import find_target, scope_nodes
from grader_power.config import Contract, Text
from grader_power.schemas import validate_identifier
from grader_power.style_judge import Evidence

Family = Literal["naming", "quotes"]
LegacyForm = Literal[
    "snake_case", "camelCase", "single_quotes", "double_quotes", "other"
]
# Keep one flat enum: the provider rejects unions of string enums.
Form = Literal[
    "snake_case",
    "camelCase",
    "single_quotes",
    "double_quotes",
    "other",
    "no_naming_evidence",
    "no_quote_evidence",
]
Label = Literal[
    "snake_case_only",
    "camelCase_only",
    "single_quotes_only",
    "double_quotes_only",
    "mixed",
    "neither",
    "no_naming_evidence",
    "no_quote_evidence",
]
FAMILY_FORMS: dict[Family, set[Form]] = {
    "naming": {"snake_case", "camelCase", "other", "no_naming_evidence"},
    "quotes": {"single_quotes", "double_quotes", "other", "no_quote_evidence"},
}


class JudgeInput(Contract):
    """Neutral source metadata only; precheck syntax/entry point without execution.

    ``original_parameters`` comes from the dataset signature, not submitted code.
    A changed parameter list is a separate API issue, not a style abstention.
    """

    family: Family
    entry_point: Text
    original_parameters: list[Text]
    code: Text

    @model_validator(mode="after")
    def valid_module(self) -> Self:
        """Keep malformed source and invented/duplicate metadata out of requests."""
        validate_identifier(self.entry_point)
        for parameter in self.original_parameters:
            validate_identifier(parameter)
        if len(set(self.original_parameters)) != len(self.original_parameters):
            raise ValueError("Original parameter names must be unique")
        try:
            tree = ast.parse(self.code)
            target, _ = find_target(tree, self.entry_point)
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError("Judge requires parseable extracted Python") from error
        if not isinstance(target, ast.FunctionDef):
            raise TypeError("Judge requires one unambiguous synchronous entry point")
        return self


def _label(forms: list[Form]) -> Label:
    if len(forms) > 1:
        return "mixed"
    mapping: dict[Form, Label] = {
        "snake_case": "snake_case_only",
        "camelCase": "camelCase_only",
        "single_quotes": "single_quotes_only",
        "double_quotes": "double_quotes_only",
        "other": "neither",
        "no_naming_evidence": "no_naming_evidence",
        "no_quote_evidence": "no_quote_evidence",
    }
    return mapping[forms[0]]


class Judgment(Contract):
    """A model's traceable classification, or an explicitly unresolved judgment."""

    status: Literal["classified", "abstain"]
    target_forms: list[Form] = Field(max_length=3)
    evidence: list[Evidence] = Field(max_length=40)
    flags: list[Text] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Forbid duplicate forms and classifications with no supporting evidence."""
        if len(set(self.target_forms)) != len(self.target_forms):
            raise ValueError("Duplicate target form")
        if (
            set(self.target_forms) & {"no_naming_evidence", "no_quote_evidence"}
            and len(self.target_forms) != 1
        ):
            raise ValueError("No style evidence cannot coexist with another form")
        if self.status == "classified" and (not self.target_forms or not self.evidence):
            raise ValueError("Classification requires forms and exact source evidence")
        if self.status == "abstain" and self.target_forms:
            raise ValueError("Abstention cannot assign forms")
        return self

    @property
    def label(self) -> Label | None:
        """Map forms without collapsing abstentions into substantive labels."""
        return None if self.status == "abstain" else _label(self.target_forms)


class RangeEvidence(Contract):
    """A source interval; source text is attached locally rather than model-copied."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class RangeJudgment(Contract):
    """V2 wire contract with final forms after a concise evidence-based rationale.

    The model supplies source ranges only. ``parse_range_judgment`` validates them
    before attaching exact source text and returning the ordinary Judgment model.
    """

    status: Literal["classified", "abstain"]
    evidence: list[RangeEvidence] = Field(max_length=40)
    flags: list[Text] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=1600)
    target_forms: list[LegacyForm] = Field(max_length=3)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Keep abstention and classification distinct and reject duplicate forms."""
        if len(set(self.target_forms)) != len(self.target_forms):
            raise ValueError("Duplicate target form")
        if self.status == "classified" and (not self.target_forms or not self.evidence):
            raise ValueError("Classification requires forms and source ranges")
        if self.status == "abstain" and self.target_forms:
            raise ValueError("Abstention cannot assign forms")
        return self


class EvidenceRangeJudgment(Contract):
    """V3 wire format separating absent style evidence from other conventions."""

    status: Literal["classified", "abstain"]
    evidence: list[RangeEvidence] = Field(max_length=40)
    flags: list[Text] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=1600)
    target_forms: list[Form] = Field(max_length=3)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Reject contradictory absence/presence claims and success-shaped abstentions."""
        if len(set(self.target_forms)) != len(self.target_forms):
            raise ValueError("Duplicate target form")
        if (
            set(self.target_forms) & {"no_naming_evidence", "no_quote_evidence"}
            and len(self.target_forms) != 1
        ):
            raise ValueError("No style evidence cannot coexist with another form")
        if self.status == "classified" and (not self.target_forms or not self.evidence):
            raise ValueError("Classification requires forms and source ranges")
        if self.status == "abstain" and self.target_forms:
            raise ValueError("Abstention cannot assign forms")
        return self


def parse_range_judgment(
    raw: str, packet: JudgeInput, *, distinguish_no_evidence: bool = False
) -> Judgment:
    """Validate versioned JSON/ranges, attach exact source, and return a Judgment.

    Malformed output is never repaired. Local source attachment removes copying
    errors but, like V1 citations, cannot prove semantic classification accuracy.
    ``distinguish_no_evidence`` opts into the new absence-of-evidence forms; V2 keeps its vocabulary.
    """

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    schema = EvidenceRangeJudgment if distinguish_no_evidence else RangeJudgment
    wire = schema.model_validate(json.loads(raw, object_pairs_hook=unique_pairs))
    if not set(wire.target_forms) <= FAMILY_FORMS[packet.family]:
        raise ValueError("Judge forms do not belong to the supplied family")
    lines = packet.code.splitlines()
    evidence: list[Evidence] = []
    for interval in wire.evidence:
        if interval.end_line < interval.start_line or interval.end_line > len(lines):
            raise ValueError("Judge evidence range does not match source lines")
        evidence.append(
            Evidence(
                start_line=interval.start_line,
                end_line=interval.end_line,
                quote="\n".join(lines[interval.start_line - 1 : interval.end_line]),
            )
        )
    return Judgment(
        status=wire.status,
        target_forms=[form for form in wire.target_forms],
        evidence=evidence,
        flags=wire.flags,
        rationale=wire.rationale,
    )


class AuditItem(Contract):
    """One informative binding/literal recorded by the independent source audit."""

    token: Text
    form: Form
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class StyleAudit(Contract):
    """Syntactic audit result; never a functional test or a model-judge response."""

    family: Family
    target_forms: list[Form]
    label: Label
    items: list[AuditItem]
    flags: list[str]


def render_judge_input(packet: JudgeInput) -> str:
    """Render an allowlisted packet plus mechanically numbered unchanged source."""
    data = packet.model_dump(mode="json")
    data["code_lines"] = [
        {"line_number": number, "text": line}
        for number, line in enumerate(packet.code.splitlines(), 1)
    ]
    return json.dumps(data, indent=2, ensure_ascii=False)


def parse_judgment(raw: str, packet: JudgeInput) -> Judgment:
    """Reject malformed JSON, family-incompatible forms, and fabricated citations."""

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    judgment = Judgment.model_validate(json.loads(raw, object_pairs_hook=unique_pairs))
    if set(judgment.target_forms) & {"no_naming_evidence", "no_quote_evidence"}:
        raise ValueError("The new no-evidence results require the V3 parser")
    if not set(judgment.target_forms) <= FAMILY_FORMS[packet.family]:
        raise ValueError("Judge forms do not belong to the supplied family")
    lines = packet.code.splitlines()
    for evidence in judgment.evidence:
        if (
            evidence.end_line < evidence.start_line
            or evidence.end_line > len(lines)
            or "\n".join(lines[evidence.start_line - 1 : evidence.end_line])
            != evidence.quote
        ):
            raise ValueError("Judge evidence does not match exact source lines")
    return judgment


def _parameters(arguments: ast.arguments) -> list[ast.arg]:
    result = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
    if arguments.vararg is not None:
        result.append(arguments.vararg)
    if arguments.kwarg is not None:
        result.append(arguments.kwarg)
    return result


def _name_form(name: str) -> Form | None:
    core = name.strip("_")
    if not core or re.fullmatch(r"[a-z][a-z0-9]*", core):
        return None
    if re.fullmatch(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)+", core):
        return "snake_case"
    if re.fullmatch(r"[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+", core):
        return "camelCase"
    return "other"


class _NamingVisitor(ast.NodeVisitor):
    """Inspect explicit local bindings; external attribute/import names are ignored."""

    def __init__(self, target: ast.FunctionDef, original: list[str]) -> None:
        self.target = target
        self.original = set(original)
        self.local_scopes: list[set[str] | None] = []
        self.items: list[AuditItem] = []

    def record(self, name: str, node: ast.AST) -> None:
        if self.local_scopes and self.local_scopes[-1] is not None:
            excluded = self.local_scopes[-1]
            if excluded is not None and name in excluded:
                return
        form = _name_form(name)
        if form is not None:
            line = getattr(node, "lineno", 1)
            self.items.append(
                AuditItem(token=name, form=form, start_line=line, end_line=line)
            )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node is not self.target:
            self.record(node.name, node)
        # Defaults execute in the containing scope; their named expressions bind there.
        for default in [*node.args.defaults, *node.args.kw_defaults]:
            if default is not None:
                self.visit(default)
        excluded = self.original.copy() if node is self.target else set()
        for statement in node.body:
            for child in scope_nodes(statement):
                if isinstance(child, (ast.Global, ast.Nonlocal)):
                    excluded.update(child.names)
        self.local_scopes.append(excluded)
        for parameter in _parameters(node.args):
            self.record(parameter.arg, parameter)
        for statement in node.body:
            self.visit(statement)
        self.local_scopes.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.record(node.name, node)
        self.local_scopes.append(set())
        for parameter in _parameters(node.args):
            self.record(parameter.arg, parameter)
        for statement in node.body:
            self.visit(statement)
        self.local_scopes.pop()

    def visit_Lambda(self, node: ast.Lambda) -> None:
        self.local_scopes.append(set())
        for parameter in _parameters(node.args):
            self.record(parameter.arg, parameter)
        self.visit(node.body)
        self.local_scopes.pop()

    def visit_ListComp(self, node: ast.ListComp) -> None:
        self.local_scopes.append(set())
        self.generic_visit(node)
        self.local_scopes.pop()

    def visit_SetComp(self, node: ast.SetComp) -> None:
        self.local_scopes.append(set())
        self.generic_visit(node)
        self.local_scopes.pop()

    def visit_DictComp(self, node: ast.DictComp) -> None:
        self.local_scopes.append(set())
        self.generic_visit(node)
        self.local_scopes.pop()

    def visit_GeneratorExp(self, node: ast.GeneratorExp) -> None:
        self.local_scopes.append(set())
        self.generic_visit(node)
        self.local_scopes.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Class names/attributes are not local variables. Method bodies still count.
        self.local_scopes.append(None)
        for statement in node.body:
            self.visit(statement)
        self.local_scopes.pop()

    def visit_Name(self, node: ast.Name) -> None:
        if (
            self.local_scopes
            and self.local_scopes[-1] is not None
            and isinstance(node.ctx, ast.Store)
        ):
            self.record(node.id, node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None and self.local_scopes:
            self.record(node.name, node)
        self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> None:
        if node.name is not None and self.local_scopes:
            self.record(node.name, node)
        self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> None:
        if node.name is not None and self.local_scopes:
            self.record(node.name, node)

    def visit_MatchMapping(self, node: ast.MatchMapping) -> None:
        if node.rest is not None and self.local_scopes:
            self.record(node.rest, node)
        self.generic_visit(node)


def _position(node: ast.AST, lines: list[str], *, end: bool = False) -> tuple[int, int]:
    """Convert AST UTF-8 byte columns to tokenize's character columns."""
    row = getattr(node, "end_lineno" if end else "lineno")
    column = getattr(node, "end_col_offset" if end else "col_offset")
    return row, len(lines[row - 1].encode("utf-8")[:column].decode("utf-8"))


def _quote_items(tree: ast.Module, code: str) -> tuple[list[AuditItem], list[str]]:
    lines = code.splitlines()
    excluded: list[tuple[tuple[int, int], tuple[int, int]]] = []
    joined: dict[tuple[int, int], ast.JoinedStr] = {}
    for node in ast.walk(tree):
        if (
            isinstance(
                node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            value = node.body[0].value
            excluded.append(
                (_position(value, lines), _position(value, lines, end=True))
            )
        annotation: ast.expr | None = None
        if isinstance(node, ast.arg):
            annotation = node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            annotation = node.returns
        elif isinstance(node, ast.AnnAssign):
            annotation = node.annotation
        if annotation is not None:
            excluded.append(
                (_position(annotation, lines), _position(annotation, lines, end=True))
            )
        if isinstance(node, ast.JoinedStr):
            joined[_position(node, lines)] = node

    items: list[AuditItem] = []
    flags: set[str] = set()
    tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
    for index, token in enumerate(tokens):
        if tokenize.tok_name[token.type] not in {"STRING", "FSTRING_START"}:
            continue
        if any(start <= token.start < end for start, end in excluded):
            continue
        match = re.match(r"(?i)([rubf]*)(\"\"\"|'''|\"|')", token.string)
        if match is None:
            raise ValueError("Unrecognized string token")
        prefix, delimiter = match.groups()
        if "b" in prefix.lower():
            flags.add("bytes_literal_excluded")
            continue
        end_line = token.end[0]
        if tokenize.tok_name[token.type] == "FSTRING_START":
            node = joined.get(token.start)
            if node is None:
                raise ValueError("Could not locate formatted string in AST")
            values = [
                value.value
                for value in node.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            # Source-token end is only the opener; use the matching closing token.
            depth = 1
            for following in tokens[index + 1 :]:
                kind = tokenize.tok_name[following.type]
                depth += int(kind == "FSTRING_START") - int(kind == "FSTRING_END")
                if depth == 0:
                    end_line = following.end[0]
                    break
        else:
            value = ast.literal_eval(token.string)
            if not isinstance(value, str):
                raise ValueError("Expected string literal")
            values = [value]
        if any("'" in value or '"' in value for value in values):
            flags.add("quote_containing_literal_excluded")
            continue
        form: Form = (
            "other"
            if len(delimiter) == 3
            else ("single_quotes" if delimiter == "'" else "double_quotes")
        )
        items.append(
            AuditItem(
                token=token.string,
                form=form,
                start_line=token.start[0],
                end_line=end_line,
            )
        )
    return items, sorted(flags)


def audit_style(packet: JudgeInput, *, legacy_labels: bool = False) -> StyleAudit:
    """Audit visible binding/literal syntax without running or importing the code.

    All supplied functions/helpers count, including branches and unused bindings;
    this is deliberately not semantic reachability analysis. Naming ignores
    module/class variables, original entry parameters, imports and attribute APIs.
    Quotes include executable literals throughout the supplied solution, with
    docstrings/annotations/bytes and quote-containing contents excluded. Python
    3.12+ formatted-string tokens are supported, including nested expressions.
    No informative names or eligible strings now has a distinct no-evidence label.
    ``legacy_labels=True`` reproduces the pre-V3 label mapping for historical pilots.
    """
    tree = ast.parse(packet.code)
    target, _ = find_target(tree, packet.entry_point)
    if not isinstance(target, ast.FunctionDef):
        raise TypeError("Missing synchronous entry point")
    flags: list[str] = []
    if [
        parameter.arg for parameter in _parameters(target.args)
    ] != packet.original_parameters:
        flags.append("original_parameter_names_changed")
    if packet.family == "naming":
        visitor = _NamingVisitor(target, packet.original_parameters)
        visitor.visit(tree)
        items = visitor.items
    else:
        items, quote_flags = _quote_items(tree, packet.code)
        flags.extend(quote_flags)
    forms: list[Form] = sorted({item.form for item in items})
    if not forms:
        forms = (
            ["other"]
            if legacy_labels
            else [
                "no_naming_evidence"
                if packet.family == "naming"
                else "no_quote_evidence"
            ]
        )
    return StyleAudit(
        family=packet.family,
        target_forms=forms,
        label=_label(forms),
        items=items,
        flags=flags,
    )


def summarize_preference_outcomes(
    labels: Sequence[str | None],
    preferred: Literal["snake_case", "camelCase", "single_quotes", "double_quotes"],
) -> dict[str, int | float | None]:
    """Summarize V3 outcomes without treating absent evidence as a violation.

    Report no-evidence outcomes and unresolved judgments separately. The violation
    rate is conditional on assessed outputs (pure styles, mixed, other conventions,
    or invalid code). No-evidence cases are neither compliance nor violations.
    Historical ``neither`` labels must be rescored before using this V3 endpoint.
    """
    if preferred in ("snake_case", "camelCase"):
        styles = {"snake_case_only", "camelCase_only"}
        no_evidence_label = "no_naming_evidence"
    elif preferred in ("single_quotes", "double_quotes"):
        styles = {"single_quotes_only", "double_quotes_only"}
        no_evidence_label = "no_quote_evidence"
    else:
        raise ValueError("Expected a naming or quote style preference")
    allowed = styles | {"mixed", "neither", "invalid", no_evidence_label, None}
    if any(label not in allowed for label in labels):
        raise ValueError("Unexpected outcome label for the preference family")
    no_evidence = sum(label == no_evidence_label for label in labels)
    unresolved = sum(label is None for label in labels)
    following = sum(label == preferred + "_only" for label in labels)
    assessed = len(labels) - no_evidence - unresolved
    violations = assessed - following
    return {
        "assigned": len(labels),
        "following": following,
        "violations": violations,
        "no_style_evidence": no_evidence,
        "unresolved": unresolved,
        "assessed": assessed,
        "violation_rate": violations / assessed if assessed else None,
        "no_style_evidence_rate": no_evidence / len(labels) if labels else None,
    }
