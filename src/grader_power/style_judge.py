"""Blinded judge inputs and strict evidence validation; no model calls or execution.

This proposed semantic rubric supplements, but never replaces, historical AST
scores. Schema-valid output is not proof that an LLM's attribution is correct.
"""

import ast
import json
from typing import Literal, Self

from pydantic import Field, model_validator

from grader_power.code_extract import find_target
from grader_power.config import Contract, Text

Form = Literal["list_comprehension", "explicit_for", "other"]


class JudgeInput(Contract):
    """Allow only neutral target metadata and extracted code, never conditions."""

    entry_point: Text
    transformation: Text
    input_contract: str
    code: Text

    @model_validator(mode="after")
    def valid_module(self) -> Self:
        """Reject syntax failures/missing targets before sending anything to a judge."""
        try:
            tree = ast.parse(self.code)
            target, _ = find_target(tree, self.entry_point)
        except (SyntaxError, ValueError, TypeError) as error:
            raise ValueError("Judge requires parseable extracted Python") from error
        if not isinstance(target, ast.FunctionDef):
            raise TypeError("Judge requires one unambiguous synchronous entry point")
        return self


class Evidence(Contract):
    """Exact source span, checked against the submitted code after parsing."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    quote: Text


class Judgment(Contract):
    """A structured claim with an explicit abstention state and no condition fields."""

    status: Literal["classified", "abstain"]
    target_forms: list[Form] = Field(max_length=3)
    evidence: list[Evidence] = Field(max_length=20)
    flags: list[Text] = Field(max_length=20)
    rationale: str = Field(min_length=1, max_length=1600)

    @model_validator(mode="after")
    def consistent(self) -> Self:
        """Reject duplicate mechanisms and success-shaped abstentions."""
        if len(set(self.target_forms)) != len(self.target_forms):
            raise ValueError("Duplicate target form")
        if self.status == "classified" and (not self.target_forms or not self.evidence):
            raise ValueError("Classification needs mechanisms and source evidence")
        if self.status == "abstain" and self.target_forms:
            raise ValueError("Abstention cannot assign mechanisms")
        return self

    @property
    def label(self) -> str | None:
        """Map measured mechanisms deterministically, preserving unresolved cases."""
        if self.status == "abstain":
            return None
        if len(self.target_forms) > 1:
            return "mixed"
        return {
            "list_comprehension": "comprehension_only",
            "explicit_for": "loop_only",
            "other": "neither",
        }[self.target_forms[0]]


def render_judge_input(packet: JudgeInput) -> str:
    """Render blinded input with a derived line table for exact source citations.

    Keep ``code`` unchanged. The render-only ``code_lines`` field numbers every
    source line, including blanks, using the same boundaries as the evidence
    validator. It contains no caller-supplied metadata and is not a JudgeInput
    field; persisted input packets retain their original four-field schema.
    """
    data = packet.model_dump(mode="json")
    data["code_lines"] = [
        {"line_number": number, "text": line}
        for number, line in enumerate(packet.code.splitlines(), start=1)
    ]
    return json.dumps(data, indent=2, ensure_ascii=False)


def parse_judgment(raw: str, packet: JudgeInput) -> Judgment:
    """Reject malformed JSON, duplicates, fences and fabricated source citations.

    Never repair a response or map a judge failure to a substantive style label.
    Matching quotations establish traceability, not semantic correctness.
    """

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=unique_pairs)
    judgment = Judgment.model_validate(value)
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
