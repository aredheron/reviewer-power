"""Combine static scorers and the optional isolated Docker correctness adapter.

No solution executes on the host. A successful AST/compile check is never
reported as a functional test pass.
"""

from dataclasses import asdict, dataclass
from typing import Any, Literal

from grader_power.code_extract import extract_code
from grader_power.datasets.evalplus_loader import DatasetName, LoadedDataset
from grader_power.datasets.oracle import evaluate_solution
from grader_power.schemas import ExtractionResult, ScoringRegion, StyleScore
from grader_power.score_style import score_style


@dataclass(frozen=True)
class OracleRequest:
    """Hidden trusted source data and a pinned local image for the Docker adapter."""

    dataset: LoadedDataset
    dataset_name: DatasetName
    task_id: str
    image: str


@dataclass(frozen=True)
class ResponseScore:
    """Lossless extraction, static outcomes, and explicit unevaluated endpoints."""

    extraction: ExtractionResult
    style: StyleScore
    compile_status: Literal["valid", "invalid", "not_evaluated"]
    compile_error: str | None
    format_compliant: bool
    correctness: dict[str, Any]
    policy_modification_check: Literal["not_evaluated"] = "not_evaluated"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fields, retaining exact source and raw text."""
        return asdict(self)


def score_response(
    raw_response: str,
    region: ScoringRegion,
    *,
    oracle: OracleRequest | None = None,
    input_kind: Literal["response", "file"] = "response",
) -> ResponseScore:
    """Extract, classify, and compile-check; optionally invoke verified Docker.

    Compilation only creates a code object; no import, exec, or function call
    from the submitted source occurs on the host. An explicit oracle request
    invokes the existing adapter, which verifies container isolation before
    starting it and has no host fallback. Response errors remain scored records.
    Invalid API inputs propagate the existing extractor/scorer validation errors.
    With input_kind='file', exact saved source is retained even if it resembles a
    Markdown response; fences and other wrappers remain syntax errors.
    """
    if input_kind not in ("response", "file"):
        raise ValueError("input_kind must be response or file")
    extraction = extract_code(raw_response, region.entry_point)
    if input_kind == "file" and extraction.code != raw_response:
        extraction = ExtractionResult(
            raw_response,
            region.entry_point,
            raw_response,
            "raw_python",
            "parse_error",
            "Saved file is invalid Python; response wrappers are not extracted",
        )
    style = score_style(extraction, region)
    compile_status: Literal["valid", "invalid", "not_evaluated"] = "not_evaluated"
    compile_error = None
    if extraction.code is not None:
        try:
            compile(extraction.code, "<model-response>", "exec", dont_inherit=True)
            compile_status = "valid"
        except (SyntaxError, ValueError, OverflowError, RecursionError) as error:
            compile_status = "invalid"
            # Do not echo source fragments in diagnostics; the exact code is
            # already retained in the extraction record.
            compile_error = type(error).__name__
    correctness: dict[str, Any] = {
        "status": "not_evaluated",
        "reason": "No pinned isolated Docker oracle was requested",
        "base": None,
        "plus": None,
        "result_type": None,
        "test_logs": None,
    }
    if oracle is not None:
        if extraction.code is None:
            correctness["reason"] = "Code extraction failed; no source can be submitted"
        else:
            outcome = evaluate_solution(
                oracle.dataset,
                oracle.dataset_name,
                oracle.task_id,
                extraction.code,
                mode="full",
                image=oracle.image,
            )
            correctness = {**asdict(outcome), "full_correct": outcome.full_correct}
            if outcome.status == "infrastructure_error":
                correctness = {
                    "status": "not_evaluated",
                    "reason": outcome.message,
                    "adapter": asdict(outcome),
                    "base": None,
                    "plus": None,
                }
    return ResponseScore(
        extraction=extraction,
        style=style,
        compile_status=compile_status,
        compile_error=compile_error,
        format_compliant=extraction.status == "ok",
        correctness=correctness,
    )
