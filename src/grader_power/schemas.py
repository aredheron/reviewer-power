"""Shared task, renderer, scoring, and validated per-attempt result contracts."""

import keyword
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import AwareDatetime, TypeAdapter, field_validator, model_validator

from grader_power.config import (
    Contract,
    FactorAssignment,
    Identifier,
    InfrastructureFailure,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    RunConfig,
    SchemaVersion,
    Sha256,
    Text,
)

StyleLabel = Literal[
    "comprehension_only", "loop_only", "mixed", "neither", "invalid_or_unparseable"
]
ExtractionStatus = Literal[
    "ok",
    "format_error",
    "ambiguous_output",
    "parse_error",
    "target_missing",
    "target_ambiguous",
]
ScoreStatus = Literal[
    "scored",
    "extraction_failed",
    "invalid_or_unparseable",
    "target_missing",
    "target_ambiguous",
    "region_ambiguous",
    "region_unsupported",
]
FeatureKind = Literal[
    "list_comprehension",
    "list_building_for",
    "set_comprehension",
    "dict_comprehension",
    "generator_expression",
    "map",
    "filter",
    "list_concatenation",
    "while",
]


def validate_identifier(value: str) -> None:
    """Reject non-string names and names that cannot be Python identifiers."""
    if not isinstance(value, str):
        raise TypeError("entry_point must be a string")
    if not value.isidentifier() or keyword.iskeyword(value):
        raise ValueError("entry_point must be a Python identifier, not a keyword")


@dataclass(frozen=True)
class ExtractionResult:
    """Raw text and exact selected source; code is retained even on parse failure."""

    raw_response: str
    entry_point: str
    code: str | None
    output_format: Literal["raw_python", "fenced_python"] | None
    status: ExtractionStatus
    reason: str


@dataclass(frozen=True)
class ScoringRegion:
    """Explicit task contract, supplied by fixture QA, never inferred from a run.

    returned_list_v1 accepts one final return expression or one local returned
    accumulator, with optional literal-empty-list guard returns. The description
    identifies the intended transformation for human review; it is not executed.
    Creating this record conveys no eligibility or human-review approval.
    """

    task_id: str
    entry_point: str
    transformation: str
    rule: Literal["returned_list_v1"]

    def __post_init__(self) -> None:
        """Validate the contract at the API boundary without coercing values."""
        validate_identifier(self.entry_point)
        for field in ("task_id", "transformation"):
            value = getattr(self, field)
            if not isinstance(value, str):
                raise TypeError(f"{field} must be a string")
            if not value.strip():
                raise ValueError(f"{field} must be nonempty")
        if self.rule != "returned_list_v1":
            raise ValueError("Unsupported scoring-region rule")


@dataclass(frozen=True)
class SourceSpan:
    """One-based lines and zero-based UTF-8 byte columns in extracted source."""

    line: int
    column: int
    end_line: int
    end_column: int


@dataclass(frozen=True)
class StyleFeature:
    """A syntactic feature in the located output construction, for blind audit."""

    kind: FeatureKind
    span: SourceSpan


@dataclass(frozen=True)
class StyleScore:
    """Style only: no correctness, preference, power, or human-review fields.

    A missing label means the region/extraction failed, not `neither`. Preserve
    these rows in downstream denominators and report the separate status.
    """

    task_id: str
    label: StyleLabel | None
    status: ScoreStatus
    reason: str
    features: tuple[StyleFeature, ...] = ()
    region_spans: tuple[SourceSpan, ...] = ()


# Validated persistence contracts. Static extraction/scoring dataclasses above
# remain condition-blind; only these records join their outputs with assignments.


class SourceTaskIdentity(Contract):
    """Original ID plus loader provenance; hash covers unnormalized source JSONL.

    Use loader source names and map manifest release_version/source sha256 to
    source_revision/source_hash. Never derive identity from canonical code style.
    """

    source: Literal["humaneval", "mbpp"]
    task_id: Text
    source_revision: Text
    source_hash: Sha256
    evalplus_version: Text

    @model_validator(mode="after")
    def check_source_id(self) -> Self:
        """Require the unchanged dataset-specific prefix and numeric source ID."""
        prefix = "HumanEval/" if self.source == "humaneval" else "Mbpp/"
        number = self.task_id.removeprefix(prefix)
        if (
            not self.task_id.startswith(prefix)
            or not number.isascii()
            or not number.isdigit()
        ):
            raise ValueError(
                "task_id must use the source's original prefix and numeric ID"
            )
        return self


class TaskIdentity(Contract):
    """Derived identity; split membership does not certify eligibility or review."""

    source: SourceTaskIdentity
    derived_task_id: Identifier
    family: Text
    split: Literal["candidate", "pilot", "main", "confirmation"]


class FixtureMetadata(Contract):
    """Provisional fixture QA metadata, excluded from model-visible task content.

    Reference hashes are explicitly null until references exist. No construction
    of this record grants human review or freezes a pilot task set.
    """

    task: TaskIdentity
    fixture_version: Text
    oracle_version: Text
    visible_contract_hash: Sha256
    oracle_hash: Sha256
    loop_ref_hash: Sha256 | None
    comp_ref_hash: Sha256 | None
    scoring_region: ScoringRegion
    screening_reasons: tuple[Text, ...]
    adaptation_notes: tuple[Text, ...]
    human_review_status: Literal["pending"]

    @model_validator(mode="after")
    def check_region_identity(self) -> Self:
        """Prevent accidentally scoring another derived task's transformation."""
        if self.scoring_region.task_id != self.task.derived_task_id:
            raise ValueError("scoring_region.task_id must match derived_task_id")
        return self


class ModelVisibleTask(Contract):
    """Allowlisted task text for renderers; rejects whole EvalPlus records.

    Supply the adapted stub/README explicitly. Instructions in these strings are
    data, not repository-development instructions. This boundary cannot detect
    a caller deliberately pasting hidden information into an allowed text field.
    """

    derived_task_id: Identifier
    target_file: Text
    readme: Text
    stub: Text

    @field_validator("target_file")
    @classmethod
    def check_target_file(cls, value: str) -> str:
        """Accept only relative Python targets under src/ without traversal."""
        if not re.fullmatch(r"src/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py", value):
            raise ValueError(
                "target_file must be a relative src/*.py path without traversal"
            )
        return value

    def to_render_task(self, identity: TaskIdentity) -> "RenderTask":
        """Join safe task text with provenance without exposing oracle metadata."""
        if identity.derived_task_id != self.derived_task_id:
            raise ValueError("visible task and identity must share derived_task_id")
        return RenderTask(
            source_task_id=identity.source.task_id,
            derived_task_id=identity.derived_task_id,
            family=identity.family,
            stub_path=self.target_file,
            stub=self.stub,
            readme=self.readme,
        )


@dataclass(frozen=True, repr=False, kw_only=True)
class OracleTask:
    """Evaluator-only normalized data; intentionally has no prompt/JSON exporter.

    Preserves native EvalPlus input types and caller-owned input lists. Do not
    mutate those lists during evaluation. Known upstream anomalies are rejected,
    never repaired. Construction performs validation only, without code execution.
    """

    source: SourceTaskIdentity
    entry_point: str
    canonical_solution: str
    contract: str
    base_input: list[Any]
    plus_input: list[Any]
    atol: float

    def __post_init__(self) -> None:
        """Check evaluator boundary types without coercing hidden inputs."""
        if not isinstance(self.source, SourceTaskIdentity):
            raise TypeError("source must be a validated SourceTaskIdentity")
        validate_identifier(self.entry_point)
        if (
            not isinstance(self.canonical_solution, str)
            or not self.canonical_solution.strip()
        ):
            raise ValueError("canonical_solution must be nonempty text")
        if not isinstance(self.contract, str):
            raise TypeError("contract must be text")
        for name in ("base_input", "plus_input"):
            if not isinstance(getattr(self, name), list):
                raise TypeError(
                    f"{name} must be an EvalPlus input list; review source anomaly"
                )
        TypeAdapter(NonNegativeFloat).validate_python(self.atol, strict=True)


SolutionFailure = Literal[
    "assertion_failure",
    "solution_timeout",
    "solution_exception",
    "result_type_mismatch",
    "invalid_solution",
]
EvaluationStatus = Literal["not_evaluated", "infrastructure_error", "passed", "failed"]


class ErrorDetails(Contract):
    """Typed cause and safe diagnostic text, excluding authentication/header data."""

    category: Literal["infrastructure", "response", "extraction", "parse", "solution"]
    code: (
        InfrastructureFailure
        | SolutionFailure
        | Literal["response_missing", "extraction_failed", "parse_error"]
    )
    message: Text

    @model_validator(mode="after")
    def check_category(self) -> Self:
        """Prevent solution failures from being marked retryable infrastructure."""
        groups = {
            "infrastructure": {
                "transport_error",
                "rate_limit",
                "provider_unavailable",
                "sandbox_unavailable",
                "oracle_failure",
            },
            "response": {"response_missing"},
            "extraction": {"extraction_failed"},
            "parse": {"parse_error"},
            "solution": {
                "assertion_failure",
                "solution_timeout",
                "solution_exception",
                "result_type_mismatch",
                "invalid_solution",
            },
        }
        if self.code not in groups[self.category]:
            raise ValueError("error code is incompatible with its category")
        return self


class CorrectnessOutcome(Contract):
    """One full base/plus suite result; unknown is neither passed nor failed.

    Counts may be unavailable after infrastructure failure. A solution timeout
    is failed; a broken sandbox/oracle is infrastructure_error. Test logs are
    evaluator-only, and must be kept in ignored storage.
    """

    status: EvaluationStatus
    tests_passed: NonNegativeInt | None
    tests_total: PositiveInt | None
    reason: Text | None
    error: ErrorDetails | None
    test_log_ref: Text | None

    @model_validator(mode="after")
    def check_outcome(self) -> Self:
        """Require evidence for pass/fail, and an explicit explanation for absence."""
        if self.tests_passed is not None and (
            self.tests_total is None or self.tests_passed > self.tests_total
        ):
            raise ValueError("tests_passed requires tests_total and cannot exceed it")
        if self.status == "not_evaluated":
            if (
                self.reason is None
                or self.error is not None
                or self.tests_passed is not None
                or self.tests_total is not None
            ):
                raise ValueError(
                    "not_evaluated requires a reason and null counts/error"
                )
        elif self.status == "infrastructure_error":
            if self.error is None or self.error.category != "infrastructure":
                raise ValueError(
                    "infrastructure_error requires infrastructure error details"
                )
        else:
            if (
                self.tests_total is None
                or self.tests_passed is None
                or self.test_log_ref is None
            ):
                raise ValueError(
                    "passed/failed requires test counts and a test_log_ref"
                )
            if self.status == "passed":
                if self.tests_passed != self.tests_total or self.error is not None:
                    raise ValueError("passed requires every test passed and no error")
            elif (
                self.tests_passed == self.tests_total
                or self.error is None
                or self.error.category != "solution"
            ):
                raise ValueError(
                    "failed requires failed tests and solution error details"
                )
        return self


class CorrectnessResults(Contract):
    """Both original suites are required; there is no implicit overall boolean."""

    base: CorrectnessOutcome
    plus: CorrectnessOutcome


class ResponseResult(Contract):
    """Raw assistant output and reference to the full provider response artifact."""

    status: Literal["not_requested", "received", "missing", "infrastructure_error"]
    raw_response: str | None
    provider_response_ref: Text | None
    error: ErrorDetails | None

    @model_validator(mode="after")
    def check_response(self) -> Self:
        """Keep empty received text intact for the extractor to classify."""
        if self.status == "received":
            if (
                self.raw_response is None
                or self.provider_response_ref is None
                or self.error is not None
            ):
                raise ValueError(
                    "received requires raw_response, provider_response_ref and no error"
                )
        else:
            if self.raw_response is not None:
                raise ValueError("only received responses may contain raw_response")
            category = {
                "not_requested": None,
                "missing": "response",
                "infrastructure_error": "infrastructure",
            }[self.status]
            if (None if self.error is None else self.error.category) != category:
                raise ValueError("response status requires matching error details")
        return self


class ExtractionOutcome(Contract):
    """Persistence wrapper around the condition-blind extractor's exact result."""

    status: Literal["not_attempted", "completed"]
    result: ExtractionResult | None
    reason: Text | None

    @model_validator(mode="after")
    def check_extraction(self) -> Self:
        """Completed means the extractor ran, not that it succeeded."""
        if self.status == "not_attempted":
            if self.result is not None or self.reason is None:
                raise ValueError("not_attempted requires a reason and null result")
        elif self.result is None:
            raise ValueError("completed extraction requires its result")
        elif self.result.status == "ok" and (
            self.result.code is None or not self.result.code.strip()
        ):
            raise ValueError("ok extraction requires nonempty code")
        return self


class StyleResult(Contract):
    """Retain scorer failures separately from both valid and invalid style labels."""

    status: Literal["not_evaluated", "completed"]
    result: StyleScore | None
    reason: Text | None

    @model_validator(mode="after")
    def check_style(self) -> Self:
        """No silent mapping of missing/unsupported region scores to neither."""
        if self.status == "not_evaluated":
            if self.result is not None or self.reason is None:
                raise ValueError("not_evaluated requires a reason and null result")
        elif self.result is None:
            raise ValueError("completed style scoring requires a result")
        elif self.result.status == "scored":
            if self.result.label not in (
                "loop_only",
                "comprehension_only",
                "mixed",
                "neither",
            ):
                raise ValueError("scored requires a valid style label")
        elif self.result.label not in (None, "invalid_or_unparseable"):
            raise ValueError("failed style scoring cannot claim a valid style label")
        return self


class LoggedMessage(Contract):
    """Exact rendered text in order, including explicit synthetic preflight context."""

    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str
    tool_call_id: Text | None


class TokenUsage(Contract):
    """Provider-reported token counts; explicit null means unavailable, not zero."""

    input_tokens: NonNegativeInt | None
    output_tokens: NonNegativeInt | None
    reasoning_tokens: NonNegativeInt | None


class TrialRecord(Contract):
    """One immutable attempt row, including absent responses and linked retries.

    Store all assigned trials. A missing style label is not implicitly zero;
    missing-response endpoint handling belongs to predeclared analysis. Cross-row
    ID uniqueness and predecessor existence are the eventual writer's job.
    AST style and compilation are independent: a scored style may coexist with
    compile_valid=False, but such code cannot have passed correctness outcomes.
    """

    schema_version: SchemaVersion
    trial_id: Identifier
    attempt_id: Identifier
    attempt_number: PositiveInt
    retry_of: Identifier | None
    retry_error: ErrorDetails | None
    config: RunConfig
    fixture: FixtureMetadata
    factors: FactorAssignment
    started_at: AwareDatetime
    finished_at: AwareDatetime
    messages: tuple[LoggedMessage, ...]
    rendered_context_hash: Sha256
    prompt_template_hash: Sha256
    response: ResponseResult
    extraction: ExtractionOutcome
    style: StyleResult
    correctness: CorrectnessResults
    compile_valid: bool | None
    format_compliant: bool | None
    attempted_policy_change: bool | None
    token_usage: TokenUsage
    wall_time_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def check_trial(self) -> Self:
        """Reject impossible stage combinations and ineligible/excessive retries."""
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.attempt_number == 1:
            if self.retry_of is not None or self.retry_error is not None:
                raise ValueError("first attempt requires null retry_of and retry_error")
        else:
            policy = self.config.infrastructure_retry
            if self.retry_of is None or self.retry_of == self.attempt_id:
                raise ValueError("retry requires a distinct predecessor attempt_id")
            if self.attempt_number > policy.max_retries + 1:
                raise ValueError(
                    "attempt_number exceeds configured infrastructure retry policy"
                )
            if (
                self.retry_error is None
                or self.retry_error.category != "infrastructure"
                or self.retry_error.code not in policy.eligible_errors
            ):
                raise ValueError("retry requires an eligible infrastructure error")
        if self.response.status == "received" and not self.messages:
            raise ValueError("received response requires exact rendered messages")
        extraction = self.extraction.result
        style = self.style.result
        outcomes = (self.correctness.base, self.correctness.plus)
        if extraction is not None:
            if (
                self.response.status != "received"
                or extraction.raw_response != self.response.raw_response
            ):
                raise ValueError("extraction requires the same raw received response")
            if extraction.entry_point != self.fixture.scoring_region.entry_point:
                raise ValueError("extraction entry_point must match fixture")
        if style is not None:
            if extraction is None:
                raise ValueError("style scoring requires an extraction result")
            if style.task_id != self.fixture.task.derived_task_id:
                raise ValueError("style task_id must match fixture")
        if extraction is None or extraction.code is None:
            if any(outcome.status != "not_evaluated" for outcome in outcomes):
                raise ValueError("correctness evaluation requires extracted code")
            if self.compile_valid is not None:
                raise ValueError("compile_valid requires extracted code")
        has_extracted_target = extraction is not None and extraction.status == "ok"
        if (not has_extracted_target or self.compile_valid is False) and any(
            outcome.status == "passed" for outcome in outcomes
        ):
            raise ValueError(
                "invalid or missing target code cannot pass correctness tests"
            )
        if (
            not has_extracted_target
            and style is not None
            and style.label
            in (
                "loop_only",
                "comprehension_only",
                "mixed",
                "neither",
            )
        ):
            raise ValueError("valid style labels require valid extracted target code")
        return self


Style = Literal["loops", "comprehensions"]
Consequence = Literal["shadow", "future_selection"]
PreflightVariant = Literal["status", "narrative"]

STYLES: tuple[Style, ...] = ("loops", "comprehensions")
CONSEQUENCES: tuple[Consequence, ...] = ("shadow", "future_selection")
PREFLIGHT_VARIANTS: tuple[PreflightVariant, ...] = ("status", "narrative")


@dataclass(frozen=True)
class RenderTask:
    """Explicit fixture projection; only stub_path, stub, and readme are visible.

    The caller must supply a style-neutral, oracle-free stub and README. This
    boundary excludes hidden fields, but cannot certify the contents of approved
    text fields. Creating this object does not accept a task for collection.
    """

    source_task_id: str
    derived_task_id: str
    family: str
    stub_path: str
    stub: str
    readme: str

    def __post_init__(self) -> None:
        """Reject missing text and unsafe/non-source fixture paths without repair."""
        for name in (
            "source_task_id",
            "derived_task_id",
            "family",
            "stub_path",
            "stub",
            "readme",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not re.fullmatch(
            r"src/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py", self.stub_path
        ):
            raise ValueError("stub_path must be a relative Python path under src/")

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "RenderTask":
        """Project only named fixture fields; ignore all other source/QA data.

        Raw EvalPlus records deliberately lack these required fixture fields.
        Missing or invalid fields raise ValueError/TypeError, without falling
        back to an upstream prompt, canonical solution, or arbitrary file map.
        """

        def required(name: str) -> str:
            """Read one mandatory projection field without coercing its value."""
            if name not in record:
                raise ValueError(f"Missing fixture field: {name}")
            value = record[name]
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            return value

        return cls(
            source_task_id=required("source_task_id"),
            derived_task_id=required("derived_task_id"),
            family=required("family"),
            stub_path=required("stub_path"),
            stub=required("stub"),
            readme=required("readme"),
        )


@dataclass(frozen=True)
class PilotCondition:
    """One cell of the soft-instruction, single-reviewer pilot factorial."""

    user_preference: Style
    grader_preference: Style
    grader_consequence: Consequence

    def __post_init__(self) -> None:
        """Validate factor levels at runtime, including untyped callers."""
        if self.user_preference not in STYLES:
            raise ValueError("Unsupported user_preference")
        if self.grader_preference not in STYLES:
            raise ValueError("Unsupported grader_preference")
        if self.grader_consequence not in CONSEQUENCES:
            raise ValueError("Unsupported grader_consequence")

    @property
    def condition_id(self) -> str:
        """Stable factor identity, independent of task, phrasing, and ordering."""
        return (
            f"pilot-v1.soft.u-{self.user_preference}"
            f".g-{self.grader_preference}.p-{self.grader_consequence}"
        )


@dataclass(frozen=True)
class VisibleFile:
    """Exact in-memory fixture file; rendering never executes its contents."""

    path: str
    content: str


@dataclass(frozen=True)
class RenderMessage:
    """Provider-neutral message; only these messages should be sent to a model."""

    role: Literal["user"]
    content: str


@dataclass(frozen=True)
class RenderMetadata:
    """Experiment provenance kept outside the model-visible messages."""

    source_task_id: str
    derived_task_id: str
    family: str
    preflight_variant: PreflightVariant
    ordering_seed: int | None
    template_sha256: str
    schema_version: Literal[1] = 1
    template_version: Literal["pilot-draft-v1"] = "pilot-draft-v1"
    instruction_strength: Literal["soft"] = "soft"
    reviewer: Literal["style-review-v2"] = "style-review-v2"
    preflight_context: Literal["simulated"] = "simulated"
    presentation: Literal["single_user_message"] = "single_user_message"
    wording_status: Literal["draft_requires_human_review"] = (
        "draft_requires_human_review"
    )
    collection_ready: Literal[False] = False


@dataclass(frozen=True)
class RenderedCondition:
    """Immutable render plus audit data; never send the whole object to a model.

    render_id identifies task × condition × phrasing, not a collection attempt.
    prompt_sha256 identifies exact message bytes under canonical JSON encoding.
    """

    condition: PilotCondition
    condition_id: str
    render_id: str
    messages: tuple[RenderMessage, ...]
    visible_files: tuple[VisibleFile, ...]
    preflight: str
    prompt_sha256: str
    metadata: RenderMetadata

    def to_factor_assignment(self) -> FactorAssignment:
        """Map the renderer's pilot cell to the validated logging vocabulary."""
        return FactorAssignment(
            kind="pilot",
            user_style=self.condition.user_preference,
            instruction_strength=self.metadata.instruction_strength,
            grader_style=self.condition.grader_preference,
            grader_consequence=self.condition.grader_consequence,
            reviewer_id=self.metadata.reviewer,
            prompt_template_variant=self.metadata.preflight_variant,
            direct_style=None,
        )
