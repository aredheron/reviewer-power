"""Validated, credential-free configuration shared by pilot components.

Use ``model_validate`` for Python mappings or ``model_validate_json`` for JSON.
All operational fields are required, including nullable settings: explicit null
means deliberately unused/unsupported, never an unresolved default. These models
neither read environment files nor initialize providers or execution sandboxes.
"""

from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Text = Annotated[str, Field(min_length=1, pattern=r"\S")]
Identifier = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SchemaVersion = Annotated[int, Field(strict=True, ge=1, le=1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]

UserStyle = Literal["loops", "comprehensions"]
GraderStyle = Literal["loops", "comprehensions"]
InstructionStrength = Literal["soft", "strict"]
GraderConsequence = Literal[
    "shadow", "future_selection", "pr_check_required", "pr_check_advisory"
]
InfrastructureFailure = Literal[
    "transport_error",
    "rate_limit",
    "provider_unavailable",
    "sandbox_unavailable",
    "oracle_failure",
]


class Contract(BaseModel):
    """Reject unknown fields/coercion and keep validated records immutable.

    Validation errors hide input values in their text; callers must likewise not
    log ``ValidationError.errors()`` with its default input inclusion. Do not use
    Pydantic's unvalidated ``model_construct`` or ``model_copy(update=...)`` at
    external boundaries. Revalidate a mapping to make changes instead.
    """

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
        revalidate_instances="always",
        hide_input_in_errors=True,
    )


class FactorAssignment(Contract):
    """One coding condition; controls remain distinguishable from primary runs.

    The primary pilot supports only the eight soft-instruction conditions.
    Secondary PR consequences and strict instructions have typed vocabulary but
    are not enabled for primary pilot trials. Probes require a separate contract.
    """

    kind: Literal["pilot", "no_grader", "direct_style"]
    user_style: UserStyle
    instruction_strength: InstructionStrength
    grader_style: GraderStyle | None
    grader_consequence: GraderConsequence | None
    reviewer_id: Identifier | None
    prompt_template_variant: Identifier
    direct_style: UserStyle | None

    @model_validator(mode="after")
    def check_condition(self) -> Self:
        """Require a complete reviewer assignment or an explicit control."""
        reviewer = (self.grader_style, self.grader_consequence, self.reviewer_id)
        if self.kind == "pilot":
            if any(value is None for value in reviewer):
                raise ValueError(
                    "pilot requires grader style, consequence, and reviewer_id"
                )
            if self.instruction_strength != "soft":
                raise ValueError("primary pilot requires soft instruction strength")
            if self.grader_consequence not in ("shadow", "future_selection"):
                raise ValueError(
                    "primary pilot supports only shadow or future_selection"
                )
        elif any(value is not None for value in reviewer):
            raise ValueError(
                "no_grader and direct_style controls require null reviewer fields"
            )
        if (self.kind == "direct_style") != (self.direct_style is not None):
            raise ValueError(
                "direct_style must be set exactly for a direct_style control"
            )
        return self


class ModelIdentity(Contract):
    """Explicit provider/model/checkpoint identifiers supplied by the operator."""

    provider: Text
    model_id: Text
    revision: Text


class ServingSettings(Contract):
    """Pinned serving choices; no arbitrary provider options or authentication.

    Hosted-provider-managed choices must be described explicitly with their
    version/identity. The endpoint is a public base URL, without userinfo, query
    parameters, or fragments. Credentials belong outside serialized contracts.
    """

    endpoint: Text
    server_version: Text
    quantization: Text
    context_length: PositiveInt
    chat_template: Text
    tool_schema: Text
    reasoning_mode: Text

    @field_validator("endpoint")
    @classmethod
    def check_endpoint(cls, value: str) -> str:
        """Reject malformed URLs and common places for embedded credentials."""
        try:
            parsed = urlsplit(value)
            port = parsed.port
        except ValueError:
            raise ValueError("endpoint must be a valid HTTP(S) base URL") from None
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or any(character.isspace() for character in value)
            or "\\" in value
            or port == 0
        ):
            raise ValueError(
                "endpoint requires HTTP(S), a host, and no credentials/query/fragment"
            )
        return value


class GenerationLimits(Contract):
    """Resolved sampling and limits; no guesses about provider token semantics.

    ``max_reasoning_tokens=None`` means no separate supported reasoning limit;
    ``seed=None`` means no provider seed. Output/reasoning accounting must be
    checked by the future provider adapter rather than silently translated here.
    """

    temperature: NonNegativeFloat
    top_p: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    seed: NonNegativeInt | None
    max_output_tokens: PositiveInt
    max_reasoning_tokens: NonNegativeInt | None
    request_timeout_seconds: PositiveFloat


class InfrastructureRetryPolicy(Contract):
    """Fixed retries after the original attempt, only for named infrastructure errors.

    Empty eligibility and zero delay explicitly disable retries. Incorrect code,
    solution timeouts, parse failures, and extraction failures are never eligible.
    Each retry must be logged as another attempt linked to its predecessor.
    """

    max_retries: NonNegativeInt
    eligible_errors: tuple[InfrastructureFailure, ...]
    delay_seconds: NonNegativeFloat

    @model_validator(mode="after")
    def check_policy(self) -> Self:
        """Reject contradictory or duplicated retry declarations."""
        if len(set(self.eligible_errors)) != len(self.eligible_errors):
            raise ValueError("eligible_errors must not contain duplicates")
        if self.max_retries == 0:
            if self.eligible_errors or self.delay_seconds != 0:
                raise ValueError(
                    "disabled retries require empty eligible_errors and zero delay"
                )
        elif not self.eligible_errors:
            raise ValueError("enabled retries require explicit eligible_errors")
        return self


class RunConfig(Contract):
    """Resolved Mode A settings, stored alongside every attempt or in its run log.

    This validates settings, not scientific readiness: task QA, prompt review,
    matched-block scheduling and Docker isolation still require independent gates.
    """

    schema_version: SchemaVersion
    run_id: Identifier
    mode: Literal["forced_exposure"]
    model: ModelIdentity
    serving: ServingSettings
    generation: GenerationLimits
    infrastructure_retry: InfrastructureRetryPolicy
    source_code_revision: Text
    renderer_version: Text
    extractor_version: Text
    scorer_version: Text

    @model_validator(mode="after")
    def check_context_limit(self) -> Self:
        """Reject an output allowance larger than the serving context window."""
        if self.generation.max_output_tokens > self.serving.context_length:
            raise ValueError("max_output_tokens must not exceed serving context_length")
        return self
