"""Offline strict validation with synthetic, deliberately nonoperational settings."""

import json
from itertools import product
from typing import Any

import pytest
from pydantic import ValidationError

from grader_power.config import FactorAssignment, RunConfig


def test_config_round_trip(run_config_data: dict[str, Any]) -> None:
    """Mapping and JSON boundaries preserve all resolved values exactly."""
    config = RunConfig.model_validate(run_config_data)
    assert RunConfig.model_validate_json(config.model_dump_json()) == config
    assert config.model_dump() == run_config_data
    assert config.generation.seed is None
    with pytest.raises(ValidationError, match="frozen"):
        config.run_id = "changed"


@pytest.mark.parametrize(
    "section", ["model", "serving", "generation", "infrastructure_retry"]
)
def test_every_operational_field_is_required(
    run_config_data: dict[str, Any], section: str
) -> None:
    """Even nullable operational settings must be supplied explicitly."""
    for field in run_config_data[section]:
        incomplete = {**run_config_data, section: dict(run_config_data[section])}
        del incomplete[section][field]
        with pytest.raises(ValidationError, match=field):
            RunConfig.model_validate(incomplete)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("generation", "temperature", -1.0),
        ("generation", "temperature", float("nan")),
        ("generation", "temperature", float("inf")),
        ("generation", "top_p", 0.0),
        ("generation", "top_p", 1.1),
        ("generation", "max_output_tokens", "128"),
        ("generation", "max_output_tokens", True),
        ("generation", "max_output_tokens", 0),
        ("generation", "max_output_tokens", 4096),
        ("generation", "max_reasoning_tokens", -1),
        ("generation", "seed", -1),
        ("generation", "request_timeout_seconds", 0.0),
        ("serving", "context_length", 0),
        ("serving", "reasoning_mode", "  "),
        ("model", "revision", None),
        ("model", "model_id", ""),
        ("model", "api_key", "synthetic-placeholder"),
        ("infrastructure_retry", "max_retries", -1),
        ("infrastructure_retry", "max_retries", 0),
        ("infrastructure_retry", "eligible_errors", ()),
        ("infrastructure_retry", "eligible_errors", ("solution_timeout",)),
        (
            "infrastructure_retry",
            "eligible_errors",
            ("transport_error", "transport_error"),
        ),
        ("infrastructure_retry", "delay_seconds", -0.1),
    ],
)
def test_invalid_configuration(
    run_config_data: dict[str, Any], section: str, field: str, value: Any
) -> None:
    """Malformed, coercible and contradictory settings fail before any provider call."""
    run_config_data[section][field] = value
    with pytest.raises(ValidationError):
        RunConfig.model_validate(run_config_data)


@pytest.mark.parametrize(
    "endpoint",
    [
        "example.invalid",
        "file:///tmp/model",
        "https://example.invalid:bad/v1",
        "https://example.invalid:0/v1",
        "https://user:synthetic-placeholder@example.invalid/v1",
        "https://example.invalid/v1?key=synthetic-placeholder",
        "https://example.invalid/#fragment",
        "https://example.invalid/\npath",
        "https://example.invalid\\path",
    ],
)
def test_reject_unsafe_endpoint(run_config_data: dict[str, Any], endpoint: str) -> None:
    """Reject embedded authentication and avoid echoing its input in error text."""
    run_config_data["serving"]["endpoint"] = endpoint
    with pytest.raises(ValidationError, match="endpoint") as raised:
        RunConfig.model_validate(run_config_data)
    assert "synthetic-placeholder" not in str(raised.value)


def test_disabled_retries_are_explicit(run_config_data: dict[str, Any]) -> None:
    """A deliberate no-retry policy is valid and survives JSON serialization."""
    run_config_data["infrastructure_retry"] = {
        "max_retries": 0,
        "eligible_errors": (),
        "delay_seconds": 0.0,
    }
    config = RunConfig.model_validate(run_config_data)
    assert (
        json.loads(config.model_dump_json())["infrastructure_retry"]["eligible_errors"]
        == []
    )


def test_all_eight_primary_conditions(factor_data: dict[str, Any]) -> None:
    """The complete pilot vocabulary represents exactly its crossed conditions."""
    for user, grader, consequence in product(
        ("loops", "comprehensions"),
        ("loops", "comprehensions"),
        ("shadow", "future_selection"),
    ):
        factors = FactorAssignment.model_validate(
            {
                **factor_data,
                "user_style": user,
                "grader_style": grader,
                "grader_consequence": consequence,
            }
        )
        assert (
            FactorAssignment.model_validate_json(factors.model_dump_json()) == factors
        )


@pytest.mark.parametrize(
    "change",
    [
        {"grader_style": None},
        {"grader_consequence": None},
        {"reviewer_id": None},
        {"instruction_strength": "strict"},
        {"grader_consequence": "pr_check_required"},
        {"user_style": "map"},
        {"kind": "no_grader"},
        {"direct_style": "loops"},
    ],
)
def test_incompatible_factor_assignment(
    factor_data: dict[str, Any], change: dict[str, Any]
) -> None:
    """Reject incomplete reviewer facts and accidental extensions of the primary pilot."""
    with pytest.raises(ValidationError):
        FactorAssignment.model_validate({**factor_data, **change})


@pytest.mark.parametrize("kind", ["no_grader", "direct_style"])
def test_separate_controls(factor_data: dict[str, Any], kind: str) -> None:
    """Controls explicitly omit all grader facts and record any direct request."""
    factors = FactorAssignment.model_validate(
        {
            **factor_data,
            "kind": kind,
            "grader_style": None,
            "grader_consequence": None,
            "reviewer_id": None,
            "direct_style": "loops" if kind == "direct_style" else None,
        }
    )
    assert factors.grader_style is None


@pytest.mark.parametrize("version", [True, "1", 1.0, 0, 2])
def test_schema_version_requires_supported_integer(
    run_config_data: dict[str, Any], version: Any
) -> None:
    """A boolean/equal float is not accepted as the integer schema version."""
    with pytest.raises(ValidationError, match="schema_version"):
        RunConfig.model_validate({**run_config_data, "schema_version": version})
