"""Explicit model access checks, separate from scientific collection and oracles."""

import asyncio
import hashlib
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from inspect_ai.model import GenerateConfig, get_model
from pydantic import Field, model_validator

from grader_power.config import Contract, ServingSettings, Text
from grader_power.datasets.fixture_builder import synthetic_region
from grader_power.scoring import score_response

SMOKE_PROMPT = (
    "Return only a complete Python implementation of "
    "def double_values(values: list[int]) -> list[int]. "
    "Return twice each input integer, in input order, without mutating the input. "
    "Do not include explanation, tests, or a main block."
)


class AccessProfile(Contract):
    """Bounded smoke settings; live endpoints and credentials resolve separately."""

    model: Text
    api_key_env: Text
    revision: Text
    base_url: str | None = None
    base_url_env: str | None = None
    repository: str | None = None
    subfolder: str | None = None
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    max_tokens: int = Field(ge=1, le=4096)
    thinking: bool | None = None

    @model_validator(mode="after")
    def check_endpoint_source(self) -> Self:
        """Require exactly one endpoint source and validate public endpoint syntax."""
        if (self.base_url is None) == (self.base_url_env is None):
            raise ValueError("Specify exactly one of base_url and base_url_env")
        if self.base_url is not None:
            ServingSettings.check_endpoint(self.base_url)
        return self

    def resolve_access(self, values: Mapping[str, str | None]) -> tuple[str, str]:
        """Resolve only named settings; never include supplied values in errors."""
        key = values.get(self.api_key_env)
        if key is None or not key.strip():
            raise ValueError(f"Missing credential: {self.api_key_env}")
        endpoint = self.base_url
        if self.base_url_env is not None:
            endpoint = values.get(self.base_url_env)
        if endpoint is None or not endpoint.strip():
            raise ValueError(f"Missing endpoint: {self.base_url_env}")
        return ServingSettings.check_endpoint(endpoint), key

    def generation_config(self) -> GenerateConfig:
        """Disable retries and use explicit thinking controls without tools."""
        extra = (
            {"chat_template_kwargs": {"enable_thinking": self.thinking}}
            if self.thinking is not None
            else None
        )
        return GenerateConfig(
            max_tokens=self.max_tokens,
            max_retries=0,
            timeout=180,
            reasoning_effort=self.reasoning_effort,
            extra_body=extra,
        )


def load_access_profiles(path: Path) -> dict[str, AccessProfile]:
    """Read the public YAML profiles and reject malformed entries before inference."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("Model profiles must be a nonempty mapping")
    profiles: dict[str, AccessProfile] = {}
    for name, record in data.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Profile names must be nonempty strings")
        profiles[name] = AccessProfile.model_validate(record)
    return profiles


async def check_model_access(
    profile: AccessProfile, values: Mapping[str, str | None]
) -> dict[str, Any]:
    """Make one bounded synthetic request and retain output without executing it.

    Credentials/endpoints are never serialized. Provider failures retain only
    their exception class and HTTP status because messages can contain secrets.
    A successful request proves generation access, not functional correctness,
    scientific readiness, or the identity of a self-hosted checkpoint's weights.
    """
    endpoint, key = profile.resolve_access(values)
    config = profile.generation_config()
    record: dict[str, Any] = {
        "purpose": "model_access_smoke_not_research_data",
        "started_at": datetime.now(UTC).isoformat(),
        "model": profile.model,
        "configured_revision": profile.revision,
        "inspect_version": version("inspect-ai"),
        "prompt": SMOKE_PROMPT,
        "prompt_sha256": hashlib.sha256(SMOKE_PROMPT.encode()).hexdigest(),
        "generation": config.model_dump(mode="json", exclude_none=True),
        "status": "failed",
        "correctness": "not_evaluated",
    }
    start = time.monotonic()
    try:
        model = get_model(
            profile.model, base_url=endpoint, api_key=key, config=config, memoize=False
        )
        output = await asyncio.wait_for(
            model.generate(SMOKE_PROMPT, cache=False), timeout=200
        )
        record["output"] = output.model_dump(mode="json")
        record["scoring"] = score_response(
            output.completion, synthetic_region()
        ).to_dict()
        stop = output.choices[0].stop_reason if output.choices else None
        record["status"] = (
            "generated"
            if output.completion.strip() and stop == "stop"
            else "incomplete"
        )
    except Exception as error:  # noqa: BLE001 -- redact provider errors at the reporting boundary
        # Deliberate reporting boundary: preserve failure without echoing secrets.
        record["error_type"] = type(error).__name__
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            record["http_status"] = status_code
    record["wall_seconds"] = time.monotonic() - start
    return record
