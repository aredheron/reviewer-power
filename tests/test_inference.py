"""Offline checks for live-access boundaries; no provider or solution execution."""

import asyncio
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ModelOutput
from pydantic import ValidationError

from grader_power.inference import (
    AccessProfile,
    check_model_access,
    load_access_profiles,
)


def profile(**changes: Any) -> AccessProfile:
    """Construct an explicit synthetic endpoint with bounded request settings."""
    return AccessProfile.model_validate(
        {
            "model": "mockllm/access-test",
            "base_url": "https://example.invalid/v1",
            "api_key_env": "TEST_API_KEY",
            "revision": "synthetic-v1",
            "max_tokens": 64,
            **changes,
        }
    )


@pytest.mark.parametrize("max_tokens", [0, 4097, "64"])
def test_reject_unbounded_or_coerced_tokens(max_tokens: Any) -> None:
    """Profiles cannot silently exceed the smoke-test completion bound."""
    with pytest.raises(ValidationError):
        profile(max_tokens=max_tokens)


def test_endpoint_selection_and_missing_credentials() -> None:
    """Missing or ambiguous access configuration fails before creating a client."""
    with pytest.raises(ValidationError):
        profile(base_url_env="LOCAL_ENDPOINT")
    with pytest.raises(ValueError, match="Missing credential"):
        profile().resolve_access({})
    local = profile(base_url=None, base_url_env="LOCAL_ENDPOINT")
    with pytest.raises(ValueError, match="Missing endpoint"):
        local.resolve_access({"TEST_API_KEY": "synthetic-test-key"})
    with pytest.raises(ValueError):
        local.resolve_access(
            {
                "TEST_API_KEY": "synthetic-test-key",
                "LOCAL_ENDPOINT": "https://u:p@example.invalid",
            }
        )


def test_public_profiles_validate() -> None:
    """Every checked-in profile specifies one endpoint and bounded retry-free settings."""
    profiles = load_access_profiles(Path("configs/models.yaml"))
    assert len(profiles) == 4
    for record in profiles.values():
        assert record.generation_config().max_retries == 0
        assert record.generation_config().timeout == 180


def test_thinking_is_explicit_for_self_hosting() -> None:
    """The checkpoint template receives the same explicit thinking control."""
    assert profile(thinking=False).generation_config().extra_body == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


@pytest.mark.parametrize(
    "content,stop,expected",
    [
        (
            "def double_values(values):\n    return [x * 2 for x in values]",
            "stop",
            "generated",
        ),
        ("def double_values", "max_tokens", "incomplete"),
        ("", "stop", "incomplete"),
    ],
)
def test_output_retention_and_incomplete_status(
    monkeypatch: pytest.MonkeyPatch, content: str, stop: Any, expected: str
) -> None:
    """Keep raw responses and separate successful generation from correctness."""
    calls: list[str] = []

    class FakeModel:
        async def generate(self, prompt: str, *, cache: bool) -> ModelOutput:
            """Return an inert response while checking caching is disabled."""
            assert cache is False
            calls.append(prompt)
            return ModelOutput.from_content(
                "synthetic-returned-id", content, stop_reason=stop
            )

    def get_model(*args: Any, **kwargs: Any) -> FakeModel:
        """Ensure access values go only to the provider constructor."""
        assert kwargs["api_key"] == "synthetic-test-key"
        assert kwargs["config"].max_retries == 0
        return FakeModel()

    monkeypatch.setattr("grader_power.inference.get_model", get_model)
    record = asyncio.run(
        check_model_access(profile(), {"TEST_API_KEY": "synthetic-test-key"})
    )
    assert len(calls) == 1
    assert record["status"] == expected
    assert record["correctness"] == "not_evaluated"
    assert record["output"]["model"] == "synthetic-returned-id"
    assert "synthetic-test-key" not in str(record)
    assert "example.invalid" not in str(record)


def test_provider_errors_do_not_echo_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failure messages can contain secrets; retain only safe diagnostic fields."""

    def fail(*args: Any, **kwargs: Any) -> None:
        """Simulate a provider exception whose message is unsafe to persist."""
        raise RuntimeError("synthetic-test-key in provider response")

    monkeypatch.setattr("grader_power.inference.get_model", fail)
    record = asyncio.run(
        check_model_access(profile(), {"TEST_API_KEY": "synthetic-test-key"})
    )
    assert record["status"] == "failed"
    assert record["error_type"] == "RuntimeError"
    assert "synthetic-test-key" not in str(record)
