"""Isolated Inspect storage and synthetic contracts; no providers or .env."""

from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(scope="session", autouse=True)
def isolate_inspect_storage(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep Inspect traces, sample buffers and caches out of user-level storage.

    Inspect caches its logging handler globally, so storage must last for the
    whole test session. Redirect only filesystem locations; evaluation, scoring
    and log writes still use the installed Inspect implementation.
    """
    root = tmp_path_factory.mktemp("inspect-storage")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(
            "inspect_ai._util.appdirs.user_data_path", lambda name: root / "data" / name
        )
        patch.setattr(
            "inspect_ai._util.appdirs.user_cache_path",
            lambda name: root / "cache" / name,
        )
        patch.delenv("INSPECT_TRACE_FILE", raising=False)
        patch.delenv("INSPECT_PY_LOGGER_FILE", raising=False)
        yield


@pytest.fixture
def run_config_data() -> dict[str, Any]:
    """Return complete synthetic values, not proposed sampling or serving choices."""
    return {
        "schema_version": 1,
        "run_id": "synthetic-run",
        "mode": "forced_exposure",
        "model": {
            "provider": "mock",
            "model_id": "synthetic-model",
            "revision": "synthetic-rev",
        },
        "serving": {
            "endpoint": "https://example.invalid/v1",
            "server_version": "synthetic-server",
            "quantization": "synthetic-none",
            "context_length": 2048,
            "chat_template": "synthetic-chat-v1",
            "tool_schema": "none",
            "reasoning_mode": "synthetic-disabled",
        },
        "generation": {
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": None,
            "max_output_tokens": 128,
            "max_reasoning_tokens": None,
            "request_timeout_seconds": 5.0,
        },
        "infrastructure_retry": {
            "max_retries": 1,
            "eligible_errors": ("transport_error",),
            "delay_seconds": 0.5,
        },
        "source_code_revision": "synthetic-source-rev",
        "renderer_version": "synthetic-render-v1",
        "extractor_version": "synthetic-extract-v1",
        "scorer_version": "synthetic-score-v1",
    }


@pytest.fixture
def factor_data() -> dict[str, Any]:
    """One provisional primary condition, without choosing prompt wording."""
    return {
        "kind": "pilot",
        "user_style": "loops",
        "instruction_strength": "soft",
        "grader_style": "comprehensions",
        "grader_consequence": "shadow",
        "reviewer_id": "synthetic-reviewer",
        "prompt_template_variant": "synthetic-template",
        "direct_style": None,
    }
