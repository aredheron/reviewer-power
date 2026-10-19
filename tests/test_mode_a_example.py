"""Offline tests of the live runner; provider and Docker boundaries are replaced."""

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, ClassVar

import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.reference_solutions import ReferencePair
from grader_power.inference import AccessProfile
from grader_power.schemas import RenderTask, ScoringRegion
from grader_power.scoring import score_response

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "live_example", ROOT / "evals/inspect_mode_a.py"
)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = demo
SPEC.loader.exec_module(demo)
IMAGE = "sha256:" + "a" * 64
CODE = "def filter_by_substring(strings, substring):\n    return [s for s in strings if substring in s]\n"


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Supply synthetic hidden records and a real Inspect mock provider, without I/O."""
    visible = RenderTask(
        "HumanEval/7",
        "example",
        "filtering",
        "src/strings.py",
        "def filter_by_substring(strings, substring):\n    raise NotImplementedError\n",
        "String utilities",
    )
    pair = ReferencePair.model_validate(
        {
            "schema_version": 1,
            "task_id": "HumanEval/7",
            "entry_point": "filter_by_substring",
            "transformation": "Filter strings",
            "implementations": {"loop": "HIDDEN_LOOP", "comprehension": "HIDDEN_COMP"},
            "review_notes": ["Synthetic only"],
        }
    )
    example = demo.Example(
        LoadedDataset(
            {
                "HumanEval/7": {
                    "canonical_solution": "HIDDEN_CANONICAL",
                    "plus_input": ["HIDDEN_PLUS"],
                }
            },
            {"variant": "full"},
        ),
        pair,
        visible,
        ScoringRegion(
            "example", "filter_by_substring", "Filter strings", "returned_list_v1"
        ),
        {"original_prompt_sha256": "synthetic"},
    )
    monkeypatch.setattr(demo, "load_example", lambda _: example)
    monkeypatch.setattr(demo, "require_verification", lambda *args: {"passed": True})
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_text(
        "test:\n  model: mockllm/example\n  base_url: https://example.invalid\n"
        "  api_key_env: TEST_API_KEY\n  revision: synthetic\n  max_tokens: 64\n"
    )
    (tmp_path / "uv.lock").write_text("synthetic")

    def fake_score(raw: str, region: ScoringRegion, **kwargs: Any) -> Any:
        """Check persistence before returning real static scores with no code execution."""
        assert list(tmp_path.glob("runs/**/raw.json"))
        assert kwargs["oracle"].dataset is example.dataset
        return score_response(raw, region)

    monkeypatch.setattr(demo, "score_response", fake_score)

    def respond(*args: Any, **kwargs: Any) -> ModelOutput:
        """Return unlimited inert responses without tokenizer downloads."""
        output = ModelOutput.from_content("synthetic", CODE)
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    monkeypatch.setattr(
        demo,
        "get_model",
        lambda *args, **kwargs: get_model(
            "mockllm/example",
            custom_outputs=respond,
            memoize=False,
        ),
    )
    return tmp_path


def test_complete_block_is_visible_only_and_immutable(setup: Path) -> None:
    """All eight assignments reach Inspect, keep raw evidence, and cannot be billed twice."""
    rows = demo.run_profile(
        setup, "run", "test", IMAGE, {"TEST_API_KEY": "synthetic-secret"}
    )
    assert len(rows) == 8
    assert len({row["condition_id"] for row in rows}) == 8
    assert all(row["status"] == "scored" for row in rows)
    assert all(row["generation_complete"] for row in rows)
    assert all(row["scoring"]["style"]["label"] == "comprehension_only" for row in rows)
    assert all(
        row["scoring"]["correctness"]["status"] == "not_evaluated" for row in rows
    )
    directory = setup / "runs/mode-a-example/run/test"
    log = read_eval_log(str(next((directory / "inspect").glob("*.eval"))))
    assert log.samples is not None and len(log.samples) == 8
    for sample in log.samples:
        assert not isinstance(sample.input, str)
        context = str(sample.input)
        assert "HIDDEN_" not in context
        assert "synthetic-secret" not in context
        assert sample.output.completion == CODE
    with pytest.raises(FileExistsError):
        demo.run_profile(
            setup, "run", "test", IMAGE, {"TEST_API_KEY": "synthetic-secret"}
        )


def test_provider_setup_failure_keeps_all_assignments(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No provider error message or credential is copied into the failure/export files."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Simulate a provider error with a sensitive message."""
        raise RuntimeError("synthetic-secret")

    monkeypatch.setattr(demo, "get_model", fail)
    rows = demo.run_profile(
        setup, "failure", "test", IMAGE, {"TEST_API_KEY": "synthetic-secret"}
    )
    assert len(rows) == 8 and all(row["status"] == "not_started" for row in rows)
    for path in (setup / "runs").rglob("*.json*"):
        assert "synthetic-secret" not in path.read_text()


def test_scoring_failure_retains_raw_output(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken scorer cannot erase the completed generation or trigger a retry."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        """Fail before invoking any oracle or submitted code."""
        raise RuntimeError("Synthetic scorer failure")

    monkeypatch.setattr(demo, "score_response", fail)
    rows = demo.run_profile(setup, "scoring", "test", IMAGE, {"TEST_API_KEY": "key"})
    assert len(rows) == 8
    assert all(row["status"] == "scoring_error" for row in rows)
    assert all(
        row["raw_output"] is not None and row["generation_complete"] for row in rows
    )
    assert all(row["scoring"] is None and row["attempt_number"] == 1 for row in rows)


def test_changed_profile_cannot_join_existing_cohort(setup: Path) -> None:
    """Sampling/model changes are rejected against the frozen block definition."""
    demo.run_profile(setup, "frozen", "test", IMAGE, {"TEST_API_KEY": "key"})
    path = setup / "configs/models.yaml"
    path.write_text(path.read_text().replace("max_tokens: 64", "max_tokens: 65"))
    with pytest.raises(ValueError, match="definition changed"):
        demo.run_profile(setup, "frozen", "test", IMAGE, {"TEST_API_KEY": "key"})


def test_stale_or_failed_sandbox_gate_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An old verification record cannot validate a changed image or adapter."""
    monkeypatch.setattr(demo, "oracle_hashes", lambda _: {"adapter": "current"})

    class Fixture:
        provenance: ClassVar[dict[str, str]] = {"task": "current"}

    path = tmp_path / demo.VERIFICATION
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "passed": True,
                "image": IMAGE,
                "example": Fixture.provenance,
                "oracle_files": {"adapter": "old"},
            }
        )
    )
    with pytest.raises(ValueError, match="Reverify"):
        demo.require_verification(tmp_path, Fixture(), IMAGE)


def test_claude_sampling_is_explicitly_provider_managed() -> None:
    """Unsupported temperature/top-p controls are absent, while retry bounds remain."""
    profile = AccessProfile(
        model="anthropic/claude-sonnet-5-5",
        api_key_env="TEST_API_KEY",
        revision="provider-managed",
        base_url="https://example.invalid",
        max_tokens=4096,
        reasoning_effort="low",
    )
    config = demo.generation_config(profile)
    assert config.temperature is None and config.top_p is None
    assert config.max_retries == 0 and config.reasoning_effort == "low"
