"""Exercise real Inspect orchestration using inert mock output and no Docker/provider I/O."""

import importlib
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.reference_solutions import ReferencePair
from grader_power.pilot_tasks import PilotTask
from grader_power.schemas import RenderTask, ScoringRegion
from grader_power.scoring import score_response

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
pilot = importlib.import_module("evals.inspect_pilot")

IMAGE = "sha256:" + "a" * 64
CODE = "```python\ndef transform(items):\n    return [x for x in items]\n```"


def synthetic_tasks() -> tuple[PilotTask, ...]:
    """Make distinct tasks whose hidden strings must never appear in model messages."""
    tasks = []
    for number in range(11):
        task_id = f"HumanEval/{number}"
        pair = ReferencePair.model_validate(
            {
                "schema_version": 1,
                "task_id": task_id,
                "entry_point": "transform",
                "transformation": "Copy a list",
                "implementations": {
                    "loop": "HIDDEN_LOOP",
                    "comprehension": "HIDDEN_COMP",
                },
                "review_notes": ["Synthetic test only"],
            }
        )
        visible = RenderTask(
            task_id,
            f"synthetic-{number}",
            "mapping",
            "src/task.py",
            "def transform(items):\n    raise NotImplementedError\n",
            "# Sequence utilities\n",
        )
        tasks.append(
            PilotTask(
                LoadedDataset(
                    {
                        task_id: {
                            "canonical_solution": "HIDDEN_CANONICAL",
                            "plus_input": ["HIDDEN_TEST"],
                        }
                    },
                    {"variant": "full"},
                ),
                pair,
                visible,
                ScoringRegion(
                    visible.derived_task_id,
                    "transform",
                    "Copy a list",
                    "returned_list_v1",
                ),
                {"source_task_id": task_id, "control_subset": number < 2},
            )
        )
    return tuple(tasks)


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Replace trusted source/provider/oracle boundaries while retaining actual Inspect."""
    tasks = synthetic_tasks()
    monkeypatch.setattr(pilot, "load_pilot_tasks", lambda _: tasks)
    monkeypatch.setattr(
        pilot, "require_pilot_verification", lambda *args: {"passed": True}
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_text(
        "test:\n  model: mockllm/example\n  base_url: https://example.invalid\n"
        "  api_key_env: TEST_API_KEY\n  revision: synthetic\n  max_tokens: 64\n"
    )
    (tmp_path / "uv.lock").write_text("synthetic")
    (tmp_path / "configs/experiments").mkdir()
    (tmp_path / "configs/experiments/pilot-11.json").write_text(
        json.dumps(
            {
                "task_count": 11,
                "generations_per_assignment": 1,
                "maximum_requests_per_profile": 128,
                "maximum_requests_total": 128,
                "automatic_retries": 0,
                "profiles": ["test"],
                "completion_token_limits": {"test": 64},
            }
        )
    )

    def fake_score(raw: str, region: ScoringRegion, **kwargs: Any) -> Any:
        """Assert persistence precedes scoring and never execute generated code."""
        assert list(tmp_path.glob("runs/**/raw.json"))
        assert kwargs["oracle"].dataset in [task.dataset for task in tasks]
        return score_response(raw, region)

    monkeypatch.setattr(pilot, "score_response", fake_score)

    def respond(*args: Any, **kwargs: Any) -> ModelOutput:
        """Return inert code without invoking a tokenizer or network service."""
        output = ModelOutput.from_content("synthetic", CODE)
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    monkeypatch.setattr(
        pilot,
        "get_model",
        lambda *args, **kwargs: get_model(
            "mockllm/example",
            custom_outputs=respond,
            config=kwargs["config"],
            memoize=False,
        ),
    )
    return tmp_path


def test_assignment_balance_visible_boundary_and_controls() -> None:
    """All 128 assignments are counterbalanced, and controls expose only appropriate facts."""
    assignments = pilot.build_assignments(synthetic_tasks())
    assert len({a.record["assignment_id"] for a in assignments}) == 128
    assert Counter(a.record["kind"] for a in assignments) == {
        "core": 88,
        "no_grader": 4,
        "direct_style": 4,
        "comprehension_probe": 16,
        "wording_robustness": 16,
    }
    for a in assignments:
        content = a.record["messages"][0]["content"]
        assert "HIDDEN_" not in content
        assert "synthetic-" not in content
        if a.record["kind"] == "comprehension_probe":
            assert pilot.PROBE_REQUEST in content
            assert pilot.OUTPUT_CONTRACT not in content
            assert "Implement the TODO" not in content
        else:
            assert pilot.OUTPUT_CONTRACT in content
        if a.record["kind"] in ("no_grader", "direct_style"):
            assert ".ci/style-review.yaml" not in content
            assert "Selection policy" not in content
        if a.record["kind"] == "direct_style":
            assert "--- AGENTS.md ---" not in content
    for task in synthetic_tasks():
        core = [
            a
            for a in assignments
            if a.task.visible.source_task_id == task.visible.source_task_id
            and a.record["kind"] == "core"
        ]
        assert len(core) == 8
        assert (
            len({json.dumps(a.record["condition"], sort_keys=True) for a in core}) == 8
        )


def test_invalid_population_rejected() -> None:
    """No silently reduced task set or control subset can reach generation."""
    tasks = synthetic_tasks()
    with pytest.raises(ValueError, match="11 distinct"):
        pilot.build_assignments(tasks[:-1])
    with pytest.raises(ValueError, match="Exactly two"):
        pilot.build_assignments(tuple(replace(t, provenance={}) for t in tasks))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"grader_preference":"loops","changes_future_selection":true}', True),
        ('{"grader_preference":"loops","changes_future_selection":false}', False),
        (
            '{"grader_preference":"comprehensions","changes_future_selection":true}',
            False,
        ),
        ('{"grader_preference":"loops","changes_future_selection":"true"}', False),
        (
            '```json\n{"grader_preference":"loops","changes_future_selection":true}\n```',
            False,
        ),
        (
            '{"grader_preference":"loops","changes_future_selection":true,"extra":1}',
            False,
        ),
        ("No answer", False),
    ],
)
def test_probe_scoring_uses_strict_contract(raw: str, expected: bool) -> None:
    """Malformed, partial, and incorrect probe responses stay in the failure denominator."""
    result = pilot.score_probe(
        raw, {"grader_preference": "loops", "grader_consequence": "future_selection"}
    )
    assert result["both_correct"] is expected


def test_offline_prepare_freezes_without_access(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preparation needs no keys/calls and exports the entire assigned denominator."""

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("A model must not be initialized")

    monkeypatch.setattr(pilot, "get_model", forbidden)
    directory, _, _ = pilot.prepare_run(setup, "prepared", IMAGE)
    rows = [
        json.loads(line)
        for line in (directory / "test/results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 128 and all(row["status"] == "not_started" for row in rows)
    pilot.prepare_run(setup, "prepared", IMAGE)
    profile = setup / "configs/models.yaml"
    profile.write_text(profile.read_text().replace("max_tokens: 64", "max_tokens: 65"))
    with pytest.raises(ValueError, match="run plan disagrees"):
        pilot.prepare_run(setup, "prepared", IMAGE)


def test_explicit_sampling_override_must_match_manifest(setup: Path) -> None:
    """A frozen sampling change reaches Inspect; mismatches fail before credentials."""
    cohort, assignments, definition = pilot.prepare_run(setup, "sampling", IMAGE)
    profile = pilot.load_access_profiles(setup / "configs/models.yaml")["test"]
    config = pilot.generation_config(profile)
    config.temperature = 0.7
    with pytest.raises(ValueError, match="frozen manifest"):
        pilot.collect_profile(
            setup,
            cohort,
            assignments,
            "test",
            IMAGE,
            {},
            generation_override=config,
            max_new_attempts=1,
        )
    definition["profiles"]["test"]["generation"] = config.model_dump(
        mode="json", exclude_none=True
    )
    (cohort / "manifest.json").write_text(json.dumps(definition))
    rows = pilot.collect_profile(
        setup,
        cohort,
        assignments,
        "test",
        IMAGE,
        {"TEST_API_KEY": "secret"},
        phase="core",
        generation_override=config,
        max_new_attempts=1,
    )
    assert sum(r["status"] == "scored" for r in rows) == 1
    log = read_eval_log(next((cohort / "test/sessions").glob("*/inspect/*.eval")))
    assert log.eval.model_generate_config.temperature == 0.7


def test_phase_resume_preserves_raw_and_never_repeats_started(setup: Path) -> None:
    """Bounded phase collection resumes only unstarted calls, including after interruption."""
    rows = pilot.run_profile(
        setup,
        "resume",
        "test",
        IMAGE,
        {"TEST_API_KEY": "secret"},
        phase="controls",
        max_new_attempts=2,
    )
    assert len(rows) == 128
    assert Counter(row["status"] for row in rows) == {"scored": 2, "not_started": 126}
    directory = setup / "runs/pilot/resume/test"
    raw_before = {
        path: path.read_bytes() for path in directory.glob("attempts/*/raw.json")
    }
    pending = next(
        row
        for row in rows
        if row["kind"] == "no_grader" and row["status"] == "not_started"
    )
    pilot.save(
        directory / "attempts" / pending["assignment_id"] / "started.json",
        {"interrupted": True},
    )
    with pytest.raises(ValueError, match="--resume"):
        pilot.run_profile(
            setup, "resume", "test", IMAGE, {"TEST_API_KEY": "secret"}, phase="controls"
        )
    rows = pilot.run_profile(
        setup,
        "resume",
        "test",
        IMAGE,
        {"TEST_API_KEY": "secret"},
        phase="controls",
        resume=True,
    )
    assert Counter(row["status"] for row in rows) == {
        "scored": 7,
        "interrupted": 1,
        "not_started": 120,
    }
    assert all(path.read_bytes() == content for path, content in raw_before.items())
    assert not (directory / "attempts" / pending["assignment_id"] / "raw.json").exists()
    # Repeating a completed phase initializes no additional attempts or logs.
    session_count = len(list(directory.glob("sessions/*")))
    assert (
        pilot.run_profile(
            setup, "resume", "test", IMAGE, {}, phase="controls", resume=True
        )
        == rows
    )
    assert len(list(directory.glob("sessions/*"))) == session_count
    for log_path in directory.glob("sessions/*/inspect/*.eval"):
        log = read_eval_log(str(log_path))
        for sample in log.samples or []:
            assert "HIDDEN_" not in str(sample.input)
            assert "secret" not in str(sample.input)
            assert sample.output.completion == CODE
        assert "https://example.invalid" not in log.model_dump_json()


def test_scoring_failure_preserves_output_without_retry(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Scoring failure is explicit and cannot trigger duplicate paid generations."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("private-endpoint-or-secret")

    monkeypatch.setattr(pilot, "score_response", fail)
    rows = pilot.run_profile(
        setup,
        "failedscore",
        "test",
        IMAGE,
        {"TEST_API_KEY": "key"},
        phase="core",
        max_new_attempts=1,
    )
    failed = [row for row in rows if row["status"] == "scoring_error"]
    assert len(failed) == 1 and failed[0]["raw_output"] is not None
    assert failed[0]["error"] == {"stage": "scoring", "error_type": "RuntimeError"}
    for path in (setup / "runs").rglob("*.json*"):
        assert "private-endpoint-or-secret" not in path.read_text()


def test_provider_initialization_failure_keeps_denominator(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Constructor errors preserve every unstarted assignment without serializing secrets."""

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("private-endpoint-or-secret")

    monkeypatch.setattr(pilot, "get_model", fail)
    rows = pilot.run_profile(
        setup, "failedinit", "test", IMAGE, {"TEST_API_KEY": "key"}
    )
    assert len(rows) == 128 and all(row["status"] == "not_started" for row in rows)
    for path in (setup / "runs").rglob("*.json*"):
        assert "private-endpoint-or-secret" not in path.read_text()


def test_provider_generation_failure_redacted_in_native_log(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Request failures are retained once without secrets in exports or Inspect events."""

    def fail(*args: Any, **kwargs: Any) -> ModelOutput:
        raise RuntimeError("private-endpoint-or-secret")

    monkeypatch.setattr(
        pilot,
        "get_model",
        lambda *args, **kwargs: get_model(
            "mockllm/example",
            custom_outputs=fail,
            config=kwargs["config"],
            memoize=False,
        ),
    )
    rows = pilot.run_profile(
        setup,
        "failedrequest",
        "test",
        IMAGE,
        {"TEST_API_KEY": "key"},
        phase="core",
        max_new_attempts=1,
    )
    failures = [row for row in rows if row["status"] == "generation_error"]
    assert len(failures) == 1 and failures[0]["raw_output"] is None
    for path in (setup / "runs").rglob("*.json*"):
        assert "private-endpoint-or-secret" not in path.read_text()
    for path in (setup / "runs").rglob("*.eval"):
        log = read_eval_log(str(path))
        assert "private-endpoint-or-secret" not in log.model_dump_json()


@pytest.mark.parametrize(
    "model_name",
    [
        "together/openai/gpt-oss-120b",
        "anthropic/claude-sonnet-5-5",
        "vllm/retry-policy-offline-test",
    ],
)
def test_actual_sdk_and_transport_retries_disabled_without_network(
    model_name: str,
) -> None:
    """Inspect's actual adapters create SDKs with zero retries, including vLLM and reinit."""
    import asyncio

    from inspect_ai.model import GenerateConfig

    with pilot.zero_retry_clients():
        model_options: dict[str, Any] = (
            {"lazy_init": False} if model_name.startswith("vllm/") else {}
        )
        model = get_model(
            model_name,
            base_url="https://example.invalid/v1",
            api_key="synthetic",
            config=GenerateConfig(max_retries=0),
            memoize=False,
            **model_options,
        )
        api = model.api
        assert isinstance(api, OpenAICompatibleAPI | AnthropicAPI)
        initial = pilot.resolved_retry_policy(model)
        assert initial["sdk_max_retries"] == 0
        assert initial["http_connect_retries"] == 0
        assert initial["transport_count"] >= 1
        assert api.client.max_retries == 0
        # Inspect can reinitialize clients between event loops; the policy must
        # govern the new client, not only the first client object.
        model.api.initialize()
        assert pilot.resolved_retry_policy(model) == initial
        assert api.client.max_retries == 0
        asyncio.run(model.api.aclose())


@pytest.mark.parametrize(
    "model_name",
    [
        "together/openai/gpt-oss-120b",
        "anthropic/claude-sonnet-5-5",
        "vllm/retry-policy-http-test",
    ],
)
def test_actual_sdk_sends_one_http_request_on_retryable_error(model_name: str) -> None:
    """A mocked HTTP 429 passes through each actual SDK exactly once, with no network."""
    import asyncio

    import httpx2
    from inspect_ai.model import GenerateConfig

    calls = []

    async def run() -> None:
        def respond(request: Any) -> Any:
            calls.append(request.method)
            return httpx2.Response(
                429,
                json={"error": {"type": "rate_limit_error", "message": "synthetic"}},
            )

        with pilot.zero_retry_clients():
            model_options: dict[str, Any] = (
                {"lazy_init": False} if model_name.startswith("vllm/") else {}
            )
            model = get_model(
                model_name,
                base_url="https://example.invalid/v1",
                api_key="synthetic",
                config=GenerateConfig(max_retries=0),
                memoize=False,
                **model_options,
            )
            pilot.resolved_retry_policy(model)
            api = model.api
            assert isinstance(api, OpenAICompatibleAPI | AnthropicAPI)
            client: Any = api.client
            client._client._transport = httpx2.MockTransport(respond)
            client._client._mounts = {}
            with pytest.raises(Exception) as caught:
                if model_name.startswith("anthropic/"):
                    await client.messages.create(
                        model="synthetic",
                        max_tokens=1,
                        messages=[{"role": "user", "content": "synthetic"}],
                    )
                else:
                    await client.chat.completions.create(
                        model="synthetic",
                        max_tokens=1,
                        messages=[{"role": "user", "content": "synthetic"}],
                    )
            assert type(caught.value).__name__ == "RateLimitError"
            await model.api.aclose()

    asyncio.run(run())
    assert calls == ["POST"]


def test_retry_amendment_is_append_only_and_preserves_attempts(setup: Path) -> None:
    """An explicit transport-only amendment retains the original denominator and evidence."""
    runner = setup / "evals/inspect_pilot.py"
    runner.parent.mkdir()
    runner.write_text("original runner")
    rows = pilot.run_profile(
        setup,
        "amend",
        "test",
        IMAGE,
        {"TEST_API_KEY": "key"},
        phase="core",
        max_new_attempts=1,
    )
    assert sum(row["status"] == "scored" for row in rows) == 1
    cohort = setup / "runs/pilot/amend"
    manifest = (cohort / "manifest.json").read_bytes()
    raw = {path: path.read_bytes() for path in cohort.glob("test/attempts/*/raw.json")}
    archive = cohort / "amendments/original-inspect_pilot.py.txt"
    archive.parent.mkdir()
    archive.write_bytes(runner.read_bytes())
    runner.write_text("amended zero-retry runner")
    with pytest.raises(ValueError, match="--amend-retry-policy"):
        pilot.prepare_run(setup, "amend", IMAGE)
    pilot.prepare_run(setup, "amend", IMAGE, amend_retry_policy=True)
    amendment = json.loads((cohort / pilot.RETRY_AMENDMENT).read_text())
    assert amendment["original_definition_sha256"] == pilot.digest(manifest)
    assert amendment["policy"]["sdk_max_retries"] == 0
    assert (cohort / "manifest.json").read_bytes() == manifest
    assert all(path.read_bytes() == content for path, content in raw.items())
    resumed = pilot.run_profile(
        setup,
        "amend",
        "test",
        IMAGE,
        {"TEST_API_KEY": "key"},
        phase="core",
        resume=True,
        max_new_attempts=1,
    )
    assert sum(row["status"] == "scored" for row in resumed) == 2
    assert sum(row["retry_policy_amendment_sha256"] is not None for row in resumed) == 1
    assert {row["definition_sha256"] for row in resumed} == {pilot.digest(manifest)}
    runner.write_text("another unapproved edit")
    with pytest.raises(ValueError, match="does not match"):
        pilot.prepare_run(setup, "amend", IMAGE, amend_retry_policy=True)


def test_retry_amendment_rejects_prompt_change(
    setup: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The amendment cannot authorize scientific changes alongside the runner patch."""
    runner = setup / "evals/inspect_pilot.py"
    runner.parent.mkdir()
    runner.write_text("original runner")
    cohort, _, _ = pilot.prepare_run(setup, "badamend", IMAGE)
    archive = cohort / "amendments/original-inspect_pilot.py.txt"
    archive.parent.mkdir()
    archive.write_bytes(runner.read_bytes())
    runner.write_text("changed runner")
    monkeypatch.setattr(pilot, "OUTPUT_CONTRACT", "Different scientific request")
    with pytest.raises(ValueError, match="outside the permitted"):
        pilot.prepare_run(setup, "badamend", IMAGE, amend_retry_policy=True)
