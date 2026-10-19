"""Provider continuation boundaries; offline and independent of actual credentials."""

import json
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.model._providers.openai_compatible import OpenAICompatibleAPI
from test_pilot_runner import synthetic_tasks

from evals import inspect_fireworks as fw
from evals import inspect_main as main
from grader_power.inference import load_access_profiles

ROOT = Path(__file__).resolve().parents[1]


def test_remaining_partition_keeps_order_and_excludes_failed_starts() -> None:
    assignments = main.build_assignments(synthetic_tasks()[:2], 2, 42)
    started = {a.record["assignment_id"] for a in assignments[:19]}
    remaining = fw.remaining_assignments(assignments, started)
    assert remaining == assignments[19:]
    assert not started & {a.record["assignment_id"] for a in remaining}
    with pytest.raises(ValueError, match="Unknown"):
        fw.remaining_assignments(assignments, {"unknown"})


def test_fireworks_request_parameters_and_no_implicit_retries() -> None:
    profile = load_access_profiles(ROOT / fw.PROFILE_PATH)[fw.PROFILE]
    config = main.main_config(ROOT, fw.PROFILE)
    with fw.pilot.zero_retry_clients():
        model = get_model(
            profile.model,
            base_url=profile.base_url,
            api_key="synthetic",
            config=config,
            memoize=False,
        )
        assert isinstance(model.api, OpenAICompatibleAPI)
        params = model.api.completion_params(config, False)
        assert params["temperature"] == 0.7
        assert params["top_p"] == 1
        assert params["reasoning_effort"] == "low"
        assert params["max_tokens"] == 4096
        policy = fw.pilot.resolved_retry_policy(model)
        assert policy["sdk_max_retries"] == 0
        assert policy["http_connect_retries"] == 0


def test_collector_routes_to_explicit_profile_and_does_not_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile_path = tmp_path / fw.PROFILE_PATH
    profile_path.parent.mkdir(parents=True)
    profile_path.write_text(
        "gpt-oss-120b:\n  model: mockllm/fireworks-test\n  base_url: https://example.invalid\n  api_key_env: SYNTHETIC_KEY\n  revision: mock\n  max_tokens: 64\n"
    )
    monkeypatch.setenv("SYNTHETIC_KEY", "synthetic")
    assignments = main.build_assignments(synthetic_tasks()[:1], 1, 42)
    cohort = tmp_path / "cohort"
    config = main.main_config(ROOT, fw.PROFILE)
    definition = {
        "purpose": "mock",
        "oracle_image": "sha256:" + "a" * 64,
        "assignments": [a.record for a in assignments],
        "profiles": {
            fw.PROFILE: {
                "model": "mockllm/fireworks-test",
                "generation": config.model_dump(mode="json", exclude_none=True),
            }
        },
    }
    fw.save(cohort / "manifest.json", definition)
    calls = []

    def model(name: str, **kwargs: Any) -> Any:
        calls.append(name)
        return get_model(
            "mockllm/fireworks-test",
            custom_outputs=[
                ModelOutput.from_content(
                    "mock",
                    "```python\ndef transform(items):\n    return [x for x in items]\n```",
                )
            ]
            * 12,
            config=kwargs["config"],
            memoize=False,
        )

    class Score:
        def to_dict(self) -> dict[str, Any]:
            return {
                "correctness": {"full_correct": True},
                "style": {"label": "comprehension_only"},
            }

    monkeypatch.setattr(fw.pilot, "get_model", model)
    monkeypatch.setattr(fw.pilot, "score_response", lambda *a, **kw: Score())
    original_loader = fw.pilot.load_access_profiles
    fw.collect(tmp_path, cohort, assignments, definition, "core", 2)
    before = {
        p: p.read_bytes() for p in (cohort / fw.PROFILE).glob("attempts/*/raw.json")
    }
    fw.collect(tmp_path, cohort, assignments, definition, "core", 1)
    assert len(before) == 2
    assert all(p.read_bytes() == content for p, content in before.items())
    assert len(list((cohort / fw.PROFILE).glob("attempts/*/started.json"))) == 3
    assert calls == ["mockllm/fireworks-test"] * 2
    assert fw.pilot.load_access_profiles is original_loader
    definition["profiles"][fw.PROFILE]["model"] = "mockllm/wrong"
    (cohort / "manifest.json").write_text(json.dumps(definition))
    with pytest.raises(ValueError, match="disagree"):
        fw.collect(tmp_path, cohort, assignments, definition, "core", 1)


def test_failed_pilot_blocks_main_before_partition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fw, "base_run", lambda _: (tmp_path, (), {}))
    monkeypatch.setattr(fw, "check_pilot", lambda _: {"passed": False})
    with pytest.raises(ValueError, match="not passed"):
        fw.prepare_continuation(tmp_path)
    assert not (tmp_path / fw.CONTINUATION_PATH).exists()


def test_combined_analysis_preserves_missing_and_counts_providers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tasks = synthetic_tasks()[:2]
    assignments = main.build_assignments(tasks, 12, 42)
    remaining = assignments[19:]
    original = tmp_path / "original"
    continuation = original / "fireworks-continuation"
    definition = {"split": {"exploratory": [t.reference.task_id for t in tasks]}}
    fw.save(original / "manifest.json", definition)
    fw.save(continuation / "manifest.json", {"test": True})
    monkeypatch.setattr(fw, "base_run", lambda _: (original, assignments, definition))
    monkeypatch.setattr(
        fw, "load_continuation", lambda _: (continuation, remaining, {})
    )

    def measured(cohort: Path, selected: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [
            {
                **a.record,
                "profile": profile,
                # Preserve one prior failed request and distinguish new results.
                "label": None
                if cohort == original
                and profile == fw.PROFILE
                and (a == assignments[0] or a in remaining)
                else "comprehension_only"
                if cohort == continuation
                else "loop_only",
                "full_correct": False,
            }
            for profile in main.PROFILES
            for a in selected
        ]

    monkeypatch.setattr(main, "measured_rows", measured)
    fw.analyze(tmp_path)
    target = original / "analysis-provider-amended-v1"
    rows = json.loads((target / "measured-rows.json").read_text())
    result = json.loads((target / "analysis.json").read_text())
    assert len(rows) == 2 * len(assignments)
    assert len({(r["profile"], r["assignment_id"]) for r in rows}) == len(rows)
    assert result["missing_labels"] == 1
    assert all(c["p_value"] is None for c in result["comparisons"])
    together = [r for r in rows if r["provider"] == "together"]
    fireworks = [r for r in rows if r["provider"] == "fireworks"]
    assert len(together) == 19
    assert sum(r["label"] is None for r in together) == 1
    assert len(fireworks) == len(remaining)
    assert {r["label"] for r in fireworks} == {"comprehension_only"}
    assert len(result["provider_amendment"]["conflict_descriptives"]) == 9
    with pytest.raises(FileExistsError):
        fw.analyze(tmp_path)
