"""Main-run preparation and mocked end-to-end boundaries; no paid providers."""

import asyncio
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ModelOutput, get_model
from test_pilot_runner import synthetic_tasks

from grader_power.main_tasks import main_tasks, source_contract, split_tasks

main = importlib.import_module("evals.inspect_main")
ROOT = Path(__file__).resolve().parents[1]


def test_real_reserved_split_and_complete_contracts() -> None:
    tasks = main_tasks(ROOT)
    split = split_tasks(tasks, "grader-power-main-32-v1")
    assert len(split["exploratory"]) == 32 and len(split["confirmation"]) == 8
    assert not set(split["exploratory"]) & set(split["confirmation"])
    assert split_tasks(tuple(reversed(tasks)), "grader-power-main-32-v1") == split
    assert Counter(t.split("/")[0] for t in split["exploratory"]) == {
        "HumanEval": 12,
        "Mbpp": 20,
    }
    for task in tasks:
        contract = task.dataset.tasks[task.visible.source_task_id].get("contract", "")
        assert contract in task.provenance["input_contract"]
    assert "assert max(numbers) > min(numbers)" in source_contract(
        {"prompt": "Rescale", "contract": "assert max(numbers) > min(numbers)"}
    )
    with pytest.raises(TypeError):
        source_contract({"prompt": "x", "contract": None})


def test_balanced_repetitions_do_not_change_prompt_or_leak_hidden_data() -> None:
    assignments = main.build_assignments(synthetic_tasks()[:2], 12, 42)
    assert (
        len(assignments) == len({a.record["assignment_id"] for a in assignments}) == 288
    )
    prompts = defaultdict(set)
    repeats = defaultdict(set)
    for a in assignments:
        r = a.record
        key = (r["source_task_id"], json.dumps(r["condition"], sort_keys=True))
        prompts[key].add(r["prompt_sha256"])
        repeats[key].add(r["repetition"])
        assert "HIDDEN_" not in str(r["messages"])
    assert len(prompts) == 24 and all(len(x) == 1 for x in prompts.values())
    assert all(x == set(range(12)) for x in repeats.values())
    assert [a.record for a in assignments] == [
        a.record for a in main.build_assignments(synthetic_tasks()[:2], 12, 42)
    ]


def test_skipped_docker_report_is_not_success(tmp_path: Path) -> None:
    report = tmp_path / "test.xml"
    report.write_text(
        "<testsuites><testsuite>" + "<testcase/>" * 9 + "</testsuite></testsuites>"
    )
    main.verify_integration(report)
    report.write_text(
        "<testsuites><testsuite>"
        + "<testcase/>" * 8
        + "<testcase><skipped/></testcase></testsuite></testsuites>"
    )
    with pytest.raises(ValueError, match="non-skipped"):
        main.verify_integration(report)


def test_frozen_artifact_rejects_changes(tmp_path: Path) -> None:
    path = tmp_path / "freeze.json"
    main.freeze(path, {"a": 1})
    main.freeze(path, {"a": 1})
    with pytest.raises(ValueError, match="differs"):
        main.freeze(path, {"a": 2})


def test_mocked_generation_judging_resume_and_tamper_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise actual Inspect collection and both provider boundaries with inert code."""
    configs = tmp_path / "configs"
    configs.mkdir()
    configs.joinpath("models.yaml").write_text(
        "".join(
            f"{name}:\n  model: mockllm/example\n  base_url: https://example.invalid\n  api_key_env: TEST_API_KEY\n  revision: synthetic\n  max_tokens: 64\n"
            for name in main.PROFILES
        )
    )
    monkeypatch.setenv("TEST_API_KEY", "synthetic-secret")
    task = synthetic_tasks()[0]
    task.dataset.tasks[task.visible.source_task_id].update(
        prompt="Copy items.", contract="assert isinstance(items, list)"
    )
    assignments = main.build_assignments((task,), 1, 42)
    cohort = tmp_path / "run"
    cohort.mkdir()
    definition = {
        "purpose": "synthetic_preflight",
        "oracle_image": "sha256:" + "a" * 64,
        "assignments": [a.record for a in assignments],
        "profiles": {
            m: {
                "model": "mockllm/example",
                "generation": main.main_config(tmp_path, m).model_dump(
                    mode="json", exclude_none=True
                ),
            }
            for m in main.PROFILES
        },
        "judge": {
            "profile": "claude-sonnet",
            "system_prompt": "Judge code only.",
            "generation": main.main_config(tmp_path, "claude-sonnet").model_dump(
                mode="json", exclude_none=True
            ),
        },
    }
    main.save(cohort / "manifest.json", definition)
    code = "def transform(items):\n    return [x for x in items]\n"

    def generation_model(*args: Any, **kwargs: Any) -> Any:
        return get_model(
            "mockllm/example",
            custom_outputs=[
                ModelOutput.from_content("mock", "```python\n" + code + "```")
            ]
            * 12,
            config=kwargs["config"],
            memoize=False,
        )

    monkeypatch.setattr(main.pilot, "get_model", generation_model)

    class Score:
        def to_dict(self) -> dict[str, Any]:
            return {
                "correctness": {"full_correct": True},
                "style": {"label": "comprehension_only"},
            }

    monkeypatch.setattr(main.pilot, "score_response", lambda *a, **k: Score())
    for profile in main.PROFILES:
        main.pilot.export_results(cohort / profile, assignments, profile, "synthetic")
        rows = main.pilot.collect_profile(
            tmp_path,
            cohort,
            assignments,
            profile,
            definition["oracle_image"],
            {"TEST_API_KEY": "synthetic-secret"},
            phase="core",
            max_new_attempts=2,
            generation_override=main.main_config(tmp_path, profile),
            export_every=48,
        )
        assert sum(r["status"] == "scored" for r in rows) == 2
        raw_before = {
            p: p.read_bytes() for p in (cohort / profile).glob("attempts/*/raw.json")
        }
        main.pilot.collect_profile(
            tmp_path,
            cohort,
            assignments,
            profile,
            definition["oracle_image"],
            {"TEST_API_KEY": "synthetic-secret"},
            phase="core",
            resume=True,
            max_new_attempts=1,
            generation_override=main.main_config(tmp_path, profile),
            export_every=48,
        )
        assert all(p.read_bytes() == data for p, data in raw_before.items())
    calls = []

    class Judge:
        async def generate(self, messages: Any, **kwargs: Any) -> ModelOutput:
            packet = json.loads(messages[1].content)
            assert set(packet) == {
                "entry_point",
                "transformation",
                "input_contract",
                "code",
                "code_lines",
            }
            assert "assert isinstance(items, list)" in packet["input_contract"]
            assert kwargs["cache"] is False and kwargs["tools"] == []
            calls.append(packet)
            return ModelOutput.from_content(
                "mock",
                json.dumps(
                    {
                        "status": "classified",
                        "target_forms": ["list_comprehension"],
                        "evidence": [
                            {
                                "start_line": 2,
                                "end_line": 2,
                                "quote": "    return [x for x in items]",
                            }
                        ],
                        "flags": [],
                        "rationale": "Returned comprehension.",
                    }
                ),
            )

    monkeypatch.setattr(main, "get_model", lambda *a, **k: Judge())
    monkeypatch.setattr(
        main.pilot,
        "resolved_retry_policy",
        lambda _: {
            "inspect_max_retries": 0,
            "sdk_max_retries": 0,
            "http_connect_retries": 0,
        },
    )
    for profile in main.PROFILES:
        asyncio.run(
            main.collect_judgments(
                tmp_path, cohort, assignments, definition, profile, 48
            )
        )
        asyncio.run(
            main.collect_judgments(
                tmp_path, cohort, assignments, definition, profile, 48
            )
        )
    assert len(calls) == 6
    measured = main.measured_rows(cohort, assignments)
    assert Counter(r["label"] for r in measured) == {None: 18, "comprehension_only": 6}
    result = next((cohort / "gpt-oss-120b/judge").glob("*/result.json"))
    data = json.loads(result.read_text())
    data["source_raw_sha256"] = "0" * 64
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="binding"):
        main.measured_rows(cohort, assignments)
    result.unlink()
    asyncio.run(
        main.collect_judgments(
            tmp_path, cohort, assignments, definition, "gpt-oss-120b", 48
        )
    )
    assert len(calls) == 6  # An interrupted, started judge request is never repeated.
    packet = json.loads((result.parent / "packet.json").read_text())
    main.save(
        result.parent / "adjudication.json",
        {
            "packet_sha256": main.digest(main.json_text(packet)),
            "blinded": True,
            "reviewer": "synthetic human",
            "rationale": "Independent test adjudication",
            "label": "loop_only",
        },
    )
    measured = main.measured_rows(cohort, assignments)
    assert sum(r["label"] == "loop_only" for r in measured) == 1
