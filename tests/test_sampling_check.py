"""Sampling diagnostic invariants without paid calls or execution of generated code."""

import importlib
from collections import Counter, defaultdict
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput
from test_pilot_runner import synthetic_tasks

sampling = importlib.import_module("evals.inspect_sampling_check")
ROOT = Path(__file__).resolve().parents[1]


def test_repetitions_have_distinct_ids_but_identical_prompts() -> None:
    assignments = sampling.build_assignments(synthetic_tasks(), 17)
    assert len({a.record["assignment_id"] for a in assignments}) == 96
    assert Counter(a.record["kind"] for a in assignments) == {
        "core": 72,
        "comprehension_probe": 24,
    }
    cells = defaultdict(list)
    for a in assignments:
        assert "HIDDEN_" not in str(a.record["messages"])
        c = a.record["condition"]
        cells[(a.record["kind"], a.record["source_task_id"], *c.values())].append(
            a.record
        )
    for key, rows in cells.items():
        expected = 3 if key[0] == "core" else 1
        assert len(rows) == expected
        assert {r["repetition"] for r in rows} == set(range(expected))
        assert len({r["prompt_sha256"] for r in rows}) == 1
    assert [a.record for a in assignments] == [
        a.record for a in sampling.build_assignments(synthetic_tasks(), 17)
    ]


def test_sampling_changes_only_temperature() -> None:
    profiles = sampling.load_access_profiles(ROOT / "configs/models.yaml")
    original = sampling.generation_config(profiles[sampling.PROFILE]).model_dump()
    actual = sampling.sampling_config(ROOT).model_dump()
    assert actual == {**original, "temperature": 0.7}


def test_judge_packet_uses_public_contract_and_rejects_invalid_output() -> None:
    task = synthetic_tasks()[0]
    task.dataset.tasks[task.visible.source_task_id]["prompt"] = "Transform a list."
    task.dataset.tasks[task.visible.source_task_id]["contract"] = (
        "assert len(items) >= 2"
    )
    raw = ModelOutput.from_content(
        "private-model-name",
        "```python\ndef transform(items):\n    return [x for x in items]\n```",
    ).model_dump(mode="json")
    packet = sampling.judge_packet(raw, task)
    assert packet is not None
    assert (
        packet.input_contract
        == "Transform a list.\n\nEvalPlus input preconditions:\nassert len(items) >= 2"
    )
    assert set(packet.model_dump()) == {
        "entry_point",
        "transformation",
        "input_contract",
        "code",
    }
    assert "private-model-name" not in str(packet)
    assert "HIDDEN_" not in str(packet)
    invalid = ModelOutput.from_content("mock", "def broken(").model_dump(mode="json")
    assert sampling.judge_packet(invalid, task) is None


def test_requires_two_controls() -> None:
    with pytest.raises(ValueError, match="two"):
        sampling.build_assignments(synthetic_tasks()[1:], 17)
