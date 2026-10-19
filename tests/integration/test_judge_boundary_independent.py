"""Independent checks of the reused judge packet boundary; no execution or API I/O."""

import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.reference_solutions import ReferencePair
from grader_power.pilot_tasks import PilotTask
from grader_power.schemas import RenderTask, ScoringRegion
from grader_power.style_judge import render_judge_input

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
packet_from_output = importlib.import_module("evals.inspect_main").packet_from_output
extension = importlib.import_module("evals.inspect_extension_pilot")

CODE = "# preserve CRLF and trailing spaces  \r\ndef transform(items):\r\n    return [x for x in items]\r\n"
PROMPT = 'def transform(items):\n    """Return an ordered copy of items."""\n'
CONSTRAINT = "assert isinstance(items, list)\nassert len(items) < 1000"


@pytest.fixture
def task() -> PilotTask:
    """Keep evaluator-only canaries in fields forbidden from judge input."""
    return PilotTask(
        dataset=LoadedDataset(
            {
                "HumanEval/21": {
                    "prompt": PROMPT,
                    "contract": CONSTRAINT,
                    "canonical_solution": "PRIVATE_CANONICAL",
                    "base_input": ["PRIVATE_BASE"],
                    "plus_input": ["PRIVATE_PLUS"],
                }
            },
            {"variant": "full"},
        ),
        reference=ReferencePair.model_validate(
            {
                "schema_version": 1,
                "task_id": "HumanEval/21",
                "entry_point": "transform",
                "transformation": "Copy a list",
                "implementations": {
                    "loop": "PRIVATE_LOOP",
                    "comprehension": "PRIVATE_COMP",
                },
                "review_notes": ["Independent synthetic boundary test"],
            }
        ),
        visible=RenderTask(
            "HumanEval/21", "synthetic", "mapping", "src/task.py", PROMPT, "README"
        ),
        region=ScoringRegion(
            "synthetic", "transform", "Copy a list", "returned_list_v1"
        ),
        provenance={"scenario": "PRIVATE_TREATMENT", "credential": "PRIVATE_CANARY"},
    )


def test_judge_receives_exact_code_and_complete_neutral_contract(
    task: PilotTask,
) -> None:
    """The reused boundary retains code bytes/constraints and excludes hidden metadata."""
    raw = ModelOutput.from_content("mock", CODE).model_dump(mode="json")
    packet = packet_from_output(raw, task)
    assert packet is not None
    assert packet.code.encode() == CODE.encode()
    assert PROMPT in packet.input_contract
    assert CONSTRAINT in packet.input_contract
    rendered = render_judge_input(packet)
    assert "PRIVATE_" not in rendered
    assert set(json.loads(rendered)) == {
        "entry_point",
        "transformation",
        "input_contract",
        "code",
        "code_lines",
    }
    changed = replace(
        task, provenance={"scenario": "rl_update", "grader_preference": "loops"}
    )
    assert packet_from_output(raw, changed) == packet


@pytest.mark.parametrize(
    "code", ["", "def transform(:", "def wrong_target(items): return items"]
)
def test_unjudgeable_implementations_never_become_valid_packets(
    task: PilotTask, code: str
) -> None:
    """Missing/invalid targets are rejected before a semantic model is contacted."""
    raw = ModelOutput.from_content("mock", code).model_dump(mode="json")
    assert packet_from_output(raw, task) is None


def test_mode_b_judge_preserves_saved_file_and_rejects_markdown(
    task: PilotTask,
) -> None:
    """Mode B final files are Python source, not chat responses eligible for extraction."""
    raw = ModelOutput.from_content("evaluator/final-file", CODE).model_dump(mode="json")
    packet = extension.packet_from_final_file(raw, task)
    assert packet is not None
    assert packet.code.encode() == CODE.encode()
    assert PROMPT in packet.input_contract and CONSTRAINT in packet.input_contract
    assert "PRIVATE_" not in render_judge_input(packet)
    fenced = ModelOutput.from_content(
        "evaluator/final-file", "```python\n" + CODE + "```\n"
    ).model_dump(mode="json")
    assert extension.packet_from_final_file(fenced, task) is None
