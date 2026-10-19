"""Minimal independently authored fixture for synthetic wiring tests only.

This is not an EvalPlus adaptation, an eligible task, or a human QA decision.
Only the explicit visible projection goes to the renderer. Mock responses live
in the harness, outside this projection.
"""

from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.schemas import RenderTask, ScoringRegion


def synthetic_task() -> RenderTask:
    """Build the small demonstration contract without reading any dataset cache."""
    return RenderTask(
        source_task_id="synthetic/double-values",
        derived_task_id="synthetic-double-values",
        family="synthetic_mapping_demo",
        stub_path="src/values.py",
        stub=(
            "def double_values(values: list[int]) -> list[int]:\n"
            '    """Return twice each input value, in input order; leave values unchanged."""\n'
            "    # TODO\n"
            "    raise NotImplementedError\n"
        ),
        readme="# Value utilities\n\nSmall utilities for integer sequences.\n",
    )


def synthetic_region() -> ScoringRegion:
    """Declare the demo's output transformation; conveys no human approval."""
    return ScoringRegion(
        task_id="synthetic-double-values",
        entry_point="double_values",
        transformation="Double each input integer in order without input mutation",
        rule="returned_list_v1",
    )


def synthetic_oracle_data() -> LoadedDataset:
    """Return hidden synthetic inputs for the existing EvalPlus Docker adapter.

    The adapter requires a HumanEval-prefixed identifier to select its comparison
    semantics. This ID is explicitly synthetic and is NOT an upstream task or a
    replacement for base/plus benchmark verification. No canonical code executes
    here, and this object must never be passed to the context renderer.
    """
    task_id = "HumanEval/synthetic-double-values"
    return LoadedDataset(
        tasks={
            task_id: {
                "task_id": task_id,
                "entry_point": "double_values",
                "prompt": "def double_values(values):\n",
                "canonical_solution": "    return [2 * value for value in values]\n",
                "base_input": [[[]], [[0]], [[1, -2, 3]]],
                "plus_input": [[[-7, -7, 4]], [[1000000, -1000000]], [[1, 1, 1]]],
                "atol": 0,
            }
        },
        provenance={"variant": "full", "source": "synthetic", "version": "demo-v1"},
    )
