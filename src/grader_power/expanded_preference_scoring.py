"""Blinded packets and verified, dataset-specific Docker scoring for new tasks."""

from functools import lru_cache
from pathlib import Path
from typing import Any

from inspect_ai.model import ModelOutput

from grader_power.code_extract import extract_code
from grader_power.datasets import apps_source, mbpp_full
from grader_power.datasets.evalplus_loader import LoadedDataset
from grader_power.datasets.task_filter import load_screening_pool
from grader_power.schemas import ScoringRegion
from grader_power.scoring import OracleRequest, score_response
from grader_power.style_judge import JudgeInput


@lru_cache(maxsize=1)
def evalplus_datasets(root: Path) -> tuple[LoadedDataset, ...]:
    """Verify the pinned EvalPlus sources once per worker process."""
    return tuple(
        load_screening_pool(
            root / "data/evalplus", root / "manifests/evalplus_source.json"
        )
    )


def score_task(
    root: Path, task: dict[str, Any], response: str, image: str
) -> dict[str, Any]:
    """Retain extraction outcomes; execute candidate code only in a verified sandbox."""
    region = ScoringRegion(
        task_id=task["task_id"],
        entry_point=task["entry_point"],
        transformation=task["transformation"],
        rule="returned_list_v1",
    )
    if task["dataset"] == "evalplus":
        dataset = next(d for d in evalplus_datasets(root) if task["task_id"] in d.tasks)
        name = "humaneval" if task["task_id"].startswith("HumanEval/") else "mbpp"
        return score_response(
            response,
            region,
            oracle=OracleRequest(dataset, name, task["task_id"], image),
        ).to_dict()
    score = score_response(response, region).to_dict()
    code = score["extraction"]["code"]
    if code is None or score["compile_status"] != "valid":
        score["correctness"] = {
            "status": "invalid_source",
            "full_correct": False,
            "tests_run": 0,
        }
        return score
    if task["dataset"] == "mbpp_full":
        result = mbpp_full.evaluate_assertions(task["source"], code, image)
        result["public_example_indices"] = task["public_test_indices"]
        result["public_example_passed"] = result["details"][0]["passed"]
        result["withheld_assertions"] = len(result["details"]) - 1
        result["withheld_all_passed"] = all(r["passed"] for r in result["details"][1:])
    elif task["dataset"] == "apps":
        result = apps_source.evaluate(task["source"], code, image)
    else:
        raise ValueError("Unknown task source; no execution fallback")
    score["correctness"] = {
        **result,
        "status": "pass" if result["passed"] else "fail",
        "full_correct": result["passed"],
    }
    return score


def packet_from_code(code: str, task: dict[str, Any]) -> JudgeInput:
    """Exclude provider, model, condition, preference, reference labels and hidden tests."""
    return JudgeInput(
        entry_point=task["entry_point"],
        transformation=task["transformation"],
        input_contract=task["input_contract"],
        code=code,
    )


def packet_from_output(raw: dict[str, Any], task: dict[str, Any]) -> JudgeInput | None:
    """Extract only the final saved code, retaining the unchanged strict judge schema."""
    output = ModelOutput.model_validate(raw)
    extraction = extract_code(output.completion, task["entry_point"])
    if extraction.status != "ok" or extraction.code is None:
        return None
    return packet_from_code(extraction.code, task)
