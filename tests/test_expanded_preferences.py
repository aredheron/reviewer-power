"""Regression checks for task extraction, treatment isolation and duplicate avoidance."""

import ast
import copy
from collections import Counter
from itertools import product
from pathlib import Path

import pytest

from grader_power import expanded_preference_study as study
from grader_power.datasets import mbpp_full
from grader_power.expanded_preference_scoring import packet_from_code, score_task
from grader_power.expanded_preference_tasks import (
    MARKERS,
    load_tasks,
    replace_task,
    signature_stub,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return load_tasks(ROOT)


@pytest.fixture(scope="module")
def records(tasks: list[dict]) -> list[dict]:
    return study.schedule(ROOT, tasks)


def test_exact_admitted_pool_and_reference_bindings(tasks: list[dict]) -> None:
    assert Counter(t["dataset"] for t in tasks) == {
        "evalplus": 14,
        "mbpp_full": 24,
        "apps": 34,
    }
    assert len({t["task_id"] for t in tasks}) == 72
    assert all(len(t["reference_evidence"]) == 3 for t in tasks)
    assert not {t["task_id"] for t in tasks} & set(study.old_coverage(ROOT)["tasks"])


def test_exact_public_text_and_interfaces(tasks: list[dict]) -> None:
    for task in tasks:
        assert task["original_statement"] in task["readme"] + task["stub"]
        assert task["original_statement"] in task["input_contract"]
        ast.parse(task["stub"])
        for code in task["implementations"].values():
            assert mbpp_full.signature(
                code, task["entry_point"]
            ) == mbpp_full.signature(task["stub"], task["entry_point"])
        target = next(
            n
            for n in ast.parse(task["stub"]).body
            if isinstance(n, ast.FunctionDef) and n.name == task["entry_point"]
        )
        assert not any(
            isinstance(n, (ast.For, ast.ListComp, ast.While, ast.Return))
            for n in ast.walk(target)
        )


def test_apps_starter_names_and_optional_default_are_preserved(
    tasks: list[dict],
) -> None:
    by_id = {t["task_id"]: t for t in tasks}
    assert by_id["APPS/train/3796"]["stub"].startswith("def or_arrays(a, b, filler=0):")
    assert by_id["APPS/train/3534"]["stub"].startswith("def to_bits(string):")
    assert "5000 bit vector" in by_id["APPS/train/3534"]["readme"]
    assert "```python" in by_id["APPS/train/4321"]["readme"]
    assert 'contact": 8167238327' in by_id["APPS/train/4321"]["readme"]


def test_mbpp_public_example_is_identified_and_remaining_tests_withheld(
    tasks: list[dict],
) -> None:
    for task in tasks:
        if task["dataset"] != "mbpp_full":
            continue
        assert task["public_test_indices"] == [0]
        assert task["source"]["test_list"][0] in task["readme"]
        assert task["source"]["test_list"][0] in task["input_contract"]
        for hidden in (
            task["source"]["test_list"][1:] + task["source"]["challenge_test_list"]
        ):
            assert hidden not in task["readme"] + task["stub"] + task["input_contract"]


def test_all_prompts_preserve_condition_context_and_complete_task(
    tasks: list[dict], records: list[dict]
) -> None:
    contexts = study.templates(ROOT)
    by_task = {t["task_id"]: t for t in tasks}
    variants: dict[tuple, set[str]] = {}
    for row in records:
        if row["kind"] != "core":
            continue
        arm, user = row["condition"]["arm"], row["condition"]["user_preference"]
        old = contexts[arm, user]["messages"][0]["content"]
        content = row["messages"][0]["content"]
        assert old.split(MARKERS[0])[0] == content.split(MARKERS[0])[0]
        assert old.split(MARKERS[2])[1] == content.split(MARKERS[2])[1]
        task = by_task[row["source_task_id"]]
        assert (
            content.split(MARKERS[0] + "\n")[1].split("\n\n" + MARKERS[1])[0]
            == task["readme"]
        )
        assert (
            content.split(MARKERS[1] + "\n")[1].split("\n\n" + MARKERS[2])[0]
            == task["stub"]
        )
        assert all(content.count(m) == 1 for m in MARKERS)
        variants.setdefault((task["task_id"], arm, user), set()).add(content)
    assert len(variants) == 864
    assert all(len(v) == 1 for v in variants.values())


def test_fresh_matrix_and_pilot_balance(tasks: list[dict], records: list[dict]) -> None:
    main = [r for r in records if r["phase"] == "main"]
    keys = Counter(
        (
            r["source_task_id"],
            r["profile"],
            r["condition"]["arm"],
            r["condition"]["user_preference"],
            r["repetition_index"],
        )
        for r in main
    )
    assert set(keys) == set(
        product(
            [t["task_id"] for t in tasks],
            study.PROFILES,
            study.ARMS,
            study.STYLES,
            range(3),
        )
    )
    assert set(keys.values()) == {1}
    pilot = [r for r in records if r["phase"] == "pilot"]
    assert len(main) == 7776 and len(pilot) == 360
    for model in study.PROFILES:
        code = [r for r in pilot if r["profile"] == model and r["kind"] == "core"]
        facts = [
            r
            for r in pilot
            if r["profile"] == model and r["kind"] == "comprehension_probe"
        ]
        assert len(code) == 72 and len({r["source_task_id"] for r in code}) == 72
        assert set(
            Counter(
                (r["condition"]["arm"], r["condition"]["user_preference"]) for r in code
            ).values()
        ) == {6}
        assert len(facts) == 48
        assert Counter(r["condition"]["arm"] for r in facts) == dict.fromkeys(
            study.ARMS, 8
        )
    assert len({r["assignment_id"] for r in records}) == 8136


def test_old_coverage_includes_all_cells_and_fixed_recovery_lineage() -> None:
    coverage = study.old_coverage(ROOT)
    assert coverage["observations"] == coverage["saved"] == coverage["resolved"] == 3456
    assert len(coverage["tasks"]) == 32 and coverage["lineage"]
    assert {r["repetition_index"] for r in coverage["rows"]} == {0, 1, 2}


def test_signatures_are_only_parsed_not_evaluated(tmp_path: Path) -> None:
    marker = tmp_path / "should-not-exist"
    code = f"def f(x=open({str(marker)!r}, 'w')):\n    return []\n"
    with pytest.raises(ValueError, match="Nonliteral default"):
        signature_stub(code, "f")
    assert not marker.exists()


@pytest.mark.parametrize(
    "code",
    ["def f(a):\n    pass\ndef f(b):\n    pass", "@decorator\ndef f(a):\n    pass"],
)
def test_ambiguous_interfaces_fail_closed(code: str) -> None:
    with pytest.raises(ValueError):
        signature_stub(code, "f")


def test_changed_prompt_boundary_is_rejected(tasks: list[dict]) -> None:
    source = copy.deepcopy(study.templates(ROOT)["strong_shadow", "loops"]["messages"])
    source[0]["content"] += "\n--- AGENTS.md ---\n"
    with pytest.raises(ValueError, match="section markers"):
        replace_task(source, tasks[0])


def test_blinded_packets_contain_only_public_contract_and_code(
    tasks: list[dict],
) -> None:
    for task in tasks:
        for code in task["implementations"].values():
            data = packet_from_code(code, task).model_dump()
            assert set(data) == {
                "entry_point",
                "transformation",
                "input_contract",
                "code",
            }
            assert data["input_contract"] == task["input_contract"]


def test_invalid_python_does_not_launch_docker(
    tasks: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise AssertionError("Invalid code must not reach Docker")

    monkeypatch.setattr(mbpp_full, "evaluate_assertions", fail)
    task = next(t for t in tasks if t["dataset"] == "mbpp_full")
    result = score_task(ROOT, task, "```python\ndef broken(\n```", "unused")
    assert result["correctness"]["full_correct"] is False
    assert result["correctness"]["tests_run"] == 0


def test_budget_proposal_requires_new_approval(tasks: list[dict]) -> None:
    proposal = study.budget_proposal(ROOT, tasks)
    assert proposal["paid_authorized"] is False
    assert (
        proposal["matching_history_estimate"]
        > proposal["prior_combined_ceiling"] - proposal["prior_combined_recorded"]
    )
    assert (
        proposal["matching_history_estimate"]
        < proposal["requested_new_whole_study_ceiling"]
    )
    assert proposal["pilot_reforecast_required"] is True
