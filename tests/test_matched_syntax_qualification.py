"""Approved exact factual qualification, fresh designation and immutable readiness."""

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grader_power import matched_syntax as study
from grader_power import matched_syntax_qualification as followup


@pytest.fixture(scope="module")
def records() -> list[dict[str, Any]]:
    return study.schedule(ROOT)[0]


@pytest.fixture
def expected() -> dict[str, Any]:
    return study.expected_facts(
        {
            "study": "naming",
            "condition": study.condition(
                "naming", "reviewer_unseen", "snake_case", "snake_case"
            ),
        }
    )


@pytest.mark.parametrize(
    "wrapper", ("plain", "whitespace", "fence", "prose", "repeated", "three")
)
def test_qualified_objects_remain_exact_and_agree(expected, wrapper):
    payload = json.dumps(expected)
    response = {
        "plain": payload,
        "whitespace": " \n" + payload + "\n",
        "fence": "```json\n" + payload + "\n```",
        "prose": payload + " trailing explanation",
        "repeated": payload + "<|assistant|> explanation </think>" + payload,
        "three": payload + " prose " + payload + " prose " + payload,
    }[wrapper]
    prior = copy.deepcopy(expected)
    result = followup.qualify_factual_format(
        response, expected, generation_complete=True
    )
    assert result["qualified_correct"] and result["all_correct"]
    assert result["first_answer"] == expected and expected == prior
    assert result["subsequent_object_count"] == {"repeated": 1, "three": 2}.get(
        wrapper, 0
    )


@pytest.mark.parametrize(
    "variant",
    (
        "empty",
        "prefix",
        "truncated",
        "duplicate_first",
        "duplicate_later",
        "wrong_first",
        "wrong_later",
        "extra",
        "wrong_type",
        "bad_later",
        "nan",
        "array",
        "fence_plus_prose",
        "truncated_later",
        "multiple_fences",
    ),
)
def test_qualification_never_repairs_semantics_or_incomplete_objects(expected, variant):
    payload = json.dumps(expected)
    wrong = json.dumps(expected | {"actor_assesses_output": True})
    duplicate = payload[:-1] + ',"actor_assesses_output":false}'
    response = {
        "empty": "",
        "prefix": "Here is the answer: " + payload,
        "truncated": payload[:-1],
        "duplicate_first": duplicate,
        "duplicate_later": payload + " prose " + duplicate,
        "wrong_first": wrong + payload,
        "wrong_later": payload + wrong,
        "extra": json.dumps(expected | {"extra": 1}),
        "wrong_type": json.dumps(expected | {"actor_mentioned": 1}),
        "bad_later": payload + " {broken}",
        "nan": payload + '{"not_a_number": NaN}',
        "array": "[" + payload + "]",
        "fence_plus_prose": "```json\n" + payload + "\n``` more text",
        "truncated_later": payload + payload[:-1],
        "multiple_fences": "```json\n"
        + payload
        + "\n```\n```json\n"
        + payload
        + "\n```",
    }[variant]
    result = followup.qualify_factual_format(
        response, expected, generation_complete=True
    )
    assert not result["qualified_correct"] and not result["all_correct"]


def test_correct_initial_json_does_not_rescue_capped_generation(expected):
    result = followup.qualify_factual_format(
        json.dumps(expected), expected, generation_complete=False
    )
    assert (
        not result["qualified_correct"] and result["reason"] == "incomplete_generation"
    )
    invalid_complete: Any = 1
    with pytest.raises(TypeError):
        followup.qualify_factual_format(
            "", expected, generation_complete=invalid_complete
        )
    with pytest.raises(ValueError):
        followup.qualify_factual_format("{}", {"wrong": True}, generation_complete=True)
    with pytest.raises(ValueError):
        followup.qualify_factual_format(
            "{}", expected | {"actor_name": "unknown"}, generation_complete=True
        )


def test_other_style_enums_and_absence_remain_exact():
    for family, choices in study.STYLES.items():
        for arm in ("absent", "reviewer_no_consequences"):
            r = {
                "study": family,
                "condition": study.condition(
                    family, arm, choices[0], None if arm == "absent" else choices[1]
                ),
            }
            answer = study.expected_facts(r)
            assert followup.qualify_factual_format(
                json.dumps(answer), answer, generation_complete=True
            )["qualified_correct"]


def test_closed_pilot_cannot_be_collected(monkeypatch, tmp_path):
    from evals import inspect_matched_syntax as runner

    monkeypatch.setattr(runner, "load", lambda *args, **kwargs: ({}, []))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    with pytest.raises(ValueError, match="pilot cohort is closed"):
        runner.collect(tmp_path, "pilot")


def test_uniform_overlay_revalidates_every_original_and_preserves_rows(
    tmp_path, monkeypatch
):
    from inspect_ai.model import ModelOutput

    from evals.inspect_mode_a import digest

    # 432 synthetic probes test complete coverage and original hashes without using live outcomes.
    record = {
        "phase": "pilot",
        "study": "quotes",
        "kind": "comprehension_probe",
        "condition": study.condition(
            "quotes", "reviewer_no_consequences", *study.STYLES["quotes"]
        ),
    }
    records = [record | {"assignment_id": str(i)} for i in range(432)]
    answer = study.expected_facts(record)
    facts = []
    folder = tmp_path / "operations/recoveries/factual-format-v1"
    folder.mkdir(parents=True)
    proposal = folder / "proposal.json"
    proposal.write_text("{}")
    approval = {
        "answer": "Apply qualification; admit GLM quotes (Recommended)",
        "proposal_path": str(proposal.relative_to(tmp_path)),
        "proposal_sha256": digest(proposal.read_bytes()),
    }
    (folder / "approval.json").write_text(json.dumps(approval))
    text = json.dumps(answer) + " explanation " + json.dumps(answer)
    raw = ModelOutput.from_content("test", text)
    raw.choices[0].stop_reason = "stop"
    for r in records:
        path = tmp_path / "requests" / r["assignment_id"]
        (path / "generation").mkdir(parents=True)
        (path / "generation/raw.json").write_text(raw.model_dump_json())
        (path / "probe.json").write_text('{"strict":false}')
        facts.append(
            {
                "assignment_id": r["assignment_id"],
                "raw_sha256": digest((path / "generation/raw.json").read_bytes()),
                "probe_sha256": digest((path / "probe.json").read_bytes()),
                "qualification": followup.qualify_factual_format(
                    text, answer, generation_complete=True
                ),
            }
        )
    saved = {
        "facts": facts,
        "approval_sha256": digest((folder / "approval.json").read_bytes()),
        "proposal_sha256": approval["proposal_sha256"],
    }
    qpath = folder / "qualification.json"
    qpath.write_text(json.dumps(saved))
    rows = [r | {"probe_correct": False} for r in records]
    before = copy.deepcopy(rows)
    qualified = followup.qualified_rows(tmp_path, rows, records)
    assert all(r["probe_correct"] for r in qualified) and rows == before
    for invalid in (facts[:-1], facts + [facts[0]]):
        qpath.write_text(json.dumps(saved | {"facts": invalid}))
        with pytest.raises(ValueError, match="432"):
            followup.qualified_rows(tmp_path, rows, records)
    qpath.write_text(json.dumps(saved))
    (tmp_path / "requests/0/probe.json").write_text("changed")
    with pytest.raises(ValueError, match="Original factual outcome changed"):
        followup.qualified_rows(tmp_path, rows, records)
