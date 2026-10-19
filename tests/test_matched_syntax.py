"""Prospective scope, prompt pairing, readiness and paired-analysis invariants."""

import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import matched_syntax as study
from grader_power import matched_syntax_analysis as analysis
from grader_power import style_preference_experiment as syntax

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records():
    return study.schedule(ROOT)[0]


def test_scope_and_public_task_identity(records):
    assert len(records) == 21600 and len({r["assignment_id"] for r in records}) == 21600
    assert Counter((r["phase"], r["kind"]) for r in records) == {
        ("main", "core"): 20736,
        ("pilot", "core"): 432,
        ("pilot", "comprehension_probe"): 432,
    }
    catalog = syntax.build_task_catalog(ROOT)
    for r in records:
        text = r["messages"][0]["content"]
        assert catalog[r["study"]][r["source_task_id"]]["stub"] in text
        assert "canonical_solution" not in text and "plus_input" not in text
        if r["condition"]["arm"] == "absent" and r["kind"] == "core":
            assert not re.search(r"\b(review(?:er)?|score)\b", text, re.IGNORECASE)
    assert all(
        len(
            {
                r["source_task_id"]
                for r in records
                if r["phase"] == "main" and r["study"] == f
            }
        )
        == 24
        for f in study.STYLES
    )


def test_only_boolean_changes_in_impactful_prompts(records):
    template = next(r for r in records if r["phase"] == "main")
    task = syntax.build_task_catalog(ROOT)[template["study"]][
        template["source_task_id"]
    ]

    def render(arm):
        r = dict(
            template,
            condition=study.condition(
                template["study"], arm, *study.STYLES[template["study"]]
            ),
        )
        return study.render_messages(r, task, root=ROOT)[0]["content"]

    baseline = render("reviewer_no_consequences")
    assert render("reviewer_rl_update") == baseline.replace(
        "review_for_training: false", "review_for_training: true"
    )
    assert render("reviewer_future_selection") == baseline.replace(
        "review_for_selection: false", "review_for_selection: true"
    )


def test_strict_facts_and_semantic_errors(records):
    for r in records:
        if r["kind"] != "comprehension_probe":
            continue
        answer = study.expected_facts(r)
        text = json.dumps(answer)
        assert study.score_probe(text, r)["strict"]["all_correct"]
        assert study.score_probe("```json\n" + text + "\n```", r)["fence_only"][
            "all_correct"
        ]
        assert not study.score_probe(text + text, r)["fence_only"]["all_correct"]
        answer["actor_assesses_output"] = not answer["actor_assesses_output"]
        assert not study.score_probe(json.dumps(answer), r)["strict"]["all_correct"]


def perfect_pilot(records):
    return [
        r
        | {
            "raw_saved": True,
            "full_correct": True,
            "probe_correct": True,
            "label": study.STYLES[r["study"]][0] + "_only",
        }
        for r in records
        if r["phase"] == "pilot"
    ]


def test_gate_is_separate_and_enforces_naming_mix(records):
    rows = perfect_pilot(records)
    assert study.pilot_gate(rows)["passed"]
    selected = [
        r
        for r in rows
        if r["study"] == "naming" and r["profile"] == "glm" and r["kind"] == "core"
    ]
    for r in selected[:15]:
        r["label"] = "mixed"
    gates = study.pilot_gate(rows)["groups"]
    assert [(g["study"], g["profile"]) for g in gates if not g["passed"]] == [
        ("naming", "glm")
    ]


def test_gate_rejects_duplicate_or_missing_cells(records):
    rows = perfect_pilot(records)
    assert not study.pilot_gate(rows[:-1] + [rows[0]])["passed"]


def test_paired_analysis_complete_missing_and_wrong_model(records):
    rows = [
        r
        | {"label": r["condition"]["user_preference"] + "_only", "full_correct": False}
        for r in records
        if r["phase"] == "main"
    ]
    result = analysis.analyze(rows, draws=100)
    assert [f["holm_family_size"] for f in result["families"]] == [6, 12, 6]
    assert all(f["inference_status"] == "complete" for f in result["families"])
    missing = next(r for r in rows if r["condition"]["arm"] == "absent")
    missing["label"] = None
    result = analysis.analyze(rows, draws=100)
    assert result["families"][0]["inference_status"] == "withheld_unresolved"
    assert all(c["p_value"] is None for c in result["families"][0]["comparisons"])
    assert result["families"][1]["inference_status"] == "complete"
    bad = copy.deepcopy(rows)
    bad[0]["model"] = "wrong"
    with pytest.raises(ValueError, match="provenance"):
        analysis.analyze(bad, draws=100)


def test_no_evidence_is_separate_observed_zero(records):
    rows = [
        r
        | {
            "label": "no_naming_evidence"
            if r["study"] == "naming"
            else "no_quote_evidence"
        }
        for r in records
        if r["phase"] == "main"
    ]
    result = analysis.analyze(rows, draws=100)
    assert result["missing_labels"] == 0
    assert all(s["opposite_style_bounds"] == [0, 0] for s in result["by_condition"])
