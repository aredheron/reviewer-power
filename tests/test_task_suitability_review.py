"""Audit authored suitability artifacts without source caches or task execution."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECOMMENDATIONS = {
    "suitable_for_reference_validation",
    "needs_resolution",
    "unsuitable",
}
CRITERIA = {
    "output_and_transformation",
    "natural_competing_implementations",
    "behavioral_equivalence",
    "determinism_and_practicality",
    "style_neutrality",
    "scoring_compatibility",
    "oracle_readiness",
}


@pytest.fixture
def records() -> list[dict[str, Any]]:
    """Read tracked metadata only; no upstream task text or hidden input access."""
    path = ROOT / "manifests/task_suitability_review.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_review_population_is_exact(records: list[dict[str, Any]]) -> None:
    """All and only the requested 84 IDs receive one authored review record."""
    candidates = [
        json.loads(line)
        for line in (ROOT / "manifests/task_candidates.jsonl").read_text().splitlines()
    ]
    selected = {
        row["task_id"]: row
        for row in candidates
        if "list_output_evidence" in row["reason_codes"]
    }
    ids = [row["task_id"] for row in records]
    assert len(selected) == len(ids) == len(set(ids)) == 84
    assert set(ids) == set(selected)
    assert Counter(row["source"]["name"] for row in records) == {
        "HumanEval+": 35,
        "MBPP+": 49,
    }
    assert "Mbpp/793" not in selected
    for row in records:
        original = selected[row["task_id"]]
        assert row["entry_point"] == original["entry_point"]
        assert row["source"] == original["source"]
        assert row["original_screening"] == {
            key: original[key] for key in row["original_screening"]
        }


def test_review_provenance_links(records: list[dict[str, Any]]) -> None:
    """Source and screening content must still match the reviewed versions."""
    paths = {
        "source_manifest": "manifests/evalplus_source.json",
        "screening_manifest": "manifests/task_candidates.jsonl",
        "screening_review": "planning/task_screening_review.md",
        "scorer": "src/grader_power/score_style.py",
    }
    hashes = {
        name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    source = json.loads((ROOT / paths["source_manifest"]).read_text())
    by_source = {item["name"]: item for item in source["datasets"]}
    for row in records:
        assert row["evalplus_version"] == source["evalplus_version"]
        for name, path in paths.items():
            assert row["provenance"][name] == {
                "path": path,
                "sha256": hashes[name],
            }
        for key in (
            "prompt_sha256",
            "canonical_solution_sha256",
            "auxiliary_contract_sha256",
        ):
            assert re.fullmatch(r"[0-9a-f]{64}", row["provenance"][key])
        expected = by_source[row["source"]["name"]]
        assert row["source"] == {key: expected[key] for key in row["source"]}
        assert row["task_id"] in expected["original_task_ids"]


def test_decision_and_verification_are_separate(records: list[dict[str, Any]]) -> None:
    """A positive agent judgment must not become approval or an oracle pass."""
    for row in records:
        assert row["recommendation"] in RECOMMENDATIONS
        assert row["reviewer_kind"] == "agent"
        assert row["statuses"] == {
            "semantic_review": "agent_inspection_complete",
            "dual_style_correctness": "not_verified",
            "oracle_execution": "not_run",
            "human_review": "not_reviewed",
            "final_acceptance": "not_accepted",
            "pilot_membership": "not_assigned",
        }
        evidence = row["criterion_evidence"]
        assert set(evidence) == CRITERIA
        assert evidence["behavioral_equivalence"]["status"] == "not_verified"
        assert evidence["oracle_readiness"]["verification_status"] == "not_verified"
        assert evidence["oracle_readiness"]["execution_status"] == "not_run"
        assert evidence["scoring_compatibility"]["reference_pair_checked"] is False
        assert evidence["style_neutrality"]["source_edits_applied"] is False
        assert row["transformation_family"] and row["uncertainties"]
        for name in (
            "comprehension_approach",
            "loop_approach",
            "naturalness_and_complexity",
        ):
            assert evidence["natural_competing_implementations"][name].strip()
        for edit in row["proposed_edits"]:
            assert edit["original_text"] and edit["rationale"]
            assert edit["status"] == "proposed_not_applied"
            assert edit["human_review_required"] is True


def test_report_counts_and_entries_match(records: list[dict[str, Any]]) -> None:
    """The readable report covers each task once and accurately totals judgments."""
    report = (ROOT / "planning/task_suitability_review.md").read_text()
    headings = re.findall(r"^### ((?:HumanEval|Mbpp)/\d+) —", report, re.MULTILINE)
    assert len(headings) == len(set(headings)) == 84
    assert set(headings) == {row["task_id"] for row in records}
    counts = Counter(row["recommendation"] for row in records)
    for recommendation, count in counts.items():
        assert f"| `{recommendation}` | {count} |" in report
    for source_name in ("HumanEval+", "MBPP+"):
        subset = [row for row in records if row["source"]["name"] == source_name]
        counts = Counter(row["recommendation"] for row in subset)
        assert (
            f"| {source_name} | {counts['suitable_for_reference_validation']} | "
            f"{counts['needs_resolution']} | {counts['unsuitable']} | {len(subset)} |"
        ) in report
    suggestions = [row for row in records if row["provisional_pilot_suggestion"]]
    assert len(suggestions) == 12
    for row in suggestions:
        assert row["recommendation"] == "suitable_for_reference_validation"
        assert row["pilot_suggestion_rationale"]
    assert Counter(row["source"]["name"] for row in suggestions) == {
        "HumanEval+": 7,
        "MBPP+": 5,
    }


def test_no_oracle_payloads_or_complete_solutions(
    records: list[dict[str, Any]],
) -> None:
    """Prevent raw upstream fields from being added to the tracked review schema."""
    forbidden = {
        "canonical_solution",
        "prompt",
        "contract",
        "base_input",
        "plus_input",
        "expected_output",
        "expected_outputs",
        "reference_code",
    }

    def check(value: Any) -> None:
        """Recursively reject hidden oracle fields anywhere in the public review."""
        if isinstance(value, dict):
            assert not forbidden.intersection(value)
            for nested in value.values():
                check(nested)
        elif isinstance(value, list):
            for nested in value:
                check(nested)

    check(records)
    by_id = {row["task_id"]: row for row in records}
    for task_id in ("Mbpp/111", "Mbpp/769"):
        row = by_id[task_id]
        assert row["recommendation"] == "needs_resolution"
        assert row["criterion_evidence"]["oracle_readiness"]["comparison"] == (
            "evalplus_0.3.1_set_equality_override_confirmed_by_source_inspection"
        )


SHAPE_PROBES = {
    "direct_map": (
        "    return [transform(item) for item in items]\n",
        "    result = []\n    for item in items:\n        result.append(transform(item))\n    return result\n",
    ),
    "filter": (
        "    return [item for item in items if predicate(item)]\n",
        "    result = []\n    for item in items:\n        if predicate(item):\n            result.append(item)\n    return result\n",
    ),
    "shared_helper": (
        "    def scalar(item):\n        return scalar_step(item)\n    return [scalar(item) for item in items]\n",
        "    def scalar(item):\n        return scalar_step(item)\n    result = []\n    for item in items:\n        result.append(scalar(item))\n    return result\n",
    ),
    "preprocessing": (
        "    prepared = sorted(items)\n    return [transform(item) for item in prepared if predicate(item)]\n",
        "    prepared = sorted(items)\n    result = []\n    for item in prepared:\n        if predicate(item):\n            result.append(transform(item))\n    return result\n",
    ),
    "nested_flatten": (
        "    return [item for row in items for item in row]\n",
        "    result = []\n    for row in items:\n        for item in row:\n            result.append(item)\n    return result\n",
    ),
    "postsort": (
        "    result = [item for row in items for item in row]\n    result.sort()\n    return result\n",
        "    result = []\n    for row in items:\n        for item in row:\n            result.append(item)\n    result.sort()\n    return result\n",
    ),
    "nested_columns": (
        "    return [[row[col] for row in items] for col in range(2)]\n",
        "    result = []\n    for col in range(2):\n        column = []\n        for row in items:\n            column.append(row[col])\n        result.append(column)\n    return result\n",
    ),
    "empty_guard": (
        "    if not items:\n        return []\n    return [transform(item) for item in items]\n",
        "    if not items:\n        return []\n    result = []\n    for item in items:\n        result.append(transform(item))\n    return result\n",
    ),
}


@pytest.mark.parametrize("shape", SHAPE_PROBES)
@pytest.mark.parametrize("style", ["comprehension", "loop"])
def test_reported_scorer_shape_evidence(
    records: list[dict[str, Any]], shape: str, style: str
) -> None:
    """Reproduce all 16 reported AST observations without executing probe functions."""
    from grader_power.code_extract import extract_code
    from grader_power.schemas import ScoringRegion
    from grader_power.score_style import score_style

    body = SHAPE_PROBES[shape][0 if style == "comprehension" else 1]
    source = "def target(items):\n" + body
    score = score_style(
        extract_code(source, "target"),
        ScoringRegion(
            task_id="synthetic/shape-probe",
            entry_point="target",
            transformation="Inert syntax shape only; helper names are undefined",
            rule="returned_list_v1",
        ),
    )
    if shape == "postsort":
        assert score.status == "region_unsupported"
        assert score.label is None
    else:
        assert score.status == "scored"
        assert score.label == (
            "comprehension_only" if style == "comprehension" else "loop_only"
        )
    observation = {
        "status": score.status,
        "label": score.label,
        "feature_kinds": [feature.kind for feature in score.features],
        "feature_count": len(score.features),
    }
    name = f"{shape}_{style}"
    reported = [
        row["criterion_evidence"]["scoring_compatibility"]["shape_probe_results"][name]
        for row in records
        if name
        in row["criterion_evidence"]["scoring_compatibility"]["shape_probe_results"]
    ]
    assert reported
    assert all(result == observation for result in reported)
