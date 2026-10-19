"""Mechanical recovery never chooses labels or fabricates evidence."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import matched_recovery as recovery


def packet():
    return {
        "entry_point": "f",
        "transformation": "Double each input value",
        "input_contract": "list of ints",
        "code": "def f(xs):\n    return [x * 2 for x in xs]",
    }


def judgment():
    return {
        "status": "classified",
        "target_forms": ["list_comprehension"],
        "evidence": [
            {
                "start_line": 2,
                "end_line": 2,
                "quote": "    return [x * 2 for x in xs]",
                "start_line_note": "line containing comprehension",
            }
        ],
        "flags": [],
        "rationale": "List comprehension performs the requested transformation.",
    }


def test_metadata_only_repair_preserves_semantics_and_exact_quote():
    j = judgment()
    fixed = recovery.normalize(json.dumps(j), packet())
    assert fixed["label"] == "comprehension_only"
    assert fixed["judgment"]["rationale"] == j["rationale"]
    assert fixed["judgment"]["evidence"][0]["quote"] == j["evidence"][0]["quote"]


def test_unique_quote_locator_repairs_missing_start_without_changing_quote():
    j = judgment()
    e = j["evidence"][0]
    e.pop("start_line")
    e["line_number"] = 2
    fixed = recovery.normalize(json.dumps(j), packet())
    assert fixed["judgment"]["evidence"][0]["start_line"] == 2


def test_semantic_conflicts_and_fabricated_quotes_stay_unresolved():
    j = judgment()
    with pytest.raises(ValueError):
        recovery.normalize(json.dumps(j) + json.dumps(j), packet())
    j["evidence"][0]["quote"] = "    return [x + 2 for x in xs]"
    with pytest.raises(ValueError):
        recovery.normalize(json.dumps(j), packet())


def test_conflicting_duplicate_is_rejected():
    text = json.dumps(judgment()).replace(
        '"start_line": 2', '"start_line": 2, "start_line": 1'
    )
    with pytest.raises(ValueError, match="Conflicting"):
        recovery.normalize(text, packet())
