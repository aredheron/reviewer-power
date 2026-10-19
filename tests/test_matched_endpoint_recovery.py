"""An endpoint override is explicit, unique, and never changes the original rows."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import matched_endpoint_recovery as endpoint


def test_only_approved_case_resolves_and_preserves_all_other_rows():
    root = Path(__file__).resolve().parents[1]
    rows = json.loads(
        (
            root
            / "runs/recovery/matched-review-recovery-v1/analysis-v1/measured-rows.json"
        ).read_text()
    )
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError):
        endpoint.analyze(rows, draws=100)
    row = next(r for r in rows if r["assignment_id"] == endpoint.IDENTITY)
    row.update(
        label="no_implementation", endpoint_value=0, original_judge_status="abstain"
    )
    snapshot = copy.deepcopy(rows)
    result = endpoint.analyze(rows, draws=100)
    assert rows == snapshot
    assert result["missing_labels"] == 0
    assert [f["holm_family_size"] for f in result["families"]] == [12, 15, 3]
    assert result["coverage"]["glm"]["labels"]["no_implementation"] == 1
    assert (
        before[
            next(
                i
                for i, r in enumerate(before)
                if r["assignment_id"] == endpoint.IDENTITY
            )
        ]["label"]
        is None
    )
    rows[0]["label"] = "no_implementation"
    with pytest.raises(ValueError):
        endpoint.analyze(rows, draws=100)
