"""Latest additive views preserve identities and both earlier outcome views."""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import latest_matched_dashboard as latest
from grader_power.results_dashboard import condition_info


def test_optional_recovery_does_not_change_historical_data(tmp_path):
    data = {"conditions": [], "assigned": 99}
    assert latest.apply(tmp_path, data) == {"conditions": [], "assigned": 99}


def test_completed_view_preserves_strict_and_prior_cells(tmp_path, monkeypatch):
    b = tmp_path / "runs/recovery/matched-gaps-v1/loop-endpoint-v1"
    b.mkdir(parents=True)
    (b / "audit.json").write_text("{}")
    root = Path(__file__).resolve().parents[1]
    rows = json.loads(
        (
            root / "runs/recovery/matched-gaps-v1/loop-endpoint-v1/measured-rows.json"
        ).read_text()
    )
    row = next(r for r in rows if r["label"] == "no_implementation")
    condition = condition_info("MATCHED", row)
    peer = next(
        r
        for r in rows
        if r["profile"] == "glm"
        and r["condition"]["user_preference"] != row["condition"]["user_preference"]
        and condition_info("MATCHED", r)["id"] == condition["id"]
    )
    strict = {"model": "glm", "user": "all", "unknown": 1}
    prior = {"model": "glm", "user": "all", "unknown": 1}
    data = {
        "assigned": 32256,
        "matched_recovery": {"first": True},
        "conditions": [{**condition, "cells": [strict], "recovery_cells": [prior]}],
    }
    before = copy.deepcopy(data)
    monkeypatch.setattr(
        latest,
        "verify",
        lambda *a, **kw: ([row, peer], {"by_condition": [], "seed": 1}),
    )
    result = latest.apply(tmp_path, data)
    cell = result["conditions"][0]
    assert cell["cells"] == before["conditions"][0]["cells"]
    assert cell["prior_recovery_cells"] == before["conditions"][0]["recovery_cells"]
    assert cell["recovery_cells"][0]["unknown"] == 0
    assert cell["recovery_cells"][0]["labels"]["no_implementation"] == 1
    assert result["assigned"] == 32256 and result["matched_prior_recovery"] == {
        "first": True
    }


def test_latest_verifier_rejects_changed_bound_source(tmp_path):
    directory = tmp_path / "latest"
    directory.mkdir()
    (tmp_path / "evidence").write_text("changed")
    (directory / "audit.json").write_text(
        json.dumps(
            {
                "verified": True,
                "source_and_retained_artifact_bindings_verified": True,
                "bindings": {"evidence": "wrong"},
            }
        )
    )
    with pytest.raises(ValueError, match="binding changed"):
        latest.verify(tmp_path, directory, endpoint=True)
