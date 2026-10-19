"""Validate the readiness cohort without inference or submitted-code execution."""

import importlib
import json
import sys
from pathlib import Path

import pytest

from grader_power.pilot_tasks import load_pilot_tasks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pilot = importlib.import_module("evals.inspect_pilot")
readiness = importlib.import_module("evals.inspect_readiness")


def test_readiness_only_changes_the_declared_metadata_line() -> None:
    originals = {
        (
            a.record["source_task_id"],
            a.record["kind"],
            json.dumps(a.record["condition"], sort_keys=True),
            a.record["direct_style"],
        ): a
        for a in pilot.build_assignments(load_pilot_tasks(ROOT))
        if a.record["kind"] in readiness.KINDS
    }
    revised = readiness.build_assignments(ROOT)
    assert len(revised) == len(originals) == 24
    for a in revised:
        r = a.record
        original = originals[
            r["source_task_id"],
            r["kind"],
            json.dumps(r["condition"], sort_keys=True),
            r["direct_style"],
        ]
        assert r["messages"][0]["content"] == original.record["messages"][0][
            "content"
        ].replace(readiness.OLD_SCORE_USE, readiness.NEW_SCORE_USE)
        assert r["assignment_id"] != original.record["assignment_id"]
        assert a.task == original.task
    assert len({a.record["assignment_id"] for a in revised}) == 24


def test_shared_collector_rejects_changed_prompts_before_credentials(
    tmp_path: Path,
) -> None:
    pilot.save(tmp_path / "manifest.json", {"assignments": []})
    with pytest.raises(ValueError, match="disagree"):
        pilot.collect_profile(
            ROOT,
            tmp_path,
            readiness.build_assignments(ROOT),
            "gpt-oss-120b",
            "unused",
            {},
        )


def test_missing_probes_never_pass_readiness() -> None:
    result = readiness.summarize([])
    assert result["comprehension_gate"]["passed"] is False
    assert result["controls_full_correct"] == 0
