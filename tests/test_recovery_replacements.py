"""The approved wrapper exception changes identities, never scientific settings."""

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import replace_interrupted_recoveries as replacement


def test_exact_five_fresh_replacements_preserve_limits():
    records = [
        {
            "assignment_id": aid,
            "recovery_kind": "generation",
            "mode": "B",
            "replaces": "original-" + aid,
            "messages": ["frozen"],
            "episode_limits": {"max_model_turns": 16},
            "profile": "gpt-oss-120b",
        }
        for aid in replacement.TARGETS
    ]
    manifest = {"records": records}
    before = copy.deepcopy(manifest)
    result = replacement.replacement_records(manifest)
    assert manifest == before
    assert len(result) == 5
    assert not replacement.TARGETS.intersection(r["assignment_id"] for r in result)
    for r in result:
        prior = next(
            x for x in records if x["assignment_id"] == r["supersedes_recovery_id"]
        )
        assert {
            k: v
            for k, v in r.items()
            if k not in ("assignment_id", "supersedes_recovery_id")
        } == {k: v for k, v in prior.items() if k != "assignment_id"}
