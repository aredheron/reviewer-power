"""Free recovery repairs cannot change a semantic label or citation quotation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import finalize_recovery_repairs as final

ROOT = Path(__file__).resolve().parents[1]


def test_additional_saved_judgments_use_approved_mechanical_operations():
    manifest = final.recovery.read(final.recovery.base(ROOT) / "manifest.json")
    records = {r["assignment_id"]: r for r in manifest["records"]}
    for aid in final.EXTRA_RANGES | {final.REDUNDANT_FIELD}:
        fixed = final.repair(ROOT, records[aid])
        final.recovery.prior.verify_hashes(ROOT, fixed["bindings"])
        assert fixed["label"] is not None
        if aid in final.EXTRA_RANGES:
            assert fixed["label"] == fixed["repair"]["audit_label"]
        else:
            assert fixed["repair"]["changes"] == [
                {
                    "action": "redundant_citation_field",
                    "index": 1,
                    "key": "line_number",
                    "value": 73,
                }
            ]
    with pytest.raises((ValueError, FileNotFoundError)):
        final.repair(ROOT, {"assignment_id": "unapproved"})


def test_empty_rationale_labels_have_separate_audit_provenance():
    manifest = final.recovery.read(final.recovery.base(ROOT) / "manifest.json")
    records = {r["assignment_id"]: r for r in manifest["records"]}
    for aid in final.AUDITED_LABELS:
        fixed = final.repair(ROOT, records[aid])
        assert fixed["method"] == "separate_syntax_audit"
        assert fixed["label"] == fixed["repair"]["audit"]["label"]
        assert fixed["repair"]["no_rationale_fabricated"] is True
        assert "judgment" not in fixed["repair"]
