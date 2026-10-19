"""Publish additive free repairs of saved recovery judgments, never new API calls."""

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspect_ai.model import ModelOutput
from pydantic import ValidationError

from evals import inspect_all_recovery as recovery
from scripts import replace_interrupted_recoveries as replacements

EXTRA_RANGES = frozenset(
    {
        "aa9716f03aa18a781ef826b5fe1ab8049856f83cd8823d9c00f6c309fb00ab20",
        "6c5cd5a766e3a984b965b6467755bf9d6c57b77659db2f69d7b9872e87cce10c",
    }
)
REDUNDANT_FIELD = "38cbdc7f028b208b8c63dba538dd990580a90e958b4d39730485ccd6f98d2b8d"

AUDITED_LABELS = frozenset(
    {
        "586859df9dd1f17b6673d8fa9cc7a026903252bea96bf9520be529b5d2b4b886",
        "030ccd0cce7e027965d696d56aed960f91cf6d6ea0d929a227fc087effaede04",
        "71cef9feca0f9022ccc1c0fead1538af66ad924c135b8a7fc34db7233d59f8c8",
        "cbd5b7c5ce3968936fc7325be4603fa38f846412c039d6d29695e2a3c675c092",
    }
)


def repair(root: Path, record: dict) -> dict:
    """Apply only the previously approved exact mechanical repair operations."""
    p = recovery.base(root) / "requests" / record["assignment_id"] / "judge"
    response = ModelOutput.model_validate(recovery.read(p / "raw.json"))
    packet = recovery.read(p / "packet.json")
    if record["assignment_id"] in EXTRA_RANGES:
        fixed = recovery.repair_reversed_ranges(response.completion, packet)
        method = "reversed_range_endpoints"
    elif record["assignment_id"] == REDUNDANT_FIELD:
        fixed = recovery.prior.normalize_judgment(
            response.completion, packet, record["design"]
        )
        if [x["action"] for x in fixed["changes"]] != ["redundant_citation_field"]:
            raise ValueError(
                "Unexpected normalization beyond redundant citation metadata"
            )
        method = "redundant_citation_field"
    elif record["assignment_id"] in AUDITED_LABELS:
        import json

        value = json.loads(response.completion)
        try:
            recovery.syntax.parse_packet(record["design"], response.completion, packet)
        except ValidationError as error:
            if [(e["loc"], e["type"]) for e in error.errors()] != [
                (("rationale",), "string_too_short")
            ]:
                raise ValueError(
                    "Code audit case has other invalid judge fields"
                ) from error
        else:
            raise ValueError("Code audit target is already a valid judgment")
        audit = recovery.preference_judge.audit_style(
            recovery.preference_judge.JudgeInput.model_validate(packet)
        )
        if (
            value["rationale"] != ""
            or value["status"] != "classified"
            or sorted(value["target_forms"]) != sorted(audit.target_forms)
        ):
            raise ValueError("Separate syntax audit disagrees with saved claimed form")
        if not all(
            1 <= e["start_line"] <= e["end_line"] <= len(packet["code"].splitlines())
            for e in value["evidence"]
        ):
            raise ValueError("Empty-rationale case also has invalid ranges")
        fixed = {
            "label": audit.label,
            "audit": audit.model_dump(mode="json"),
            "original_judge_status": "judge_error",
            "original_failure": "empty_rationale",
            "no_rationale_fabricated": True,
        }
        method = "separate_syntax_audit"
    else:
        raise ValueError("Unapproved free repair target")
    return {
        "assignment_id": record["assignment_id"],
        "replaces": record["replaces"],
        "study": record["origin_study"],
        "method": method,
        "label": fixed["label"],
        "repair": fixed,
        "bindings": {
            str((p / name).relative_to(root)): recovery.digest((p / name).read_bytes())
            for name in ("raw.json", "packet.json", "result.json")
        },
    }


def finalize(root: Path, report: Path) -> None:
    """Recompute affected study statistics while preserving complete assigned rows."""
    original, supplement = replacements.verify(root)
    b = recovery.base(root)
    prior_index = b / "analysis-index-v2.json"
    index = recovery.read(prior_index)
    if not index["complete"]:
        raise ValueError("Paid recovery analysis must finish first")
    records = {
        r["assignment_id"]: r for r in original["records"] + supplement["records"]
    }
    repairs = [
        repair(root, records[aid])
        for aid in sorted(EXTRA_RANGES | {REDUNDANT_FIELD} | AUDITED_LABELS)
    ]
    cases = recovery.ET.parse(report).getroot().findall(".//testcase")
    if not cases or any(
        c.find(t) is not None for c in cases for t in ("failure", "error", "skipped")
    ):
        raise ValueError("Free-repair tests must pass")
    d = b / "analysis/postprocessing-v1"
    manifest = {
        "repairs": repairs,
        "previous_index_sha256": recovery.digest(prior_index.read_bytes()),
        "approval_bindings": {
            str(p.relative_to(root)): recovery.digest(p.read_bytes())
            for p in (
                b / "operations/approval.json",
                b / "operations/recoveries/terminal-outcomes-v1/approval.json",
                b / "operations/code-audit-approval.json",
            )
        },
        "policy": "Approved mechanical repairs and separately attributed complete syntax audits of empty-rationale judgments; original failures preserved, no fabricated judge rationales or new paid calls.",
        "reports": {
            str(report.relative_to(root)): recovery.digest(report.read_bytes())
        },
        "sources": {
            name: recovery.digest((root / name).read_bytes())
            for name in (
                "scripts/finalize_recovery_repairs.py",
                "tests/test_recovery_postprocessing.py",
            )
        },
    }
    recovery.freeze(d / "manifest.json", manifest)
    index = copy.deepcopy(index)
    for study in sorted({x["study"] for x in repairs}):
        relative = recovery.RELATIVE_RUNS[study]
        binding = index["sources"][relative]
        rows = recovery.read(root / binding["path"])
        lineage = recovery.read(root / binding["lineage"])
        selected = {x["replaces"]: x for x in repairs if x["study"] == study}
        for row in rows:
            fixed = selected.get(row["assignment_id"])
            if fixed is None:
                continue
            if (
                row["all_recovery_id"] != fixed["assignment_id"]
                or row["label"] is not None
            ):
                raise ValueError(
                    "Free repair is not the designated unresolved judgment"
                )
            row.update(
                label=fixed["label"],
                judge_status="resolved_by_code_audit"
                if fixed["method"] == "separate_syntax_audit"
                else "classified",
                all_judgment_repair=fixed["method"],
                label_source="independent_code_audit"
                if fixed["method"] == "separate_syntax_audit"
                else "mechanically_repaired_sonnet",
            )
            for link in lineage:
                if link["assignment_id"] == row["assignment_id"]:
                    link.update(label=fixed["label"], postprocessing=fixed["method"])
        destination = b / "analysis/final-v3" / study
        recovery.freeze(destination / "measured-rows.json", rows)
        recovery.freeze(destination / "lineage.json", lineage)
        if study == "syntax":
            from grader_power import style_preference_experiment

            analysis = style_preference_experiment.analyze(rows, repetitions=3)
        else:
            analysis = recovery.previous.statistics(study, rows)
        analysis.update(
            recovery_note=manifest["policy"],
            measured_rows_sha256=recovery.digest(
                (destination / "measured-rows.json").read_bytes()
            ),
        )
        recovery.freeze(destination / "analysis.json", analysis)
        binding.update(
            path=str((destination / "measured-rows.json").relative_to(root)),
            sha256=recovery.digest((destination / "measured-rows.json").read_bytes()),
            lineage=str((destination / "lineage.json").relative_to(root)),
            lineage_sha256=recovery.digest((destination / "lineage.json").read_bytes()),
        )
    index.update(
        postprocessing_manifest=str((d / "manifest.json").relative_to(root)),
        postprocessing_sha256=recovery.digest((d / "manifest.json").read_bytes()),
    )
    recovery.freeze(b / "analysis-index-v3.json", index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    finalize(Path(__file__).resolve().parents[1], args.report.resolve())


if __name__ == "__main__":
    main()
