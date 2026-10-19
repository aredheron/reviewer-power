"""Approved follow-up scope, original provenance, designation and no-requeue checks."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput

ROOT = (
    Path.cwd()
    if os.environ.get("MATCHED_STAGED_TESTS") == "1"
    else Path(__file__).resolve().parents[1]
)
sys.path.insert(0, str(ROOT))
from grader_power import matched_review as study
from grader_power import matched_review_followup as followup
from grader_power.spending import nanodollars

if os.environ.get("MATCHED_STAGED_TESTS") == "1":
    module_spec = importlib.util.spec_from_file_location(
        "matched_followup_staged", "/tmp/inspect_matched_review_followup.py"
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("Staged runner is missing")
    runner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(runner)
else:
    runner = importlib.import_module("evals.inspect_matched_review")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(runner.json_text(value))


@pytest.fixture(scope="module")
def records():
    # Only original pilot cells are needed for these readiness-specific fixtures.
    return [r for r in study.schedule(ROOT)[0] if r["phase"] == "pilot"]


@pytest.fixture
def prepared(tmp_path, records):
    base = study.base(tmp_path)
    write(base / "assignments.json", records)
    original_approval = {"approved": True}
    write(base / "operations/human-approval.json", original_approval)
    manifest = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "median_continuation_limit_usd": study.MEDIAN_LIMIT,
        "assignments_sha256": runner.digest((base / "assignments.json").read_bytes()),
        "approval_sha256": runner.digest(
            (base / "operations/human-approval.json").read_bytes()
        ),
        "profiles": study.PROFILES,
        "cost_floor_per_response": {p: [0.001, 0.002] for p in study.PROFILES},
    }
    write(base / "manifest.json", manifest)
    write(base / "pilot-gate.json", {"passed": False, "strict_original": True})
    approval = json.loads(
        (study.base(ROOT) / followup.DIRECTORY / "approval.json").read_text()
    )
    proposal_path = base / approval["proposal_path"]
    proposal_path.parent.mkdir(parents=True, exist_ok=True)
    proposal_path.write_bytes(
        (study.base(ROOT) / approval["proposal_path"]).read_bytes()
    )
    assert (
        runner.digest((base / approval["proposal_path"]).read_bytes())
        == approval["proposal_sha256"]
    )
    write(base / followup.DIRECTORY / "approval.json", approval)
    helper = tmp_path / "src/grader_power/matched_review_followup.py"
    helper.parent.mkdir(parents=True)
    helper.write_bytes(
        (ROOT / "src/grader_power/matched_review_followup.py").read_bytes()
    )
    for row in records:
        path = runner.path_for(base, row)
        if row["kind"] == "comprehension_probe":
            text = json.dumps(study.expected_facts(row))
            if row["profile"] == "glm":
                text += "\nThe supplied configuration determines these answers."
            raw = ModelOutput.from_content("fixture", text).model_dump(mode="json")
            write(path / "generation/raw.json", raw)
            write(path / "probe.json", study.score_probe(text, row))
        else:
            write(
                path / "generation/raw.json",
                ModelOutput.from_content("fixture", "code").model_dump(mode="json"),
            )
            write(path / "scoring.json", {"correctness": {"full_correct": True}})
            write(
                path / "judge/result.json",
                {"label": "loop_only", "status": "classified"},
            )
    plan = runner.prepare_followup(tmp_path)
    batch = json.loads((base / followup.DIRECTORY / "probe2.json").read_text())
    amendment = {
        "approved_followup": followup.DIRECTORY,
        "audit_bindings": plan["audit_bindings"],
    }
    return tmp_path, base, manifest, records, batch["assignments"], amendment, plan


def test_preparation_freezes_full_scope_without_ledger_or_request_dispatch(prepared):
    _, base, manifest, records, fresh, amendment, plan = prepared
    assert len(fresh) == 168
    assert plan["free_qualification_records"] == 504
    assert plan["fresh_factual_bound_usd"] == 0.736524008
    assert plan["all_original_facts_qualified_counts"] == {
        p: 168 for p in study.PROFILES
    }
    assert not (base / "spending.sqlite3").exists()
    assert all(not runner.path_for(base, row).exists() for row in fresh)
    assert runner.probe2_records(base, manifest, records, amendment) == fresh
    assert json.loads((base / "pilot-gate.json").read_text())["passed"] is False
    assert runner.ledger_for(base, manifest).scope == runner.digest(
        runner.json_text(manifest)
    )


def test_fresh_factual_missing_never_falls_back_to_correct_original(prepared):
    _, base, _, old, fresh, _, _ = prepared
    rows = runner.qualified_pilot_rows(base, old + fresh)
    gate = study.pilot_gate(rows)
    assert {r["profile"] for r in gate["groups"] if r["passed"]} == {"glm", "deepseek"}
    gpt = [
        r
        for r in rows
        if r["profile"] == "gpt-oss-120b" and r["kind"] == "comprehension_probe"
    ]
    assert len(gpt) == 168 and all(not r["raw_saved"] for r in gpt)
    assert {r["assignment_id"] for r in gpt} == {r["assignment_id"] for r in fresh}


def test_fresh_facts_keep_original_strict_parser(prepared):
    _, base, _, old, fresh, _, _ = prepared
    row = fresh[0]
    text = json.dumps(study.expected_facts(row)) + " trailing explanation"
    path = runner.path_for(base, row)
    write(
        path / "generation/raw.json",
        ModelOutput.from_content("fixture", text).model_dump(mode="json"),
    )
    write(path / "probe.json", study.score_probe(text, row))
    measured = next(
        r
        for r in runner.qualified_pilot_rows(base, old + fresh)
        if r["assignment_id"] == row["assignment_id"]
    )
    assert measured["raw_saved"] and not measured["probe_correct"]


@pytest.mark.parametrize("target", ["glm", "gpt-oss-120b"])
def test_every_original_qualification_is_revalidated_even_designated_away(
    prepared, target
):
    _, base, _, old, fresh, _, _ = prepared
    row = next(
        r for r in old if r["profile"] == target and r["kind"] == "comprehension_probe"
    )
    path = runner.generation_path(base, row)
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="source"):
        runner.qualified_pilot_rows(base, old + fresh)


def test_original_strict_scores_cannot_be_silently_replaced(prepared):
    _, base, _, old, fresh, _, _ = prepared
    row = next(r for r in old if r["kind"] == "comprehension_probe")
    write(runner.path_for(base, row) / "probe.json", {"correct": True})
    with pytest.raises(ValueError, match="strict"):
        runner.qualified_pilot_rows(base, old + fresh)


def test_partial_batch_and_changed_approval_are_not_admitted(prepared):
    _, base, manifest, records, _, amendment, _ = prepared
    path = base / followup.DIRECTORY / "probe2.json"
    value = json.loads(path.read_text())
    value["assignments"].pop()
    write(path, value)
    amendment = copy.deepcopy(amendment)
    amendment["audit_bindings"][str(path.relative_to(base))] = runner.digest(
        path.read_bytes()
    )
    with pytest.raises(ValueError, match="scope"):
        runner.probe2_records(base, manifest, records, amendment)


def test_original_pilot_requeue_is_rejected_before_models_initialize(
    prepared, monkeypatch
):
    root, _, manifest, old, fresh, _, _ = prepared
    monkeypatch.setattr(runner, "load", lambda *args, **kwargs: (manifest, old + fresh))
    monkeypatch.setattr(runner, "guard", lambda base: None)
    monkeypatch.setattr(runner, "initialize_pacing", lambda *args: None)
    with pytest.raises(ValueError, match="original pilot"):
        runner.collect(root, "pilot")


def test_run_selects_probe2_then_main_without_original_pilot(prepared, monkeypatch):
    _, _, manifest, old, fresh, _, _ = prepared
    monkeypatch.setattr(runner, "load", lambda *args, **kwargs: (manifest, old + fresh))
    monkeypatch.setattr(runner, "isolate_collection_process", lambda: None)
    seen = []
    monkeypatch.setattr(runner, "collect", lambda root, phase: seen.append(phase))
    monkeypatch.setattr(sys, "argv", ["run_matched_review.py", "run"])
    runner.main()
    assert seen == ["probe2", "main"]


def test_followup_counts_and_forecast_have_no_new_coding_judgments(prepared):
    root, base, manifest, _, fresh, _, _ = prepared
    measured = [runner.measure(base, row) for row in fresh]
    count = runner.counts(measured)["probe2/gpt-oss-120b/matched_review"]
    assert count["assigned"] == 168 and count["coding_assigned"] == 0
    forecast = runner.median_forecast(root, manifest, fresh, draws=100)
    assert forecast["median_total_usd"] == pytest.approx(0.168)
    assert forecast["models"]["gpt-oss-120b"]["judge"]["future_requests"] == 0
    assert runner.ledger_for(base, manifest).cap == nanodollars(550)


def test_prior_analysis_archive_preserves_bytes_and_refuses_replacement(tmp_path):
    base = study.base(tmp_path)
    write(base / "manifest.json", {"original": "manifest"})
    write(base / "analysis-v1/analysis.json", {"models": ["deepseek"]})
    write(base / "analysis-v1/measured-rows.json", [{"label": "loop_only"}])
    index = runner.archive_prior_analysis(tmp_path)
    assert index is not None and len(index["bindings"]) == 2
    for name in ("analysis.json", "measured-rows.json"):
        assert (base / "analysis-v1" / name).read_bytes() == (
            base / followup.DIRECTORY / "prior-analysis-v1" / name
        ).read_bytes()
    assert runner.archive_prior_analysis(tmp_path) == index
    write(base / "analysis-v1/analysis.json", {"different": True})
    with pytest.raises(ValueError, match="different"):
        runner.archive_prior_analysis(tmp_path)


@pytest.fixture
def amended(prepared, monkeypatch):
    root, base, manifest, records, fresh, _, plan = prepared
    old_source = "evals/inspect_matched_review.py"
    frozen_science = "src/grader_power/matched_review.py"
    added = {
        "src/grader_power/matched_review_followup.py",
        "tests/test_matched_review_followup.py",
        "tests/test_matched_review_followup_runner.py",
        "planning/matched_review_followup_v1.md",
    }
    manifest = copy.deepcopy(manifest)
    manifest.update(
        code_sha256={
            old_source: runner.digest(b"old runner"),
            frozen_science: runner.digest(b"science"),
        },
        verification_reports={},
        historical_bindings={},
    )
    write(base / "manifest.json", manifest)
    snapshot = base / "source-snapshot" / old_source
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(b"old runner")
    current = (
        manifest["code_sha256"]
        | {old_source: runner.digest(b"amended runner")}
        | {name: runner.digest(name) for name in added}
    )
    monkeypatch.setattr(runner, "sources", lambda root: dict(current))
    for name in ("probe2.json", "format-overlay.json"):
        path = base / followup.DIRECTORY / name
        value = json.loads(path.read_text())
        value["manifest_sha256"] = runner.digest(runner.json_text(manifest))
        write(path, value)
    bindings = {
        name: runner.digest((base / name).read_bytes())
        for name in plan["audit_bindings"]
    }
    report = base / "verification/focused.xml"
    report.parent.mkdir(parents=True)
    report.write_text(
        '<testsuites><testsuite><testcase name="passed"/></testsuite></testsuites>'
    )
    retained = next(base.glob("requests/*/generation/raw.json"))
    amendment = {
        "approved_followup": followup.DIRECTORY,
        "manifest_sha256": runner.digest(runner.json_text(manifest)),
        "code_sha256": dict(current),
        "audit_bindings": bindings,
        "verification_reports": {
            str(report.relative_to(base)): runner.digest(report.read_bytes())
        },
        "retained_artifacts": {
            str(retained.relative_to(base)): runner.digest(retained.read_bytes())
        },
    }
    path = base / followup.DIRECTORY / "amendment.json"
    write(path, amendment)
    write(
        base / "operations/recovery-state.json",
        {
            "source_amendment": str(path.relative_to(base)),
            "source_amendment_sha256": runner.digest(path.read_bytes()),
        },
    )
    return root, base, manifest, records, fresh, current, retained


def test_amended_load_adds_only_designated_cohort_with_original_ledger_identity(
    amended,
):
    root, base, manifest, records, fresh, _, _ = amended
    loaded_manifest, loaded = runner.load(root, retained=True)
    assert loaded_manifest == manifest and loaded == records + fresh
    assert runner.ledger_for(base, loaded_manifest).scope == runner.digest(
        runner.json_text(manifest)
    )
    assert len(loaded) == len(records) + 168


def test_amended_load_still_rejects_frozen_science_change(amended):
    root, _, _, _, _, current, _ = amended
    current["src/grader_power/matched_review.py"] = runner.digest(
        b"changed frozen science"
    )
    with pytest.raises(ValueError, match="frozen science"):
        runner.load(root, retained=True)


def test_amended_load_rejects_unapproved_added_sources(amended):
    root, _, _, _, _, current, _ = amended
    current["src/grader_power/unapproved.py"] = runner.digest(b"unapproved")
    with pytest.raises(ValueError, match="frozen science"):
        runner.load(root, retained=True)


def test_retained_load_checks_original_artifacts_after_followup(amended):
    root, _, _, _, _, _, retained = amended
    retained.write_bytes(retained.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Retained outcome"):
        runner.load(root, retained=True)


@pytest.mark.parametrize("name", ["probe2.json", "format-overlay.json"])
def test_embedded_approval_link_must_match_even_when_artifact_rebound(prepared, name):
    _, base, manifest, records, _, amendment, _ = prepared
    relative = followup.DIRECTORY + "/" + name
    path = base / relative
    value = json.loads(path.read_text())
    value["approval_sha256"] = "0" * 64
    write(path, value)
    amendment["audit_bindings"][relative] = runner.digest(path.read_bytes())
    with pytest.raises(ValueError, match="scope, gates or budget"):
        runner.probe2_records(base, manifest, records, amendment)


def test_amendment_requires_followup_approval_even_if_binding_missing(amended):
    root, base, _, _, _, _, _ = amended
    relative = followup.DIRECTORY + "/approval.json"
    path = base / followup.DIRECTORY / "amendment.json"
    amendment = json.loads(path.read_text())
    del amendment["audit_bindings"][relative]
    write(path, amendment)
    state = base / "operations/recovery-state.json"
    recovery = json.loads(state.read_text())
    recovery["source_amendment_sha256"] = runner.digest(path.read_bytes())
    write(state, recovery)
    (base / relative).unlink()
    with pytest.raises(FileNotFoundError):
        runner.load(root, retained=True)
