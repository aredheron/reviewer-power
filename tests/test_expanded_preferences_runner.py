"""Admission, accounting, no-replay and analysis tests for the expansion runner."""

import asyncio
import copy
import json
import shutil
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput, ModelUsage

from evals import inspect_expanded_preferences as runner
from grader_power import expanded_preference_study as study
from grader_power.expanded_preference_tasks import load_tasks
from grader_power.spending import SpendingLimit, nanodollars

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def records() -> list[dict]:
    tasks = load_tasks(ROOT)
    return study.calibration_records(tasks) + study.schedule(ROOT, tasks)


def ready(rows: list[dict]) -> list[dict]:
    return [
        copy.deepcopy(r)
        | {
            "raw_saved": r["kind"] != "reference_judgment",
            "full_correct": True,
            "probe_correct": True,
            "label": r.get("expected_label", "loop_only"),
        }
        for r in rows
    ]


def test_pilot_admission_is_per_model_and_never_behavioral(records: list[dict]) -> None:
    rows = ready(records)
    assert study.pilot_gate(rows)["passed"]
    for row in rows:
        if (
            row["phase"] == "pilot"
            and row["kind"] == "comprehension_probe"
            and row["profile"] == "glm"
            and row["condition"]["arm"] == "strong_shadow"
        ):
            row["probe_correct"] = False
    gate = study.pilot_gate(rows)
    assert [g["profile"] for g in gate["groups"] if g["passed"]] == [
        "qwen",
        "gpt-oss-120b",
    ]
    assert not gate["behavior_used_to_gate"]


def test_missing_or_duplicate_pilot_cannot_qualify(records: list[dict]) -> None:
    rows = [r for r in ready(records) if r["phase"] == "pilot"]
    assert not study.pilot_gate(rows[:-1])["passed"]
    rows[-1] = rows[-2]
    assert not study.pilot_gate(rows)["passed"]


def test_reference_calibration_holds_semantic_disagreement(records: list[dict]) -> None:
    rows = [r for r in ready(records) if r["phase"] == "calibration"]
    assert study.calibration_gate(rows)["passed"]
    rows[0]["label"] = "mixed"
    assert not study.calibration_gate(rows)["passed"]
    rows[0]["label"] = None
    assert study.calibration_gate(rows)["passed"]
    other = next(
        r for r in rows[1:] if r["source_task_id"] == rows[0]["source_task_id"]
    )
    other["label"] = None
    assert not study.calibration_gate(rows)["passed"]


def test_fact_parser_preserves_complete_strict_schema(records: list[dict]) -> None:
    row = next(r for r in records if r["kind"] == "comprehension_probe")
    text = json.dumps(study.expected_facts(row))
    assert study.score_probe(text, row)["strict"]["all_correct"]
    assert study.score_probe("```json\n" + text + "\n```", row)["fence_only"][
        "all_correct"
    ]
    assert not study.score_probe(text + " extra", row)["fence_only"]["all_correct"]


def pipeline(tmp_path: Path, rows: list[dict]) -> runner.Pipeline:
    base = study.base(tmp_path)
    base.mkdir(parents=True, exist_ok=True)
    (base / "tasks-private.json").write_text("[]")
    manifest = {
        "cap_usd": 275,
        "judge_profile": study.JUDGE_PROFILE,
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "cost_floor_per_response": {p: [0.001, 0.02] for p in study.PROFILES},
    }
    return runner.Pipeline(tmp_path, manifest, rows, "pilot")


def test_saved_request_is_never_replayed(tmp_path: Path, records: list[dict]) -> None:
    row = next(r for r in records if r["phase"] == "main")
    p = pipeline(tmp_path, [row])
    output = ModelOutput.from_content("test", "safe")
    output.usage = ModelUsage(input_tokens=100, output_tokens=20, total_tokens=120)
    client = AsyncMock()
    client.generate.return_value = output
    p.models[row["profile"]] = client

    async def work() -> None:
        first = await p.request(row, "generation", [ChatMessageUser(content="test")])
        assert first == await p.request(
            row, "generation", [ChatMessageUser(content="test")]
        )

    asyncio.run(work())
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["outstanding"] == 0


@pytest.mark.parametrize("failure", ["timeout", "missing_usage", "credential_echo"])
def test_terminal_errors_keep_reservations(
    tmp_path: Path, records: list[dict], failure: str
) -> None:
    row = next(r for r in records if r["phase"] == "main")
    p = pipeline(tmp_path, [row])
    client = AsyncMock()
    p.models[row["profile"]] = client
    p.keys.add("synthetic-test-marker")
    if failure == "timeout":
        client.generate.side_effect = TimeoutError("uncertain acceptance")
    else:
        client.generate.return_value = ModelOutput.from_content(
            "test", "synthetic-test-marker" if failure == "credential_echo" else "safe"
        )
    asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert client.generate.await_count == 1
    assert p.ledger.snapshot()["outstanding"] > 0
    if failure == "credential_echo":
        assert not (runner.path_for(p.base, row) / "generation/raw.json").exists()


def test_capped_saved_output_is_terminal_without_operational_failure(
    tmp_path: Path, records: list[dict]
) -> None:
    row = next(r for r in records if r["phase"] == "main")
    p = pipeline(tmp_path, [row])
    output = ModelOutput.from_content("test", "partial")
    output.choices[0].stop_reason = "max_tokens"
    output.usage = ModelUsage(input_tokens=100, output_tokens=4096, total_tokens=4196)
    client = AsyncMock()
    client.generate.return_value = output
    p.models[row["profile"]] = client
    asyncio.run(p.request(row, "generation", [ChatMessageUser(content="test")]))
    assert not p.history[-1]["failed"]
    assert p.measurements[row["assignment_id"]]["generation_status"] == "incomplete"
    assert p.measurements[row["assignment_id"]]["raw_saved"]


def test_calibration_never_calls_generation_or_counts_as_response(
    tmp_path: Path, records: list[dict]
) -> None:
    row = next(r for r in records if r["phase"] == "calibration")
    p = pipeline(tmp_path, [row])
    asyncio.run(p.generation_stage(row))
    assert p.judging.get_nowait() == row
    measured = runner.measure(p.base, row)
    assert measured["generation_model"] is None and not measured["started"]
    forecast = runner.median_forecast(tmp_path, p.manifest, [row], draws=100)
    assert forecast["median_total_usd"] == 0.02
    assert sum(r["future_requests"] for r in forecast["profiles"].values()) == 0


def test_guard_and_human_stop_block_admission(
    tmp_path: Path, records: list[dict]
) -> None:
    rows = [r for r in records if r["phase"] == "main"][:5]
    p = pipeline(tmp_path, rows)
    for row in rows:
        p.event(row, "judge", True, "ValidationError")
    assert p.stop.is_set()
    with pytest.raises(RuntimeError):
        p.check()


def test_short_tail_and_held_models_drain(
    tmp_path: Path, records: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [r for r in records if r["phase"] == "main"][:10]
    p = pipeline(tmp_path, rows)
    p.phase = "main"
    p.eligible_models = ["qwen"]
    seen = []

    async def generation(row: dict) -> None:
        seen.append(row["assignment_id"])

    monkeypatch.setattr(p, "generation_stage", generation)
    monkeypatch.setattr(p, "initialize_models", lambda: None)
    monkeypatch.setattr(p, "status", lambda state: {})
    asyncio.run(asyncio.wait_for(p.run(), timeout=3))
    assert set(seen) == {r["assignment_id"] for r in rows if r["profile"] == "qwen"}


def test_flat_new_cap_and_forecast_no_double_count(
    tmp_path: Path, records: list[dict]
) -> None:
    rows = [next(r for r in records if r["phase"] == "main")]
    p = pipeline(tmp_path, rows)
    before = runner.median_forecast(tmp_path, p.manifest, rows, draws=100)
    assert before["median_total_usd"] == pytest.approx(0.021)
    p.ledger.reserve({"unknown": nanodollars(275)})
    with pytest.raises(SpendingLimit):
        p.ledger.reserve({"extra": 1})
    after = runner.median_forecast(tmp_path, p.manifest, rows, draws=100)
    assert after["median_total_usd"] == pytest.approx(275.021)
    assert after["outstanding_usd"] == 275 and after["budget_notification_required"]


def test_analysis_requires_full_matrix_and_separate_nine_contrast_families(
    records: list[dict],
) -> None:
    rows = [
        r
        | {
            "label": "loop_only"
            if r["condition"]["user_preference"] == "loops"
            else "comprehension_only",
            "full_correct": False,
        }
        for r in records
        if r["phase"] == "main"
    ]
    result = study.analyze_cohort(rows, cohort="new72", task_count=72, draws=100)
    assert result["assigned"] == 7776
    assert all(
        len(f["comparisons"]) == f["holm_family_size"] == 9
        for f in result["families"].values()
    )
    assert all(
        c["estimate"] == 0
        for f in result["families"].values()
        for c in f["comparisons"]
    )
    soft = next(r for r in rows if r["condition"]["arm"] == "soft_absent")
    soft["label"] = None
    result = study.analyze_cohort(rows, cohort="new72", task_count=72, draws=100)
    assert result["families"]["primary"]["unresolved_in_family"] == 0
    assert result["families"]["secondary"]["unresolved_in_family"] == 1
    assert all(
        c["p_value"] is None for c in result["families"]["secondary"]["comparisons"]
    )
    with pytest.raises(ValueError):
        study.analyze_cohort(rows[:-1], cohort="new72", task_count=72, draws=100)


def test_paired_task_effect_and_missingness_bound_direction(
    records: list[dict],
) -> None:
    rows: list[dict] = [
        r
        | {
            "label": (
                "loop_only"
                if r["condition"]["user_preference"] == "comprehensions"
                else "comprehension_only"
            )
            if r["condition"]["arm"] == "strong_shadow"
            else "neither"
        }
        for r in records
        if r["phase"] == "main"
    ]
    result = study.analyze_cohort(rows, cohort="new72", task_count=72, draws=100)
    values = result["families"]["primary"]["comparisons"]
    assert [round(c["estimate"]) for c in values] == [1, -1, -1] * 3
    row = next(r for r in rows if r["condition"]["arm"] == "strong_shadow")
    row["label"] = None
    result = study.analyze_cohort(rows, cohort="new72", task_count=72, draws=100)
    c = next(
        c
        for c in result["families"]["primary"]["comparisons"]
        if c["model"] == row["profile"]
        and c["contrast"] == "strong_shadow_minus_strong_absent"
    )
    assert c["identification_bounds"] == pytest.approx([1 - 1 / 432, 1])


@pytest.fixture
def reference_review(tmp_path: Path) -> Path:
    """Copy the approved, hash-bound proposal while isolating mutation tests."""
    source = study.base(ROOT) / runner.REFERENCE_REVIEW
    target = tmp_path / runner.REFERENCE_REVIEW
    target.mkdir(parents=True)
    for name in ("proposal.json", "approval.json"):
        (target / name).write_bytes((source / name).read_bytes())
    return tmp_path


def test_four_corrected_controls_preserve_every_original_record(
    reference_review: Path, records: list[dict]
) -> None:
    before = copy.deepcopy(records)
    fresh = runner.corrected_reference_records(reference_review, records)
    assert records == before
    assert len(fresh) == 4
    assert {r["phase"] for r in fresh} == {"calibration2"}
    assert {r["assignment_id"] for r in fresh}.isdisjoint(
        r["assignment_id"] for r in records
    )
    assert len({r["replaces_assignment_id"] for r in fresh}) == 4


@pytest.mark.parametrize("mutation", ["approval", "proposal"])
def test_reference_approval_and_exact_code_are_binding(
    reference_review: Path, records: list[dict], mutation: str
) -> None:
    path = reference_review / runner.REFERENCE_REVIEW / (mutation + ".json")
    value = json.loads(path.read_text())
    if mutation == "approval":
        value["fresh_judge_requests"] = 5
    else:
        value["four_references"][0]["proposed_code"] += "# changed\n"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="matching approval"):
        runner.corrected_reference_records(reference_review, records)


def test_new_calibration_gate_preserves_old_failure_and_applies_same_thresholds(
    reference_review: Path, records: list[dict]
) -> None:
    old = ready([r for r in records if r["phase"] == "calibration"])
    fresh = ready(runner.corrected_reference_records(reference_review, records))
    targets = {r["replaces_assignment_id"] for r in fresh}
    for row in old:
        if row["assignment_id"] in targets:
            row["label"] = "mixed"
    unresolved = next(r for r in old if r["source_task_id"] == "APPS/train/2715")
    unresolved["label"] = None
    assert not study.calibration_gate(old)["passed"]
    designated = runner.qualified_calibration_rows(old + fresh)
    assert len(designated) == 144
    gate = study.calibration_gate(designated)
    assert gate["passed"] and gate["unresolved"] == 1
    assert sum(r["label"] == "mixed" for r in old) == 4
    fresh[0]["label"] = "mixed"
    assert not study.calibration_gate(runner.qualified_calibration_rows(old + fresh))[
        "passed"
    ]


def test_missing_fresh_controls_never_fall_back_to_old_success(
    reference_review: Path, records: list[dict]
) -> None:
    old = ready([r for r in records if r["phase"] == "calibration"])
    fresh = ready(runner.corrected_reference_records(reference_review, records))
    for row in fresh:
        row["label"] = None
    selected = runner.qualified_calibration_rows(old + fresh)
    assert sum(r["label"] is None for r in selected) == 4
    assert not study.calibration_gate(selected)["passed"]
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        runner.qualified_calibration_rows(old + fresh[:-1])
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        runner.qualified_calibration_rows(old + fresh[:-1] + fresh[:1])


def test_corrected_control_counts_are_separate_and_have_no_generations(
    reference_review: Path, records: list[dict]
) -> None:
    fresh = runner.corrected_reference_records(reference_review, records)
    measured = [runner.measure(reference_review, r) for r in records + fresh]
    counts = runner.counts(measured)
    original = counts["calibration/reference/reviewer_observation"]
    corrected = counts["calibration2/reference/reviewer_observation"]
    assert original["reference_judgments_assigned"] == 144
    assert corrected["reference_judgments_assigned"] == 4
    assert corrected["pending_judges"] == 4
    assert corrected["started"] == corrected["raw_saved"] == 0
    assert all(
        r["generation_model"] is None for r in measured if r["phase"] == "calibration2"
    )


def test_saved_corrected_judge_is_not_retried(
    reference_review: Path, records: list[dict], monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = runner.corrected_reference_records(reference_review, records)
    p = pipeline(reference_review, fresh)
    row = fresh[0]
    path = runner.path_for(p.base, row) / "judge/started.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    request = AsyncMock()
    monkeypatch.setattr(p, "request", request)
    asyncio.run(p.judging_stage(row))
    request.assert_not_awaited()


@pytest.fixture
def cleanup_base(tmp_path: Path, records: list[dict]) -> Path:
    """Isolate the approved saved evidence without issuing any model calls."""
    source = study.base(ROOT)
    for name in (runner.PILOT_PROPOSAL, runner.PILOT_CLEANUP + "/approval.json"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    for rec in records:
        if rec["phase"] == "pilot":
            shutil.copytree(
                runner.path_for(source, rec), runner.path_for(tmp_path, rec)
            )
    return tmp_path


def test_approved_cleanup_is_readiness_only_and_keeps_gpt_held(
    cleanup_base: Path, records: list[dict]
) -> None:
    original = [runner.measure(cleanup_base, r) for r in records]
    before = copy.deepcopy(original)
    qualification = runner.pilot_qualification(cleanup_base, records)
    selected = runner.qualified_pilot_rows(original, qualification)
    assert original == before
    assert not study.pilot_gate(original)["passed"]
    gate = study.pilot_gate(selected)
    assert [g["profile"] for g in gate["groups"] if g["passed"]] == ["glm", "qwen"]
    assert qualification["main_models"] == ["glm", "qwen"]
    assert qualification["held_models"] == ["gpt-oss-120b"]
    assert [g["coding_correct"] for g in gate["groups"]] == [65, 69, 60]
    assert [g["coding_resolved"] for g in gate["groups"]] == [72, 69, 72]
    assert (
        sum(
            a["probe_correct"] != b["probe_correct"] for a, b in zip(original, selected)
        )
        == 10
    )
    assert sum(a["label"] != b["label"] for a, b in zip(original, selected)) == 4
    assert all(a == b for a, b in zip(original, selected) if a["phase"] != "pilot")
    assert [runner.measure(cleanup_base, r) for r in records] == original


@pytest.mark.parametrize(
    "file", ["generation/raw.json", "judge/packet.json", "judge/result.json"]
)
def test_cleanup_rejects_changed_saved_evidence(
    cleanup_base: Path, records: list[dict], file: str
) -> None:
    proposal = json.loads((cleanup_base / runner.PILOT_PROPOSAL).read_text())
    aid = proposal["judge_repairs"][0]["assignment_id"]
    path = cleanup_base / "requests" / aid / file
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="evidence changed"):
        runner.pilot_qualification(cleanup_base, records)


def test_cleanup_requires_exact_approval_and_complete_designations(
    cleanup_base: Path, records: list[dict]
) -> None:
    qualification = runner.pilot_qualification(cleanup_base, records)
    rows = [runner.measure(cleanup_base, r) for r in records]
    aid = next(iter(qualification["facts"]))
    with pytest.raises(ValueError, match="Missing or duplicate"):
        runner.qualified_pilot_rows(
            [r for r in rows if r["assignment_id"] != aid], qualification
        )
    with pytest.raises(ValueError, match="Missing or duplicate"):
        runner.qualified_pilot_rows(rows + [rows[0]], qualification)
    next(r for r in rows if r["assignment_id"] == aid)["phase"] = "main"
    with pytest.raises(ValueError, match="non-pilot"):
        runner.qualified_pilot_rows(rows, qualification)
    path = cleanup_base / runner.PILOT_CLEANUP / "approval.json"
    approval = json.loads(path.read_text())
    approval["main_models"].append("gpt-oss-120b")
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="matching approval"):
        runner.pilot_qualification(cleanup_base, records)


def test_pilot_metadata_cleanup_does_not_relax_strict_parser(
    cleanup_base: Path, records: list[dict]
) -> None:
    proposal = json.loads((cleanup_base / runner.PILOT_PROPOSAL).read_text())
    item = proposal["judge_repairs"][0]
    path = cleanup_base / "requests" / item["assignment_id"] / "judge"
    output = ModelOutput.model_validate(json.loads((path / "raw.json").read_text()))
    packet = json.loads((path / "packet.json").read_text())
    with pytest.raises(ValueError):
        runner.parse_packet("reviewer_observation", output.completion, packet)
    cleaned, _ = runner.remove_approved_evidence_notes(output.completion)
    assert (
        runner.parse_packet("reviewer_observation", cleaned, packet).label
        == "comprehension_only"
    )
    duplicate = output.completion.replace(
        '"start_line_note":', '"start_line_note": null, "start_line_note":', 1
    )
    with pytest.raises(ValueError, match="Duplicate"):
        runner.remove_approved_evidence_notes(duplicate)
    with pytest.raises(ValueError, match="metadata differs"):
        runner.remove_approved_evidence_notes(cleaned)
    data = json.loads(cleaned)
    data["evidence"][0]["quote"] = "invented source"
    with pytest.raises(ValueError, match="exact source"):
        runner.parse_packet("reviewer_observation", json.dumps(data), packet)
    data = json.loads(cleaned)
    data["extra"] = True
    with pytest.raises(ValueError):
        runner.parse_packet("reviewer_observation", json.dumps(data), packet)


@pytest.fixture
def main_format_case(
    tmp_path: Path, records: list[dict]
) -> tuple[Path, dict, dict, str]:
    """Use bound synthetic source evidence and the exact real cleanup approval."""
    base = study.base(tmp_path)
    approval = json.loads(
        (study.base(ROOT) / runner.MAIN_FORMAT / "approval.json").read_text()
    )
    for name in (runner.MAIN_FORMAT + "/approval.json", approval["proposal"]):
        target = base / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((study.base(ROOT) / name).read_bytes())
    rec = next(r for r in records if r["phase"] == "main" and r["profile"] == "qwen")
    code = "def f(xs):\n    return [x for x in xs]\n"
    packet = {
        "code": code,
        "entry_point": "f",
        "transformation": "Construct the returned list",
        "input_contract": "xs is a list",
    }
    text = json.dumps(
        {
            "status": "classified",
            "target_forms": ["list_comprehension"],
            "evidence": [
                {
                    "start_line": 2,
                    "end_line": 2,
                    "start_line_note": None,
                    "quote": "    return [x for x in xs]",
                }
            ],
            "flags": [],
            "rationale": "The return is a list comprehension.",
        }
    )
    path = runner.path_for(base, rec)
    source = ModelOutput.from_content("test", code).model_dump(mode="json")
    output = ModelOutput.from_content("test", text).model_dump(mode="json")
    result = {
        "status": "judge_error",
        "label": None,
        "error_type": "ValidationError",
        "source_raw_sha256": runner.digest(runner.json_text(source)),
        "packet_sha256": runner.digest(runner.json_text(packet)),
    }
    for name, data in (
        ("generation/raw.json", source),
        ("judge/raw.json", output),
        ("judge/packet.json", packet),
        ("judge/result.json", result),
    ):
        runner.original.freeze(path / name, data)
    return base, rec, packet, text


def test_main_cleanup_preserves_raw_and_strict_labels(main_format_case: tuple) -> None:
    base, rec, packet, text = main_format_case
    path = runner.path_for(base, rec)
    original = {str(p): p.read_bytes() for p in path.rglob("*.json")}
    with pytest.raises(ValueError):
        runner.parse_packet(rec["study"], text, packet)
    candidate = runner.save_main_format_qualification(base, rec)
    assert candidate is not None and candidate["label"] == "comprehension_only"
    assert runner.save_main_format_qualification(base, rec) == candidate
    assert all(Path(name).read_bytes() == data for name, data in original.items())
    strict = runner.measure(base, rec)
    qualified = runner.measure(base, rec, qualified=True)
    assert strict["label"] is None and strict["judge_status"] == "judge_error"
    assert (
        qualified["label"] == "comprehension_only" and qualified["strict_label"] is None
    )
    assert qualified["label_source"] == "approved_main_judge_format_v1"
    assert (
        runner.counts([qualified])["main/qwen/reviewer_observation"]["strict_resolved"]
        == 0
    )
    assert runner.main_format_candidate(base, {**rec, "phase": "pilot"}) is None


@pytest.mark.parametrize(
    "replacement",
    [
        '"start_line": 2, "start_line": 3',
        '"start_line": 2, "start_line": 2.0',
        '"start_line": true, "start_line": 1',
        '"start_line": 2, "end_line": 2',
        '"start_line": 2, "start_line_note": "duplicate note"',
    ],
)
def test_main_cleanup_rejects_other_or_conflicting_duplicates(
    main_format_case: tuple, replacement: str
) -> None:
    _, _, _, text = main_format_case
    with pytest.raises(ValueError):
        runner.normalize_main_judge_format(
            text.replace('"start_line": 2', replacement, 1)
        )


@pytest.mark.parametrize("suffix", [" extra", "{}", "\n```", " NaN"])
def test_main_cleanup_rejects_trailing_json_and_text(
    main_format_case: tuple, suffix: str
) -> None:
    _, _, _, text = main_format_case
    with pytest.raises(ValueError):
        runner.normalize_main_judge_format(text + suffix)


def test_main_cleanup_duplicate_operation_is_location_and_type_limited(
    main_format_case: tuple,
) -> None:
    _, rec, packet, text = main_format_case
    clean, edits = runner.normalize_main_judge_format(
        text.replace('"start_line": 2', '"start_line": 2, "start_line": 2')
    )
    assert (
        runner.parse_packet(rec["study"], clean, packet).label == "comprehension_only"
    )
    assert {e["operation"] for e in edits} == {
        "remove_noncontract_metadata",
        "collapse_identical_duplicate_start_line",
    }
    for raw in (
        '{"start_line":2,"start_line":2}',
        '{"foo":{"start_line":2,"start_line":2}}',
        '{"evidence":[{"start_line_note":7}]}',
        '{"evidence":[{"start_line_note":NaN}]}',
    ):
        with pytest.raises(ValueError):
            runner.normalize_main_judge_format(raw)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_evidence", "extra_field", "empty_rationale", "max_tokens", "abstention"],
)
def test_main_cleanup_leaves_unapproved_failures_unresolved(
    main_format_case: tuple, mutation: str
) -> None:
    base, rec, _, text = main_format_case
    path = runner.path_for(base, rec) / "judge/raw.json"
    data = json.loads(text)
    if mutation == "wrong_evidence":
        data["evidence"][0]["quote"] = "fabricated source"
    elif mutation == "extra_field":
        data["extra"] = True
    elif mutation == "empty_rationale":
        data["rationale"] = ""
    elif mutation == "abstention":
        data.update(status="abstain", target_forms=[])
    output = ModelOutput.from_content("test", json.dumps(data))
    if mutation == "max_tokens":
        output.choices[0].stop_reason = "max_tokens"
    path.write_text(output.model_dump_json())
    assert runner.main_format_candidate(base, rec) is None
    assert not (runner.path_for(base, rec) / runner.MAIN_FORMAT_RESULT).exists()


@pytest.mark.parametrize(
    "name",
    [
        "generation/raw.json",
        "judge/raw.json",
        "judge/packet.json",
        "judge/result.json",
        "judge/qualified-format-v1.json",
    ],
)
def test_main_cleanup_detects_tampered_bindings(
    main_format_case: tuple, name: str
) -> None:
    base, rec, _, _ = main_format_case
    runner.save_main_format_qualification(base, rec)
    path = runner.path_for(base, rec) / name
    data = json.loads(path.read_text())
    if name == runner.MAIN_FORMAT_RESULT:
        data["label"] = "loop_only"
    else:
        data["tampered"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        runner.measure(base, rec, qualified=True)


def test_main_cleanup_requires_exact_approval(main_format_case: tuple) -> None:
    base, rec, _, _ = main_format_case
    path = base / runner.MAIN_FORMAT / "approval.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="approval or proposal"):
        runner.main_format_candidate(base, rec)


def test_future_qualification_retains_strict_failure_and_never_replays(
    main_format_case: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, rec, packet, _ = main_format_case
    raw = json.loads((runner.path_for(base, rec) / "judge/raw.json").read_text())
    shutil.rmtree(runner.path_for(base, rec) / "judge")
    p = pipeline(tmp_path, [rec])
    p.phase = "main"
    p.manifest["judges"] = {
        rec["study"]: {"system_prompt": "unchanged synthetic rubric"}
    }
    p.tasks_by_id[rec["source_task_id"]] = {}
    monkeypatch.setattr(
        runner,
        "packet_from_output",
        lambda *_: runner.style_judge.JudgeInput.model_validate(packet),
    )

    async def saved_response(*_args: object) -> dict:
        runner.original.freeze(runner.path_for(base, rec) / "judge/raw.json", raw)
        return raw

    request = AsyncMock(side_effect=saved_response)
    monkeypatch.setattr(p, "request", request)
    asyncio.run(p.judging_stage(rec))
    asyncio.run(p.judging_stage(rec))
    assert request.await_count == 1
    assert p.history[-1]["strict_failed"] is True and p.history[-1]["failed"] is False
    assert p.history[-1]["format_qualified"] is True
    assert p.measurements[rec["assignment_id"]]["label"] == "comprehension_only"
    assert runner.measure(base, rec)["label"] is None
    # Unqualified strict errors still trigger the unchanged five-in-48 guard.
    for i in range(5):
        p.event(
            {**rec, "assignment_id": "synthetic-" + str(i)},
            "judge",
            True,
            "ValidationError",
        )
    assert p.stop.is_set()


def test_analysis_keeps_strict_and_qualified_views_separate(
    main_format_case: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, rec, _, _ = main_format_case
    runner.save_main_format_qualification(base, rec)
    old = {"assignment_id": "retained", "label": "loop_only"}
    runner.write_json(
        base / "retained-main.json", {"rows": [old], "lineage": {"fixed": True}}
    )
    manifest = {"cap_usd": 275}
    monkeypatch.setattr(runner, "load", lambda *_args, **_kw: (manifest, [rec]))
    seen = []

    def analysis(rows: list[dict], *, cohort: str, task_count: int) -> dict:
        seen.append((cohort, copy.deepcopy(rows), task_count))
        return {
            "assigned": len(rows),
            "missing_labels": sum(r["label"] is None for r in rows),
        }

    monkeypatch.setattr(study, "analyze_cohort", analysis)
    monkeypatch.setattr(study, "task_sensitivity", lambda rows: {"assigned": len(rows)})
    result = runner.analyze(tmp_path)
    assert len(seen) == 6
    assert seen[0][1][0]["label"] is None
    assert seen[3][1][0]["label"] == "comprehension_only"
    assert seen[1][1] == seen[4][1] == [old]
    assert result["main_format_qualification"]["recovered_labels"] == 1
    assert (
        json.loads((base / "analysis-v1/new72/measured-rows.json").read_text())[0][
            "label"
        ]
        is None
    )
    assert (
        json.loads((base / "analysis-v2/new72/measured-rows.json").read_text())[0][
            "strict_label"
        ]
        is None
    )
