"""Verify prompt identities, blinded audit routing, bounded resumes and contrasts."""

import importlib
import json
import sys
from collections import Counter
from itertools import product
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
study = importlib.import_module("grader_power.reversal")
from grader_power.spending import SpendingLedger, SpendingLimit, nanodollars

run = importlib.import_module("evals.inspect_reversal")
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def schedule():
    return run.schedule(ROOT)


def test_frozen_endpoints_and_balanced_quartets(schedule):
    main = [r.record for n, rows in schedule if n.startswith("main-") for r in rows]
    pilot = [r.record for n, rows in schedule if n.startswith("pilot-") for r in rows]
    assert len(main) == len({r["assignment_id"] for r in main}) == 1024
    assert len(pilot) == 32
    assert len({r["source_task_id"] for r in main}) == 32
    assert {r["source_task_id"] for r in pilot} == {"HumanEval/87", "Mbpp/413"}
    assert Counter(
        (
            r["condition"]["grader_consequence"],
            r["condition"]["user_preference"],
            r["condition"]["variant"],
        )
        for r in main
    ) == {x: 64 for x in product(("shadow", "rl_update"), study.LABELS, study.VARIANTS)}
    selection = json.loads((ROOT / "manifests/main_selection.json").read_text())
    assert not {r["source_task_id"] for r in main} & set(selection["confirmation"])
    for _, rows in schedule:
        assert len({study.matched_key(r.record) for r in rows}) == 1
        versions = {r.record["condition"]["variant"]: r.record for r in rows}
        assert set(versions) == set(study.VARIANTS)
        old = versions["original"]["messages"][0]["content"]
        mention = versions["mention_only"]["messages"][0]["content"]
        definitions = versions["definitions_only"]["messages"][0]["content"]
        both = versions["both"]["messages"][0]["content"]
        assert mention.replace(" or map/filter pipelines", "") == old
        assert both.replace(" or map/filter pipelines", "") == definitions
        addition = (
            "--- .ci/style-definitions.yaml ---\n" + run.study.DEFINITIONS + "\n\n"
            if hasattr(run.study, "DEFINITIONS")
            else ""
        )
        assert addition and definitions.replace(addition, "") == old


def test_audit_selection_and_blinding(tmp_path):
    slots, retained = run.prepare_audit(ROOT, tmp_path)
    assert len(slots) == 256
    assert Counter(
        (s["historical_study"], s["purpose"], s["direction"]) for s in slots
    ) == {x: 32 for x in product(("old", "new"), ("shadow", "rl_update"), study.LABELS)}
    for slot in slots:
        path = tmp_path / "audit/packets" / (slot["id"] + ".json")
        assert path.exists() == slot["eligible"]
        if path.exists():
            assert set(json.loads(path.read_text())) == {
                "entry_point",
                "transformation",
                "input_contract",
                "code",
            }
    assert retained
    assert (
        len(json.loads((tmp_path / "audit/inspection-selection.json").read_text()))
        == 16
    )


def gate_rows():
    return [
        {
            "eligible": True,
            "historical_study": s,
            "purpose": p,
            "direction": d,
            "original_label": study.LABELS[d] if s == "old" else "neither",
            "v2_label": study.LABELS[d] if s == "old" else "neither",
        }
        for s, p, d, _ in product(
            ("old", "new"), ("shadow", "rl_update"), study.LABELS, range(32)
        )
    ]


def test_audit_routes_measurement_disagreement_before_new_generation():
    rows = gate_rows()
    assert run.audit_gate(rows)["passed"]
    for r in rows[:14]:
        r["v2_label"] = None
    assert not run.audit_gate(rows)["passed"]
    rows = gate_rows()
    for r in rows[:14]:
        r["v2_label"] = "neither"
    assert not run.audit_gate(rows)["passed"]
    rows = gate_rows()
    for r in rows:
        r["v2_label"] = r["original_label"] = study.LABELS[r["direction"]]
    assert not run.audit_gate(rows)["passed"]


def synthetic():
    return [
        {
            "assignment_id": str((t, p, d, v, i)),
            "source_task_id": t,
            "condition": {
                "grader_consequence": p,
                "user_preference": d,
                "grader_preference": next(x for x in study.LABELS if x != d),
                "variant": v,
            },
            "label": study.LABELS[d]
            if v in ("original", "mention_only")
            else "neither",
            "started": True,
            "raw_saved": True,
            "scored": True,
            "full_correct": True,
        }
        for t, p, d, v, i in product(
            range(32), ("shadow", "rl_update"), study.LABELS, study.VARIANTS, range(2)
        )
    ]


def test_analysis_identifies_definitions_not_mention_or_interaction():
    output = study.analyze(synthetic(), draws=200)
    assert output["counts"]["assigned"] == 1024
    assert len(output["cells"]) == 16
    assert len(output["contrasts"]) == 8
    expected = {
        "both_minus_original": -1,
        "mention": 0,
        "definitions": -1,
        "interaction": 0,
    }
    for c in output["contrasts"]:
        assert c["primary_inference"]["estimate"] == expected[c["name"]]
        assert "p_holm" in c["primary_inference"]


def test_missing_is_not_review_adherence_and_withholds_family():
    rows = synthetic()
    rows[0]["label"] = None
    rows[0]["raw_saved"] = False
    output = study.analyze(rows, draws=200)
    assert output["counts"]["unresolved_or_missing"] == 1
    assert output["counts"]["reviewer_following"] == 0
    assert all(c["primary_inference"] is None for c in output["contrasts"])
    assert all(c["resolved_only_sensitivity"] is not None for c in output["contrasts"])
    assert any(
        c["identification_bounds"][0] != c["identification_bounds"][1]
        for c in output["contrasts"]
    )


def test_pilot_gate_ignores_style_direction():
    rows = synthetic()[:32]
    assert run.pilot_gate(rows)["passed"]
    for r in rows:
        r["label"] = "neither"
    assert run.pilot_gate(rows)["passed"]
    rows[0]["label"] = rows[1]["label"] = None
    assert not run.pilot_gate(rows)["passed"]
    rows = synthetic()[:32]
    for r in rows[:4]:
        r["full_correct"] = False
    assert not run.pilot_gate(rows)["passed"]


def test_single_ledger_counts_audit_generation_judge_and_unknown(tmp_path):
    ledger = SpendingLedger(
        tmp_path / "spending.json", cap=nanodollars(100), scope="fixed-investigation"
    )
    ledger.reserve(
        {
            "audit": nanodollars(20),
            "generation": nanodollars(30),
            "judge": nanodollars(49),
        }
    )
    ledger.settle({"audit": nanodollars(10)})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"extra": nanodollars(12)})
    assert ledger.snapshot()["outstanding"] == nanodollars(79)
    with pytest.raises(ValueError):
        SpendingLedger(
            ledger.path, cap=nanodollars(101), scope="fixed-investigation"
        ).snapshot()


def test_started_generation_is_not_replayed_and_unknown_charge_retained(
    tmp_path, monkeypatch, schedule
):
    name, assignments = schedule[0]
    row = assignments[0]
    base = run.directory(tmp_path)
    path = (
        base / "blocks" / name / run.PROFILE / "attempts" / row.record["assignment_id"]
    )
    path.mkdir(parents=True)
    (path / "started.json").write_text('{"unknown_accepted_request":true}')
    manifest = {"cap_usd": 100}
    monkeypatch.setattr(run, "child_permission", lambda *a: None)
    monkeypatch.setattr(run, "load", lambda *a: manifest)
    monkeypatch.setattr(
        run, "block", lambda *a: ({"oracle_image": "frozen", "plan": {}}, (row,))
    )
    monkeypatch.setattr(run.extension, "access_profiles", lambda *a: {})
    monkeypatch.setattr(run.extension, "enforce_judge_limit", lambda *a: None)
    monkeypatch.setattr(run.original, "collect_judgments", AsyncMock())
    monkeypatch.setattr(run.previous, "measure", lambda *a: [])

    def forbidden(*a, **k):
        raise AssertionError("A started generation was replayed")

    monkeypatch.setattr(run.pilot, "collect_profile", forbidden)
    run.worker(tmp_path, name, 1000)
    balance = run.ledger_for(tmp_path, manifest).snapshot()
    identity = name + "/" + row.record["assignment_id"]
    assert balance["entries"][identity + "/generation"]["settled"] is None
    assert balance["entries"][identity + "/judge"]["settled"] == 0
    assert (base / "blocks" / name / "complete.json").exists()


def test_started_audit_requests_never_replayed(tmp_path, monkeypatch):
    base = run.directory(tmp_path)
    packet = {
        "entry_point": "f",
        "transformation": "double items",
        "input_contract": "list of ints",
        "code": "def f(items):\n    return [2*x for x in items]\n",
    }
    identity = "a" * 64
    slot = {
        "id": identity,
        "eligible": True,
        "packet_sha256": run.digest(run.json_text(packet)),
    }
    manifest = {"audit": [slot]}
    run.original.freeze(base / "audit/packets" / (identity + ".json"), packet)
    for rubric in ("binary", "v2"):
        run.original.freeze(
            base / "audit/results" / identity / rubric / "started.json",
            {"uncertain": True},
        )
    monkeypatch.setattr(run, "child_permission", lambda *a: None)
    monkeypatch.setattr(run, "load", lambda *a: manifest)
    calls = AsyncMock(side_effect=AssertionError("Started audit replayed"))
    monkeypatch.setattr(run, "request_audit_judgment", calls)
    run.audit_worker(tmp_path, identity, 1000)
    calls.assert_not_called()
    balance = run.ledger_for(tmp_path, manifest).snapshot()
    assert balance["spent"] == 0
    assert balance["outstanding"] == 2 * run.BOUND


def test_phase_only_wrapper_uses_unchanged_frozen_judge_bound(
    tmp_path, monkeypatch, schedule
):
    row = schedule[0][1][0]
    aid = row.record["assignment_id"]
    path = tmp_path / run.PROFILE / "attempts" / aid
    path.mkdir(parents=True)
    (path / "raw.json").write_text("{}")
    packet = run.binary_judge.JudgeInput(
        entry_point="f",
        transformation="increment list",
        input_contract="",
        code="def f(xs):\n    return [x+1 for x in xs]\n",
    )
    monkeypatch.setattr(run.original, "packet_from_output", lambda *a: packet)
    frozen = {"plan": {"phase": "pilot"}, "judge": {"system_prompt": "Classify code."}}
    run.enforce_frozen_judge_limit(tmp_path, row, frozen)
    result_path = tmp_path / run.PROFILE / "judge" / aid / "result.json"
    assert not result_path.exists()
    assert "limits" not in frozen["plan"]
    frozen["judge"]["system_prompt"] = "x" * 32768
    run.enforce_frozen_judge_limit(tmp_path, row, frozen)
    result = json.loads(result_path.read_text())
    assert result["status"] == "judge_input_limit" and result["label"] is None
    assert result["input_bytes"] > 32768
