"""Verified matched-review exports retain assignment denominators and paired inference."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power import matched_review as study
from grader_power import matched_review_analysis as analysis
from grader_power import results_dashboard as dashboard


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dashboard_cli_resolves_repository_analysis_in_isolated_python(tmp_path: Path):
    """Exercise the real CLI without PYTHONPATH, while replacing only file export."""
    script = Path(__file__).resolve().parents[1] / "scripts/build_results_dashboard.py"
    code = """
import runpy
import sys
from grader_power import results_dashboard
def export(root, output):
    from grader_power import matched_review_analysis
    assert len(matched_review_analysis.FAMILY_CONTRASTS) == 3
    return {'assigned': 0, 'conditions': [], 'groups': [], 'missing': []}
results_dashboard.write_dashboard = export
sys.argv = [sys.argv[1], '--output', 'unused.html']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "0 observations" in result.stdout
    assert not (tmp_path / "unused.html").exists()


@pytest.fixture(scope="module")
def template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Full synthetic matrix with one required unknown and a mixed resolved outcome."""
    root = tmp_path_factory.mktemp("matched-dashboard-template")
    run = root / "runs/reviewer-observation/matched-review-v1"
    rows = []
    for model, task, arm, user, rep in product(
        study.PROFILES, range(32), study.ARMS, study.STYLES, range(8)
    ):
        for reviewer in (None,) if arm == "absent" else study.STYLES:
            identity = f"{model}-{task}-{arm}-{user}-{reviewer}-{rep}"
            rows.append(
                {
                    "assignment_id": identity,
                    "profile": model,
                    "provider": study.PROFILES[model]["provider"],
                    "model": study.PROFILES[model]["model"],
                    "generation_model": study.PROFILES[model]["model"],
                    "phase": "main",
                    "kind": "core",
                    "mode": "A",
                    "study": "matched_review",
                    "workflow": study.RUN,
                    "source_task_id": f"Task/{task:02}",
                    "historical_assignment_id": f"old-{task}",
                    "repetition_index": rep,
                    "condition": study.condition(arm, user, reviewer),
                    "label": "loop_only",
                    "full_correct": False,
                }
            )
    rows[0]["label"] = None  # Shared absence: primary and unseen families withheld.
    rows[1]["label"] = "mixed"  # Resolved zero for the narrow endpoint, not unknown.
    records = [
        row | {"messages": [{"role": "user", "content": json.dumps(row["condition"])}]}
        for row in rows
    ]
    pilots = [
        {
            "assignment_id": f"pilot-{model}-{kind}-{index}",
            "profile": model,
            "phase": "pilot",
            "kind": kind,
        }
        for model, kind, index in product(
            study.PROFILES, ("core", "comprehension_probe"), range(168)
        )
    ]
    write(run / "assignments.json", records + pilots)
    manifest = {
        "run": study.RUN,
        "main_assignments": 32256,
        "assignments_sha256": sha(run / "assignments.json"),
        "profiles": study.PROFILES,
    }
    write(run / "manifest.json", manifest)
    manifest_sha = sha(run / "manifest.json")
    result = analysis.analyze(rows)
    result["manifest_sha256"] = manifest_sha
    write(run / "analysis-v1/measured-rows.json", rows)
    write(run / "analysis-v1/analysis.json", result)
    folder = run / dashboard.MATCHED_FOLLOWUP
    write(folder / "approval.json", {"explicit_approval": True})
    common = {
        "approval_sha256": sha(folder / "approval.json"),
        "manifest_sha256": manifest_sha,
    }
    write(
        folder / "probe2.json",
        common
        | {
            "assignments": [
                {
                    "assignment_id": f"fresh-{row['assignment_id']}",
                    "probe2_of": row["assignment_id"],
                    "phase": "probe2",
                    "kind": "comprehension_probe",
                    "profile": "gpt-oss-120b",
                }
                for row in pilots
                if row["profile"] == "gpt-oss-120b"
                and row["kind"] == "comprehension_probe"
            ]
        },
    )
    write(
        folder / "format-overlay.json",
        common
        | {
            "rows": {
                row["assignment_id"]: {}
                for row in pilots
                if row["kind"] == "comprehension_probe"
            }
        },
    )
    write(folder / "implementation-plan.json", common)
    write(folder / "probe2-gate.json", {"passed": True})
    write(folder / "qualified-pilot-gate.json", {"passed": True})
    amendment = "operations/recoveries/followup/amendment.json"
    write(
        run / amendment,
        {
            "manifest_sha256": manifest_sha,
            "approved_followup": dashboard.MATCHED_FOLLOWUP,
        },
    )
    lineage = [str(path.relative_to(run)) for path in folder.glob("*.json")] + [
        amendment
    ]
    write(
        run / "operations/completion/analysis-audit.json",
        {
            "verified": True,
            "source_and_retained_artifact_bindings_verified": True,
            "selected_main_assignments": 32256,
            "families": dashboard.MATCHED_FAMILIES,
            "bindings": {name: sha(run / name) for name in dashboard.MATCHED_EXPORTS},
            "lineage_bindings": {name: sha(run / name) for name in lineage},
        },
    )
    return root


@pytest.fixture
def fixture(tmp_path: Path, template: Path) -> tuple[Path, Path, Path]:
    shutil.copytree(template / "runs", tmp_path / "runs")
    run = tmp_path / "runs/reviewer-observation/matched-review-v1"
    return tmp_path, run, run / "analysis-v1/measured-rows.json"


def rebind(run: Path, relative: str, *, lineage: bool = False) -> None:
    path = run / "operations/completion/analysis-audit.json"
    audit = dashboard.read_json(path)
    audit["lineage_bindings" if lineage else "bindings"][relative] = sha(run / relative)
    write(path, audit)


def test_matched_source_single_absence_and_assigned_rate_contract(fixture, monkeypatch):
    root, run, path = fixture
    monkeypatch.setattr(
        dashboard,
        "SOURCES",
        (
            (
                "MATCHED",
                "reviewer-observation/matched-review-v1",
                "analysis-v1/measured-rows.json",
            ),
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "all_recovery_path",
        lambda *args: pytest.fail("New matched study must bypass historical recovery"),
    )
    data = dashboard.build_data(root)
    assert data["assigned"] == 32256 and len(data["sources"]) == 1
    assert len(data["conditions"]) == 21 and len(data["groups"]) == 8
    absence = next(c for c in data["conditions"] if c["relation"] == "absent")
    assert sum(c["relation"] == "absent" for c in data["conditions"]) == 1
    assert all(group["baselines"] == [absence["id"]] for group in data["groups"])
    assert all(c["baseline"] == absence["id"] for c in data["conditions"])
    cells = [s for c in data["conditions"] for s in c["cells"]]
    assert {s["assigned"] for s in cells if s["user"] == "all"} == {512}
    assert {s["assigned"] for s in cells if s["user"] != "all"} == {256}
    assert all(s["endpoint"] == "per_assigned" for s in cells)
    unknown = next(
        s
        for s in absence["cells"]
        if s["model"] == "gpt-oss-120b" and s["user"] == "loops"
    )
    assert unknown["unknown"] == 1 and unknown["mixed"] == 1
    assert unknown["estimate"] is None and unknown["interval95"]["alternative"] is None
    assert unknown["identification_bounds"] == [0, 1 / 256]
    assert data["matched_review"]["analysis"] == dashboard.read_json(
        run / "analysis-v1/analysis.json"
    )
    assert data["sources"][0]["path"] == str(path.relative_to(root))
    assert data["matched_review"]["analysis_source"] == str(
        path.with_name("analysis.json").relative_to(root)
    )
    assert data["matched_review"]["shared_absence"] is True


@pytest.mark.parametrize(
    "change",
    [
        "hash",
        "factor",
        "duplicate",
        "family",
        "withholding",
        "bounds",
        "label",
        "nonfinite",
    ],
)
def test_matched_export_rejects_changed_evidence(fixture, change):
    _, run, path = fixture
    if change in {"hash", "factor", "duplicate", "label"}:
        rows = dashboard.read_json(path)
        if change in {"hash", "label"}:
            rows[0]["label"] = "comprehension_only"
        elif change == "factor":
            rows[0]["condition"]["actor"] = "scorer"
        else:
            rows[0] = copy.deepcopy(rows[1])
        write(path, rows)
        if change != "hash":
            rebind(run, "analysis-v1/measured-rows.json")
    else:
        analysis_path = run / "analysis-v1/analysis.json"
        value = dashboard.read_json(analysis_path)
        if change == "family":
            value["families"][0]["comparisons"].pop()
        elif change == "withholding":
            value["families"][0]["comparisons"][0]["p_value"] = 0.01
        elif change == "bounds":
            value["by_condition"][0]["opposite_style_bounds"] = [0, 1]
        else:
            value["extra_cost"] = float("nan")
        write(analysis_path, value)
        rebind(run, "analysis-v1/analysis.json")
    with pytest.raises(ValueError, match="Matched|Invalid|Duplicate|Require"):
        dashboard.verify_matched_analysis(run, path)


@pytest.mark.parametrize(
    "change", ["missing_lineage", "embedded_approval", "fresh_designation"]
)
def test_matched_followup_lineage_cannot_be_dropped_or_rebound(fixture, change):
    _, run, path = fixture
    if change == "missing_lineage":
        audit_path = run / "operations/completion/analysis-audit.json"
        value = dashboard.read_json(audit_path)
        del value["lineage_bindings"][dashboard.MATCHED_FOLLOWUP + "/approval.json"]
        write(audit_path, value)
    else:
        relative = dashboard.MATCHED_FOLLOWUP + "/probe2.json"
        value = dashboard.read_json(run / relative)
        if change == "embedded_approval":
            value["approval_sha256"] = "0" * 64
        else:
            value["assignments"].pop()
        write(run / relative, value)
        rebind(run, relative, lineage=True)
    with pytest.raises(ValueError, match="Matched"):
        dashboard.verify_matched_analysis(run, path)


def test_matched_directional_intervals_are_exact_frozen_values(fixture):
    _, run, path = fixture
    rows = dashboard.read_json(path)
    frozen = dashboard.read_json(run / "analysis-v1/analysis.json")
    summary = next(
        s
        for s in frozen["by_condition"]
        if s["model"] == "glm"
        and s["arm"] == "absent"
        and s["user_preference"] == "comprehensions"
    )
    selected = [
        r
        for r in rows
        if r["profile"] == "glm"
        and r["condition"]["arm"] == "absent"
        and r["condition"]["user_preference"] == "comprehensions"
    ]
    cell = dashboard.matched_tally(selected, frozen["by_condition"])
    assert cell["estimate"] == summary["task_weighted_opposite_style_rate"] == 1
    assert cell["interval95"]["alternative"] == summary["task_cluster_interval95"]
    assert cell["identification_bounds"] == summary["opposite_style_bounds"]


def test_recovery_switch_adds_no_observations_and_keeps_strict_cells(
    fixture, monkeypatch
):
    """A failed designated replacement is shown even if the strict original succeeded."""
    root, _parent, path = fixture
    run = root / "runs/recovery/matched-review-recovery-v1"
    strict = dashboard.read_json(path)
    rows = copy.deepcopy(strict)
    designated = []
    for i, row in enumerate(rows[:147]):
        identity = f"recovery-{i}"
        designated.append({"assignment_id": identity, "replaces": row["assignment_id"]})
        row.update(
            recovery_assignment_id=identity, original_label=row["label"], label=None
        )
    repairs = [{"assignment_id": r["assignment_id"]} for r in rows[147:233]]
    write(run / "manifest.json", {"run": "matched-review-recovery-v1"})
    write(run / "designated-records.json", designated)
    write(run / "free-repairs-v1.json", {"repairs": repairs})
    write(run / "analysis-v1/measured-rows.json", rows)
    result = analysis.analyze(rows) | {"pending_recoveries": []}
    write(run / "analysis-v1/analysis.json", result)
    names = [
        "manifest.json",
        "designated-records.json",
        "free-repairs-v1.json",
        "analysis-v1/measured-rows.json",
        "analysis-v1/analysis.json",
    ]
    audit = {
        "verified": True,
        "source_and_retained_artifact_bindings_verified": True,
        "assigned": 32256,
        "families": [12, 15, 3],
        "bootstrap_draws": 10000,
        "parent_rows_sha256": sha(path),
        "bindings": {n: sha(run / n) for n in names},
    }
    write(run / "operations/completion/dashboard-audit.json", audit)
    monkeypatch.setattr(
        dashboard,
        "SOURCES",
        (
            (
                "MATCHED",
                "reviewer-observation/matched-review-v1",
                "analysis-v1/measured-rows.json",
            ),
        ),
    )
    data = dashboard.build_data(root)
    assert data["assigned"] == 32256 and len(data["conditions"]) == 21
    assert len(data["sources"]) == 1
    assert data["matched_recovery"]["adds_observations"] == 0
    assert dashboard.read_json(path) == strict
    for field, expected in [("cells", 1), ("recovery_cells", 147)]:
        cells = [s for c in data["conditions"] for s in c[field] if s["user"] == "all"]
        assert sum(c["assigned"] for c in cells) == 32256
        assert sum(c["unknown"] for c in cells) == expected
    rows[300]["label"] = None
    write(run / "analysis-v1/measured-rows.json", rows)
    audit["bindings"]["analysis-v1/measured-rows.json"] = sha(
        run / "analysis-v1/measured-rows.json"
    )
    write(run / "operations/completion/dashboard-audit.json", audit)
    with pytest.raises(ValueError, match="Untreated"):
        dashboard.verify_exports(run, syntax=False)
