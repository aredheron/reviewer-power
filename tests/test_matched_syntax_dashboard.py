"""Fresh robustness exports preserve unknowns, task sets and whole inference families."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grader_power import matched_syntax as study
from grader_power import matched_syntax_analysis as analysis
from grader_power import results_dashboard as dashboard
from grader_power.matched_dashboard import verify_exports


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def bind(run):
    names = [
        "manifest.json",
        "assignments.json",
        "analysis-v1/analysis.json",
        "analysis-v1/measured-rows.json",
    ]
    write(
        run / "operations/completion/dashboard-audit.json",
        {
            "verified": True,
            "source_and_retained_artifact_bindings_verified": True,
            "assigned": 20736,
            "families": [6, 12, 6],
            "bootstrap_draws": 10000,
            "bindings": {
                n: hashlib.sha256((run / n).read_bytes()).hexdigest() for n in names
            },
        },
    )


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    root = tmp_path_factory.mktemp("matched-syntax-dashboard")
    run = root / "runs/reviewer-observation/matched-syntax-v1"
    records = study.schedule(Path(__file__).resolve().parents[1])[0]
    rows = [
        r | {"label": study.STYLES[r["study"]][0] + "_only", "full_correct": False}
        for r in records
        if r["phase"] == "main"
    ]
    for r in rows:
        if r["profile"] == "glm" and r["study"] == "naming":
            r["label"] = None
    rows[0]["label"] = "no_" + rows[0]["study"].replace("quotes", "quote") + "_evidence"
    write(run / "manifest.json", {"run": study.RUN})
    write(run / "assignments.json", records)
    write(run / "analysis-v1/measured-rows.json", rows)
    write(run / "analysis-v1/analysis.json", analysis.analyze(rows))
    bind(run)
    return root, run, rows


def test_fresh_syntax_has_own_task_denominators_and_shared_absence(
    completed, monkeypatch
):
    root, _run, rows = completed
    monkeypatch.setattr(
        dashboard,
        "SOURCES",
        (
            (
                "MATCHED_SYNTAX",
                "reviewer-observation/matched-syntax-v1",
                "analysis-v1/measured-rows.json",
            ),
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "all_recovery_path",
        lambda *args: pytest.fail("Cannot use historical syntax loader"),
    )
    data = dashboard.build_data(root)
    assert data["assigned"] == 20736
    assert len(data["conditions"]) == 18 and len(data["groups"]) == 4
    assert len(data["sources"]) == 1
    assert sum(c["relation"] == "absent" for c in data["conditions"]) == 2
    for c in data["conditions"]:
        for cell in c["cells"]:
            assert cell["endpoint"] == "per_assigned"
            assert cell["tasks"] == 24
            assert cell["assigned"] == (384 if cell["user"] == "all" else 192)
            assert cell["bootstrap_draws"] == 10000
            if c["family"] == "MATCHED_NAMING" and cell["model"] == "glm":
                assert cell["unknown"] == cell["assigned"]
                assert cell["estimate"] is None and cell["identification_bounds"] == [
                    0,
                    1,
                ]
    assert [
        f["holm_family_size"] for f in data["matched_syntax"]["analysis"]["families"]
    ] == [6, 12, 6]
    assert all(
        f["inference_status"] == "withheld_unresolved"
        for f in data["matched_syntax"]["analysis"]["families"]
    )
    evidence = [r for r in rows if r["assignment_id"] == rows[0]["assignment_id"]]
    # Explicitly keep resolved no-evidence outcomes in the assigned denominator.
    cell = dashboard.matched_tally(
        evidence,
        [
            {
                "study": evidence[0]["study"],
                "model": evidence[0]["profile"],
                **evidence[0]["condition"],
                "task_weighted_opposite_style_rate": 0,
                "task_cluster_interval95": [0, 0],
            }
        ],
    )
    assert (
        cell["no_evidence"] == 1 and cell["unknown"] == 0 and cell["alternative"] == 0
    )


@pytest.mark.parametrize("change", ["hash", "factor", "family", "withholding"])
def test_syntax_verifier_rejects_changed_exports(completed, tmp_path, change):
    _, original, _ = completed
    import shutil

    run = tmp_path / "run"
    shutil.copytree(original, run)
    target = run / (
        "analysis-v1/measured-rows.json"
        if change in {"hash", "factor"}
        else "analysis-v1/analysis.json"
    )
    value = copy.deepcopy(dashboard.read_json(target))
    if change == "hash":
        value[0]["label"] = None
    elif change == "factor":
        value[0]["source_task_id"] = "Wrong/task"
    elif change == "family":
        value["families"][0]["comparisons"].pop()
    else:
        value["families"][0]["comparisons"][0]["p_value"] = 0.01
    write(target, value)
    if change != "hash":
        bind(run)
    with pytest.raises(ValueError):
        verify_exports(run, syntax=True)
