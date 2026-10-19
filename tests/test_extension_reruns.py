"""Authorized reruns cannot broaden selection or replace existing valid outcomes."""

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
rerun = importlib.import_module("scripts.rerun_extension_missing")


def original_rows() -> list[dict[str, Any]]:
    """Build a complete denominator with exactly the ten authorized gaps."""
    ids = sorted(rerun.AUTHORIZED_IDS) + [f"saved-{i}" for i in range(12278)]
    return [
        {
            "assignment_id": aid,
            "raw_saved": aid not in rerun.AUTHORIZED_IDS,
            "mode": "A",
            "condition": {"user_preference": "loops"},
            "profile": "gpt-oss-120b",
            "provider": "fireworks",
            "repetition_index": 1,
            "scenario": "rl_update",
            "source_task_id": "HumanEval/29",
            "label": None if aid in rerun.AUTHORIZED_IDS else "loops",
        }
        for aid in ids
    ]


def test_only_the_ten_outcomes_change_and_original_stays_intact() -> None:
    original = original_rows()
    updates = [{**r, "raw_saved": True, "label": "loops"} for r in original[:10]]
    combined = rerun.overlay(original, updates)
    assert len(combined) == 12288
    assert all(r["raw_saved"] for r in combined)
    assert combined[10:] == original[10:]
    assert not any(r["raw_saved"] for r in original[:10])


@pytest.mark.parametrize(
    "kind", ["extra_missing", "new_response", "duplicate", "mode_b"]
)
def test_changed_original_scope_is_rejected(kind: str) -> None:
    rows = original_rows()
    if kind == "extra_missing":
        rows[10]["raw_saved"] = False
    elif kind == "new_response":
        rows[0]["raw_saved"] = True
    elif kind == "duplicate":
        rows[-1] = rows[-2]
    else:
        rows[0]["mode"] = "B"
    with pytest.raises(ValueError):
        rerun.select_missing(rows)


@pytest.mark.parametrize("kind", ["duplicate", "extra", "provider", "condition"])
def test_overlay_rejects_new_assignments_or_changed_treatments(kind: str) -> None:
    rows = original_rows()
    updates = [dict(r) for r in rows[:10]]
    if kind == "duplicate":
        updates[-1] = updates[0]
    elif kind == "extra":
        updates.append(rows[10])
    elif kind == "provider":
        updates[0]["provider"] = "other"
    else:
        updates[0]["condition"] = {"user_preference": "comprehensions"}
    with pytest.raises(ValueError):
        rerun.overlay(rows, updates)


def test_cli_cannot_collect_without_explicit_paid_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["rerun", "run", "--amendment", "unused.json"])
    monkeypatch.setattr(
        rerun, "run", lambda *args: pytest.fail("No paid authorization")
    )
    with pytest.raises(SystemExit) as error:
        rerun.main()
    assert error.value.code == 2
