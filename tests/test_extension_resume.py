"""An authorized recovery retains history, STOP and the five-in-48 safeguard."""

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
resume = importlib.import_module("scripts.resume_extension_main")


def history(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Represent a stopped run with five reviewed failures and intact raw history."""
    path = tmp_path / "operations/request-events.jsonl"
    path.parent.mkdir(parents=True)
    payload = b"".join(
        (json.dumps({"id": str(i), "failed": True}) + "\n").encode() for i in range(5)
    )
    path.write_bytes(payload)
    return path, {
        "reviewed_prefix_bytes": len(payload),
        "reviewed_prefix_sha256": resume.runner.digest(payload),
    }


def append(path: Path, events: list[dict[str, Any]]) -> None:
    """Append synthetic stages without modifying earlier evidence."""
    with path.open("a") as f:
        for event in events:
            f.write(json.dumps(event) + "\n")


def test_reset_only_excludes_reviewed_history_and_still_stops(tmp_path: Path) -> None:
    path, record = history(tmp_path)
    original = path.read_bytes()
    resume.recovery_guard(tmp_path, record)
    append(path, [{"id": str(i + 5), "failed": True} for i in range(4)])
    resume.recovery_guard(tmp_path, record)
    append(path, [{"id": "9", "failed": True}])
    with pytest.raises(RuntimeError, match="Five failed"):
        resume.recovery_guard(tmp_path, record)
    assert path.read_bytes().startswith(original)


def test_recovery_honors_stop_and_detects_truncated_or_changed_history(
    tmp_path: Path,
) -> None:
    path, record = history(tmp_path)
    stop = tmp_path / "operations/STOP"
    stop.touch()
    with pytest.raises(RuntimeError, match="STOP"):
        resume.recovery_guard(tmp_path, record)
    stop.unlink()
    path.write_bytes(path.read_bytes().replace(b"true", b"null", 1))
    with pytest.raises(ValueError, match="history changed"):
        resume.recovery_guard(tmp_path, record)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="offset"):
        resume.recovery_guard(tmp_path, record)


def test_window_ages_normally_and_partial_or_invalid_records_fail(
    tmp_path: Path,
) -> None:
    path, record = history(tmp_path)
    append(path, [{"failed": True}] * 4 + [{"failed": False}] * 48)
    resume.recovery_guard(tmp_path, record)
    append(path, [{"failed": "false"}])
    with pytest.raises(ValueError, match="Malformed"):
        resume.recovery_guard(tmp_path, record)


def test_amendment_must_be_bound_before_collector_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = resume.runner.directory(tmp_path)
    p = base / "operations/recoveries/test.json"
    p.parent.mkdir(parents=True)
    p.write_text("{}")
    monkeypatch.setattr(
        resume.runner,
        "collect",
        lambda _: pytest.fail("Invalid amendment must not collect"),
    )
    with pytest.raises(ValueError, match="binding mismatch"):
        resume.run(tmp_path, p)
