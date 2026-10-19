"""Deterministic lock ownership, orphan checks and paid-order gates; no providers."""

import fcntl
import json
import os
from pathlib import Path

import pytest

from grader_power import collection_lock as locks


@pytest.fixture
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model a dedicated process group without changing pytest's own session."""
    monkeypatch.setattr(locks.os, "getpgrp", lambda: os.getpid())
    monkeypatch.setattr(locks, "process_group_members", lambda group: {os.getpid()})


def test_mutual_exclusion_and_release(tmp_path: Path, isolated: None) -> None:
    """Concurrent callers cannot enter, and normal completion leaves released metadata."""
    with (
        locks.collection_lock(tmp_path, owner="auto1", run_id="synthetic"),
        pytest.raises(RuntimeError, match="holds the lock"),
        locks.collection_lock(tmp_path, owner="auto1", run_id="duplicate"),
    ):
        pytest.fail("Concurrent lock was acquired")
    path = tmp_path / "runs/extension-pilot/coordination/owner.json"
    assert json.loads(path.read_text())["state"] == "released"


def test_surviving_child_blocks_restart(
    tmp_path: Path, isolated: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not infer readiness merely because the parent's flock disappeared."""
    monkeypatch.setattr(
        locks, "process_group_members", lambda group: {os.getpid(), 987654}
    )
    with (
        pytest.raises(RuntimeError, match="surviving child"),
        locks.collection_lock(tmp_path, owner="auto1", run_id="synthetic"),
    ):
        pass
    with (
        pytest.raises(RuntimeError, match="Previous collection"),
        locks.collection_lock(tmp_path, owner="auto1", run_id="restart"),
    ):
        pass
    monkeypatch.setattr(locks, "process_group_members", lambda group: set())
    with locks.collection_lock(tmp_path, owner="auto1", run_id="restart"):
        pass


def test_phase_exception_releases_lock(tmp_path: Path, isolated: None) -> None:
    """A Python exception releases ownership only after child inspection succeeds."""
    with (
        pytest.raises(ValueError, match="synthetic"),
        locks.collection_lock(tmp_path, owner="auto1", run_id="synthetic"),
    ):
        raise ValueError("synthetic")
    path = tmp_path / "runs/extension-pilot/coordination/collection.lock"
    with path.open() as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_order_gate(tmp_path: Path, isolated: None) -> None:
    """Offline preparation may overlap, but later paid phases need predecessor completion."""
    with (
        pytest.raises(RuntimeError, match="requires auto1"),
        locks.collection_lock(tmp_path, owner="auto5", run_id="synthetic"),
    ):
        pass
    directory = tmp_path / "runs/extension-pilot/coordination"
    (directory / "auto1-complete.json").write_text('{"complete": true}')
    with locks.collection_lock(tmp_path, owner="auto5", run_id="synthetic"):
        pass
    with (
        pytest.raises(RuntimeError, match="requires auto5"),
        locks.collection_lock(tmp_path, owner="auto4", run_id="synthetic"),
    ):
        pass


@pytest.mark.parametrize(
    "owner,predecessor",
    [
        ("exploration-v1", "auto4"),
        ("named-files-v1", "exploration-v1"),
        ("named-files-v2", "named-files-v1"),
        ("development-six-v1", "named-files-v2"),
    ],
)
def test_exploration_waits_for_prior_pilot_and_shares_lock(
    tmp_path: Path, isolated: None, owner: str, predecessor: str
) -> None:
    """Follow-up runs use a fresh owner without bypassing existing paid work."""
    with (
        pytest.raises(RuntimeError, match=f"requires {predecessor}"),
        locks.collection_lock(tmp_path, owner=owner, run_id="synthetic"),
    ):
        pass
    directory = tmp_path / "runs/extension-pilot/coordination"
    (directory / f"{predecessor}-complete.json").write_text('{"complete": true}')
    with (
        locks.collection_lock(tmp_path, owner=owner, run_id="synthetic"),
        pytest.raises(RuntimeError, match="holds the lock"),
        locks.collection_lock(tmp_path, owner="auto1", run_id="other"),
    ):
        pass


def test_real_process_inspection_excludes_its_own_ps_child(tmp_path: Path) -> None:
    """A dedicated real worker releases cleanly without mistaking ps for an orphan."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; from grader_power.collection_lock import isolate_collection_process, collection_lock; import sys; isolate_collection_process();\nwith collection_lock(Path(sys.argv[1]), owner='auto1', run_id='synthetic'): pass",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    assert not result.stdout
    assert (
        json.loads(
            (tmp_path / "runs/extension-pilot/coordination/owner.json").read_text()
        )["state"]
        == "released"
    )
