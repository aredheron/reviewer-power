"""Offline Mode A integration, durability, and confidentiality regression tests."""

import fcntl
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.log import read_eval_log

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "offline_demo", ROOT / "evals" / "inspect_oneshot.py"
)
assert SPEC is not None and SPEC.loader is not None
demo = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(demo)


@pytest.fixture
def storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Put generated output under pytest storage and forbid dotenv/origin reads."""
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    monkeypatch.setattr(
        demo,
        "code_revision",
        lambda: {
            "commit": "test-commit",
            "working_tree_dirty": True,
            "source_sha256": "test-source",
        },
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        """Fail if offline evaluation tries to discover credentials or Git origins."""
        raise AssertionError("dotenv and Git-origin inspection are forbidden")

    monkeypatch.setattr("inspect_ai._util.dotenv.find_dotenv", forbidden)
    monkeypatch.setattr("inspect_ai._util.dotenv.dotenv_values", forbidden)
    monkeypatch.setattr("inspect_ai._util.git.git_context", forbidden)
    return tmp_path / "runs" / "synthetic" / "demo"


def rows(path: Path) -> list[dict[str, Any]]:
    """Read the reproducible export without parsing Inspect internals."""
    return [
        json.loads(line) for line in (path / "results.jsonl").read_text().splitlines()
    ]


def test_complete_block_logs_resume_and_retry(storage: Path) -> None:
    """Real Inspect mock generation retains eight assignments, failures, and retries."""
    first = demo.run_demo("demo", "status")
    assert first["new_attempts"] == 8
    initial = rows(storage)
    assert len(initial) == 8
    assert len({row["trial_id"] for row in initial}) == 8
    assert (
        len(
            {
                json.dumps(row["rendered"]["condition"], sort_keys=True)
                for row in initial
            }
        )
        == 8
    )
    assert sum(row["status"] == "infrastructure_error" for row in initial) == 1
    assert {row["scoring"]["style"]["label"] for row in initial if row["scoring"]} >= {
        "comprehension_only",
        "loop_only",
        "mixed",
        "neither",
        "invalid_or_unparseable",
    }
    for row in initial:
        assert row["code_revision"]["working_tree_dirty"] is True
        attempt_dir = storage / "attempts" / row["trial_id"] / "0001"
        log = read_eval_log(str(attempt_dir / row["inspect_log"]))
        assert log.samples is not None
        sample = log.samples[0]
        assert not isinstance(sample.input, str)
        assert [message.content for message in sample.input] == [
            message["content"] for message in row["rendered"]["messages"]
        ]
        if row["scoring"]:
            assert sample.output.completion == row["raw_response"]
            assert row["scoring"]["extraction"]["raw_response"] == row["raw_response"]
            assert row["scoring"]["correctness"]["status"] == "not_evaluated"

    snapshots = {
        path: path.read_bytes() for path in storage.rglob("*") if path.is_file()
    }
    assert demo.run_demo("demo", "status")["new_attempts"] == 0
    assert all(path.read_bytes() == content for path, content in snapshots.items())
    assert demo.run_demo("demo", "status", retry_failures=True)["new_attempts"] == 1
    retried = rows(storage)
    assert len(retried) == 9
    retry = next(row for row in retried if row["attempt_number"] == 2)
    original = next(row for row in initial if row["attempt_id"] == retry["retry_of"])
    assert original["status"] == "infrastructure_error"
    assert retry["original_attempt_id"] == original["attempt_id"]
    assert retry["trial_id"] == original["trial_id"]
    assert retry["status"] == "scored"
    assert all(row in retried for row in initial)
    assert demo.run_demo("demo", "status", retry_failures=True)["new_attempts"] == 0


def test_interrupted_trial_is_retained_without_regeneration(
    storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash leaves all assignments and explicit uncertainty about generation."""
    actual = demo.execute_attempt

    def crash(*args: Any, **kwargs: Any) -> None:
        """Interrupt after assignment to exercise recovery of an unfinished attempt."""
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "execute_attempt", crash)
    with pytest.raises(KeyboardInterrupt):
        demo.run_demo("demo", "status")
    assert len(rows(storage)) == 8
    assert all(row["status"] == "pending" for row in rows(storage))
    monkeypatch.setattr(demo, "execute_attempt", actual)
    summary = demo.run_demo("demo", "status")
    assert summary["new_attempts"] == 7
    assert len(rows(storage)) == 8
    assert sum(row["status"] == "interrupted" for row in rows(storage)) == 1


def test_changed_definition_and_corrupt_storage_fail_closed(
    storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Never overwrite a run after prompt/config/source changes or corrupt JSON."""
    monkeypatch.setattr(
        demo, "execute_attempt", lambda *args: {"status": "scored", "scoring": None}
    )
    demo.run_demo("demo", "status")
    with pytest.raises(ValueError, match="changed"):
        demo.run_demo("demo", "narrative")
    result = next(storage.glob("attempts/*/*/result.json"))
    result.write_text("broken")
    with pytest.raises(json.JSONDecodeError):
        demo.run_demo("demo", "status")
    assert result.read_text() == "broken"


def test_atomic_attempts_cannot_be_replaced(tmp_path: Path) -> None:
    """Immutable publication fails on collisions; export replacement is explicit."""
    path = tmp_path / "record.json"
    demo.atomic_write(path, "original")
    with pytest.raises(FileExistsError):
        demo.atomic_write(path, "replacement")
    assert path.read_text() == "original"
    assert list(tmp_path.iterdir()) == [path]


def test_interrupted_raw_response_is_retained(
    storage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between response persistence and final scoring never loses raw text."""
    raw = "synthetic response saved before interruption"

    def crash(trial: Any, number: int, directory: Path, image: str | None) -> None:
        """Interrupt after raw persistence to verify that recovery retains the output."""
        output = demo.ModelOutput.from_content(demo.MODEL, raw)
        demo.atomic_write(
            directory / "raw.json", demo.json_text(output.model_dump(mode="json"))
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(demo, "execute_attempt", crash)
    with pytest.raises(KeyboardInterrupt):
        demo.run_demo("demo", "status")
    monkeypatch.setattr(
        demo, "execute_attempt", lambda *args: {"status": "scored", "scoring": None}
    )
    demo.run_demo("demo", "status")
    interrupted = next(row for row in rows(storage) if row["status"] == "interrupted")
    assert interrupted["raw_response"] == raw
    assert interrupted["raw_output"]["choices"]
    assert interrupted["attempt_number"] == 1


def test_concurrent_writer_fails_before_generating(storage: Path) -> None:
    """A second writer cannot publish duplicate attempts for the same run."""
    storage.mkdir(parents=True)
    with (storage / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            demo.run_demo("demo", "status")
    assert not (storage / "manifest.json").exists()


def test_code_revision_records_dirty_untracked_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HEAD and dirty state remain separate, and hashes include uncommitted source."""
    monkeypatch.setattr(demo, "ROOT", tmp_path)
    source = tmp_path / "src" / "grader_power" / "synthetic.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic untracked module\n")
    commands: list[list[str]] = []

    def git(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Record requested metadata operations and supply deterministic Git output."""
        commands.append(args)
        return subprocess.CompletedProcess(
            args,
            0,
            "test-head\n"
            if "rev-parse" in args
            else "?? src/grader_power/synthetic.py\n",
        )

    monkeypatch.setattr(demo.subprocess, "run", git)
    revision = demo.code_revision()
    assert revision["commit"] == "test-head"
    assert revision["working_tree_dirty"] is True
    assert "src/grader_power/synthetic.py" in revision["file_sha256"]
    assert len(commands) == 2
    assert all(
        "remote" not in command and "diff" not in command for command in commands
    )
    source.write_text("# Changed untracked module\n")
    assert demo.code_revision()["source_sha256"] != revision["source_sha256"]


def test_hidden_material_is_not_rendered(storage: Path) -> None:
    """No solutions, hidden tests, scores, or execution metadata enter the input."""
    definition = demo.build_definition("demo", "status")
    for trial in definition["trials"]:
        text = json.dumps(trial["rendered"]["messages"])
        for hidden in (
            "canonical_solution",
            "base_input",
            "plus_input",
            "mock_case",
            "source_sha256",
            "result.append",
            "lambda value",
        ):
            assert hidden not in text
        assert "not_reviewed" not in text


def test_launcher_drops_parent_secrets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The launcher passes a fixed environment rather than filtering secret names."""
    specification = importlib.util.spec_from_file_location(
        "launcher", ROOT / "scripts" / "run_dry_run.py"
    )
    assert specification is not None and specification.loader is not None
    launcher = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(launcher)
    monkeypatch.setenv("UNRECOGNIZED_CREDENTIAL", "synthetic-test-sentinel")
    monkeypatch.setattr(
        sys, "argv", ["run_dry_run.py", "--run-id", "demo", "--preflight", "status"]
    )

    def capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        """Check the worker's environment and directory without starting a process."""
        assert "synthetic-test-sentinel" not in json.dumps(kwargs)
        assert set(kwargs["env"]) == {
            "PATH",
            "HOME",
            "TMPDIR",
            "LANG",
            "PYTHON_DOTENV_DISABLED",
        }
        assert kwargs["cwd"] != str(ROOT)
        return subprocess.CompletedProcess([], 0)

    monkeypatch.setattr(launcher.subprocess, "run", capture)
    with pytest.raises(SystemExit) as result:
        launcher.main()
    assert result.value.code == 0
