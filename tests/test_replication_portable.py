"""Portable planning, admission, blinding, and request replay protection."""

import asyncio
import shutil
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from inspect_ai.model import ChatMessageUser, ModelOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from grader_power import replication as r

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "configs/replication", tmp_path / "configs/replication")
    for folder in ("src", "prompts/presentation"):
        shutil.copytree(ROOT / folder, tmp_path / folder)
    shutil.copyfile(ROOT / "uv.lock", tmp_path / "uv.lock")
    return tmp_path


@pytest.mark.parametrize(
    ("study", "count"),
    [("loops", 336), ("naming", 144), ("quotes", 144), ("strong", 64)],
)
def test_offline_plan_complete_balanced_unique(
    root: Path, study: str, count: int
) -> None:
    run = r.prepare(root, study, study, "gpt-oss-120b", "pilot")
    rows = r.measured(run)
    assert len(rows) == len({x["assignment_id"] for x in rows}) == count
    assert sum(x["kind"] == "core" for x in rows) == count // 2
    assert not r.pilot_gate(rows, study)["passed"]
    assert not list(run.glob("attempts/*"))
    with pytest.raises(FileExistsError):
        r.prepare(root, study, study, "gpt-oss-120b", "pilot")


def test_main_requires_matching_successful_pilot(root: Path) -> None:
    pilot = r.prepare(root, "pilot", "loops", "glm", "pilot")
    with pytest.raises(ValueError, match="admission failed"):
        r.prepare(root, "main", "loops", "glm", "main", "pilot")
    for row in r.measured(pilot):
        r.save(
            pilot / "attempts" / row["assignment_id"] / "result.json",
            {
                "raw_saved": True,
                "full_correct": True,
                "label": "loop_only",
                "probe_correct": True,
            },
        )
    main = r.prepare(root, "main", "loops", "glm", "main", "pilot")
    assert len(r.measured(main)) == 10752
    with pytest.raises(ValueError, match="differ"):
        r.prepare(root, "wrong-model", "loops", "deepseek", "main", "pilot")


def test_changed_packets_and_plans_fail_closed(root: Path) -> None:
    run = r.prepare(root, "pilot", "strong", "glm", "pilot")
    (run / "plan.json").write_text("{}")
    with pytest.raises(ValueError, match="plan changed"):
        r.measured(run)
    (root / "configs/replication/strong-packets.json.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        r.packets(root, "strong")
    with pytest.raises(ValueError):
        r.run_path(root, "../../escape")


def test_blind_judge_never_gets_condition_or_model() -> None:
    row = next(p for p in r.packets(ROOT, "loops") if p["kind"] == "core")
    code = f"def {row['entry_point']}(*args):\n    return []\n"
    packet = r.judge_packet(row, code).model_dump()
    assert set(packet) == {"entry_point", "transformation", "input_contract", "code"}
    assert "condition" not in packet and "profile" not in packet


def test_request_limits_saved_raw_reuse_and_ambiguous_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = AsyncMock()
    client.generate.return_value = ModelOutput.from_content("mock", "safe")
    monkeypatch.setattr(r, "get_model", lambda *args, **kwargs: client)
    messages = [ChatMessageUser(content="synthetic")]
    request = r.Requests(1)
    output = asyncio.run(request.call(tmp_path / "one", "mock", {}, messages))
    assert output.completion == "safe"
    asyncio.run(request.call(tmp_path / "one", "mock", {}, messages))
    assert client.generate.call_count == 1
    with pytest.raises(r.RequestLimit):
        asyncio.run(request.call(tmp_path / "two", "mock", {}, messages))
    assert not (tmp_path / "two/started.json").exists()
    client.generate.side_effect = RuntimeError("SENSITIVE_ERROR_TEXT")
    with pytest.raises(RuntimeError, match="withheld"):
        asyncio.run(r.Requests(1).call(tmp_path / "failed", "mock", {}, messages))
    assert "SENSITIVE_ERROR_TEXT" not in (tmp_path / "failed/error.json").read_text()
    with pytest.raises(RuntimeError, match="Started request"):
        asyncio.run(r.Requests(1).call(tmp_path / "failed", "mock", {}, messages))
    assert client.generate.call_count == 2


@pytest.mark.parametrize("study", ["loops", "naming", "quotes", "strong"])
def test_collect_generates_scores_and_resumes_without_duplicate_calls(
    root: Path, monkeypatch: pytest.MonkeyPatch, study: str
) -> None:
    """Exercise the full adapter pipeline using fake providers and a fake isolated oracle."""
    from dataclasses import dataclass

    @dataclass
    class Outcome:
        status: str = "pass"
        full_correct: bool = True

    run = r.prepare(root, "tiny", study, "gpt-oss-120b", "pilot")
    plan = r.read_json(run / "plan.json")
    row = next(x for x in plan["assignments"] if x["kind"] == "core")
    plan["assignments"] = [row]
    (run / "plan.json").write_text(r.encoded(plan))
    (run / "plan.sha256").write_text(r.digest(run / "plan.json"))
    code = f"def {row['entry_point']}(*args):\n    return []"
    judge = {
        "status": "classified",
        "target_forms": ["other"],
        "evidence": [{"start_line": 2, "end_line": 2, "quote": "    return []"}],
        "flags": [],
        "rationale": "The function returns a list literal.",
    }
    expected_label = "neither"
    if study in ("naming", "quotes"):
        expected_label = (
            "no_naming_evidence" if study == "naming" else "no_quote_evidence"
        )
        judge["target_forms"] = [expected_label]
        judge["evidence"] = [{"start_line": 1, "end_line": 2}]
    client = AsyncMock()
    client.generate.side_effect = [
        ModelOutput.from_content("mock", "```python\n" + code + "\n```"),
        ModelOutput.from_content("mock", r.encoded(judge)),
    ]
    monkeypatch.setattr(r, "get_model", lambda *args, **kwargs: client)
    monkeypatch.setattr(r, "require_sandbox", lambda *args: None)
    monkeypatch.setattr(r, "load_dataset", lambda *args: None)
    executions = []

    def oracle(*args, **kwargs):
        executions.append(args)
        return Outcome()

    monkeypatch.setattr(r, "evaluate_solution", oracle)
    monkeypatch.setenv("FIREWORKS_API_KEY", "synthetic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic")
    with pytest.raises(r.RequestLimit):
        asyncio.run(r.collect(root, run, "synthetic", 1))
    assert client.generate.call_count == len(executions) == 1
    asyncio.run(r.collect(root, run, "synthetic", 1))
    assert client.generate.call_count == 2 and len(executions) == 1
    assert r.measured(run)[0]["label"] == expected_label
    asyncio.run(r.collect(root, run, "synthetic", 1))
    assert client.generate.call_count == 2
    (root / "prompts/presentation/judge-loops.txt").write_text("changed")
    with pytest.raises(ValueError, match="judge prompts changed"):
        asyncio.run(r.collect(root, run, "synthetic", 1))
