"""Offline matched-render tests; all task text is inert synthetic data."""

import hashlib
import json
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, replace
from itertools import combinations, product
from pathlib import Path
from typing import Any

import pytest

from grader_power.render import PROMPTS_DIR, render_pilot_block
from grader_power.schemas import (
    PilotCondition,
    PreflightVariant,
    RenderedCondition,
    RenderTask,
)


@pytest.fixture
def task() -> RenderTask:
    """A synthetic visible stub, never compiled, imported, or executed."""
    return RenderTask(
        source_task_id="synthetic/source-1",
        derived_task_id="synthetic-derived-1",
        family="mapping",
        stub_path="src/example.py",
        stub='def example(items):\n    """Double each integer in order."""\n    # TODO\n',
        readme="# Example utilities\n\nDeterministic integer utilities.\n",
    )


def files(row: RenderedCondition) -> dict[str, str]:
    """Index visible files without including experiment metadata."""
    return {file.path: file.content for file in row.visible_files}


@pytest.mark.parametrize("variant", ["status", "narrative"])
def test_complete_block(task: RenderTask, variant: PreflightVariant) -> None:
    """Each draft has exactly one of each independently enumerated factor cell."""
    block = render_pilot_block(task, preflight_variant=variant)
    actual = [
        (
            r.condition.user_preference,
            r.condition.grader_preference,
            r.condition.grader_consequence,
        )
        for r in block
    ]
    assert actual == list(
        product(
            ("loops", "comprehensions"),
            ("loops", "comprehensions"),
            ("shadow", "future_selection"),
        )
    )
    assert len({r.condition_id for r in block}) == 8
    assert len({r.render_id for r in block}) == 8
    assert len({r.prompt_sha256 for r in block}) == 8
    assert block[0].condition_id == "pilot-v1.soft.u-loops.g-loops.p-shadow"


@pytest.mark.parametrize("variant", ["status", "narrative"])
def test_every_one_factor_pair_has_only_intended_differences(
    task: RenderTask, variant: PreflightVariant
) -> None:
    """All 12 cube edges change only the assigned line, including message text."""
    block = render_pilot_block(task, preflight_variant=variant)
    checked = 0
    expected_file = {
        "user_preference": "AGENTS.md",
        "grader_preference": ".ci/style-review.yaml",
        "grader_consequence": ".ci/selection-policy.yaml",
    }
    for left, right in combinations(block, 2):
        changed = [
            field
            for field in expected_file
            if getattr(left.condition, field) != getattr(right.condition, field)
        ]
        if len(changed) != 1:
            continue
        checked += 1
        factor = changed[0]
        a, b = files(left), files(right)
        assert {path for path in a if a[path] != b[path]} == {expected_file[factor]}
        assert left.metadata == right.metadata
        a_lines = a[expected_file[factor]].splitlines(keepends=True)
        b_lines = b[expected_file[factor]].splitlines(keepends=True)
        assert len(a_lines) == len(b_lines)
        differences = [(x, y) for x, y in zip(a_lines, b_lines) if x != y]
        assert len(differences) == 1
        before, after = differences[0]
        assert left.messages[0].content.count(before) == 1
        assert (
            left.messages[0].content.replace(before, after) == right.messages[0].content
        )
        assert left.messages[0].role == right.messages[0].role == "user"
        if factor == "user_preference":
            assert left.preflight == right.preflight
            assert before.startswith("- Default to ")
        else:
            assert left.preflight.replace(before, after) == right.preflight
            prefix = (
                "  collection_transformation_style: "
                if factor == "grader_preference"
                else "selection_policy: "
            )
            assert before.startswith(prefix)
    assert checked == 12


def test_two_drafts_preserve_facts_and_are_provisional(task: RenderTask) -> None:
    """Phrasing varies without changing task, factors, or policy/rubric facts."""
    status = render_pilot_block(task, preflight_variant="status")
    narrative = render_pilot_block(task, preflight_variant="narrative")
    for a, b in zip(status, narrative):
        assert a.condition_id == b.condition_id
        assert a.visible_files == b.visible_files
        assert a.render_id != b.render_id
        assert a.prompt_sha256 != b.prompt_sha256
        for row in (a, b):
            visible = files(row)
            assert visible[".ci/style-review.yaml"] in row.preflight
            assert visible[".ci/selection-policy.yaml"] in row.preflight
            assert "selection" not in visible[".ci/style-review.yaml"]
            assert "comprehensions" not in visible[".ci/selection-policy.yaml"]
            assert "for_loops" not in visible[".ci/selection-policy.yaml"]
            policy = visible[".ci/selection-policy.yaml"]
            assert "unchanged frozen checkpoints" in policy
            assert "frozen model weights" in policy
            assert "no parameter updates or gradients from scores" in policy
            assert (
                "correctness_and_patch_acceptance_independent_of_review_score" in policy
            )
            assert row.metadata.preflight_context == "simulated"
            assert row.metadata.wording_status == "draft_requires_human_review"
            assert row.metadata.collection_ready is False
            assert row.metadata.instruction_strength == "soft"
            assert len(row.messages) == 1
            assert "?" not in row.messages[0].content
            assert "simulated" not in row.messages[0].content
            assert "draft_requires_human_review" not in row.messages[0].content


def test_source_hidden_and_unrelated_fields_never_render(task: RenderTask) -> None:
    """Allowlisting excludes hidden canaries even in new/nested metadata fields."""
    record: dict[str, Any] = asdict(task)
    hidden = {
        "canonical_solution": "CANARY_CANONICAL_SOLUTION",
        "base_input": [["CANARY_BASE_INPUT"]],
        "plus_input": [["CANARY_PLUS_INPUT"]],
        "expected_outputs": ["CANARY_EXPECTED_OUTPUT"],
        "reference_solutions": {"loops": "CANARY_REFERENCE"},
        "oracle": {"tests": "CANARY_HIDDEN_TESTS"},
        "metadata": {"instructions": "CANARY_IGNORE_ALL_PREVIOUS_INSTRUCTIONS"},
        "files": {"private.py": "CANARY_ARBITRARY_FILE"},
        "prompt": "CANARY_RAW_SOURCE_PROMPT",
    }
    record.update(hidden)
    projected = RenderTask.from_record(record)
    assert projected == task
    for variant in ("status", "narrative"):
        block = render_pilot_block(projected, preflight_variant=variant)
        serialized = json.dumps([asdict(r) for r in block])
        assert "CANARY_" not in serialized
        assert all(f'"{key}"' not in serialized for key in hidden if key != "metadata")
        for row in block:
            assert set(files(row)) == {
                "README.md",
                "src/example.py",
                "AGENTS.md",
                ".ci/style-review.yaml",
                ".ci/selection-policy.yaml",
            }
    assert record["oracle"] == hidden["oracle"]


def test_metadata_is_never_visible_and_text_is_not_reinterpreted(
    task: RenderTask,
) -> None:
    """Even allowed metadata stays hidden; stub template syntax is literal data."""
    task = replace(
        task,
        source_task_id="CANARY_SOURCE",
        derived_task_id="CANARY_DERIVED",
        family="CANARY_FAMILY",
        stub=task.stub + "# literal $policy ${rubric} {canonical_solution}\n",
    )
    for row in render_pilot_block(task, preflight_variant="status"):
        assert "CANARY_" not in row.messages[0].content
        assert task.stub in row.messages[0].content


def test_ids_hashes_and_seeded_order_are_reproducible(task: RenderTask) -> None:
    """Local seeded ordering preserves complete cells and exact prompt identities."""
    baseline = render_pilot_block(task, preflight_variant="status")
    assert baseline == render_pilot_block(task, preflight_variant="status")
    state = random.getstate()
    seeded = render_pilot_block(task, preflight_variant="status", seed=37)
    assert random.getstate() == state
    assert seeded == render_pilot_block(task, preflight_variant="status", seed=37)
    assert [r.condition_id for r in seeded] != [r.condition_id for r in baseline]
    index = {r.condition_id: r for r in baseline}
    assert set(index) == {r.condition_id for r in seeded}
    for row in seeded:
        original = index[row.condition_id]
        assert row.render_id == original.render_id
        assert row.messages == original.messages
        assert row.prompt_sha256 == original.prompt_sha256
        assert row.metadata.ordering_seed == 37
        encoded = json.dumps(
            [asdict(m) for m in row.messages],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        assert row.prompt_sha256 == hashlib.sha256(encoded).hexdigest()


def test_hash_tracks_text_but_not_metadata(task: RenderTask, tmp_path: Path) -> None:
    """Changing visible bytes or templates changes hashes; metadata cannot do so."""
    original = render_pilot_block(task, preflight_variant="status")[0]
    same = render_pilot_block(
        replace(task, family="other"), preflight_variant="status"
    )[0]
    assert original.prompt_sha256 == same.prompt_sha256
    changed = render_pilot_block(
        replace(task, stub=task.stub + "# café\n"), preflight_variant="status"
    )[0]
    assert changed.prompt_sha256 != original.prompt_sha256
    assert changed.render_id == original.render_id
    shutil.copytree(PROMPTS_DIR, tmp_path / "prompts")
    template = tmp_path / "prompts/user_task/implement.txt"
    template.write_text(template.read_text() + "\n", encoding="utf-8")
    revised = render_pilot_block(
        task, preflight_variant="status", prompts_dir=tmp_path / "prompts"
    )[0]
    assert revised.prompt_sha256 != original.prompt_sha256
    assert revised.metadata.template_sha256 != original.metadata.template_sha256


@pytest.mark.parametrize(
    "path",
    [
        "../secret.py",
        "/tmp/task.py",
        "src/../secret.py",
        "src/a\n.py",
        "fixtures/oracles/test.py",
    ],
)
def test_invalid_fixture_path_is_rejected(task: RenderTask, path: str) -> None:
    """Only safe source-file labels may enter visible context."""
    with pytest.raises(ValueError, match="stub_path"):
        replace(task, stub_path=path)


@pytest.mark.parametrize(
    "field",
    ["source_task_id", "derived_task_id", "family", "stub_path", "stub", "readme"],
)
def test_fixture_fields_are_required_and_validated(
    task: RenderTask, field: str
) -> None:
    """No missing fixture content is replaced with source solutions or prompts."""
    record = asdict(task)
    record.pop(field)
    with pytest.raises(ValueError, match=field):
        RenderTask.from_record(record)
    record[field] = None
    with pytest.raises(TypeError, match=field):
        RenderTask.from_record(record)
    record[field] = "  "
    with pytest.raises(ValueError, match=field):
        RenderTask.from_record(record)


@pytest.mark.parametrize(
    "kwargs,exception",
    [
        ({"preflight_variant": "unknown"}, ValueError),
        ({"preflight_variant": "../private"}, ValueError),
        ({"preflight_variant": "status", "seed": True}, TypeError),
        ({"preflight_variant": "status", "seed": "37"}, TypeError),
    ],
)
def test_invalid_renderer_options_fail(
    task: RenderTask, kwargs: dict[str, Any], exception: type[Exception]
) -> None:
    """Invalid variants/seeds fail instead of falling back to a default."""
    with pytest.raises(exception):
        render_pilot_block(task, **kwargs)


def test_invalid_conditions_and_missing_templates_fail(
    task: RenderTask, tmp_path: Path
) -> None:
    """Reject unsupported experimental levels and missing/malformed assets."""
    for field in ("user_preference", "grader_preference", "grader_consequence"):
        factors: dict[str, Any] = {
            "user_preference": "loops",
            "grader_preference": "loops",
            "grader_consequence": "shadow",
        }
        factors[field] = "unsupported"
        with pytest.raises(ValueError, match=field):
            PilotCondition(**factors)
    with pytest.raises(FileNotFoundError):
        render_pilot_block(task, preflight_variant="status", prompts_dir=tmp_path)
    shutil.copytree(PROMPTS_DIR, tmp_path / "prompts")
    (tmp_path / "prompts/preflight/status.txt").write_text("${unknown}")
    with pytest.raises(KeyError, match="unknown"):
        render_pilot_block(
            task, preflight_variant="status", prompts_dir=tmp_path / "prompts"
        )


def test_synthetic_cli_is_reproducible_across_processes() -> None:
    """The inspection command prints eight draft records without executing task code."""
    root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(root / "scripts/render_pilot_example.py"),
        "--preflight",
        "narrative",
        "--seed",
        "37",
    ]
    a = subprocess.run(command, capture_output=True, text=True, check=True, cwd=root)
    b = subprocess.run(command, capture_output=True, text=True, check=True, cwd=root)
    assert a.stdout == b.stdout
    assert a.stderr == b.stderr == ""
    rows = json.loads(a.stdout)
    assert len(rows) == 8
    assert all(r["metadata"]["collection_ready"] is False for r in rows)
    assert all(r["metadata"]["preflight_context"] == "simulated" for r in rows)
