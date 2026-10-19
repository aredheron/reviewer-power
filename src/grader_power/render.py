"""Render whole provisional pilot blocks without inference or code execution."""

import hashlib
import json
import random
from dataclasses import asdict
from itertools import product
from pathlib import Path
from string import Template

from grader_power.schemas import (
    CONSEQUENCES,
    PREFLIGHT_VARIANTS,
    STYLES,
    PilotCondition,
    PreflightVariant,
    RenderedCondition,
    RenderMessage,
    RenderMetadata,
    RenderTask,
    VisibleFile,
)

# This renderer is for the research checkout; callers outside it can explicitly
# supply its prompts directory. No fixture directory is walked or globbed.
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
TEMPLATE_PATHS = (
    "user_task/implement.txt",
    "agents_md/soft.md",
    "grader_rubric/style-review.yaml",
    "decision_policy/selection-policy.yaml",
    "decision_policy/shadow.txt",
    "decision_policy/future_selection.txt",
    "preflight/status.txt",
    "preflight/narrative.txt",
)


def _digest(value: object) -> str:
    """Hash canonical UTF-8 JSON without whitespace or ASCII escaping."""
    content = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def render_pilot_block(
    task: RenderTask,
    *,
    preflight_variant: PreflightVariant,
    seed: int | None = None,
    prompts_dir: Path = PROMPTS_DIR,
) -> tuple[RenderedCondition, ...]:
    """Return all eight cells, holding task and phrasing constant within a block.

    A variant must be chosen explicitly; neither draft is preferred. With no
    seed, order is user × grader × consequence using schema level order. An
    integer seed shuffles the complete block with an isolated random generator;
    it never changes IDs or prompts. No model calls, source-cache reads, hidden
    file reads, task-code execution, or task-eligibility decisions occur.

    Raises TypeError/ValueError for invalid inputs. Missing templates and invalid
    substitutions fail explicitly. Templates are read once per block as UTF-8.
    """
    if not isinstance(task, RenderTask):
        raise TypeError("task must be a RenderTask with explicitly permitted fields")
    if preflight_variant not in PREFLIGHT_VARIANTS:
        raise ValueError("preflight_variant must be 'status' or 'narrative'")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise TypeError("seed must be an integer or None")
    templates = {
        name: (prompts_dir / name).read_text(encoding="utf-8")
        for name in TEMPLATE_PATHS
    }
    metadata = RenderMetadata(
        source_task_id=task.source_task_id,
        derived_task_id=task.derived_task_id,
        family=task.family,
        preflight_variant=preflight_variant,
        ordering_seed=seed,
        template_sha256=_digest(templates),
    )
    request = Template(templates["user_task/implement.txt"]).substitute(
        stub_path=task.stub_path
    )
    block: list[RenderedCondition] = []
    for user, grader, consequence in product(STYLES, STYLES, CONSEQUENCES):
        condition = PilotCondition(user, grader, consequence)
        preferred, other = (
            ("explicit for-loops", "list comprehensions")
            if user == "loops"
            else ("list comprehensions", "explicit for-loops")
        )
        guidance = Template(templates["agents_md/soft.md"]).substitute(
            preferred=preferred, other=other
        )
        rubric = Template(templates["grader_rubric/style-review.yaml"]).substitute(
            grader_style=(
                "explicit_for_loops" if grader == "loops" else "list_comprehensions"
            )
        )
        policy = Template(
            templates["decision_policy/selection-policy.yaml"]
        ).substitute(
            selection_rule=templates[f"decision_policy/{consequence}.txt"].rstrip("\n")
        )
        preflight = Template(
            templates[f"preflight/{preflight_variant}.txt"]
        ).substitute(rubric=rubric, policy=policy)
        files = (
            VisibleFile("README.md", task.readme),
            VisibleFile(task.stub_path, task.stub),
            VisibleFile("AGENTS.md", guidance),
            VisibleFile(".ci/style-review.yaml", rubric),
            VisibleFile(".ci/selection-policy.yaml", policy),
        )
        # The two CI files are exposed inside preflight, once each. Metadata and
        # IDs are deliberately absent from model messages. Do not serialize task.
        context = "\n\n".join(
            f"--- {file.path} ---\n{file.content}" for file in files[:3]
        )
        messages = (RenderMessage("user", f"{request}\n{context}\n\n{preflight}"),)
        block.append(
            RenderedCondition(
                condition=condition,
                condition_id=condition.condition_id,
                render_id=_digest(
                    [
                        task.source_task_id,
                        task.derived_task_id,
                        condition.condition_id,
                        preflight_variant,
                    ]
                ),
                messages=messages,
                visible_files=files,
                preflight=preflight,
                prompt_sha256=_digest([asdict(message) for message in messages]),
                metadata=metadata,
            )
        )
    if seed is not None:
        random.Random(seed).shuffle(block)
    return tuple(block)
