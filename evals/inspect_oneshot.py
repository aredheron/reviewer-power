"""Offline synthetic Mode A: Inspect generation/scoring plus immutable attempts.

Run through scripts/run_dry_run.py, which starts a credential-free worker. This
module intentionally exposes no provider, endpoint, real-task, or Mode B option.
"""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast
from unittest.mock import patch

from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    ModelUsage,
    get_model,
)
from inspect_ai.scorer import Score, Scorer, Target, scorer
from inspect_ai.solver import Generate, Solver, TaskState, solver

from grader_power.datasets.fixture_builder import (
    synthetic_oracle_data,
    synthetic_region,
    synthetic_task,
)
from grader_power.render import TEMPLATE_PATHS, render_pilot_block
from grader_power.schemas import PreflightVariant
from grader_power.scoring import OracleRequest, score_response

ROOT = Path(__file__).resolve().parents[1]
MODEL = "mockllm/grader-power-synthetic"
CASES = (
    "comprehension",
    "loop",
    "mixed",
    "alternative",
    "syntax_error",
    "ambiguous_fences",
    "missing_target",
    "model_error",
)
MOCK_RESPONSES = {
    "comprehension": "def double_values(values):\n    return [2 * value for value in values]\n",
    "loop": (
        "```python\ndef double_values(values):\n    result = []\n"
        "    for value in values:\n        result.append(2 * value)\n"
        "    return result\n```\n"
    ),
    "mixed": (
        "def double_values(values):\n    result = [2 * value for value in values]\n"
        "    for value in values:\n        result.append(2 * value)\n"
        "    return result\n"
    ),
    "alternative": "def double_values(values):\n    return list(map(lambda value: 2 * value, values))\n",
    "syntax_error": "def double_values(values)\n    return [2 * value for value in values]\n",
    "ambiguous_fences": "```python\ndef double_values(values):\n    return []\n```\n```python\nx = 1\n```\n",
    "missing_target": "def different_function(values):\n    return []\n",
}


def json_text(value: object) -> str:
    """Canonical serialization for manifests, records, and content hashes."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n"


def atomic_write(path: Path, content: str, *, replace: bool = False) -> None:
    """Publish flushed UTF-8 bytes atomically; immutable records cannot clobber.

    Only the reproducible export projection uses replace=True. Hard linking the
    temporary file publishes complete contents and fails if a record exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read an object record; malformed storage fails rather than being repaired."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Expected an object in run storage")
    return value


def code_revision() -> dict[str, Any]:
    """Record HEAD, dirty state, and code hashes without reading diffs or remotes.

    The allowlist includes untracked implementation files, but no caches, .env,
    or raw run outputs. HEAD alone never represents an uncommitted checkout.
    """

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        """Read local Git metadata without raising for unavailable repository state."""
        return subprocess.run(
            ["git", "-C", str(ROOT), *args],
            capture_output=True,
            text=True,
            check=False,
        )

    head = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=all")
    paths = sorted(
        {
            *ROOT.glob("src/grader_power/**/*.py"),
            *ROOT.glob("evals/*.py"),
            *ROOT.glob("scripts/*.py"),
            *(ROOT / "prompts" / name for name in TEMPLATE_PATHS),
            ROOT / "configs" / "sandbox" / "runner.py",
            ROOT / "configs" / "sandbox" / "Dockerfile",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
        }
    )
    hashes = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file() and not path.is_symlink()
    }
    return {
        "commit": head.stdout.strip() if head.returncode == 0 else None,
        "working_tree_dirty": bool(status.stdout) if status.returncode == 0 else None,
        "source_sha256": hashlib.sha256(json_text(hashes).encode()).hexdigest(),
        "file_sha256": hashes,
    }


def generation_config() -> GenerateConfig:
    """Resolve mock settings explicitly, without provider/environment options."""
    return GenerateConfig(temperature=0, max_tokens=256, seed=0, max_retries=0)


def build_definition(
    run_id: str, preflight: PreflightVariant, oracle_image: str | None = None
) -> dict[str, Any]:
    """Resolve only allowlisted offline settings and a complete eight-cell block."""
    block = render_pilot_block(synthetic_task(), preflight_variant=preflight, seed=0)
    cases = dict(zip(sorted(row.condition_id for row in block), CASES, strict=True))
    return {
        "schema_version": 1,
        "run_id": run_id,
        "purpose": "synthetic_software_demonstration_not_research_data",
        "human_review_status": "not_reviewed",
        "prompt_status": "provisional",
        "mode": "A",
        "source_split": "synthetic",
        "fixture_version": "synthetic-double-values-v1",
        "oracle_version": {
            "image": oracle_image,
            "source": "synthetic_not_upstream_EvalPlus",
            "fixture_sha256": hashlib.sha256(
                json_text(synthetic_oracle_data().tasks).encode()
            ).hexdigest(),
            "mode": "full",
            "wall_seconds": 180,
        },
        "model": {"id": MODEL, "provider": "mockllm", "revision": "scripted-v1"},
        "generation": generation_config().model_dump(mode="json"),
        "inspect_settings": {
            "log_format": "eval",
            "display": "none",
            "notification": False,
            "log_samples": True,
            "log_model_api": True,
            "log_refusals": False,
            "fail_on_error": False,
            "retry_on_error": 0,
            "max_samples": 1,
        },
        "inspect_version": version("inspect-ai"),
        "python_version": platform.python_version(),
        "code_revision": code_revision(),
        "retry_policy": {
            "automatic_retries": 0,
            "explicit_retry_limit": 1,
            "eligible_statuses": ["infrastructure_error", "interrupted"],
            "response_failures_eligible": False,
        },
        "correctness": "Docker_requested"
        if oracle_image
        else "not_evaluated_no_pinned_Docker_image",
        "trials": [
            {
                "trial_id": "synthetic-" + row.render_id,
                "rendered": asdict(row),
                "mock_case": cases[row.condition_id],
            }
            for row in block
        ],
    }


@solver
def persist_response(path: str) -> Solver:
    """Generate once with Inspect and save raw output before any scoring."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        """Persist exact provider output before a scorer can fail or be interrupted."""
        state = await generate(state, tool_calls="none")
        atomic_write(Path(path), json_text(state.output.model_dump(mode="json")))
        return state

    return solve


@scorer(metrics=[])
def static_score(oracle_image: str | None = None) -> Scorer:
    """Adapt condition-blind project scoring to Inspect's native scorer contract."""

    async def score(state: TaskState, target: Target) -> Score:
        """Return static evidence and explicitly requested isolated correctness results."""
        oracle = (
            OracleRequest(
                synthetic_oracle_data(),
                "humaneval",
                "HumanEval/synthetic-double-values",
                oracle_image,
            )
            if oracle_image is not None
            else None
        )
        result = score_response(
            state.output.completion, synthetic_region(), oracle=oracle
        )
        return Score(
            value=int(result.style.label == "comprehension_only"),
            metadata=result.to_dict(),
        )

    return score


@task
def inspect_oneshot(
    trial: dict[str, Any], raw_path: str, oracle_image: str | None = None
) -> Task:
    """One assigned synthetic attempt; the persisted manifest supplies its block.

    Each attempt has its own native Inspect log to make interrupted recovery
    unambiguous. Only rendered message contents become model-visible input.
    """
    messages = trial["rendered"]["messages"]
    if any(message["role"] != "user" for message in messages):
        raise ValueError("The current provisional renderer requires user-role context")
    return Task(
        name="synthetic_mode_a",
        dataset=[
            Sample(
                id=trial["trial_id"],
                input=[
                    ChatMessageUser(content=message["content"]) for message in messages
                ],
                metadata={"trial_id": trial["trial_id"], "synthetic": True},
            )
        ],
        solver=persist_response(raw_path),
        scorer=static_score(oracle_image),
        config=generation_config(),
    )


def execute_attempt(
    trial: dict[str, Any], number: int, directory: Path, oracle_image: str | None = None
) -> dict[str, Any]:
    """Use Inspect's mock model, preserving even failed generations and scores.

    The pinned Inspect startup unconditionally calls init_dotenv and captures
    Git remotes. Disable these two metadata conveniences for this offline path;
    explicit safe provenance is recorded by our manifest instead. No general
    Inspect behavior, solver scheduling, or provider implementation is replaced.
    """
    if version("inspect-ai") != "0.3.277":
        raise RuntimeError("Revalidate offline initialization for this Inspect version")
    case = trial["mock_case"]

    def respond(*args: Any, **kwargs: Any) -> ModelOutput:
        """A fixed response per assigned case, independent of model-visible factors."""
        if case == "model_error" and number == 1:
            raise RuntimeError("Synthetic model failure for retry demonstration")
        output = ModelOutput.from_content(
            MODEL, MOCK_RESPONSES["loop" if case == "model_error" else case]
        )
        output.usage = ModelUsage(input_tokens=0, output_tokens=0, total_tokens=0)
        return output

    model = get_model(MODEL, custom_outputs=respond, memoize=False)
    with (
        patch("inspect_ai._eval.context.init_dotenv"),
        patch("inspect_ai._eval.task.log.git_context", return_value=None),
    ):
        logs = eval(
            inspect_oneshot(trial, str(directory / "raw.json"), oracle_image),
            model=model,
            log_dir=str(directory / "inspect"),
            log_format="eval",
            display="none",
            notification=False,
            log_samples=True,
            log_model_api=True,
            log_refusals=False,
            fail_on_error=False,
            retry_on_error=0,
            max_samples=1,
        )
    log = logs[0]
    sample = log.samples[0] if log.samples else None
    result: dict[str, Any] = {
        "status": "infrastructure_error",
        "error": "Inspect did not produce a scored sample",
        "raw_response": None,
        "scoring": None,
        "usage": None,
        "usage_kind": "synthetic_zero_tokens_not_measured",
        "inspect_log": str(Path(log.location).relative_to(directory)),
    }
    if sample is not None:
        result["raw_response"] = (
            sample.output.completion if sample.output.choices else None
        )
        result["usage"] = (
            sample.output.usage.model_dump() if sample.output.usage else None
        )
        if sample.error is not None:
            result["error"] = sample.error.message
        elif sample.scores:
            result.update(
                status="scored",
                error=None,
                scoring=next(iter(sample.scores.values())).metadata,
            )
    return result


def export_results(run_dir: Path, definition: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild an atomic JSONL projection; retain every attempt and pending trial."""
    rows: list[dict[str, Any]] = []
    for trial in definition["trials"]:
        results = sorted(
            (run_dir / "attempts" / trial["trial_id"]).glob("*/result.json")
        )
        if results:
            for path in results:
                assigned = read_json(path.with_name("assigned.json"))
                result = read_json(path)
                if (
                    assigned.get("run_id") != definition["run_id"]
                    or assigned.get("trial_id") != trial["trial_id"]
                    or any(result.get(key) != value for key, value in assigned.items())
                ):
                    raise ValueError("Attempt identity differs from its assignment")
                rows.append(result)
        else:
            rows.append(
                {
                    "run_id": definition["run_id"],
                    **trial,
                    "status": "pending",
                    "attempt_id": None,
                    "synthetic": True,
                }
            )
    content = "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    )
    atomic_write(run_dir / "results.jsonl", content, replace=True)
    return rows


def run_demo(
    run_id: str,
    preflight: PreflightVariant,
    *,
    retry_failures: bool = False,
    oracle_image: str | None = None,
) -> dict[str, Any]:
    """Resume a synthetic block; terminal attempts are never silently rerun.

    Crashed intents become explicit interrupted records. One opt-in retry of an
    infrastructure failure/interruption is allowed; invalid responses are final.
    An OS advisory lock excludes concurrent writers and releases after crashes.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must be 1-64 ASCII letters/digits/underscores/hyphens")
    if oracle_image is not None and not re.fullmatch(
        r"sha256:[0-9a-f]{64}", oracle_image
    ):
        raise ValueError("oracle-image must be an immutable local sha256 image ID")
    run_dir = ROOT / "runs" / "synthetic" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        definition = build_definition(run_id, preflight, oracle_image)
        manifest = run_dir / "manifest.json"
        if manifest.exists():
            if json_text(read_json(manifest)) != json_text(definition):
                raise ValueError(
                    "Run definition or source code changed; use a new run-id"
                )
        else:
            atomic_write(manifest, json_text(definition))
        export_results(run_dir, definition)
        executed = 0
        for trial in definition["trials"]:
            trial_dir = run_dir / "attempts" / trial["trial_id"]
            previous: dict[str, Any] | None = None
            intents = sorted(trial_dir.glob("*/assigned.json"))
            for index, intent_path in enumerate(intents, start=1):
                if index > 2 or intent_path.parent.name != f"{index:04d}":
                    raise ValueError(
                        "Attempt sequence is inconsistent with retry policy"
                    )
                result_path = intent_path.with_name("result.json")
                if not result_path.exists():
                    interrupted = {
                        **read_json(intent_path),
                        "status": "interrupted",
                        "error": "Previous process stopped before recording its outcome",
                        "scoring": None,
                        "raw_response": None,
                        "completed_at": datetime.now(UTC).isoformat(),
                    }
                    raw_path = intent_path.with_name("raw.json")
                    if raw_path.exists():
                        interrupted["raw_output"] = read_json(raw_path)
                        interrupted["raw_response"] = ModelOutput.model_validate(
                            interrupted["raw_output"]
                        ).completion
                    atomic_write(result_path, json_text(interrupted))
                previous = read_json(result_path)
            if previous is not None and not (
                retry_failures
                and len(intents) == 1
                and previous["status"] in ("infrastructure_error", "interrupted")
            ):
                continue
            number = len(intents) + 1
            attempt_id = f"{trial['trial_id']}/a{number}"
            directory = trial_dir / f"{number:04d}"
            intent = {
                "run_id": run_id,
                "synthetic": True,
                **trial,
                "attempt_id": attempt_id,
                "attempt_number": number,
                "original_attempt_id": f"{trial['trial_id']}/a1",
                "retry_of": previous["attempt_id"] if previous else None,
                "started_at": datetime.now(UTC).isoformat(),
                "resolved_config_sha256": hashlib.sha256(
                    json_text(definition).encode()
                ).hexdigest(),
                "code_revision": definition["code_revision"],
            }
            atomic_write(directory / "assigned.json", json_text(intent))
            started = time.monotonic()
            try:
                outcome = execute_attempt(trial, number, directory, oracle_image)
            except (OSError, RuntimeError) as error:
                # Reporting policy: keep the assigned attempt even if Inspect's
                # initialization/logging fails. Do not serialize arbitrary error
                # text, which can contain environment values in third-party code.
                outcome = {
                    "status": "infrastructure_error",
                    "error": type(error).__name__,
                    "raw_response": None,
                    "scoring": None,
                }
            result = {
                **intent,
                **outcome,
                "completed_at": datetime.now(UTC).isoformat(),
                "wall_seconds": time.monotonic() - started,
            }
            if (directory / "raw.json").exists():
                result["raw_output"] = read_json(directory / "raw.json")
                result["raw_response"] = ModelOutput.model_validate(
                    result["raw_output"]
                ).completion
            atomic_write(directory / "result.json", json_text(result))
            executed += 1
            export_results(run_dir, definition)
        rows = export_results(run_dir, definition)
    return {
        "run_directory": str(run_dir),
        "assigned_trials": 8,
        "new_attempts": executed,
        "retained_rows": len(rows),
        "scored_attempts": sum(row["status"] == "scored" for row in rows),
        "failed_or_interrupted_attempts": sum(
            row["status"] in ("infrastructure_error", "interrupted") for row in rows
        ),
        "correctness": "see per-attempt adapter outcomes"
        if oracle_image
        else "not evaluated",
    }


def main() -> None:
    """Worker entry point; options cannot select providers or research datasets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preflight", choices=("status", "narrative"), required=True)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument(
        "--oracle-image",
        help="Optional verified local sha256 image ID; never pulls/builds",
    )
    args = parser.parse_args()
    print(
        json_text(
            run_demo(
                args.run_id,
                cast(PreflightVariant, args.preflight),
                retry_failures=args.retry_failures,
                oracle_image=args.oracle_image,
            )
        ),
        end="",
    )


if __name__ == "__main__":
    main()
