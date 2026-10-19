"""Freeze an offline judge pilot; --generate explicitly permits bounded API calls."""

import argparse
import asyncio
import fcntl
import json
import os
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal

from dotenv import dotenv_values
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ResponseSchema,
    get_model,
)
from inspect_ai.util import json_schema

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.inspect_pilot import resolved_retry_policy, zero_retry_clients
from grader_power.inference import load_access_profiles
from grader_power.pilot_tasks import digest, save
from grader_power.preference_judge import (
    EvidenceRangeJudgment,
    JudgeInput,
    RangeJudgment,
    audit_style,
    parse_judgment,
    parse_range_judgment,
    render_judge_input,
)
from grader_power.preference_tasks import load_bundles, visible_task
from grader_power.spending import SpendingLedger, SpendingLimit, nanodollars, usage_cost

CAP = nanodollars(5)
MAX_TOKENS = 2048
GENERATION = GenerateConfig(
    max_tokens=MAX_TOKENS,
    max_retries=0,
    max_connections=4,
    parallel_tool_calls=False,
    reasoning_effort="low",
    timeout=180,
)
PACKET_FIELDS = ("family", "entry_point", "original_parameters", "code")
PromptVersion = Literal["v1", "v2", "v3"]


def generation_for(version: PromptVersion) -> GenerateConfig:
    """Select each version's wire schema while preserving historical settings."""
    if version == "v1":
        return GENERATION.model_copy(deep=True)
    return GENERATION.model_copy(
        update={
            "reasoning_effort": "medium",
            "response_schema": ResponseSchema(
                name=f"style_judgment_{version}",
                json_schema=json_schema(
                    EvidenceRangeJudgment if version == "v3" else RangeJudgment
                ),
            ),
        }
    )


def freeze(path: Path, value: Any) -> None:
    """Create evidence once, allowing only identical content on resume."""
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError("Frozen evidence changed; choose a fresh output directory")
    else:
        save(path, value)


def packet_for(case: dict[str, Any]) -> JudgeInput:
    """Allowlist neutral fields so expected answers never enter the request."""
    return JudgeInput(**{key: case[key] for key in PACKET_FIELDS})


def build_cases(root: Path, version: PromptVersion = "v1") -> list[dict[str, Any]]:
    """Combine authored synthetic cases, pilot references, and fixed repeats."""
    cases = json.loads(
        (root / "fixtures/style_preferences/judge_calibration.json").read_text()
    )
    for case in cases:
        case["origin"] = "agent_authored_synthetic"
    bundles, datasets = load_bundles(root)
    for bundle in bundles:
        for task in bundle.tasks:
            if task.split != "pilot":
                continue
            visible = visible_task(task, datasets[task.dataset].tasks[task.task_id])
            for style, code in task.implementations.items():
                cases.append(
                    {
                        "id": f"reference-{bundle.family}-{task.task_id.replace('/', '-')}-{style}",
                        "family": bundle.family,
                        "entry_point": task.entry_point,
                        "original_parameters": visible["original_parameters"],
                        "code": code,
                        "expected_label": style + "_only",
                        "notes": "Verified paired pilot reference",
                        "origin": "agent_authored_reference",
                    }
                )
    for family in ("naming", "quotes"):
        originals = [case for case in cases if case["family"] == family]
        synthetic = [
            case
            for case in originals
            if case["origin"] == "agent_authored_synthetic"
            and case["expected_label"] != "invalid"
        ][:2]
        references = [
            case for case in originals if case["origin"] == "agent_authored_reference"
        ][:2]
        if len(synthetic) != 2 or len(references) != 2:
            raise ValueError(
                "Each family requires two valid synthetic cases and two references"
            )
        for original in synthetic + references:
            packet_for(original)
            cases.append(
                {
                    **original,
                    "id": "repeat-" + original["id"],
                    "repeat_of": original["id"],
                }
            )
    if version in ("v2", "v3"):
        fresh = json.loads(
            (root / "fixtures/style_preferences/judge_validation_v2.json").read_text()
        )
        if len(fresh) != 16 or any(
            sum(case["family"] == family for case in fresh) != 8
            for family in ("naming", "quotes")
        ):
            raise ValueError(
                "V2 requires sixteen fixed validation cases, eight per family"
            )
        if {case["code"] for case in fresh} & {case["code"] for case in cases}:
            raise ValueError("V2 validation source must be distinct from calibration")
        cases.extend(
            {
                **case,
                "origin": "independent_validation"
                if version == "v2"
                else "reused_validation",
            }
            for case in fresh
        )
    if version == "v3":
        revision = json.loads(
            (root / "fixtures/style_preferences/judge_evidence_v3.json").read_text()
        )
        overrides = revision["label_overrides"]
        if not set(overrides) <= {case["id"] for case in cases}:
            raise ValueError("Unknown V3 label override")
        for case in cases:
            if case["id"] in overrides:
                if case["expected_label"] != "neither" or overrides[case["id"]] != (
                    "no_naming_evidence"
                    if case["family"] == "naming"
                    else "no_quote_evidence"
                ):
                    raise ValueError(
                        "V3 overrides only split missing style evidence from historical neither"
                    )
                case["previous_expected_label"] = case["expected_label"]
                case["expected_label"] = overrides[case["id"]]
        cases.extend(
            {**case, "origin": "independent_validation"} for case in revision["cases"]
        )
    identities = [case["id"] for case in cases]
    if len(set(identities)) != len(identities) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in identities
    ):
        raise ValueError("Case IDs must be unique safe directory names")
    return cases


def prepare(root: Path, output: Path, version: PromptVersion = "v1") -> dict[str, Any]:
    """Freeze prompts, generation settings, source hashes, packets, and static audits."""
    profile = load_access_profiles(root / "configs/models.yaml")["claude-sonnet"]
    prompt_paths = {
        family: f"prompts/style_judge/{family}-{version}.txt"
        for family in ("naming", "quotes")
    }
    systems = {
        family: (root / path).read_text() for family, path in prompt_paths.items()
    }
    paths = [
        "scripts/pilot_preference_judges.py",
        "src/grader_power/preference_judge.py",
        "src/grader_power/preference_tasks.py",
        "src/grader_power/spending.py",
        "src/grader_power/code_extract.py",
        "evals/inspect_pilot.py",
        "configs/models.yaml",
        "fixtures/style_preferences/judge_calibration.json",
        "fixtures/style_preferences/naming.json",
        "fixtures/style_preferences/quotes.json",
        "manifests/evalplus_source.json",
        *prompt_paths.values(),
    ]
    if version in ("v2", "v3"):
        paths.append("fixtures/style_preferences/judge_validation_v2.json")
    if version == "v3":
        paths.append("fixtures/style_preferences/judge_evidence_v3.json")
    definition = {
        "schema_version": 1,
        "prompt_version": version,
        "purpose": "agent_authored_judge_pilot_not_human_validation",
        "profile": profile.model_dump(mode="json"),
        "generation": generation_for(version).model_dump(
            mode="json", exclude_none=True
        ),
        "systems": systems,
        "cases": build_cases(root, version),
        "cap_nanodollars": CAP,
        "rates_usd_per_million": {"input": 4, "output": 20},
        "input_overhead_bound": 2048,
        "inspect_version": package_version("inspect-ai"),
        "proposed_agreement_threshold": 0.95,
        "source_sha256": {name: digest((root / name).read_bytes()) for name in paths},
    }
    freeze(output / "definition.json", definition)
    for case in definition["cases"]:
        path = output / "cases" / case["id"]
        try:
            packet = packet_for(case)
        except (ValueError, TypeError, KeyError) as error:
            freeze(
                path / "outcome.json",
                {
                    "status": "precheck_invalid",
                    "label": None,
                    "error_type": type(error).__name__,
                },
            )
        else:
            freeze(path / "packet.json", packet.model_dump(mode="json"))
            freeze(
                path / "audit.json",
                audit_style(packet, legacy_labels=version != "v3").model_dump(
                    mode="json"
                ),
            )
    return definition


def reservation(
    system: str, packet: JudgeInput, schema: dict[str, Any] | None = None
) -> int:
    """Bound input by UTF-8 bytes plus overhead, charging full output allowance."""
    input_bound = (
        len((system + render_judge_input(packet) + json.dumps(schema or {})).encode())
        + 2048
    )
    return usage_cost(
        {"input_tokens": input_bound, "output_tokens": MAX_TOKENS},
        input_rate=4,
        output_rate=20,
    )


async def generate_cases(
    output: Path,
    definition: dict[str, Any],
    model: Any,
    ledger: SpendingLedger,
    secret: str,
) -> None:
    """Dispatch at most four requests; never repeat started or uncertain attempts."""
    semaphore = asyncio.Semaphore(4)
    config = GenerateConfig.model_validate(
        definition.get("generation", GENERATION.model_dump(mode="json"))
    )

    async def attempt(case: dict[str, Any]) -> None:
        async with semaphore:
            path = output / "cases" / case["id"]
            if (path / "started.json").exists() or (path / "outcome.json").exists():
                return
            packet = packet_for(case)
            system = definition["systems"][case["family"]]
            try:
                ledger.reserve(
                    {
                        case["id"]: reservation(
                            system,
                            packet,
                            definition.get("generation", {}).get("response_schema"),
                        )
                    }
                )
            except SpendingLimit:
                return
            resolved_retry_policy(model)
            save(
                path / "started.json",
                {
                    "packet_sha256": digest(render_judge_input(packet)),
                    "started_at": datetime.now(UTC).isoformat(),
                },
            )
            outcome: dict[str, Any] = {"status": "request_error", "label": None}
            try:
                response = await asyncio.wait_for(
                    model.generate(
                        [
                            ChatMessageSystem(content=system),
                            ChatMessageUser(content=render_judge_input(packet)),
                        ],
                        tools=[],
                        tool_choice="none",
                        cache=False,
                        config=config,
                    ),
                    timeout=200,
                )
                raw = response.model_dump(mode="json")
                if secret and secret in json.dumps(raw):
                    raise ValueError("Credential echo rejected")
                freeze(path / "raw.json", raw)
                try:
                    usage = raw.get("usage")
                    if not isinstance(usage, dict):
                        raise TypeError("Complete usage is unavailable")
                    amount = usage_cost(usage, input_rate=4, output_rate=20)
                except (KeyError, TypeError, ValueError):
                    outcome["usage_complete"] = False
                else:
                    ledger.settle({case["id"]: amount})
                    outcome["usage_complete"] = True
                outcome["status"] = "incomplete_response"
                if response.choices and response.choices[0].stop_reason == "stop":
                    outcome["status"] = "invalid_response"
                    if definition.get("prompt_version", "v1") in ("v2", "v3"):
                        parsed = parse_range_judgment(
                            response.completion,
                            packet,
                            distinguish_no_evidence=definition["prompt_version"]
                            == "v3",
                        )
                    else:
                        parsed = parse_judgment(response.completion, packet)
                    outcome.update(
                        status=parsed.status,
                        label=parsed.label,
                        judgment=parsed.model_dump(mode="json"),
                    )
            except Exception as error:  # noqa: BLE001 -- redact provider errors, retaining reservations
                outcome["error_type"] = type(error).__name__
            freeze(path / "outcome.json", outcome)

    pending = [
        case
        for case in definition["cases"]
        if not (output / "cases" / case["id"] / "started.json").exists()
        and not (output / "cases" / case["id"] / "outcome.json").exists()
    ]
    if pending:
        # Check one real case before parallel dispatch, catching provider/schema rejection.
        await attempt(pending[0])
        first_outcome = output / "cases" / pending[0]["id"] / "outcome.json"
        if (
            first_outcome.exists()
            and json.loads(first_outcome.read_text())["status"] == "request_error"
        ):
            raise RuntimeError(
                "Initial provider request failed; remaining cases were not attempted"
            )
        await asyncio.gather(*(attempt(case) for case in pending[1:]))


def summarize(output: Path, definition: dict[str, Any]) -> dict[str, Any]:
    """Keep missing/error/precheck outcomes separate from substantive classifications."""
    results = []
    for case in definition["cases"]:
        path = output / "cases" / case["id"]
        outcome = (
            json.loads((path / "outcome.json").read_text())
            if (path / "outcome.json").exists()
            else {
                "status": "started_without_outcome"
                if (path / "started.json").exists()
                else "not_attempted",
                "label": None,
            }
        )
        results.append(
            {
                "id": case["id"],
                "family": case["family"],
                "expected": case["expected_label"],
                "repeat_of": case.get("repeat_of"),
                "origin": case.get("origin", "agent_authored_synthetic"),
                "injection": "injection"
                in (case["id"] + str(case.get("notes", ""))).lower(),
                "status": outcome["status"],
                "observed": outcome.get("label"),
                "correct": outcome["status"] == "classified"
                and outcome.get("label") == case["expected_label"],
            }
        )
    families = {}
    by_id = {row["id"]: row for row in results}
    for family in ("naming", "quotes"):
        rows = [
            row
            for row in results
            if row["family"] == family and row["repeat_of"] is None
        ]
        substantive = [row for row in rows if row["status"] == "classified"]
        eligible = [row for row in rows if row["expected"] != "invalid"]
        confusion: dict[str, dict[str, int]] = {}
        for row in substantive:
            counts = confusion.setdefault(row["expected"], {})
            counts[row["observed"]] = counts.get(row["observed"], 0) + 1
        repeats = [
            row
            for row in results
            if row["family"] == family and row["repeat_of"] is not None
        ]
        paired = [
            row
            for row in repeats
            if row["status"] == "classified"
            and by_id[row["repeat_of"]]["status"] == "classified"
        ]
        injections = [row for row in eligible if row["injection"]]
        correct = sum(row["correct"] for row in eligible)
        all_injections = bool(injections) and all(row["correct"] for row in injections)
        prechecks_ok = all(
            (row["status"] == "precheck_invalid") == (row["expected"] == "invalid")
            for row in rows
        )
        fresh = [row for row in eligible if row["origin"] == "independent_validation"]
        fresh_classified = [row for row in fresh if row["status"] == "classified"]
        families[family] = {
            "assigned_originals": len(rows),
            "statuses": dict(Counter(row["status"] for row in rows)),
            "confusion": confusion,
            "label_counts": dict(Counter(row["observed"] for row in substantive)),
            "no_style_evidence": sum(
                row["observed"] in ("no_naming_evidence", "no_quote_evidence")
                for row in substantive
            ),
            "substantive_agreement": sum(row["correct"] for row in substantive)
            / len(substantive)
            if substantive
            else None,
            "correct_of_all_valid_assignments": correct,
            "valid_assignments": len(eligible),
            "all_valid_agreement": correct / len(eligible) if eligible else None,
            "injection_cases": len(injections),
            "all_injections_correct": all_injections,
            "prechecks_match_expected": prechecks_ok,
            "independent_validation": {
                "assigned": len(fresh),
                "correct": sum(row["correct"] for row in fresh),
                "all_valid_agreement": sum(row["correct"] for row in fresh) / len(fresh)
                if fresh
                else None,
                "substantive_agreement": sum(row["correct"] for row in fresh_classified)
                / len(fresh_classified)
                if fresh_classified
                else None,
                "statuses": dict(Counter(row["status"] for row in fresh)),
            },
            "repeat_stability": {
                "assigned": len(repeats),
                "classified_pairs": len(paired),
                "same_label": sum(
                    row["observed"] == by_id[row["repeat_of"]]["observed"]
                    for row in paired
                ),
            },
            "proposed_gate_passed": bool(eligible)
            and correct / len(eligible) >= 0.95
            and all_injections
            and prechecks_ok,
        }
    return {
        "validation_status": "agent_authored_fixtures_not_human_validation",
        "prompt_version": definition.get("prompt_version", "v1"),
        "threshold": 0.95,
        "families": families,
        "results": results,
    }


def main() -> None:
    """Prepare offline by default; serialize collection with a directory lock."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--prompt-version", choices=("v1", "v2", "v3"), default="v3")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".collection.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        definition = prepare(ROOT, args.output, args.prompt_version)
        scope = digest(json.dumps(definition, sort_keys=True))
        ledger = SpendingLedger(args.output / "spending.json", cap=CAP, scope=scope)
        if args.generate:
            profile = load_access_profiles(ROOT / "configs/models.yaml")[
                "claude-sonnet"
            ]
            endpoint, key = profile.resolve_access(
                {**dotenv_values(ROOT / ".env"), **os.environ}
            )
            with zero_retry_clients():
                model = get_model(
                    profile.model,
                    base_url=endpoint,
                    api_key=key,
                    config=GenerateConfig.model_validate(definition["generation"]),
                    memoize=False,
                )
                freeze(args.output / "retry-policy.json", resolved_retry_policy(model))
                asyncio.run(generate_cases(args.output, definition, model, ledger, key))
        report = {**summarize(args.output, definition), "spending": ledger.snapshot()}
        summary_path = args.output / (
            "summary-" + digest(json.dumps(report, sort_keys=True))[:16] + ".json"
        )
        freeze(summary_path, report)
        temporary = args.output / f".summary-{os.getpid()}.tmp"
        save(temporary, report)
        os.replace(temporary, args.output / "summary.json")
        print(
            json.dumps(
                {
                    "summary": str(args.output / "summary.json"),
                    "families": report["families"],
                    "spent_nanodollars": report["spending"]["spent"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:  # noqa: BLE001 -- CLI must not expose provider/credential details
        print(f"Pilot failed: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None
