"""Charge blinded judge calibration to the same study ledger before coding pilots."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)

from evals import inspect_extension_pilot as extension
from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals.inspect_framing import guard, stage_event
from evals.inspect_mode_a import digest, json_text
from grader_power.adaptive_pacing import AdaptivePacer
from grader_power.pacing import WaitClock, paced_http
from grader_power.spending import SpendingLedger, nanodollars, usage_cost
from grader_power.three_style_judge import (
    JudgeInput,
    parse_judgment,
    render_judge_input,
)

BOUND = nanodollars(((32768 + 2048) * 4 + 4096 * 10) / 1e6)
SUPPLEMENT_GATE = "at-least-23-valid-zero-valid-incorrect-v1"


def verify_supplement(root: Path, base: Path, spec: dict[str, Any]) -> Path:
    """Verify the approved fresh cohort and all retained original outcome bindings."""
    if spec["gate"] != SUPPLEMENT_GATE:
        raise ValueError("Unapproved supplemental calibration gate")
    cohort = spec.get("cohort", "calibration-supplement-v1")
    if cohort not in ("calibration-supplement-v1", "calibration-rubric-v2"):
        raise ValueError("Unapproved calibration cohort")
    retained = (
        ("calibration",)
        if cohort == "calibration-supplement-v1"
        else ("calibration", "calibration-supplement-v1")
    )
    fixture = root / spec["fixtures"]
    if (
        fixture.resolve()
        != (root / "fixtures/three_style" / (cohort + ".json")).resolve()
        or digest(fixture.read_bytes()) != spec["fixtures_sha256"]
    ):
        raise ValueError("Supplemental fixture binding changed")
    for name, expected in spec["original_artifacts"].items():
        path = base / name
        if not any(
            path.resolve().is_relative_to((base / old).resolve()) for old in retained
        ):
            raise ValueError("Original calibration artifact outside cohort")
        if digest(path.read_bytes()) != expected:
            raise ValueError("Original calibration outcome changed")
    original_paths = {
        str(path.relative_to(base))
        for old in retained
        for path in (base / old).rglob("*")
        if path.is_file()
    }
    if set(spec["original_artifacts"]) != original_paths:
        raise ValueError("Original calibration evidence is not fully bound")
    if cohort == "calibration-rubric-v2":
        for previous in retained:
            path = root / "fixtures/three_style" / (previous + ".json")
            if digest(path.read_bytes()) != spec["retired_fixtures_sha256"][previous]:
                raise ValueError("Retired calibration fixture changed")
    old = [
        row
        for previous in retained
        for row in json.loads(
            (root / "fixtures/three_style" / (previous + ".json")).read_text()
        )
    ]
    new = json.loads(fixture.read_text())
    if (
        len(new) != 24
        or len({r["id"] for r in new}) != 24
        or len({r["code"] for r in new}) != 24
        or {r["id"] for r in old} & {r["id"] for r in new}
        or {r["code"] for r in old} & {r["code"] for r in new}
        or any(
            not r["id"].startswith(
                "rubric2-" if cohort == "calibration-rubric-v2" else "supplement-"
            )
            or "/" in r["id"]
            for r in new
        )
    ):
        raise ValueError("Supplement must contain 24 fresh unique examples")
    return fixture


def gate_passes(results: list[dict[str, Any]], *, supplemental: bool) -> bool:
    """Keep the original gate; apply only the explicitly approved supplemental gate."""
    if supplemental:
        valid = [r for r in results if r["observed"] is not None]
        return (
            len(results) == 24 and len(valid) >= 23 and all(r["correct"] for r in valid)
        )
    return sum(r["correct"] for r in results) >= 23 and all(
        r["correct"] for r in results if r["expected"] in ("map_filter_only", "mixed")
    )


def calibrate(
    root: Path,
    base: Path,
    judge: dict[str, Any],
    whole: SpendingLedger,
    *,
    supplement: dict[str, Any] | None = None,
) -> None:
    """Never resend a started calibration request; preserve malformed/ambiguous results."""
    fixture_path = (
        root / "fixtures/three_style/calibration.json"
        if supplement is None
        else verify_supplement(root, base, supplement)
    )
    fixtures = json.loads(fixture_path.read_text())
    cohort_name = (
        "calibration"
        if supplement is None
        else supplement.get("cohort", "calibration-supplement-v1")
    )
    entry = "judge-" + cohort_name
    cohort = base / cohort_name
    ledger = SpendingLedger(
        cohort / "spending.json",
        cap=BOUND * len(fixtures),
        scope=digest(json_text({"fixtures": fixtures, "judge": judge})),
    )
    whole.reserve({entry: BOUND * len(fixtures)})
    profile = extension.access_profiles(root)[judge["profile"]]
    endpoint, key = profile.resolve_access(
        {**dotenv_values(root / ".env"), **os.environ}
    )
    pacer = AdaptivePacer(base / "operations/pacing")

    async def run() -> None:
        with pilot.zero_retry_clients():
            model = get_model(
                profile.model,
                base_url=endpoint,
                api_key=key,
                config=GenerateConfig.model_validate(judge["generation"]),
                memoize=False,
            )
            original.freeze(
                cohort / "retry-policy.json", pilot.resolved_retry_policy(model)
            )
            for fixture in fixtures:
                guard(base)
                identity = fixture["id"]
                path = cohort / identity
                packet = JudgeInput(
                    **{
                        k: fixture[k]
                        for k in (
                            "entry_point",
                            "transformation",
                            "input_contract",
                            "code",
                        )
                    }
                )
                if (
                    len((judge["system_prompt"] + render_judge_input(packet)).encode())
                    > 32768
                ):
                    raise ValueError("Calibration prompt exceeds reserved bound")
                ledger.reserve({identity: BOUND})
                if (path / "started.json").exists():
                    continue
                original.freeze(path / "packet.json", packet.model_dump())
                original.freeze(
                    path / "started.json",
                    {"packet_sha256": digest(json_text(packet.model_dump()))},
                )
                outcome: dict[str, Any] = {"status": "judge_error", "label": None}
                clock = WaitClock()
                try:
                    with paced_http(
                        pacer, path / "transport", clock, lambda: guard(base)
                    ):
                        response = await clock.wait_for(
                            model.generate(
                                [
                                    ChatMessageSystem(content=judge["system_prompt"]),
                                    ChatMessageUser(content=render_judge_input(packet)),
                                ],
                                tools=[],
                                tool_choice="none",
                                cache=False,
                            ),
                            timeout=200,
                        )
                    raw = response.model_dump(mode="json")
                    if key in json_text(raw):
                        raise ValueError("Credential echo rejected")
                    original.freeze(path / "raw.json", raw)
                    pacer.usage("anthropic", raw)
                    try:
                        amount = usage_cost(raw["usage"], input_rate=2, output_rate=10)
                    except (KeyError, ValueError, TypeError):
                        pass  # Retain full reservation for incomplete usage.
                    else:
                        ledger.settle({identity: amount})
                    if response.choices and response.choices[0].stop_reason == "stop":
                        parsed = parse_judgment(response.completion, packet)
                        outcome = {
                            "status": parsed.status,
                            "label": parsed.label,
                            "judgment": parsed.model_dump(mode="json"),
                        }
                    else:
                        outcome["status"] = "incomplete_judgment"
                except Exception as error:  # noqa: BLE001 -- preserve failures without provider secrets
                    outcome["error_type"] = type(error).__name__
                original.freeze(path / "result.json", outcome)
                stage_event(
                    base,
                    cohort_name + "/" + identity,
                    "judge",
                    outcome["status"] not in ("classified", "abstain"),
                    outcome.get("error_type"),
                )
                if ledger.snapshot()["overrun"]:
                    raise RuntimeError("Calibration reservation exceeded")

    asyncio.run(run())
    balance = ledger.snapshot()
    whole.settle({entry: balance["committed"]})
    results = []
    for fixture in fixtures:
        path = cohort / fixture["id"]
        result = pilot.read_optional(path / "result.json") or {}
        results.append(
            {
                "id": fixture["id"],
                "expected": fixture["expected_label"],
                "observed": result.get("label"),
                "correct": result.get("label") == fixture["expected_label"],
            }
        )
    passed = gate_passes(results, supplemental=supplement is not None)
    original.freeze(
        cohort / "gate.json",
        {
            "passed": passed,
            "results": results,
            "correct": sum(r["correct"] for r in results),
            "assigned": len(fixtures),
        },
    )
    if not passed:
        raise RuntimeError(
            "Prospective judge calibration failed; coding pilot remains unstarted"
        )
