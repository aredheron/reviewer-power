"""One bounded asynchronous pipeline for the approved reviewer-observation and instruction-strength study."""

import argparse
import asyncio
import json
import os
import random
import shutil
import signal
import sqlite3
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path
from statistics import median
from typing import Any

from dotenv import dotenv_values
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)

from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals import inspect_reviewer_observation as historical
from evals.inspect_framing import guard
from evals.inspect_mode_a import digest, json_text
from evals.inspect_three_style import scoring_infrastructure_issue
from grader_power import strong_preference_replication as study
from grader_power import style_judge
from grader_power.main_tasks import main_tasks
from grader_power.pacing import WaitClock, append_event, write_json
from grader_power.pilot_tasks import load_pilot_tasks, require_pilot_verification
from grader_power.replication_spending import RequestLedger
from grader_power.replication_transport import (
    AccountRPM,
    RequestContext,
    concurrent_transport,
    request_scope,
)
from grader_power.replication_transport import (
    ModelPacer as SharedModelPacer,
)
from grader_power.scoring import OracleRequest, score_response
from grader_power.spending import nanodollars, usage_cost
from grader_power.strong_preference_lock import (
    collection_lock,
    isolate_collection_process,
)

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = (
    "src/grader_power/strong_preference_replication.py",
    "src/grader_power/strong_preference_lock.py",
    "evals/inspect_strong_preferences.py",
    "scripts/run_strong_preferences.py",
    "planning/strong_preference_protocol.md",
    "tests/test_strong_preferences.py",
)


class ModelPacer(SharedModelPacer):
    """Bound unverified discovery for the newly requested serving tiers."""

    def adjust(self, provider: str, lane: dict[str, Any], now: float) -> None:
        super().adjust(provider, lane, now)
        if provider == "fireworks" and self.profile in ("minimax", "kimi"):
            scale = 2 if self.profile == "minimax" else 1
            for metric, limit in {
                "input": 21600000,
                "uncached": 5400000,
                "output": 216000,
            }.items():
                if (
                    metric not in lane["verified_limits"]
                    and lane["limits"][metric] > limit * scale
                ):
                    lane["limits"][metric] = float(limit * scale)
                    self.event(
                        provider,
                        "unverified_tier_ceiling",
                        metric=metric,
                        ceiling=limit * scale,
                    )


def sources(root: Path) -> dict[str, str]:
    """Bind all reused science, transport and sandbox sources plus this versioned runner."""
    return {
        **historical.sources(root),
        **{n: digest((root / n).read_bytes()) for n in NEW_SOURCES},
    }


def ledger_for(base: Path, manifest: dict[str, Any]) -> RequestLedger:
    """Use one flat ledger; no overlapping block reservations."""
    return RequestLedger(
        base / "spending.sqlite3",
        cap=nanodollars(study.CAP),
        scope=digest(json_text(manifest)),
    )


def path_for(base: Path, r: dict[str, Any]) -> Path:
    """Locate only the new run's immutable outcomes."""
    return base / "requests" / r["assignment_id"]


def measure(base: Path, r: dict[str, Any]) -> dict[str, Any]:
    """Count saved records, separating terminal labels, pending stages and probes."""
    path = path_for(base, r)
    generation = pilot.read_optional(path / "generation/result.json") or {}
    scoring = pilot.read_optional(path / "scoring.json") or {}
    judged = pilot.read_optional(path / "judge/result.json") or {}
    probe = pilot.read_optional(path / "probe.json") or {}
    return {
        k: r[k]
        for k in (
            "assignment_id",
            "historical_assignment_id",
            "profile",
            "study",
            "phase",
            "kind",
            "source_task_id",
            "condition",
            "mode",
            "repetition_index",
        )
    } | {
        "provider": study.PROFILES[r["profile"]]["provider"],
        "reused_generation": r.get("supplement_stage") == "judge",
        "generation_model": study.PROFILES[r["profile"]]["model"],
        "started": (path / "generation/started.json").exists(),
        "attempt_complete": bool(generation),
        "raw_saved": (path / "generation/raw.json").exists(),
        "scored": bool(scoring),
        "full_correct": scoring.get("correctness", {}).get("full_correct"),
        "judge_started": (path / "judge/started.json").exists(),
        "judge_result": bool(judged),
        "label": judged.get("label"),
        "probe_correct": probe.get("fence_only", {}).get("all_correct"),
        "probe_correct_strict": probe.get("strict", {}).get("all_correct"),
        "generation_status": generation.get("status"),
        "judge_status": judged.get("status"),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep model, experiment, cohort and missingness explicit in status reports."""
    result = {}
    for phase in ("pilot", "main"):
        for profile in study.PROFILES:
            for design in ("reviewer_observation",):
                selected = [
                    r
                    for r in rows
                    if (r["phase"], r["profile"], r["study"])
                    == (phase, profile, design)
                ]
                code = [r for r in selected if r["kind"] != "comprehension_probe"]
                result[f"{phase}/{profile}/{design}"] = {
                    "assigned": len(selected),
                    "coding_assigned": len(code),
                    "reused_generations": sum(
                        r.get("reused_generation", False) for r in selected
                    ),
                    **{
                        k: sum(bool(r[k]) for r in selected)
                        for k in ("started", "attempt_complete", "raw_saved", "scored")
                    },
                    "resolved": sum(r["label"] is not None for r in code),
                    "terminal_unresolved": sum(
                        r["judge_result"] and r["label"] is None for r in code
                    ),
                    "inflight_judges": sum(
                        r["judge_started"] and not r["judge_result"] for r in code
                    ),
                    "pending_judges": sum(
                        r["raw_saved"]
                        and not r["judge_started"]
                        and not r["judge_result"]
                        for r in code
                    ),
                    "facts_correct": sum(r["probe_correct"] is True for r in selected),
                }
    return result


def generation_path(base: Path, r: dict[str, Any]) -> Path:
    """Read only this study's generation; no supplemental or replayed outcomes."""
    return path_for(base, r) / "generation/raw.json"


def render_packet(design: str, data: dict[str, Any]) -> str:
    """Render the original blinded binary judge packet."""
    if design != "reviewer_observation":
        raise ValueError("Unknown study")
    return style_judge.render_judge_input(style_judge.JudgeInput.model_validate(data))


def parse_packet(design: str, text: str, data: dict[str, Any]) -> style_judge.Judgment:
    """Preserve the original strict binary judge parser."""
    if design != "reviewer_observation":
        raise ValueError("Unknown study")
    return style_judge.parse_judgment(text, style_judge.JudgeInput.model_validate(data))


class Pipeline:
    """Overlap lightweight API tasks with four verified Docker scoring threads."""

    def __init__(
        self,
        root: Path,
        manifest: dict[str, Any],
        records: list[dict[str, Any]],
        phase: str,
    ) -> None:
        self.root, self.manifest, self.records, self.phase = (
            root,
            manifest,
            records,
            phase,
        )
        self.eligible_models = list(study.PROFILES)
        self.base = study.base(root)
        self.operations = self.base / "operations"
        self.ledger = ledger_for(self.base, manifest)
        self.stop = asyncio.Event()
        self.finishing = asyncio.Event()
        self.reason: str | None = None
        self.tasks_by_id = {
            t.visible.source_task_id: t
            for t in (*main_tasks(root), *load_pilot_tasks(root))
        }
        self.measurements = {r["assignment_id"]: measure(self.base, r) for r in records}
        self.history = (
            [
                json.loads(s)
                for s in (self.operations / "stage-events.jsonl")
                .read_text()
                .splitlines()
            ]
            if (self.operations / "stage-events.jsonl").exists()
            else []
        )
        self.events_seen = {(r["identity"], r["stage"]) for r in self.history}
        self.queues: dict[str, asyncio.Queue[Any]] = {
            k: asyncio.Queue() for k in study.PROFILES
        }
        self.scoring: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.judging: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.account = AccountRPM(self.operations / "pacing")
        self.models: dict[str, Any] = {}
        self.keys: set[str] = set()
        self.active = Counter()
        self.started_at = time.time()
        self.targets = {k: 8 for k in (*study.PROFILES, "judge")}
        self.completions: list[dict[str, Any]] = []
        self.last_cost_forecast_at = 0.0
        self.cost_forecast: dict[str, Any] | None = None

    def check(self) -> None:
        """Reject new paid work immediately after any admission stop."""
        if self.stop.is_set():
            raise RuntimeError("Admission stopped: " + str(self.reason))
        if (self.base / "STOP").exists() or (self.operations / "STOP").exists():
            self.pause("Human STOP")
            raise RuntimeError("Human STOP")

    def pause(self, reason: str) -> None:
        """Persist first stop; accepted requests drain without further paid admission."""
        if not self.stop.is_set():
            self.reason = reason
            write_json(
                self.operations / "controller-stop.json",
                {"reason": reason, "at": time.time(), "pid": os.getpid()},
            )
            append_event(
                self.operations / "events.jsonl",
                {"event": "admission_stopped", "reason": reason},
            )
            self.stop.set()

    def event(
        self, r: dict[str, Any], stage: str, failed: bool, error: str | None = None
    ) -> None:
        """Append each terminal stage once and preserve the five-in-48 safeguard."""
        identity = r["assignment_id"]
        if (identity, stage) not in self.events_seen:
            e = {
                "identity": identity,
                "stage": stage,
                "failed": failed,
                "error_type": error,
                "profile": r["profile"],
                "at": time.time(),
            }
            self.history.append(e)
            self.events_seen.add((identity, stage))
            append_event(self.operations / "stage-events.jsonl", e)
            recovery = (
                pilot.read_optional(self.operations / "recovery-state.json") or {}
            )
            if (
                sum(
                    x["failed"]
                    for x in self.history[recovery.get("error_window_origin", 0) :][
                        -48:
                    ]
                )
                >= 5
            ):
                self.pause("Five failed stages in last 48")
        self.measurements[identity] = measure(self.base, r)

    def initialize_models(self) -> None:
        """Construct verified zero-retry clients once; never serialize credentials."""
        values = {**dotenv_values(self.root / ".env"), **os.environ}
        specs = {k: (v, v["generation"]) for k, v in self.manifest["profiles"].items()}
        specs.update(
            {
                "judge/" + k: (study.JUDGE_PROFILE, v["generation"])
                for k, v in self.manifest["judges"].items()
            }
        )
        for name, (spec, config) in specs.items():
            key = values.get(spec["api_key_env"])
            if not key:
                raise ValueError("Missing credential: " + spec["api_key_env"])
            self.keys.add(key)
            model = get_model(
                spec["model"],
                base_url=spec["base_url"],
                api_key=key,
                config=GenerateConfig.model_validate(config),
                memoize=False,
            )
            original.freeze(
                self.operations / "retry-policies" / (name.replace("/", "-") + ".json"),
                pilot.resolved_retry_policy(model),
            )
            self.models[name] = model

    async def request(
        self, r: dict[str, Any], stage: str, messages: list[Any]
    ) -> dict[str, Any] | None:
        """Reserve, persist intent, dispatch once, retain raw usage and uncertain charges."""
        self.check()
        name = r["profile"] if stage == "generation" else "judge/" + r["study"]
        profile = r["profile"] if stage == "generation" else "sonnet"
        spec = (
            self.manifest["profiles"][profile]
            if stage == "generation"
            else self.manifest["judge_profile"]
        )
        path = path_for(self.base, r) / stage
        if (path / "started.json").exists():
            return pilot.read_optional(path / "raw.json")
        content = [m.model_dump(mode="json") for m in messages]
        size = len(json_text(content).encode())
        if stage == "judge" and size > 32768:
            raise ValueError("Judge input exceeds frozen 32768-byte bound")
        bound = nanodollars(
            ((size + 2048) * spec["input_rate"] * 2 + 4096 * spec["output_rate"]) / 1e6
        )
        identity = r["assignment_id"] + "/" + stage
        self.ledger.reserve({identity: bound})
        original.freeze(
            path / "started.json",
            {
                "at": time.time(),
                "model": spec["model"],
                "messages_sha256": digest(json_text(content)),
                "reserved": bound,
            },
        )
        pacer = ModelPacer(self.operations / "pacing" / profile, profile=profile)
        clock = WaitClock()
        context = RequestContext(
            profile,
            spec["provider"],
            spec["model"].split("/", 1)[1],
            path,
            self.operations,
            pacer,
            self.account,
            clock,
            self.check,
        )
        outcome: dict[str, Any] = {"status": "request_error"}
        raw = None
        start = time.time()
        try:
            with request_scope(context):
                response = await clock.wait_for(
                    self.models[name].generate(
                        messages, tools=[], tool_choice="none", cache=False
                    ),
                    timeout=200,
                )
            candidate = response.model_dump(mode="json")
            if any(k in json_text(candidate) for k in self.keys):
                self.pause("Credential echo rejected; integrity review required")
                raise ValueError("Credential echo rejected")
            raw = candidate
            original.freeze(path / "raw.json", raw)
            try:
                if not isinstance(raw.get("usage"), dict):
                    raise TypeError("Missing complete usage")
                amount = usage_cost(
                    raw["usage"],
                    input_rate=spec["input_rate"],
                    output_rate=spec["output_rate"],
                )
            except (KeyError, TypeError, ValueError):
                self.pause("Complete usage missing; reservation retained")
                raise
            pacer.usage(spec["provider"], raw)
            self.ledger.settle({identity: amount})
            if self.ledger.snapshot()["overrun"]:
                self.pause("Recorded usage exceeded reserved bound")
            outcome["status"] = (
                "completed"
                if response.choices and response.choices[0].stop_reason == "stop"
                else "incomplete"
            )
            self.completions.append(
                {
                    "at": time.time(),
                    "profile": profile,
                    "stage": stage,
                    "output_tokens": raw["usage"]["output_tokens"],
                    "active_seconds": time.time() - start - clock.waiting(),
                }
            )
        except Exception as error:  # noqa: BLE001 -- durable redacted terminal API failure
            outcome["error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                outcome["http_status"] = status
            body = getattr(error, "body", None)
            if isinstance(body, dict):
                detail = body.get("error", body)
                if isinstance(detail, dict):
                    outcome["provider_error"] = {}
                    for field in ("type", "code", "message"):
                        if isinstance(detail.get(field), (str, int)):
                            value = str(detail[field])
                            for key in self.keys:
                                value = value.replace(key, "[REDACTED]")
                            outcome["provider_error"][field] = value[:2000]
        outcome.update(
            at=time.time(),
            seconds=time.time() - start,
            explicit_wait_seconds=clock.waiting(),
        )
        original.freeze(path / "response-status.json", outcome)
        if stage == "generation":
            original.freeze(path / "result.json", outcome)
            self.event(
                r, stage, outcome["status"] != "completed", outcome.get("error_type")
            )
        return raw

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Never dispatch a started request again; retain incomplete/missing responses."""
        path = path_for(self.base, r)
        raw = await self.request(
            r, "generation", [ChatMessageUser(content=r["messages"][0]["content"])]
        )
        if raw is None:
            return
        output = ModelOutput.model_validate(raw)
        if r["kind"] == "comprehension_probe":
            if not (path / "probe.json").exists():
                original.freeze(
                    path / "probe.json",
                    study.score_probe(output.completion, r),
                )
            self.measurements[r["assignment_id"]] = measure(self.base, r)
            return
        await self.scoring.put(r)

    async def scoring_stage(self, r: dict[str, Any]) -> None:
        """Execute only in the pinned Docker oracle; interrupted scoring never replays."""
        path = path_for(self.base, r)
        raw = json.loads(generation_path(self.base, r).read_text())
        output = ModelOutput.model_validate(raw)
        if not (path / "scoring-started.json").exists():
            original.freeze(
                path / "scoring-started.json",
                {"at": time.time(), "raw_sha256": digest(json_text(raw))},
            )
            task = self.tasks_by_id[r["source_task_id"]]
            score = await asyncio.to_thread(
                score_response,
                output.completion,
                task.region,
                oracle=OracleRequest(
                    task.dataset,
                    task.dataset_name,
                    task.visible.source_task_id,
                    self.manifest["oracle_image"],
                ),
            )
            original.freeze(path / "scoring.json", score.to_dict())
        score_data = pilot.read_optional(path / "scoring.json")
        if score_data is None or scoring_infrastructure_issue(score_data):
            self.event(r, "scoring", True, "SandboxInfrastructure")
            self.pause("Sandbox scoring requires review")
            return
        self.event(r, "scoring", False)
        await self.judging.put(r)

    async def judging_stage(self, r: dict[str, Any]) -> None:
        """Use each study's frozen blinded rubric; preserve invalid quotes and abstentions."""
        path = path_for(self.base, r)
        if (path / "judge/started.json").exists() or (
            path / "judge/result.json"
        ).exists():
            return
        raw = json.loads(generation_path(self.base, r).read_text())
        output = ModelOutput.model_validate(raw)
        packet = original.packet_from_output(raw, self.tasks_by_id[r["source_task_id"]])
        binding = {"source_raw_sha256": digest(json_text(raw))}
        if (
            packet is None
            or not output.choices
            or output.choices[0].stop_reason != "stop"
        ):
            original.freeze(
                path / "judge/result.json",
                {
                    **binding,
                    "status": "invalid_or_unparseable",
                    "label": "invalid_or_unparseable",
                },
            )
            self.event(r, "judge", False)
            return
        data = packet.model_dump(mode="json")
        original.freeze(path / "judge/packet.json", data)
        request = render_packet(r["study"], data)
        raw_judge = await self.request(
            r,
            "judge",
            [
                ChatMessageSystem(
                    content=self.manifest["judges"][r["study"]]["system_prompt"]
                ),
                ChatMessageUser(content=request),
            ],
        )
        result: dict[str, Any] = {
            **binding,
            "packet_sha256": digest(json_text(data)),
            "label": None,
            "status": "judge_error",
        }
        try:
            if raw_judge is not None:
                response = ModelOutput.model_validate(raw_judge)
                if response.choices and response.choices[0].stop_reason == "stop":
                    parsed = parse_packet(r["study"], response.completion, data)
                    result.update(
                        label=parsed.label,
                        status=parsed.status,
                        judgment=parsed.model_dump(mode="json"),
                    )
        except (ValueError, TypeError) as error:
            result["error_type"] = type(error).__name__
        original.freeze(path / "judge/result.json", result)
        self.event(
            r,
            "judge",
            result["status"] not in ("classified", "abstain"),
            result.get("error_type"),
        )

    async def worker(
        self, queue: asyncio.Queue[Any], stage: str, index: int, lane: str
    ) -> None:
        """Bound stage concurrency and drain already accepted work after a stop."""
        while True:
            while (
                stage != "scoring"
                and index >= self.targets[lane]
                and not self.stop.is_set()
                and not self.finishing.is_set()
            ):
                await asyncio.sleep(0.25)
            r = await queue.get()
            try:
                if r is None:
                    return
                if self.stop.is_set():
                    continue
                self.check()
                self.active[lane] += 1
                try:
                    if stage == "generation":
                        await self.generation_stage(r)
                    elif stage == "scoring":
                        await self.scoring_stage(r)
                    else:
                        await self.judging_stage(r)
                finally:
                    self.active[lane] -= 1
            except Exception as error:  # noqa: BLE001 -- controller boundary drains all paid tasks
                if r is not None:
                    self.event(r, stage, True, type(error).__name__)
                self.pause(type(error).__name__ + " in " + stage)
            finally:
                queue.task_done()

    def status(self, state: str) -> dict[str, Any]:
        """Report fresh counts and realized completions separately from quota estimates."""
        now = time.time()
        self.completions = [r for r in self.completions if r["at"] > now - 300]
        balance = self.ledger.snapshot()
        if self.phase == "main" and now - self.last_cost_forecast_at >= 300:
            self.cost_forecast = median_forecast(
                self.root, self.manifest, self.records, list(self.measurements.values())
            )
            self.last_cost_forecast_at = now
        output = {
            "budget_forecast": self.cost_forecast,
            "state": state,
            "phase": self.phase,
            "pid": os.getpid(),
            "updated_at": now,
            "counts": counts(list(self.measurements.values())),
            "recorded_usd": balance["spent"] / 1e9,
            "request_outstanding_usd": balance["outstanding"] / 1e9,
            "cap_usd": study.CAP,
            "active": dict(self.active),
            "concurrency_targets": self.targets,
            "queues": {
                **{k: q.qsize() for k, q in self.queues.items()},
                "scoring": self.scoring.qsize(),
                "judging": self.judging.qsize(),
            },
            "free_disk_bytes": shutil.disk_usage(self.root).free,
            "pacing": {
                k: ModelPacer(self.operations / "pacing" / k, profile=k).utilization()
                for k in tuple(dict.fromkeys((*study.PROFILES, "sonnet")))
            },
            "last_minute": {
                k: {
                    "completions": sum(
                        r["profile"] == k and r["at"] > now - 60
                        for r in self.completions
                    ),
                    "generation_completions": sum(
                        r["profile"] == k
                        and r["stage"] == "generation"
                        and r["at"] > now - 60
                        for r in self.completions
                    ),
                    "output_tokens": sum(
                        r["output_tokens"]
                        for r in self.completions
                        if r["profile"] == k and r["at"] > now - 60
                    ),
                }
                for k in tuple(dict.fromkeys((*study.PROFILES, "sonnet")))
            },
        }
        write_json(self.operations / "status.json", output)
        append_event(self.operations / "throughput-events.jsonl", output)
        return output

    def tune(self, report: dict[str, Any]) -> None:
        """Grow judge capacity despite upstream backpressure; bound generation demand."""
        for lane in self.targets:
            profile = "sonnet" if lane == "judge" else lane
            p = report["pacing"][profile].get(
                (
                    study.JUDGE_PROFILE
                    if profile == "sonnet"
                    else study.PROFILES[profile]
                )["provider"],
                {},
            )
            demand = (
                self.judging.qsize() if lane == "judge" else self.queues[lane].qsize()
            )
            utilization = max(p.get("utilization", {}).values(), default=0)
            if (
                demand
                and not p.get("queued")
                and utilization < 0.75
                and (lane == "judge" or self.scoring.qsize() < 48)
            ):
                self.targets[lane] = min(32, self.targets[lane] + 4)

    async def monitor(self) -> None:
        """Verify bindings and tune lightweight network slots without expanding Docker."""
        ticks = 0
        while True:
            await asyncio.sleep(10)
            try:
                self.check()
                guard(self.base)
                if ticks % 3 == 0:
                    load(self.root)
                report = self.status("draining" if self.stop.is_set() else "running")
                if report["free_disk_bytes"] < 3 * 1024**3:
                    self.pause("Disk headroom below 3 GiB")
                if ticks % 3 == 2 and not self.stop.is_set():
                    self.tune(report)
                ticks += 1
            except Exception as error:  # noqa: BLE001 -- cease admission, leave paid work draining
                self.pause(type(error).__name__ + " during monitor")
                self.status("draining")

    async def run(self) -> None:
        """Run all phase queues, then wait for paid tasks and Docker work before return."""
        self.initialize_models()
        for r in self.records:
            if r["phase"] == self.phase and (
                self.phase != "main" or r["profile"] in self.eligible_models
            ):
                self.queues[r["profile"]].put_nowait(r)
        workers = []
        for profile, q in self.queues.items():
            workers.extend(
                asyncio.create_task(self.worker(q, "generation", i, profile))
                for i in range(32)
            )
        workers.extend(
            asyncio.create_task(self.worker(self.scoring, "scoring", i, "scoring"))
            for i in range(4)
        )
        workers.extend(
            asyncio.create_task(self.worker(self.judging, "judge", i, "judge"))
            for i in range(32)
        )
        watcher = asyncio.create_task(self.monitor())
        self.status("running")
        try:
            await asyncio.gather(*(q.join() for q in self.queues.values()))
            await self.scoring.join()
            await self.judging.join()
        finally:
            self.finishing.set()
            for q in self.queues.values():
                for _ in range(32):
                    q.put_nowait(None)
            for _ in range(4):
                await self.scoring.put(None)
            for _ in range(32):
                await self.judging.put(None)
            await asyncio.gather(*workers)
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        self.status("paused" if self.stop.is_set() else self.phase + "_complete")
        if self.stop.is_set():
            raise RuntimeError(str(self.reason))


def analyze(root: Path) -> dict[str, Any]:
    """Reconstruct every main assignment from durable records and verify label bindings."""
    manifest, records = load(root, retained=True)
    base = study.base(root)
    rows = [measure(base, r) for r in records if r["phase"] == "main"]
    for r in records:
        path = path_for(base, r)
        judged = pilot.read_optional(path / "judge/result.json")
        if judged and judged.get("source_raw_sha256") != digest(
            json_text(json.loads(generation_path(base, r).read_text()))
        ):
            raise ValueError("Judgment source binding changed")
        if judged and judged.get("status") in ("classified", "abstain"):
            packet = json.loads((path / "judge/packet.json").read_text())
            raw = ModelOutput.model_validate(
                json.loads((path / "judge/raw.json").read_text())
            )
            parsed = parse_packet(r["study"], raw.completion, packet)
            if (
                parsed.label != judged["label"]
                or digest(json_text(packet)) != judged["packet_sha256"]
            ):
                raise ValueError("Raw and parsed judge labels differ")
    result = study.analyze(rows)
    result.update(
        manifest_sha256=digest(json_text(manifest)),
        recorded_usd=ledger_for(base, manifest).snapshot()["spent"] / 1e9,
    )
    write_json(base / "analysis-v1/measured-rows.json", rows)
    write_json(base / "analysis-v1/analysis.json", result)
    return result


def initialize_pacing(root: Path, base: Path) -> None:
    """Seed matching prior lanes once, preserving cap provenance and cooldowns."""
    now = time.time()
    audit: dict[str, Any] = {}
    for profile in tuple(dict.fromkeys((*study.PROFILES, "sonnet"))):
        target = base / "operations/pacing" / profile / "rate-state.json"
        if target.exists():
            continue
        candidates = [
            root / study.PARENT / "operations/pacing" / profile / "rate-state.json",
            root
            / "runs/replication/simple-models-v1/operations/pacing"
            / profile
            / "rate-state.json",
            root
            / "runs/replication/compact-models-v1/operations/pacing"
            / profile
            / "rate-state.json",
        ]
        if profile == "gpt-oss-120b":
            candidates = [
                root
                / "runs/three-style/three-style-v1/operations/pacing/rate-state.json"
            ]
        if profile == "sonnet":
            candidates.append(
                root
                / "runs/recovery/judge-recovery-v1/operations/pacing/sonnet/rate-state.json"
            )
        candidates.append(
            root
            / "runs/style-preferences/syntax-preferences-v1/operations/pacing"
            / profile
            / "rate-state.json"
        )
        available = [p for p in candidates if p.is_file()]
        if not available:
            continue
        source = max(available, key=lambda p: p.stat().st_mtime)
        data = json.loads(source.read_text())
        provider = (
            study.JUDGE_PROFILE if profile == "sonnet" else study.PROFILES[profile]
        )["provider"]
        if provider not in data:
            raise ValueError("Prior pacing lane lacks matching provider")
        lane = data[provider]
        # Old tickets cannot own this controller's new FIFO. Retain recent usage,
        # historic throttle state, numeric evidence, and all cooldown timestamps.
        lane["queue"] = []
        lane["window"] = [r for r in lane.get("window", []) if r["at"] > now - 60]
        write_json(target, {provider: lane})
        audit[profile] = {
            "source": str(source.relative_to(root)),
            "source_sha256": digest(source.read_bytes()),
            "provider": provider,
            "source_modified_at": source.stat().st_mtime,
            "retained_numeric_limits": lane.get("verified_limits", {}),
            "local_allowances_not_current_quota": lane["limits"],
            "cooldown_until": lane["cooldown_until"],
        }
    account_target = base / "operations/pacing/account-rpm.json"
    if not account_target.exists():
        candidates = [
            root / relative / "operations/pacing/account-rpm.json"
            for relative in (
                study.PARENT,
                "runs/style-preferences/syntax-preferences-v1",
                "runs/replication/simple-models-v1",
                "runs/replication/compact-models-v1",
                "runs/recovery/audited-outcomes-v1",
                "runs/recovery/judge-recovery-v1",
            )
        ]
        available = [p for p in candidates if p.is_file()]
        if available:
            source = max(available, key=lambda p: p.stat().st_mtime)
            account = json.loads(source.read_text())
            cap = account.get("cap", 6000)
            if type(cap) not in (int, float) or not 0 < cap <= 6000:
                raise ValueError("Prior shared account RPM cap is invalid")
            account["window"] = [
                r for r in account.get("window", []) if r["at"] > now - 60
            ]
            write_json(account_target, account)
            audit["shared_account_rpm"] = {
                "source": str(source.relative_to(root)),
                "source_sha256": digest(source.read_bytes()),
                "cap": cap,
                "reported_at": account.get("reported_at"),
            }
    if audit:
        original.freeze(
            base / "operations/pacing-initialization.json", {"at": now, "lanes": audit}
        )


def parent_readiness(root: Path) -> list[dict[str, Any]]:
    """Reuse only the parent's approved GLM pilot cells, including its frozen qualification."""
    _, records = historical.load(root, retained=True)
    return [
        r
        for r in historical.qualified_pilot_rows(root / study.PARENT, records)
        if r["profile"] == "glm" and r["condition"]["arm"] in study.ARMS
    ]


def empirical_floors(root: Path) -> dict[str, list[float]]:
    """Reprice saved strong-condition token usage without calling any model."""
    parent = root / study.PARENT
    records = json.loads((parent / "assignments.json").read_text())
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for row in records:
        if row["phase"] != "main" or row["condition"]["arm"] not in study.ARMS:
            continue
        for stage in ("generation", "judge"):
            p = parent / "requests" / row["assignment_id"] / stage / "raw.json"
            if p.exists():
                usage = json.loads(p.read_text()).get("usage")
                if isinstance(usage, dict):
                    pools.setdefault(row["profile"], {}).setdefault(stage, []).append(
                        usage
                    )
    result = {}
    for profile, spec in study.PROFILES.items():
        source_profiles = ["glm"] if profile == "glm" else list(pools)
        means = []
        for stage in ("generation", "judge"):
            rates = spec if stage == "generation" else study.JUDGE_PROFILE
            means.append(
                max(
                    sum(
                        usage_cost(
                            u,
                            input_rate=rates["input_rate"],
                            output_rate=rates["output_rate"],
                        )
                        / 1e9
                        for u in pools[p][stage]
                    )
                    / len(pools[p][stage])
                    for p in source_profiles
                )
            )
        result[profile] = means
    return result


def prepare(root: Path, reports: list[Path]) -> None:
    """Freeze exact prompts, inherited readiness and a cap within the unused prior ceiling."""
    base = study.base(root)
    if (base / "manifest.json").exists():
        raise ValueError("Run already frozen")
    if len(reports) < 2:
        raise ValueError("Offline and Docker verification reports required")
    for path in reports:
        cases = ET.parse(path).getroot().findall(".//testcase")
        if not cases or any(
            c.find(t) is not None
            for c in cases
            for t in ("failure", "error", "skipped")
        ):
            raise ValueError("Verification report failed")
    original.verify_integration(reports[-1])
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    parent = root / study.PARENT
    complete = json.loads(
        (parent / "operations/completion/completion-audit.json").read_text()
    )
    if (
        complete["state"] != "complete"
        or complete["ledger"]["recorded_usd"] != study.PARENT_RECORDED
        or complete["ledger"]["outstanding_usd"] != 0
    ):
        raise ValueError("Prior whole-study commitment changed")
    with closing(
        sqlite3.connect(f"file:{parent}/spending.sqlite3?mode=ro", uri=True)
    ) as db:
        amounts = list(db.execute("SELECT reserved, settled FROM entries"))
    if any(s is None for _, s in amounts) or sum(s for _, s in amounts) != nanodollars(
        study.PARENT_RECORDED
    ):
        raise ValueError("Parent accounting no longer matches reserved headroom")
    if nanodollars(study.PARENT_RECORDED) + nanodollars(study.CAP) != nanodollars(
        study.COMBINED_CAP
    ):
        raise ValueError("Combined spending ceiling changed")
    records, retained = study.schedule(root)
    reused = parent_readiness(root)
    original.freeze(base / "retained-glm-readiness.json", reused)
    old = json.loads((root / "runs/main/main-32-v1/manifest.json").read_text())
    judge = {
        **old["judge"],
        "generation": {**old["judge"]["generation"], "max_connections": 64},
    }
    if (
        judge["model"] != study.JUDGE_PROFILE["model"]
        or judge["generation"]["max_tokens"] != 4096
    ):
        raise ValueError("Original blinded judge changed")
    original.freeze(base / "assignments.json", records)
    manifest = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "combined_cap_usd": study.COMBINED_CAP,
        "parent_recorded_usd": study.PARENT_RECORDED,
        "authorization": study.AUTHORIZATION,
        "main_assignments": 3840,
        "pilot_assignments": 128,
        "reused_glm_readiness": 64,
        "assignments_sha256": digest((base / "assignments.json").read_bytes()),
        "readiness_sha256": digest((base / "retained-glm-readiness.json").read_bytes()),
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "judge_profile": study.JUDGE_PROFILE,
        "judges": {"reviewer_observation": judge},
        "oracle_image": image,
        "code_sha256": sources(root),
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
        "historical_bindings": retained,
        "cost_floor_per_response": empirical_floors(root),
        "prospective_gates": {
            "per_model_saved": 64,
            "coding_correct_min": 29,
            "coding_resolved_min": 31,
            "factual_min_each_arm": "7/8",
            "factual_parser": "strict plus one complete JSON fence; no trailing-text repair",
            "behavior_used_to_gate": False,
            "admission": "per model; failed new models held while qualified GLM may continue",
        },
        "prospective_analysis": {
            "main_assignments": 3840,
            "repetitions": study.REPETITIONS,
            "holm_family_size": 15,
            "contrasts_per_model": study.COMPARISONS,
            "bootstrap_draws": 10000,
            "task_clusters": 32,
            "withhold_primary_family_if_unresolved": True,
            "parent_pooling": False,
        },
        "operational_limits": {
            "network_workers_per_model": 32,
            "initial_active_per_lane": 8,
            "judge_workers": 32,
            "scoring_workers": 4,
            "stage_queue_size": 64,
        },
    }
    original.freeze(base / "manifest.json", manifest)
    for name, sha in manifest["code_sha256"].items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed during freeze")
        p = base / "source-snapshot" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("xb") as stream:
            stream.write(payload)
    ledger_for(base, manifest).snapshot()
    write_json(
        base / "operations/recovery-state.json",
        {
            "retry_blocked": False,
            "failure_streaks": {},
            "ladder_position": 0,
            "next_check_due_at": None,
        },
    )
    original.freeze(
        base / "preflight-forecast.json", median_forecast(root, manifest, records)
    )


def load(
    root: Path, *, retained: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fail closed on scope/source/outcome changes; permit hash-bound operational repairs only."""
    base = study.base(root)
    m = json.loads((base / "manifest.json").read_text())
    current = sources(root)
    if (
        m["run"] != study.RUN
        or m["cap_usd"] != study.CAP
        or m["combined_cap_usd"] != study.COMBINED_CAP
        or m["parent_recorded_usd"] != study.PARENT_RECORDED
    ):
        raise ValueError("Frozen spending scope changed")
    if m["code_sha256"] != current:
        recovery = json.loads((base / "operations/recovery-state.json").read_text())
        p = base / recovery["source_amendment"]
        if digest(p.read_bytes()) != recovery["source_amendment_sha256"]:
            raise ValueError("Amendment hash changed")
        amendment = json.loads(p.read_text())
        allowed = {
            "evals/inspect_strong_preferences.py",
            "src/grader_power/strong_preference_lock.py",
            "tests/test_strong_preferences.py",
        }
        changed = {n for n in current if current[n] != m["code_sha256"].get(n)}
        if (
            set(current) != set(m["code_sha256"])
            or not changed <= allowed
            or amendment["manifest_sha256"] != digest(json_text(m))
            or amendment["code_sha256"] != current
        ):
            raise ValueError("Amendment cannot change frozen science/shared sources")
        for name in changed:
            if (
                digest((base / "source-snapshot" / name).read_bytes())
                != m["code_sha256"][name]
            ):
                raise ValueError("Original source snapshot changed")
        for name, sha in amendment["audit_bindings"].items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Repair evidence changed")
        for name, sha in amendment["verification_reports"].items():
            p = base / name
            cases = ET.parse(p).getroot().findall(".//testcase")
            if (
                digest(p.read_bytes()) != sha
                or not cases
                or any(
                    c.find(t) is not None
                    for c in cases
                    for t in ("failure", "error", "skipped")
                )
            ):
                raise ValueError("Repair checks failed")
        if retained:
            for name, sha in amendment["retained_artifacts"].items():
                if digest((base / name).read_bytes()) != sha:
                    raise ValueError("Retained outcome changed")
    for name, key in (
        ("assignments.json", "assignments_sha256"),
        ("retained-glm-readiness.json", "readiness_sha256"),
    ):
        if digest((base / name).read_bytes()) != m[key]:
            raise ValueError("Frozen assignments/readiness changed")
    for name, sha in {
        **m["verification_reports"],
        **(m["historical_bindings"] if retained else {}),
    }.items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Bound report/reference changed: " + name)
    return m, json.loads((base / "assignments.json").read_text())


def readiness_rows(base: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep reused GLM readiness separate from paid new-model pilot observations."""
    return json.loads((base / "retained-glm-readiness.json").read_text()) + [
        measure(base, r) for r in records if r["phase"] == "pilot"
    ]


def median_forecast(
    root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    measurements: list[dict[str, Any]] | None = None,
    *,
    draws: int = 1000,
) -> dict[str, Any]:
    """Predict new and combined commitments, retaining unknown reserves and empirical cost floors."""
    if draws < 100:
        raise ValueError("At least 100 forecast draws required")
    base = study.base(root)
    balance = ledger_for(base, manifest).snapshot()
    by_id = {r["assignment_id"]: r for r in records}
    samples: dict[tuple[str, str, str], list[float]] = {}
    with closing(sqlite3.connect(base / "spending.sqlite3")) as db:
        for identity, settled in db.execute(
            "SELECT identity,settled FROM entries WHERE settled IS NOT NULL"
        ):
            aid, stage = identity.rsplit("/", 1)
            r = by_id.get(aid)
            if r and r["kind"] == "core" and stage in ("generation", "judge"):
                samples.setdefault((r["profile"], stage, r["phase"]), []).append(
                    settled / 1e9
                )
    measured = (
        measurements
        if measurements is not None
        else [measure(base, r) for r in records]
    )
    pending = Counter()
    for r in measured:
        if not r["started"]:
            pending[r["profile"], "generation"] += 1
        if (
            r["kind"] == "core"
            and not r["judge_started"]
            and not r["judge_result"]
            and (not r["started"] or r["raw_saved"] or not r["attempt_complete"])
        ):
            pending[r["profile"], "judge"] += 1
    simulations = [balance["committed"] / 1e9] * draws
    randomizer = random.Random(study.SEED)
    models = {}
    for profile in study.PROFILES:
        models[profile] = {}
        for stage in ("generation", "judge"):
            main = samples.get((profile, stage, "main"), [])
            pilot_values = samples.get((profile, stage, "pilot"), [])
            floor = manifest["cost_floor_per_response"][profile][
                0 if stage == "generation" else 1
            ]
            values = main if len(main) >= 32 else pilot_values or [floor]
            for i in range(draws):
                simulations[i] += pending[profile, stage] * max(
                    0 if len(main) >= 32 else floor,
                    sum(randomizer.choices(values, k=len(values))) / len(values),
                )
            models[profile][stage] = {
                "sample": "main"
                if len(main) >= 32
                else "pilot"
                if pilot_values
                else "historical token repricing",
                "sample_size": len(main) if len(main) >= 32 else len(pilot_values),
                "mean_request_usd": sum(values) / len(values),
                "cost_floor_usd": floor,
                "future_requests": pending[profile, stage],
            }
    simulations.sort()
    estimate = median(simulations)
    return {
        "manifest_sha256": digest(json_text(manifest)),
        "median_incremental_usd": estimate,
        "median_total_usd": study.PARENT_RECORDED + estimate,
        "incremental_range10_90_usd": [
            simulations[draws // 10],
            simulations[9 * draws // 10],
        ],
        "recorded_usd": balance["spent"] / 1e9,
        "outstanding_usd": balance["outstanding"] / 1e9,
        "parent_recorded_usd": study.PARENT_RECORDED,
        "hard_cap_usd": study.COMBINED_CAP,
        "incremental_cap_usd": study.CAP,
        "budget_notification_required": study.PARENT_RECORDED + estimate >= 350,
        "models": models,
        "at": time.time(),
        "draws": draws,
        "seed": study.SEED,
    }


def collect(root: Path, phase: str) -> None:
    """Collect qualified models only, under the common lock and remaining combined budget."""
    manifest, records = load(root, retained=True)
    base = study.base(root)
    guard(base)
    initialize_pacing(root, base)
    eligible = list(study.PROFILES)
    if phase == "main":
        gate = study.pilot_gate(readiness_rows(base, records))
        eligible = [g["profile"] for g in gate["groups"] if g["passed"]]
        if not eligible:
            raise ValueError("No model passed readiness")
        audit = median_forecast(root, manifest, records)
        if audit["median_total_usd"] > study.COMBINED_CAP:
            raise ValueError("Median forecast exceeds combined ceiling")
        tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
        original.freeze(
            base / "operations/admissions" / f"{tag}.json",
            {"gate": gate, "eligible_models": eligible, "forecast": audit},
        )
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}-{phase}"
    launch = base / "operations/launches" / tag
    original.freeze(
        launch / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "phase": phase,
            "eligible_models": eligible,
            "manifest_sha256": digest(json_text(manifest)),
            "authorization": study.AUTHORIZATION,
        },
    )
    owner = root / "runs/extension-pilot/coordination/owner.json"
    if owner.exists():
        original.freeze(launch / "previous-owner.json", json.loads(owner.read_text()))
    with collection_lock(root, owner=study.RUN, run_id=study.RUN):
        require_pilot_verification(root, manifest["oracle_image"])
        image = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                manifest["oracle_image"],
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        containers = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        if (
            image.stdout.strip() != manifest["oracle_image"]
            or containers.stdout.strip()
        ):
            raise ValueError("Pinned sandbox not clean/verified")
        original.freeze(
            launch / "preflight.json",
            {"image": image.stdout.strip(), "oracle_containers": [], "at": time.time()},
        )
        pipeline = Pipeline(root, manifest, records, phase)
        pipeline.eligible_models = eligible
        with (
            pilot.zero_retry_clients(),
            concurrent_transport(),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):

            async def execute() -> None:
                loop = asyncio.get_running_loop()
                loop.set_default_executor(executor)
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(
                        sig, pipeline.pause, "Operator signal; drain accepted work"
                    )
                await pipeline.run()

            try:
                asyncio.run(execute())
            finally:
                result = subprocess.run(
                    [
                        "docker",
                        "ps",
                        "-aq",
                        "--filter",
                        "label=grader-power.oracle=true",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=20,
                )
                original.freeze(
                    launch / "cleanup.json",
                    {"oracle_containers": result.stdout.split(), "at": time.time()},
                )
                if result.stdout.strip():
                    raise RuntimeError("Sandbox cleanup not verified")
        if phase == "pilot":
            original.freeze(
                base / "pilot-gate.json",
                study.pilot_gate(readiness_rows(base, records)),
            )
        else:
            analyze(root)
            pipeline.status(
                "complete"
                if len(eligible) == len(study.PROFILES)
                else "qualified_main_complete_others_held"
            )


def main() -> None:
    """Prepare without inference; run new-model pilots and admitted additional main repetitions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "collect", "run", "status", "analyze")
    )
    parser.add_argument("--phase", choices=("pilot", "main"), default="pilot")
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(ROOT, args.report)
    elif args.command in ("collect", "run"):
        isolate_collection_process()
        if args.command == "run":
            collect(ROOT, "pilot")
            collect(ROOT, "main")
        else:
            collect(ROOT, args.phase)
    elif args.command == "analyze":
        analyze(ROOT)
    else:
        manifest, records = load(ROOT, retained=True)
        print(
            json.dumps(
                {
                    "counts": counts([measure(study.base(ROOT), r) for r in records]),
                    "ledger": ledger_for(study.base(ROOT), manifest).snapshot(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
