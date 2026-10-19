"""Additive same-limit recovery, serialized after matched syntax collection."""

import argparse
import asyncio
import json
import os
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
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput

from evals import inspect_main as original
from evals import inspect_matched_review as parent
from evals import inspect_pilot as pilot
from evals.inspect_framing import guard
from evals.inspect_mode_a import digest, json_text
from grader_power import matched_recovery as study
from grader_power import matched_review as original_study
from grader_power import matched_review_analysis as analysis
from grader_power.main_tasks import main_tasks
from grader_power.matched_recovery_lock import (
    collection_lock,
    isolate_collection_process,
    process_group_members,
)
from grader_power.pacing import append_event, write_json
from grader_power.pilot_tasks import load_pilot_tasks, require_pilot_verification
from grader_power.replication_spending import RequestLedger
from grader_power.replication_transport import (
    AccountRPM,
    ModelPacer,
    concurrent_transport,
)
from grader_power.spending import nanodollars

ROOT = Path(__file__).resolve().parents[1]


def ledger_for(base: Path, manifest: dict[str, Any]) -> RequestLedger:
    return RequestLedger(
        base / "spending.sqlite3",
        cap=nanodollars(study.CAP),
        scope=digest(json_text(manifest)),
    )


def path_for(base: Path, r: dict[str, Any]) -> Path:
    return base / "requests" / r["assignment_id"]


def generation_path(base: Path, r: dict[str, Any]) -> Path:
    return (
        ROOT / r["source_path"] if r["recovery_stage"] == "judge" else path_for(base, r)
    ) / "generation/raw.json"


def measure(base: Path, r: dict[str, Any]) -> dict[str, Any]:
    result = parent.measure(base, r)
    return result | {
        "recovery_stage": r["recovery_stage"],
        "replaces": r["replaces"],
        "reused_generation": r["recovery_stage"] == "judge",
    }


def recovery_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate fresh generation totals from judgments of reused responses."""
    return {
        stage: parent.counts([r for r in rows if r["recovery_stage"] == stage])
        for stage in ("generation", "judge")
    }


def recovery_pending(base: Path, r: dict[str, Any]) -> bool:
    """A failed terminal generation needs no further judgment or replay."""
    path = path_for(base, r)
    if (path / "judge/result.json").exists():
        return False
    return not (
        r["recovery_stage"] == "generation"
        and (path / "generation/result.json").exists()
        and not (path / "generation/raw.json").exists()
    )


def passed_report(path: Path) -> bool:
    """Require nonempty executed tests with no failure, error or skip."""
    cases = ET.parse(path).getroot().findall(".//testcase")
    return bool(cases) and not any(
        c.find(tag) is not None
        for c in cases
        for tag in ("failure", "error", "skipped")
    )


def initialize_pacing(root: Path, base: Path) -> None:
    """Retain the latest matching numeric limits, usage windows and cooldowns."""
    now = time.time()
    audit = {}
    for profile in (*study.PROFILES, "sonnet", "account-rpm"):
        relative = (
            "account-rpm.json"
            if profile == "account-rpm"
            else profile + "/rate-state.json"
        )
        target = base / "operations/pacing" / relative
        if target.exists():
            continue
        choices = [
            root / "runs/reviewer-observation" / run / "operations/pacing" / relative
            for run in ("matched-review-v1", "matched-syntax-v1")
        ]
        source = max(
            (p for p in choices if p.is_file()), key=lambda p: p.stat().st_mtime
        )
        payload = source.read_bytes()
        data = json.loads(payload)
        if profile == "account-rpm":
            if not 0 < data["cap"] <= 6000:
                raise ValueError("Invalid shared RPM cap")
            data["window"] = [r for r in data.get("window", []) if r["at"] > now - 60]
        else:
            provider = (
                study.JUDGE_PROFILE if profile == "sonnet" else study.PROFILES[profile]
            )["provider"]
            lane = data[provider]
            lane["queue"] = []
            lane["window"] = [r for r in lane.get("window", []) if r["at"] > now - 60]
            data = {provider: lane}
        write_json(target, data)
        audit[profile] = {
            "source": str(source.relative_to(root)),
            "sha256": digest(payload),
            "copied_state": data,
        }
    if audit:
        original.freeze(
            base / "operations/pacing-initialization.json", {"at": now, "lanes": audit}
        )


def other_committed(root: Path) -> float:
    total = 0
    for directory in ("matched-review-v1", "matched-syntax-v1"):
        b = root / "runs/reviewer-observation" / directory
        m = json.loads((b / "manifest.json").read_text())
        with closing(
            sqlite3.connect(f"file:{b / 'spending.sqlite3'}?mode=ro", uri=True)
        ) as db:
            scope = db.execute("SELECT cap,identity FROM scope WHERE id=1").fetchone()
            if scope != (nanodollars(m["cap_usd"]), digest(json_text(m))):
                raise ValueError("Other ledger scope changed")
            spent, reserved = db.execute(
                "SELECT COALESCE(SUM(settled),0),COALESCE(SUM(CASE WHEN settled IS NULL THEN reserved ELSE 0 END),0) FROM entries"
            ).fetchone()
            total += (spent + reserved) / 1e9
    return total


def sources(root: Path) -> dict[str, str]:
    return parent.sources(root) | {
        name: digest((root / name).read_bytes())
        for name in (
            "src/grader_power/matched_recovery.py",
            "src/grader_power/matched_recovery_lock.py",
            "evals/inspect_matched_recovery.py",
            "scripts/run_matched_recovery.py",
            "scripts/prepare_matched_recovery.py",
            "tests/test_matched_recovery.py",
            "tests/test_matched_recovery_runner.py",
            "evals/inspect_outcome_recovery.py",
            "planning/matched_recovery_protocol.md",
        )
    }


def load(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    b = study.base(root)
    m = json.loads((b / "manifest.json").read_text())
    if (
        m["run"] != study.RUN
        or m["cap_usd"] != study.CAP
        or m["source_hashes"] != sources(root)
    ):
        raise ValueError("Recovery source changed")
    for name, sha in m["bindings"].items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Recovery source outcome changed")
    for name, sha in m["verification_reports"].items():
        path = root / name
        if digest(path.read_bytes()) != sha or not passed_report(path):
            raise ValueError("Recovery verification report changed or failed")
    return m, m["records"]


def forecast(root: Path, m: dict[str, Any]) -> dict[str, Any]:
    b = study.base(root)
    balance = ledger_for(b, m).snapshot()
    remaining = 0.0
    with closing(sqlite3.connect(b / "spending.sqlite3")) as db:
        reserved = {row[0] for row in db.execute("SELECT identity FROM entries")}
    for r in m["records"]:
        p = path_for(b, r)
        if (
            r["recovery_stage"] == "generation"
            and r["assignment_id"] + "/generation" not in reserved
            and not (p / "generation/started.json").exists()
        ):
            remaining += m["forecast_costs"][r["profile"]][0]
        if (
            r["assignment_id"] + "/judge" not in reserved
            and not (p / "judge/result.json").exists()
        ):
            remaining += m["forecast_costs"][r["profile"]][1]
    estimate = balance["committed"] / 1e9 + remaining
    return {
        "median_total_usd": estimate,
        "combined_median_total_usd": estimate + other_committed(root),
        "recorded_usd": balance["spent"] / 1e9,
        "outstanding_usd": balance["outstanding"] / 1e9,
        "at": time.time(),
    }


class Pipeline(parent.Pipeline):
    """Reuse unchanged request transport and sandbox stages with isolated state."""

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
        if self.cost_forecast and (
            self.cost_forecast["median_total_usd"] > study.MEDIAN_LIMIT
            or self.cost_forecast["combined_median_total_usd"] > 500
        ):
            self.pause("Recovery/combined cost forecast exceeds authorized limit")
        if self.stop.is_set():
            raise RuntimeError(str(self.reason))
        if (self.base / "STOP").exists() or (self.operations / "STOP").exists():
            self.pause("Human STOP")
            raise RuntimeError("Human STOP")

    def event(
        self, r: dict[str, Any], stage: str, failed: bool, error: str | None = None
    ) -> None:
        super().event(r, stage, failed, error)
        self.measurements[r["assignment_id"]] = measure(self.base, r)

    async def generation_stage(self, r: dict[str, Any]) -> None:
        if r["recovery_stage"] == "judge":
            await self.judging.put(r)
        else:
            await super().generation_stage(r)

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
        if r["recovery_stage"] == "judge":
            prior_packet = json.loads(
                (self.root / r["source_path"] / "judge/packet.json").read_text()
            )
            if data != prior_packet:
                raise ValueError("Reused response's blinded packet changed")
        original.freeze(path / "judge/packet.json", data)
        request = parent.render_packet(r["study"], data)
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
                    parsed = parent.parse_packet(r["study"], response.completion, data)
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

    def status(self, state: str) -> dict[str, Any]:
        """Report fresh counts and realized completions separately from quota estimates."""
        now = time.time()
        self.completions = [r for r in self.completions if r["at"] > now - 300]
        balance = self.ledger.snapshot()
        if now - self.last_cost_forecast_at >= 300:
            self.cost_forecast = forecast(self.root, self.manifest)
            self.last_cost_forecast_at = now
            if self.cost_forecast["median_total_usd"] > study.MEDIAN_LIMIT:
                self.pause(
                    "Median whole-study forecast exceeds $500 continuation limit"
                )
        output = {
            "budget_forecast": self.cost_forecast,
            "state": state,
            "phase": self.phase,
            "pid": os.getpid(),
            "updated_at": now,
            "counts": recovery_counts(list(self.measurements.values())),
            "counts_by_arm": parent.counts(
                list(self.measurements.values()), by_arm=True
            ),
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
                for k in tuple(dict.fromkeys((*original_study.PROFILES, "sonnet")))
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
                for k in tuple(dict.fromkeys((*original_study.PROFILES, "sonnet")))
            },
        }
        write_json(self.operations / "status.json", output)
        append_event(self.operations / "throughput-events.jsonl", output)
        return output

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


def prepare(root: Path) -> None:
    """Bind reviewed additive outcomes, unchanged prompts and conservative recovery costs."""
    b = study.base(root)
    if (b / "manifest.json").exists():
        raise ValueError("Recovery already frozen")
    parent_manifest, parent_records = parent.load(root, retained=True)
    reports = [
        b / "operations/recovery-tests.xml",
        root / "runs/reviewer-observation/matched-syntax-v1/verification/docker.xml",
    ]
    if not all(passed_report(p) for p in reports):
        raise ValueError("Passing focused and unchanged Docker reports required")
    original.verify_integration(reports[-1])
    require_pilot_verification(root, parent_manifest["oracle_image"])
    records = study.recovery_records(root)
    proof = study.free_repairs(root)
    bindings = {}
    for p in (
        b / "operations/initial-audit.json",
        b / "free-repairs-v1.json",
        b / "designated-records.json",
        b / "operations/human-approval.json",
        original_study.base(root) / "manifest.json",
        original_study.base(root) / "analysis-v1/measured-rows.json",
    ):
        bindings[str(p.relative_to(root))] = digest(p.read_bytes())
    for item in proof["repairs"]:
        bindings.update(item["bindings"])
    for r in records:
        source = root / r["source_path"]
        for p in source.rglob("*"):
            if p.is_file():
                bindings[str(p.relative_to(root))] = digest(p.read_bytes())
    floors = {}
    for profile in study.PROFILES:
        costs = []
        for stage in ("generation", "judge"):
            values = []
            for r in parent_records:
                if r["phase"] != "main" or r["profile"] != profile:
                    continue
                p = (
                    original_study.base(root)
                    / "requests"
                    / r["assignment_id"]
                    / stage
                    / "raw.json"
                )
                if p.exists():
                    values.append(
                        parent.request_cost(
                            json.loads(p.read_text())["usage"],
                            study.PROFILES[profile]
                            if stage == "generation"
                            else study.JUDGE_PROFILE,
                            no_cache=True,
                        )
                        / 1e9
                    )
            costs.append(sum(values) / len(values))
        floors[profile] = costs
    m = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "authorization": study.AUTHORIZATION,
        "records": records,
        "profiles": parent_manifest["profiles"],
        "judge_profile": parent_manifest["judge_profile"],
        "judges": parent_manifest["judges"],
        "oracle_image": parent_manifest["oracle_image"],
        "source_hashes": sources(root),
        "verification_reports": {
            str(p.relative_to(root)): digest(p.read_bytes()) for p in reports
        },
        "bindings": bindings,
        "forecast_costs": floors,
        "designated_attempts": True,
        "parent_pooling": False,
    }
    original.freeze(b / "manifest.json", m)
    for name, sha in m["source_hashes"].items():
        content = (root / name).read_bytes()
        if digest(content) != sha:
            raise ValueError("Source changed during recovery freeze")
        target = b / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(content)
    ledger_for(b, m).snapshot()
    original.freeze(b / "preflight-forecast.json", forecast(root, m))


def analyze(root: Path) -> dict[str, Any]:
    """Publish original strict rows and a separate fully designated recovery view."""
    m, records = load(root)
    b = study.base(root)
    old = original_study.base(root)
    rows = json.loads((old / "analysis-v1/measured-rows.json").read_text())
    index = {r["assignment_id"]: r for r in rows}
    for item in study.free_repairs(root)["repairs"]:
        target = index[item["assignment_id"]]
        target.update(
            original_label=target["label"],
            original_judge_status=target["judge_status"],
            label=item["repair"]["label"],
            label_source="separate_precedent_mechanical_repair",
            qualified_judge_status="classified",
        )
    pending = []
    for r in records:
        p = path_for(b, r)
        judged = pilot.read_optional(p / "judge/result.json")
        if recovery_pending(b, r):
            pending.append(r["assignment_id"])
        selected = measure(b, r)
        label = None if judged is None else judged["label"]
        repair = None
        if judged is not None:
            raw_generation = json.loads(generation_path(b, r).read_text())
            if judged["source_raw_sha256"] != digest(json_text(raw_generation)):
                raise ValueError("Recovered judgment generation binding changed")
            if (p / "judge/packet.json").exists():
                packet = json.loads((p / "judge/packet.json").read_text())
                if judged["packet_sha256"] != digest(json_text(packet)):
                    raise ValueError("Recovered judgment packet binding changed")
                raw_response = pilot.read_optional(p / "judge/raw.json")
                if label is not None:
                    if raw_response is None:
                        raise ValueError("Resolved judgment has no raw response")
                    output = ModelOutput.model_validate(raw_response)
                    if not output.choices or output.choices[0].stop_reason != "stop":
                        raise ValueError("Resolved judgment is incomplete")
                    if (
                        parent.parse_packet(r["study"], output.completion, packet).label
                        != label
                    ):
                        raise ValueError("Recovered label differs from strict reparse")
        if judged and label is None and (p / "judge/raw.json").exists():
            raw = ModelOutput.model_validate(
                json.loads((p / "judge/raw.json").read_text())
            )
            if raw.choices and raw.choices[0].stop_reason == "stop":
                try:
                    repair = study.normalize(
                        raw.completion,
                        json.loads((p / "judge/packet.json").read_text()),
                    )
                except (ValueError, TypeError):
                    pass
                else:
                    label = repair["label"]
                    original.freeze(
                        p / "judge/qualified-format.json",
                        {
                            **repair,
                            "raw_sha256": digest((p / "judge/raw.json").read_bytes()),
                            "packet_sha256": digest(
                                (p / "judge/packet.json").read_bytes()
                            ),
                            "authorization": study.AUTHORIZATION,
                        },
                    )
        target = index[r["replaces"]]
        target.update(
            original_label=target["label"],
            original_judge_status=target["judge_status"],
            label=label,
            label_source="fresh_recovery_format"
            if repair
            else "designated_fresh_recovery",
            recovery_assignment_id=r["assignment_id"],
            judge_status=selected["judge_status"],
            judge_started=selected["judge_started"],
            judge_result=selected["judge_result"],
            qualified_judge_status="classified" if repair else selected["judge_status"],
        )
        if r["recovery_stage"] == "generation":
            target.update(
                {
                    k: selected[k]
                    for k in (
                        "started",
                        "attempt_complete",
                        "raw_saved",
                        "scored",
                        "full_correct",
                        "generation_status",
                        "judge_status",
                    )
                }
            )
    result = analysis.analyze(rows)
    result.update(
        recovery_run=study.RUN,
        designated_recoveries=len(records),
        free_repairs=86,
        pending_recoveries=pending,
        recovery_counts=recovery_counts([measure(b, r) for r in records]),
        budget=forecast(root, m),
        original_analysis_sha256=digest(
            (old / "analysis-v1/measured-rows.json").read_bytes()
        ),
    )
    write_json(b / "analysis-v1/measured-rows.json", rows)
    write_json(b / "analysis-v1/analysis.json", result)
    return result


def collect(root: Path) -> None:
    """Require all preceding collectors finished; never compete with the healthy syntax run."""
    m, records = load(root)
    b = study.base(root)
    guard(b)
    owner = root / "runs/extension-pilot/coordination/owner.json"
    previous = json.loads(owner.read_text())
    if previous["state"] != "released" or process_group_members(previous["pgid"]):
        raise ValueError("Previous collector and every paid child must exit first")
    new = root / "runs/reviewer-observation/matched-syntax-v1"
    status = json.loads((new / "operations/status.json").read_text())
    if status["state"] not in ("complete", "qualified_main_complete_others_held"):
        raise ValueError(
            "Wait for the syntax collector and its downstream stages to finish"
        )
    syntax_records = json.loads((new / "assignments.json").read_text())
    for row in syntax_records:
        if (
            row["phase"] == "main"
            and (row["study"], row["profile"]) != ("naming", "glm")
            and not (
                new / "requests" / row["assignment_id"] / "generation/result.json"
            ).exists()
        ):
            raise ValueError(
                "Approved syntax main cohorts, including GLM quotes, are not finished"
            )
    if other_committed(root) + study.CAP > 550:
        raise ValueError("Insufficient combined recovery allowance")
    f = forecast(root, m)
    if (
        f["median_total_usd"] > study.MEDIAN_LIMIT
        or f["combined_median_total_usd"] > 500
    ):
        raise ValueError("Recovery forecast over limit")
    tag = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + f"-{os.getpid()}"
    launch = b / "operations/launches" / tag
    original.freeze(
        launch / "process.json",
        {
            "pid": os.getpid(),
            "pgid": os.getpgrp(),
            "manifest_sha256": digest(json_text(m)),
            "authorization": study.AUTHORIZATION,
            "forecast": f,
            "previous_owner": previous,
        },
    )
    with collection_lock(root, owner=study.RUN, run_id=study.RUN):
        initialize_pacing(root, b)
        require_pilot_verification(root, m["oracle_image"])
        image = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", m["oracle_image"]],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        ).stdout.strip()
        containers = parent.sandbox_containers()
        if image != m["oracle_image"] or any(containers.values()):
            raise ValueError("Pinned sandbox not clean/verified")
        original.freeze(
            launch / "preflight.json", {"image": image, **containers, "at": time.time()}
        )
        pipe = Pipeline(root, m, records, "main")
        with (
            pilot.zero_retry_clients(),
            concurrent_transport(),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):

            async def execute() -> None:
                loop = asyncio.get_running_loop()
                loop.set_default_executor(executor)
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(sig, pipe.pause, "Operator signal")
                await pipe.run()

            try:
                asyncio.run(execute())
            finally:
                cleanup = parent.sandbox_containers()
                original.freeze(launch / "cleanup.json", cleanup)
                if any(cleanup.values()):
                    raise RuntimeError("Sandbox cleanup failed")
        result = analyze(root)
        if result["pending_recoveries"]:
            raise RuntimeError("Recovery has unfinished downstream stages")
        pipe.status("complete")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "collect", "analyze"))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(ROOT)
    elif args.command == "check":
        m, records = load(ROOT)
        print(json.dumps({"records": len(records), "forecast": forecast(ROOT, m)}))
    elif args.command == "analyze":
        analyze(ROOT)
    else:
        isolate_collection_process()
        collect(ROOT)


if __name__ == "__main__":
    main()
