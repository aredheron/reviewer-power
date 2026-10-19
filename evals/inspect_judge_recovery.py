"""One authorized, additive recovery of 90 saved-code judgments under a new $5 cap."""

import argparse
import asyncio
import copy
import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageSystem, ChatMessageUser, ModelOutput

from evals import inspect_framing as framing
from evals import inspect_outcome_recovery as prior
from grader_power import (
    compact_replication,
    extension_analysis,
    main_analysis,
    reversal,
    reward_conflict_analysis,
    simple_replication_analysis,
    three_style_analysis,
)
from grader_power.main_tasks import main_tasks
from grader_power.replication_spending import RequestLedger
from grader_power.scoring import OracleRequest, score_response
from grader_power.spending import nanodollars

RUN = "judge-recovery-v1"
CAP = 5
AUTHORIZATION = 'User: "Sounds good, please do that. I authorize a $5 budget." Scope: one new blinded judgment on each of 90 saved implementations, including seven offline sandbox rescoring recoveries. Forty format/evidence failures use line ranges with caller-extracted quotations; semantic rubric unchanged. Review two ambiguities without forcing labels; retain 32 interrupted episodes. Refresh statistical summaries and dashboard. No new generations, repeated failed recovery calls or old ledger changes.'
CATEGORIES = {
    "judge_proven_unsent": 24,
    "judge_transport_failure": 19,
    "judgment_blocked_by_scoring": 7,
    "malformed_json": 20,
    "evidence_quote_mismatch": 20,
}
RELATIVE_RUNS = {
    "original_main": "main/main-32-v1",
    "extension": "extension-main/gpt-extension-main-v1",
    "framing_names": "framing-main/framing-names-v1",
    "reward_conflict": "reward-conflict/reward-conflict-v1",
    "three_style": "three-style/three-style-v1",
    "reversal": "reversal/reversal-v1",
    "compact_replication": "replication/compact-models-v1",
    "simple_replication": "replication/simple-models-v1",
}
NEW_SOURCES = (
    "evals/inspect_judge_recovery.py",
    "scripts/recover_saved_judgments.py",
    "tests/test_judge_recovery.py",
    "src/grader_power/simple_replication_analysis.py",
    "src/grader_power/results_dashboard.py",
    "tests/test_results_dashboard.py",
)
read = prior.read
freeze = prior.original.freeze
digest = prior.digest
json_text = prior.json_text


def base(root: Path) -> Path:
    """Use a separate run, request namespace and budget."""
    return root / "runs/recovery" / RUN


def range_prompt(prompt: str) -> str:
    """Change only the response schema; preserve all semantic rubric text verbatim."""
    prefix, contract = prompt.split("OUTPUT CONTRACT\n", 1)
    contract = contract.replace(', "quote": "exact source lines"', "")
    start = contract.index("For quote, copy the\n")
    end = contract.index("Cite enough source", start)
    contract = (
        contract[:start]
        + "Do not return quote fields. The caller extracts the exact source text from each\ninclusive line range.\n"
        + contract[end:]
    )
    contract = contract.replace(
        "Do not invent line numbers or truncate quotes.",
        "Do not invent line numbers or return out-of-range spans.",
    )
    return prefix + "OUTPUT CONTRACT\n" + contract


def parse_ranges(text: str, packet: dict[str, Any], design: str) -> Any:
    """Reject duplicates, coercions, extra fields and invalid ranges, then validate normally."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate response key")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(value, dict) or not isinstance(value.get("evidence"), list):
        raise TypeError("Expected judgment object with evidence array")
    lines = packet["code"].splitlines()
    for item in value["evidence"]:
        if not isinstance(item, dict) or set(item) != {"start_line", "end_line"}:
            raise ValueError("Evidence must contain exactly two line numbers")
        start, end = item["start_line"], item["end_line"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 1 <= start <= end <= len(lines)
        ):
            raise ValueError("Invalid evidence range")
        item["quote"] = "\n".join(lines[start - 1 : end])
    return prior.compact.parse_packet(design, json_text(value), packet)


def catalog(root: Path) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    """Verify the preceding recovery and bind exact saved implementations and blinded packets."""
    previous = prior.verify(root)
    directory = prior.base(root)
    cases = read(directory / "remaining-unresolved.json")
    selected = [c for c in cases if c["category"] in CATEGORIES]
    if Counter(c["category"] for c in selected) != CATEGORIES:
        raise ValueError("Authorized 90-case scope changed")
    bindings = {
        str(p.relative_to(root)): digest(p.read_bytes())
        for p in directory.glob("analysis/replacements/*/*.json")
    }
    for name in ("manifest.json", "completion.json", "remaining-unresolved.json"):
        bindings[str((directory / name).relative_to(root))] = digest(
            (directory / name).read_bytes()
        )
    tasks = {t.visible.source_task_id: t for t in main_tasks(root)}
    rows = {
        study: {
            r["assignment_id"]: r
            for r in read(
                directory / "analysis/replacements" / study / "measured-rows.json"
            )
        }
        for study in RELATIVE_RUNS
    }
    records = []
    for c in selected:
        prior.verify_hashes(root, c["evidence"])
        bindings.update(c["evidence"])
        source = [
            p for p in c["evidence"] if p.endswith("/raw.json") and "/judge/" not in p
        ]
        if len(source) != 1:
            raise ValueError("Missing or ambiguous original generation")
        raw = read(root / source[0])
        reconstructed = prior.original.packet_from_output(raw, tasks[c["task"]])
        if reconstructed is None:
            raise ValueError("Saved implementation has no valid blinded packet")
        packet = reconstructed.model_dump(mode="json")
        packet_path = root / c["judge_path"] / "packet.json"
        if packet_path.exists() and packet != read(packet_path):
            raise ValueError("Original packet differs from reconstructed code/contract")
        old = rows[c["study"]][c["assignment_id"]]
        if old["label"] is not None:
            raise ValueError("Selected judgment already resolved")
        design = old.get(
            "study",
            "reversal" if c["study"] in ("three_style", "reversal") else "mapping",
        )
        records.append(
            {
                "assignment_id": digest(
                    json_text([RUN, c["study"], c["assignment_id"], 1])
                ),
                "replaces": c["assignment_id"],
                "origin_study": c["study"],
                "original_run": "runs/" + RELATIVE_RUNS[c["study"]],
                "profile": c["profile"],
                "source_task_id": c["task"],
                "study": design,
                "phase": "recovery",
                "category": c["category"],
                "source_raw": source[0],
                "source_raw_sha256": digest(json_text(raw)),
                "packet": packet,
                "needs_scoring": c["category"] == "judgment_blocked_by_scoring",
                "line_ranges": c["category"]
                in ("malformed_json", "evidence_quote_mismatch"),
            }
        )
    # Source code is deliberately not copied into a new generation/raw.json: these
    # are new judgments, never new generation responses or re-executed episodes.
    return records, bindings, previous


def prepare(root: Path, reports: list[Path]) -> None:
    """Freeze the new authorization, sources, inputs, semantic rubric and test evidence."""
    records, bindings, previous = catalog(root)
    verified = {}
    for path in reports:
        cases = ET.parse(path).getroot().findall(".//testcase")
        if not cases or any(
            c.find(t) is not None
            for c in cases
            for t in ("failure", "error", "skipped")
        ):
            raise ValueError("A required verification report did not pass")
        verified[str(path.relative_to(root))] = digest(path.read_bytes())
    if len(verified) < 2:
        raise ValueError("Require offline and live Docker verification")
    pacing = prior.base(root) / "operations/pacing/sonnet/rate-state.json"
    bindings[str(pacing.relative_to(root))] = digest(pacing.read_bytes())
    judges = copy.deepcopy(previous["judges"])
    range_prompts = {k: range_prompt(v["system_prompt"]) for k, v in judges.items()}
    manifest = {
        "run": RUN,
        "authorization": AUTHORIZATION,
        "cap_usd": CAP,
        "records": records,
        "profiles": {"sonnet": previous["profiles"]["sonnet"]},
        "judges": judges,
        "range_prompts": range_prompts,
        "oracle_image": previous["oracle_image"],
        "source_hashes": {
            **previous["source_hashes"],
            **{p: digest((root / p).read_bytes()) for p in NEW_SOURCES},
        },
        "retained_evidence": bindings,
        "verification_reports": verified,
        "pacing_source": str(pacing.relative_to(root)),
        "analysis_input": str(
            (prior.base(root) / "analysis/replacements").relative_to(root)
        ),
        "protocol": "One judge call per selected saved implementation, without original failures or treatment/model metadata in the prompt. Only 40 formatting failures use range-only evidence. Original semantic rubric and label parser remain unchanged. New malformed/ambiguous answers remain unresolved.",
    }
    freeze(base(root) / "manifest.json", manifest)
    for name, value in manifest["source_hashes"].items():
        payload = (root / name).read_bytes()
        if digest(payload) != value:
            raise ValueError("Source changed during preparation")
        path = base(root) / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("Snapshot drift")
        path.write_bytes(payload)


def verify(root: Path) -> dict[str, Any]:
    """Fail closed on any changed scientific/source/input/report binding."""
    manifest = read(base(root) / "manifest.json")
    if (
        manifest["run"] != RUN
        or manifest["cap_usd"] != CAP
        or len(manifest["records"]) != 90
    ):
        raise ValueError("Recovery scope changed")
    expected = dict(manifest["source_hashes"])
    amendment_path = base(root) / "operations/analysis-amendment.json"
    if amendment_path.exists():
        amendment = read(amendment_path)
        allowed = {
            "evals/inspect_judge_recovery.py",
            "tests/test_judge_recovery.py",
            "src/grader_power/results_dashboard.py",
            "tests/test_results_dashboard.py",
        }
        if (
            amendment["kind"] != "offline-model-identity-fix"
            or amendment["manifest_sha256"] != digest(json_text(manifest))
            or not set(amendment["sources"]) <= allowed
        ):
            raise ValueError("Unapproved offline analysis amendment")
        for name, value in amendment["sources"].items():
            old = base(root) / "source-snapshot" / name
            if digest(old.read_bytes()) != expected[name]:
                raise ValueError("Original analysis source snapshot changed")
            expected[name] = value
        prior.verify_hashes(root, amendment["verification_reports"])
        audit_path = base(root) / amendment["audit"]
        if digest(audit_path.read_bytes()) != amendment["audit_sha256"]:
            raise ValueError("Offline repair audit changed")
        prior.verify_hashes(base(root), read(audit_path)["retained_outcomes"])
    prior.verify_hashes(root, expected)
    for key in ("retained_evidence", "verification_reports"):
        prior.verify_hashes(root, manifest[key])
    return manifest


def outcome(directory: Path, r: dict[str, Any]) -> dict[str, Any]:
    """Count only new recovery work, not referenced original generation responses."""
    p = directory / "requests" / r["assignment_id"]
    result = prior.pilot.read_optional(p / "judge/result.json") or {}
    return {
        "judge_started": (p / "judge/started.json").exists(),
        "judge_result": bool(result),
        "label": result.get("label"),
        "judge_status": result.get("status"),
        "rescored": (p / "scoring.json").exists(),
    }


class JudgePipeline(prior.RecoveryPipeline):
    """Reuse bounded, audited transport with no generation queue and a separate ledger."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root, self.manifest, self.records, self.phase = (
            root,
            manifest,
            manifest["records"],
            "recovery",
        )
        self.base, self.operations = base(root), base(root) / "operations"
        self.ledger = RequestLedger(
            self.base / "spending.sqlite3",
            cap=nanodollars(CAP),
            scope=digest(json_text(manifest)),
        )
        self.stop, self.finishing = asyncio.Event(), asyncio.Event()
        self.reason: str | None = None
        self.tasks_by_id = {t.visible.source_task_id: t for t in main_tasks(root)}
        events = self.operations / "stage-events.jsonl"
        self.history = (
            [json.loads(s) for s in events.read_text().splitlines()]
            if events.exists()
            else []
        )
        self.events_seen = {(e["identity"], e["stage"]) for e in self.history}
        self.measurements: dict[str, Any] = {}
        self.queues: dict[str, asyncio.Queue[Any]] = {}
        self.scoring: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.judging: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.account = prior.AccountRPM(self.operations / "pacing")
        self.models: dict[str, Any] = {}
        self.keys: set[str] = set()
        self.active: Counter[str] = Counter()
        self.started_at = time.time()
        self.targets = {"judge": 4}
        self.completions: list[dict[str, Any]] = []

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Prohibit generation regardless of any misrouted record."""
        raise RuntimeError("Generation is outside this recovery authorization")

    async def scoring_stage(self, r: dict[str, Any]) -> None:
        """Rescore seven saved implementations only, inside the unchanged pinned oracle."""
        if not r["needs_scoring"]:
            raise ValueError("Unexpected scoring request")
        path = self.base / "requests" / r["assignment_id"]
        if not (path / "scoring-started.json").exists():
            freeze(
                path / "scoring-started.json",
                {"at": time.time(), "source_raw_sha256": r["source_raw_sha256"]},
            )
            task = self.tasks_by_id[r["source_task_id"]]
            raw = ModelOutput.model_validate(read(self.root / r["source_raw"]))
            score = await asyncio.to_thread(
                score_response,
                raw.completion,
                task.region,
                oracle=OracleRequest(
                    task.dataset,
                    task.dataset_name,
                    task.visible.source_task_id,
                    self.manifest["oracle_image"],
                ),
            )
            freeze(path / "scoring.json", score.to_dict())
        score = prior.pilot.read_optional(path / "scoring.json")
        if score is None or prior.compact.scoring_infrastructure_issue(score):
            self.event(r, "scoring", True, "SandboxInfrastructure")
            self.pause("Sandbox scoring requires review")
            return
        self.event(r, "scoring", False)
        await self.judging.put(r)

    async def judging_stage(self, r: dict[str, Any]) -> None:
        """Judge unchanged code once; retain strict failures and ambiguous labels."""
        p = self.base / "requests" / r["assignment_id"] / "judge"
        if (p / "started.json").exists() or (p / "result.json").exists():
            return
        packet = r["packet"]
        freeze(p / "packet.json", packet)
        prompt = (
            self.manifest["range_prompts"][r["study"]]
            if r["line_ranges"]
            else self.manifest["judges"][r["study"]]["system_prompt"]
        )
        messages = [
            ChatMessageSystem(content=prompt),
            ChatMessageUser(content=prior.compact.render_packet(r["study"], packet)),
        ]
        raw = await self.request(r, "judge", messages)
        result = {
            "source_raw_sha256": r["source_raw_sha256"],
            "packet_sha256": digest(json_text(packet)),
            "status": "judge_error",
            "label": None,
            "line_ranges": r["line_ranges"],
        }
        try:
            if raw is not None:
                output = ModelOutput.model_validate(raw)
                if output.choices and output.choices[0].stop_reason == "stop":
                    parsed = (
                        parse_ranges(output.completion, packet, r["study"])
                        if r["line_ranges"]
                        else prior.compact.parse_packet(
                            r["study"], output.completion, packet
                        )
                    )
                    result.update(
                        status=parsed.status,
                        label=parsed.label,
                        judgment=parsed.model_dump(mode="json"),
                    )
        except (ValueError, TypeError) as error:
            result["error_type"] = type(error).__name__
        freeze(p / "result.json", result)
        self.event(
            r,
            "judge",
            result["status"] not in ("classified", "abstain"),
            result.get("error_type"),
        )

    def status(self, state: str) -> dict[str, Any]:
        """Report recovery counts, live activity and nonoverlapping spend/reservations."""
        rows = [outcome(self.base, r) for r in self.records]
        result = {
            "at": time.time(),
            "pid": os.getpid(),
            "state": state,
            "reason": self.reason,
            "assigned_judgments": len(rows),
            "started": sum(r["judge_started"] for r in rows),
            "completed": sum(r["judge_result"] for r in rows),
            "resolved": sum(r["label"] is not None for r in rows),
            "rescored": sum(r["rescored"] for r in rows),
            "active": dict(self.active),
            "usage": self.ledger.snapshot(),
        }
        prior.write_json(self.operations / "status.json", result)
        return result

    async def run(self) -> None:
        """Feed bounded downstream queues and drain accepted work on any stop."""
        self.initialize_models()
        workers = [
            asyncio.create_task(self.worker(self.judging, "judge", i, "judge"))
            for i in range(4)
        ] + [
            asyncio.create_task(self.worker(self.scoring, "scoring", i, "scoring"))
            for i in range(4)
        ]
        watcher = asyncio.create_task(self.monitor())
        self.status("running")
        try:
            for r in self.records:
                await (self.scoring if r["needs_scoring"] else self.judging).put(r)
            await self.scoring.join()
            await self.judging.join()
        finally:
            self.finishing.set()
            for q in (self.scoring, self.judging):
                for _ in range(4):
                    await q.put(None)
            await asyncio.gather(*workers)
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        self.status("paused" if self.stop.is_set() else "complete")
        if self.stop.is_set():
            raise RuntimeError(self.reason)


def sandbox_clean(image: str) -> None:
    """Require the verified image and no surviving experiment containers."""
    found = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    containers = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if found.stdout.strip() != image or containers.stdout.strip():
        raise RuntimeError("Sandbox identity/cleanup failed")


def collect(root: Path) -> None:
    """Acquire the global lock and run only the newly authorized judgment attempts."""
    manifest = verify(root)
    if (base(root) / "operations/analysis-amendment.json").exists():
        raise RuntimeError(
            "Offline analysis amendment does not authorize paid collection"
        )
    prior.isolate_collection_process()
    launch = (
        base(root)
        / "operations/launches"
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
    )
    owner = root / "runs/extension-pilot/coordination/owner.json"
    if owner.exists():
        freeze(launch / "previous-owner.json", read(owner))
    with prior.collection_lock(root, owner="compact-models-v1", run_id=RUN):
        prior.require_pilot_verification(root, manifest["oracle_image"])
        sandbox_clean(manifest["oracle_image"])
        freeze(
            launch / "authorized-launch.json",
            {
                "at": time.time(),
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "authorization": AUTHORIZATION,
                "manifest_sha256": digest(json_text(manifest)),
                "sandbox_clean": True,
            },
        )
        target = base(root) / "operations/pacing/sonnet/rate-state.json"
        if not target.exists():
            prior.write_json(target, read(root / manifest["pacing_source"]))
        pipeline = JudgePipeline(root, manifest)
        with (
            prior.pilot.zero_retry_clients(),
            prior.concurrent_transport(),
            ThreadPoolExecutor(max_workers=6) as executor,
        ):

            async def execute() -> None:
                loop = asyncio.get_running_loop()
                loop.set_default_executor(executor)
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.add_signal_handler(
                        sig, pipeline.pause, "Operator signal; drain paid work"
                    )
                await pipeline.run()

            try:
                asyncio.run(execute())
            finally:
                sandbox_clean(manifest["oracle_image"])
                freeze(
                    launch / "drained.json",
                    {
                        "at": time.time(),
                        "sandbox_clean": True,
                        "usage": pipeline.ledger.snapshot(),
                    },
                )
        verify(root)
        freeze(
            launch / "complete.json", {"all_bindings_verified": True, "at": time.time()}
        )


def framing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Recompute the original descriptive framing analysis without changing its denominator."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        c = r["condition"]
        groups[
            (
                c["grader_consequence"],
                r["framing"],
                r["role_name"],
                c["user_preference"] != c["grader_preference"],
            )
        ].append(r)
    summaries = []
    for (purpose, frame, name, conflict), rs in sorted(groups.items()):
        known = [r for r in rs if r["label"] is not None]
        result = {
            "purpose": purpose,
            "framing": frame,
            "role_name": name,
            "conflict": conflict,
            **framing.counts(rs),
            "labels": dict(Counter(r["label"] for r in known)),
            "correct": sum(r["full_correct"] is True for r in rs),
        }
        for field, preference in (
            ("user_following", "user_preference"),
            ("grader_following", "grader_preference"),
        ):
            n = sum(
                framing.style_matches(r["label"], r["condition"][preference])
                for r in known
            )
            result.update(
                {
                    field: n,
                    field + "_resolved_rate": n / len(known) if known else None,
                    field + "_assigned_bounds": [
                        n / len(rs),
                        (n + len(rs) - len(known)) / len(rs),
                    ],
                }
            )
        summaries.append(result)
    return {"counts": framing.counts(rows), "groups": summaries}


def statistics(study: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Reuse the prospective analysis functions, tasks and multiplicity families."""
    tasks = sorted({r["source_task_id"] for r in rows})
    if study == "original_main":
        return main_analysis.analyze_rows(rows, tasks, 12)
    if study == "extension":
        return extension_analysis.analyze_rows(rows, tasks, 12)
    if study == "simple_replication":
        return simple_replication_analysis.analyze_rows(
            [{**r, "repetition": r["repetition_index"]} for r in rows], tasks, 4
        )
    if study == "compact_replication":
        return compact_replication.analyze(rows)
    if study == "three_style":
        return three_style_analysis.summarize(rows)
    if study == "reward_conflict":
        return reward_conflict_analysis.summarize(rows)
    if study == "reversal":
        return reversal.analyze(rows)
    if study == "framing_names":
        return framing_summary(rows)
    raise ValueError("Unknown analysis family")


def analyze(root: Path) -> None:
    """Validate recovered labels and publish full-denominator additive rows/statistics."""
    manifest = verify(root)
    records = {(r["origin_study"], r["replaces"]): r for r in manifest["records"]}
    states = [outcome(base(root), r) for r in manifest["records"]]
    if not all(s["judge_result"] for s in states):
        raise ValueError("Do not publish an incomplete recovery")
    index, summary = {}, {}
    for study, relative in RELATIVE_RUNS.items():
        source = root / manifest["analysis_input"] / study / "measured-rows.json"
        rows = read(source)
        lineage = []
        for row in rows:
            r = records.get((study, row["assignment_id"]))
            if r is None:
                continue
            path = base(root) / "requests" / r["assignment_id"]
            result = read(path / "judge/result.json")
            if result["source_raw_sha256"] != r["source_raw_sha256"] or result[
                "packet_sha256"
            ] != digest(json_text(r["packet"])):
                raise ValueError("Recovered evidence binding changed")
            if result["status"] in ("classified", "abstain"):
                raw = ModelOutput.model_validate(read(path / "judge/raw.json"))
                parsed = (
                    parse_ranges(raw.completion, r["packet"], r["study"])
                    if r["line_ranges"]
                    else prior.compact.parse_packet(
                        r["study"], raw.completion, r["packet"]
                    )
                )
                if (
                    parsed.label != result["label"]
                    or parsed.model_dump(mode="json") != result["judgment"]
                ):
                    raise ValueError("Recovered judgment does not match raw evidence")
            row.update(
                judge_started=(path / "judge/started.json").exists(),
                judge_result=True,
                judge_status=result["status"],
                label=result["label"],
                judgment_recovery_id=r["assignment_id"],
                judgment_recovery_method="new_blinded_line_ranges"
                if r["line_ranges"]
                else "new_blinded_original_schema",
            )
            if r["needs_scoring"]:
                score = read(path / "scoring.json")
                row.update(
                    scored=True, full_correct=score["correctness"].get("full_correct")
                )
            lineage.append(
                {
                    "assignment_id": row["assignment_id"],
                    "recovery_id": r["assignment_id"],
                    "original_raw": r["source_raw"],
                    "original_failure": r["category"],
                    "new_label": row["label"],
                }
            )
        if len(
            {(r.get("profile", "gpt-oss-120b"), r["assignment_id"]) for r in rows}
        ) != len(rows):
            raise ValueError("Analysis has duplicate assignments")
        directory = base(root) / "analysis" / study
        freeze(directory / "measured-rows.json", rows)
        freeze(directory / "lineage.json", lineage)
        result = statistics(study, rows)
        result.update(
            recovery_note="Supplemental analysis after approved mechanical, generation and judge recoveries; original outcomes retained. Replacement generations are explicitly linked. No unresolved label is imputed.",
            measured_rows_sha256=digest(
                (directory / "measured-rows.json").read_bytes()
            ),
        )
        freeze(directory / "analysis.json", result)
        index[relative] = {
            "path": str((directory / "measured-rows.json").relative_to(root)),
            "sha256": digest((directory / "measured-rows.json").read_bytes()),
            "rows": len(rows),
        }
        summary[study] = {
            "assigned": len(rows),
            "resolved": sum(r["label"] is not None for r in rows),
            "unresolved": sum(r["label"] is None for r in rows),
            "new_judgments": len(lineage),
            "new_resolved": sum(r["new_label"] is not None for r in lineage),
        }
    freeze(base(root) / "analysis/summary.json", summary)
    freeze(
        base(root) / "analysis-index.json",
        {
            "complete": True,
            "manifest_sha256": digest(json_text(manifest)),
            "sources": index,
            "note": "Uses additive audited repairs and linked replacements; historical analyses remain intact.",
        },
    )
    freeze(
        base(root) / "completion.json",
        {
            "new_judgments": 90,
            "new_resolved": sum(s["label"] is not None for s in states),
            "remaining_unresolved": sum(s["unresolved"] for s in summary.values()),
            "usage": JudgePipeline(root, manifest).ledger.snapshot(),
            "all_original_bindings_verified": True,
        },
    )


def main() -> None:
    """Prepare, collect, verify or analyze the authorized additive judge recovery."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "analyze", "verify"))
    parser.add_argument("--report", action="append", type=Path, default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        prepare(root, [p.resolve() for p in args.report])
    elif args.action == "collect":
        collect(root)
    elif args.action == "analyze":
        analyze(root)
    else:
        verify(root)
    print(args.action + " completed")


if __name__ == "__main__":
    main()
