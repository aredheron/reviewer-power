"""Additive, explicitly authorized recovery of audited main-study outcomes."""

import argparse
import asyncio
import copy
import json
import os
import shutil
import signal
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from inspect_ai.model import ChatMessageUser

from evals import inspect_compact_replication as compact
from evals import inspect_main as original
from evals import inspect_pilot as pilot
from evals import inspect_reward_conflict as conflict
from evals import inspect_three_style as three
from evals.inspect_mode_a import digest, json_text
from grader_power import style_judge, three_style_judge
from grader_power.compact_replication_lock import (
    collection_lock,
    isolate_collection_process,
)
from grader_power.main_tasks import main_tasks
from grader_power.pacing import append_event, write_json
from grader_power.pilot_tasks import require_pilot_verification
from grader_power.replication_spending import RequestLedger
from grader_power.replication_transport import AccountRPM, concurrent_transport
from grader_power.spending import nanodollars

RUN = "audited-outcomes-v1"
AUDIT = "runs/audits/unresolved-20261016/inventory.json"
CAP = 5
AUTHORIZATION = (
    'User: "yes, please run the 120 free repairs. You may also rerun any missing '
    'generations; you may have a $5 budget for that but let me know if you need more." '
    "Scope: exactly the 120 audited main mechanical repairs and one fresh linked "
    "attempt for each of the 80 audited missing main Mode A generations, including "
    "normal scoring and first judging of replacements. No other failed judges, "
    "retired pilots, partial episodes, or repeat attempts are authorized here."
)
NEW_SOURCES = (
    "evals/inspect_outcome_recovery.py",
    "scripts/recover_audited_outcomes.py",
    "tests/test_outcome_recovery.py",
)


def base(root: Path) -> Path:
    """Keep recovery evidence and its new budget separate from every original run."""
    return root / "runs/recovery" / RUN


def read(path: Path) -> Any:
    """Read required JSON without defaulting missing scientific evidence."""
    return json.loads(path.read_text())


def verify_hashes(root: Path, bindings: dict[str, str]) -> None:
    """Fail closed on any changed source, report, packet or original outcome."""
    for name, expected in bindings.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Evidence path escapes repository")
        if digest(path.read_bytes()) != expected:
            raise ValueError("Bound evidence changed: " + name)


def normalize_judgment(
    text: str, packet: dict[str, Any], design: str
) -> dict[str, Any]:
    """Normalize only approved metadata/identical duplicates/exact-quote citations.

    Conflicting duplicate keys, changed semantic fields, fabricated quotations,
    ambiguous quote locations and remaining strict validation errors fail closed.
    The live parser and original response are never modified.
    """
    changes: list[dict[str, Any]] = []

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                if result[key] != value:
                    raise ValueError("Conflicting duplicate key")
                changes.append(
                    {"action": "identical_duplicate", "key": key, "value": value}
                )
            result[key] = value
        return result

    parsed = json.loads(text, object_pairs_hook=pairs)
    if not isinstance(parsed, dict):
        raise TypeError("Judgment must be a JSON object")
    candidate = copy.deepcopy(parsed)
    module = style_judge if design == "mapping" else three_style_judge
    for key in list(candidate):
        if key not in module.Judgment.model_fields and candidate[key] in (
            None,
            "",
            [],
            {},
        ):
            changes.append(
                {"action": "empty_metadata", "key": key, "value": candidate.pop(key)}
            )
    lines = packet["code"].splitlines()
    for index, evidence in enumerate(candidate.get("evidence", [])):
        if not isinstance(evidence, dict):
            raise TypeError("Evidence must be an object")
        for key in list(evidence):
            if key not in ("start_line", "end_line", "quote") and (
                evidence[key] in (None, "", [], {})
                or key in ("start_line_note", "start_line_text")
            ):
                changes.append(
                    {
                        "action": "incidental_evidence_metadata",
                        "index": index,
                        "key": key,
                        "value": evidence.pop(key),
                    }
                )
        quote = evidence.get("quote")
        if not isinstance(quote, str) or not quote:
            continue
        quoted = quote.splitlines()
        matches = [
            i + 1
            for i in range(len(lines) - len(quoted) + 1)
            if lines[i : i + len(quoted)] == quoted
        ]
        if len(matches) == 1:
            for key in ("line_number", "start", "text"):
                if key in evidence and (key != "text" or evidence[key] == quote):
                    changes.append(
                        {
                            "action": "redundant_citation_field",
                            "index": index,
                            "key": key,
                            "value": evidence.pop(key),
                        }
                    )
            start, end = matches[0], matches[0] + len(quoted) - 1
            if (evidence.get("start_line"), evidence.get("end_line")) != (start, end):
                changes.append(
                    {
                        "action": "unique_exact_quote_range",
                        "index": index,
                        "old": [evidence.get("start_line"), evidence.get("end_line")],
                        "new": [start, end],
                    }
                )
                evidence.update(start_line=start, end_line=end)
    for field in ("status", "target_forms", "flags", "rationale"):
        if candidate.get(field) != parsed.get(field):
            raise ValueError("Normalization changed semantic judgment")
    if [e.get("quote") for e in candidate.get("evidence", [])] != [
        e.get("quote") for e in parsed.get("evidence", [])
    ]:
        raise ValueError("Normalization changed evidence quote")
    result = compact.parse_packet(design, json_text(candidate), packet)
    if not changes or result.status != "classified" or result.label is None:
        raise ValueError("No approved classified mechanical repair")
    return {
        "status": result.status,
        "label": result.label,
        "judgment": result.model_dump(mode="json"),
        "changes": changes,
    }


def repair_catalog(root: Path, audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify and resolve exactly the 120 user-approved candidates in memory first."""
    cases = [
        c
        for c in audit["cases"]
        if c["scope"] == "main" and c["category"] == "mechanical_repair_candidate"
    ]
    if len(cases) != 120:
        raise ValueError("Mechanical repair scope changed")
    result = []
    for c in cases:
        verify_hashes(root, c["evidence"])
        path = root / c["judge_path"]
        raw, packet = read(path / "raw.json"), read(path / "packet.json")
        # The approved audit's label determines which frozen parser must validate;
        # source study identifies three-style cases, including compact reversal.
        design = "reversal" if c["study"] in ("three_style", "reversal") else "mapping"
        if c["study"] == "compact_replication":
            rows = read(root / "runs/replication/compact-models-v1/assignments.json")
            design = next(
                r["study"] for r in rows if r["assignment_id"] == c["assignment_id"]
            )
        fixed = normalize_judgment(raw["completion"], packet, design)
        if fixed["label"] != c["judge"]["proposed_label"]:
            raise ValueError("Repair differs from approved audit candidate")
        result.append(
            {
                **fixed,
                "origin_study": c["study"],
                "assignment_id": c["assignment_id"],
                "design": design,
                "original_evidence": c["evidence"],
                "original_judge_result": read(path / "result.json"),
            }
        )
    return result


def build_replacements(
    root: Path, audit: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, str]]:
    """Copy exact frozen inputs; introduce only fresh IDs and recovery lineage."""
    cm, originals = compact.load(root, retained=True)
    three_manifest = three.load(root)
    index = {r["assignment_id"]: r for r in originals}
    bindings: dict[str, str] = {}
    conflict_manifest = legacy_manifest(
        root, "reward-conflict/reward-conflict-v1", bindings
    )
    legacy_manifest(root, "framing-main/framing-names-v1", bindings)
    profiles = copy.deepcopy(cm["profiles"])
    judges = copy.deepcopy(cm["judges"])
    records = []
    targets = [
        c for c in audit["cases"] if c["scope"] == "main" and c["stage"] == "generation"
    ]
    if len(targets) != 80 or any(c["mode"] != "A" for c in targets):
        raise ValueError("Replacement scope must be exactly 80 Mode A assignments")
    for c in targets:
        verify_hashes(root, c["evidence"])
        aid = c["assignment_id"]
        if c["study"] == "compact_replication":
            row = index[aid]
            parent = root / "runs/replication/compact-models-v1"
            manifest_path = parent / "manifest.json"
            if (root / c["attempt_path"] / "generation/raw.json").exists():
                raise ValueError("Original response is no longer missing")
        else:
            attempt = root / c["attempt_path"]
            block = next(p for p in attempt.parents if p.parent.name == "blocks")
            parent, manifest_path = block.parent.parent, block / "manifest.json"
            frozen = read(manifest_path)
            if c["study"] == "reward_conflict":
                frozen, _ = conflict.block(root, conflict_manifest, block.name)
            elif c["study"] == "three_style":
                frozen, _ = three.block(root, three_manifest, block.name)
            else:
                # This study used an audited runner/pacing amendment; preserve its
                # exact original block rather than re-rendering any prompt.
                top = read(parent / "manifest.json")
                entry = next(b for b in top["blocks"] if b["name"] == block.name)
                if digest(json_text(frozen)) != entry["manifest_sha256"]:
                    raise ValueError("Framing block binding changed")
            row = next(r for r in frozen["assignments"] if r["assignment_id"] == aid)
            design = "reversal" if c["study"] == "three_style" else "mapping"
            if frozen["judge"]["system_prompt"] != judges[design]["system_prompt"]:
                raise ValueError("Current approved judge rubric differs")
            row = {**row, "profile": "gpt-oss-120b", "study": design}
            spec = {
                "provider": "fireworks",
                "base_url": "https://api.fireworks.ai/inference/v1",
                "api_key_env": "FIREWORKS_API_KEY",
                "input_rate": 0.15,
                "output_rate": 0.6,
                **frozen["profiles"]["gpt-oss-120b"],
            }
            if "gpt-oss-120b" in profiles and profiles["gpt-oss-120b"] != spec:
                raise ValueError("Frozen GPT settings differ")
            profiles["gpt-oss-120b"] = spec
            if (attempt / "raw.json").exists():
                raise ValueError("Original response is no longer missing")
            if frozen["oracle_image"] != cm["oracle_image"]:
                raise ValueError("Frozen sandbox images differ")
        bindings[str(manifest_path.relative_to(root))] = digest(
            manifest_path.read_bytes()
        )
        fresh = digest(json_text([RUN, c["study"], aid, "generation", 1]))
        records.append(
            {
                **copy.deepcopy(row),
                "assignment_id": fresh,
                "historical_assignment_id": aid,
                "origin_study": c["study"],
                "original_run": str(parent.relative_to(root)),
                "phase": "recovery",
                "replaces": aid,
                "recovery_attempt": 1,
            }
        )
    locked = set(read(root / "manifests/main_selection.json")["confirmation"])
    if (
        locked & {r["source_task_id"] for r in records}
        or len({r["assignment_id"] for r in records}) != 80
    ):
        raise ValueError("Locked task exposure or duplicate replacement")
    return (
        records,
        {"profiles": profiles, "judges": judges, "oracle_image": cm["oracle_image"]},
        bindings,
    )


def legacy_manifest(root: Path, run: str, bindings: dict[str, str]) -> dict[str, Any]:
    """Validate historical inputs without resuming superseded source-enumerating loaders.

    Later-added files are irrelevant to the frozen file list. The framing lock's
    sole change adds the next study's owner; this recovery uses the independently
    verified current compact lock and never resumes either historical controller.
    """
    directory = root / "runs" / run
    manifest = read(directory / "manifest.json")
    expected = manifest["code_sha256"]
    state_path = directory / "operations/recovery-state.json"
    state = read(state_path)
    reports = dict(manifest["verification_reports"])
    paths = [directory / "manifest.json", state_path]
    if state.get("operational_amendment"):
        amendment_path = directory / state["operational_amendment"]
        if digest(amendment_path.read_bytes()) != state["operational_amendment_sha256"]:
            raise ValueError("Historical amendment changed")
        amendment = read(amendment_path)
        if amendment["manifest_sha256"] != digest(json_text(manifest)):
            raise ValueError("Historical amendment manifest changed")
        allowed = {
            "evals/inspect_reward_conflict.py",
            "scripts/run_reward_conflict.py",
            "tests/test_reward_conflict.py",
            "tests/integration/test_reward_conflict_ownership.py",
        }
        amended = amendment["code_sha256"]
        if set(amended) != set(expected) or any(
            amended[k] != v and k not in allowed for k, v in expected.items()
        ):
            raise ValueError("Historical science changed")
        expected = amended
        reports.update(amendment["verification_reports"])
        paths.append(amendment_path)
    for name, value in expected.items():
        current = root / name
        if digest(current.read_bytes()) != value:
            old = directory / "source-snapshot" / name
            if (
                run != "framing-main/framing-names-v1"
                or name != "src/grader_power/collection_lock.py"
                or digest(old.read_bytes()) != value
            ):
                raise ValueError("Unreviewed historical source difference: " + name)
            if (
                current.read_text().replace('        "reward-conflict-v1",\n', "", 1)
                != old.read_text()
            ):
                raise ValueError("Historical lock differs beyond added owner")
            paths.append(old)
        paths.append(current)
    verify_hashes(root, reports)
    bindings.update(reports)
    bindings.update({str(p.relative_to(root)): digest(p.read_bytes()) for p in paths})
    return manifest


def pacing_sources(root: Path) -> dict[str, str]:
    """Preserve observed caps, cooldowns and allowances from the last completed lanes."""
    simple = "runs/replication/simple-models-v1/operations/pacing/"
    return {
        **{
            p: simple + p + "/rate-state.json"
            for p in ("sonnet", "deepseek", "qwen", "glm")
        },
        "gpt-oss-120b": "runs/three-style/three-style-v1/operations/pacing/rate-state.json",
        "account": simple + "account-rpm.json",
    }


def seed_pacing(root: Path, manifest: dict[str, Any]) -> None:
    """Copy rate evidence under the collection lock; preserve all window reservations."""
    for profile, source in manifest["pacing_sources"].items():
        target = (
            base(root)
            / "operations/pacing"
            / (
                "account-rpm.json"
                if profile == "account"
                else profile + "/rate-state.json"
            )
        )
        if not target.exists():
            write_json(target, read(root / source))


def prepare(root: Path, reports: list[Path]) -> dict[str, Any]:
    """Freeze approval, evidence, settings, implementation and passed test reports."""
    audit = read(root / AUDIT)
    verify_hashes(root, audit["source_hashes"])
    verify_hashes(root, audit["verification"]["parser_sources"])
    repairs = repair_catalog(root, audit)
    records, configuration, bindings = build_replacements(root, audit)
    for c in audit["cases"]:
        if c["scope"] == "main":
            bindings.update(c["evidence"])
    bindings.update(audit["source_hashes"])
    bindings[AUDIT] = digest((root / AUDIT).read_bytes())
    pacing = pacing_sources(root)
    bindings.update({p: digest((root / p).read_bytes()) for p in pacing.values()})
    sources = {
        **compact.sources(root),
        **{name: digest((root / name).read_bytes()) for name in NEW_SOURCES},
    }
    verified = {}
    import xml.etree.ElementTree as ET

    for report in reports:
        cases = ET.parse(report).getroot().findall(".//testcase")
        if not cases or any(
            c.find(t) is not None
            for c in cases
            for t in ("failure", "error", "skipped")
        ):
            raise ValueError("Recovery verification report did not pass")
        verified[str(report.relative_to(root))] = digest(report.read_bytes())
    if len(verified) < 2:
        raise ValueError("Focused and live sandbox reports required")
    manifest = {
        "run": RUN,
        "authorization": AUTHORIZATION,
        "cap_usd": CAP,
        **configuration,
        "records": records,
        "repairs": repairs,
        "source_hashes": sources,
        "retained_evidence": bindings,
        "verification_reports": verified,
        "pacing_sources": pacing,
        "canonical_analyses": audit["source_hashes"],
        "operational_change": "Separate asynchronous recovery pipeline; judge connection pool is 64 as in current compact collector. Scientific generation settings, prompts, rubrics and limits are unchanged. Historical source lists are verified explicitly: reward-conflict predates two added three-style prompt files; framing's sole lock change adds the reward-conflict owner. Current compact lock is separately verified. All rate evidence and cooldowns are copied, not reset.",
    }
    original.freeze(base(root) / "manifest.json", manifest)
    return manifest


def verify(root: Path) -> dict[str, Any]:
    """Recheck every recovery binding before any transformation or paid launch."""
    manifest = read(base(root) / "manifest.json")
    if manifest["cap_usd"] != CAP or manifest["run"] != RUN:
        raise ValueError("Recovery scope/ceiling changed")
    for key in ("source_hashes", "retained_evidence", "verification_reports"):
        verify_hashes(root, manifest[key])
    return manifest


def apply_repairs(root: Path) -> None:
    """Save all approved repairs as immutable overlays, leaving originals untouched."""
    manifest = verify(root)
    for repair in manifest["repairs"]:
        original.freeze(
            base(root)
            / "mechanical"
            / repair["origin_study"]
            / (repair["assignment_id"] + ".json"),
            {
                **repair,
                "authorization_sha256": digest(json_text(manifest)),
                "method": "user-approved mechanical normalization; no API call",
            },
        )
    analyze(root, "mechanical")


def measure(directory: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Read recovery statuses without assuming a response or successful stage."""
    path = directory / "requests" / record["assignment_id"]
    gen = pilot.read_optional(path / "generation/result.json") or {}
    score = pilot.read_optional(path / "scoring.json") or {}
    judge = pilot.read_optional(path / "judge/result.json") or {}
    return {
        "started": (path / "generation/started.json").exists(),
        "attempt_complete": bool(gen),
        "raw_saved": (path / "generation/raw.json").exists(),
        "scored": bool(score),
        "full_correct": score.get("correctness", {}).get("full_correct"),
        "judge_started": (path / "judge/started.json").exists(),
        "judge_result": bool(judge),
        "label": judge.get("label"),
        "judge_status": judge.get("status"),
        "generation_status": gen.get("status"),
    }


def analyze(root: Path, version: str) -> dict[str, Any]:
    """Write full-denominator supplemental rows and explicit repair/replacement lineage."""
    manifest = verify(root)
    audit = read(root / AUDIT)
    source_by_study = {
        name: next(
            p
            for p in manifest["canonical_analyses"]
            if "/"
            + {
                "original_main": "main-32-v1",
                "extension": "gpt-extension-main-v1",
                "framing_names": "framing-names-v1",
                "reward_conflict": "reward-conflict-v1",
                "three_style": "three-style-v1",
                "reversal": "reversal-v1",
                "compact_replication": "compact-models-v1",
                "simple_replication": "simple-models-v1",
            }[name]
            + "/"
            in p
        )
        for name in audit["counts"]
    }
    repairs = {(r["origin_study"], r["assignment_id"]): r for r in manifest["repairs"]}
    replacements = {(r["origin_study"], r["replaces"]): r for r in manifest["records"]}
    summary = {}
    for study, source in source_by_study.items():
        rows = read(root / source)
        lineage = []
        for row in rows:
            key = study, row["assignment_id"]
            if key in repairs:
                r = repairs[key]
                path = (
                    base(root) / "mechanical" / study / (r["assignment_id"] + ".json")
                )
                if path.exists():
                    row.update(
                        label=r["label"],
                        judge_status=r["status"],
                        judge_result=True,
                        recovery_method="mechanical_normalization",
                    )
                    lineage.append(
                        {
                            "assignment_id": row["assignment_id"],
                            "method": "mechanical_normalization",
                            "evidence": str(path.relative_to(root)),
                        }
                    )
            if version != "mechanical" and key in replacements:
                r = replacements[key]
                outcome = measure(base(root), r)
                if outcome["attempt_complete"]:
                    row.update(
                        outcome,
                        recovery_method="new_generation_attempt",
                        recovery_assignment_id=r["assignment_id"],
                    )
                    lineage.append(
                        {
                            "assignment_id": row["assignment_id"],
                            "replacement_id": r["assignment_id"],
                            "original_remains_missing": True,
                            "method": "new_generation_attempt",
                        }
                    )
        summary[study] = {
            "assigned": len(rows),
            "original_unresolved": audit["counts"][study]["unresolved"],
            "remaining_unresolved": sum(r["label"] is None for r in rows),
            "mechanical_repairs": sum(
                x["method"] == "mechanical_normalization" for x in lineage
            ),
            "replacement_attempts": sum(
                x["method"] == "new_generation_attempt" for x in lineage
            ),
            "labels": dict(Counter(str(r["label"]) for r in rows)),
        }
        destination = base(root) / "analysis" / version / study
        original.freeze(destination / "measured-rows.json", rows)
        original.freeze(destination / "lineage.json", lineage)
    original.freeze(base(root) / "analysis" / version / "summary.json", summary)
    return summary


class RecoveryPipeline(compact.Pipeline):
    """Reuse verified request/scoring/judging stages with an independent $5 ledger."""

    def __init__(self, root: Path, manifest: dict[str, Any]) -> None:
        self.root, self.manifest, self.records, self.phase = (
            root,
            manifest,
            manifest["records"],
            "recovery",
        )
        self.base = base(root)
        self.operations = self.base / "operations"
        self.ledger = RequestLedger(
            self.base / "spending.sqlite3",
            cap=nanodollars(CAP),
            scope=digest(json_text(manifest)),
        )
        self.stop, self.finishing = asyncio.Event(), asyncio.Event()
        self.reason: str | None = None
        self.tasks_by_id = {t.visible.source_task_id: t for t in main_tasks(root)}
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
        self.measurements = {
            r["assignment_id"]: measure(self.base, r) for r in self.records
        }
        self.queues: dict[str, asyncio.Queue[Any]] = {
            k: asyncio.Queue() for k in manifest["profiles"]
        }
        self.scoring: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.judging: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.account = AccountRPM(self.operations / "pacing")
        self.models: dict[str, Any] = {}
        self.keys: set[str] = set()
        self.active: Counter[str] = Counter()
        self.started_at = time.time()
        self.targets = {k: 4 for k in (*manifest["profiles"], "judge")}
        self.completions: list[dict[str, Any]] = []

    def check(self) -> None:
        """Preserve all human stops and stop admission after any local fatal error."""
        super().check()
        if (self.operations / "controller-stop.json").exists():
            raise RuntimeError("Recovery stop requires review")
        for r in self.records:
            for path in (
                self.root / r["original_run"] / "STOP",
                self.root / r["original_run"] / "operations/STOP",
            ):
                if path.exists():
                    self.pause("Original-study human STOP")
                    raise RuntimeError("Original-study human STOP")

    def event(
        self, r: dict[str, Any], stage: str, failed: bool, error: str | None = None
    ) -> None:
        """Preserve every failure and the same five-failed-stages-in-48 safeguard."""
        key = r["assignment_id"], stage
        if key not in self.events_seen:
            e = {
                "identity": key[0],
                "stage": stage,
                "failed": failed,
                "error_type": error,
                "at": time.time(),
            }
            self.history.append(e)
            self.events_seen.add(key)
            append_event(self.operations / "stage-events.jsonl", e)
            if sum(e["failed"] for e in self.history[-48:]) >= 5:
                self.pause("Five failed stages in last 48")
        self.measurements[r["assignment_id"]] = measure(self.base, r)

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Issue one fresh bounded generation; never replay a recovery intent."""
        if r["messages"] != [{"role": "user", "content": r["messages"][0]["content"]}]:
            raise ValueError("Unexpected frozen message structure")
        raw = await self.request(
            r, "generation", [ChatMessageUser(content=r["messages"][0]["content"])]
        )
        if raw is not None:
            await self.scoring.put(r)

    def status(self, state: str) -> dict[str, Any]:
        """Record exact saved/completed/resolved counts and nonoverlapping reservations."""
        rows = [measure(self.base, r) for r in self.records]
        report = {
            "state": state,
            "at": time.time(),
            "pid": os.getpid(),
            "assigned": len(rows),
            "counts": {
                k: sum(bool(r[k]) for r in rows)
                for k in (
                    "started",
                    "attempt_complete",
                    "raw_saved",
                    "scored",
                    "judge_started",
                    "judge_result",
                )
            },
            "resolved": sum(r["label"] is not None for r in rows),
            "recorded_and_outstanding": self.ledger.snapshot(),
            "active": dict(self.active),
            "recent_completions": [
                r for r in self.completions if r["at"] > time.time() - 60
            ],
            "reason": self.reason,
        }
        write_json(self.operations / "status.json", report)
        return report

    async def monitor(self) -> None:
        """Refresh durable progress and fail closed on integrity or disk pressure."""
        while True:
            await asyncio.sleep(15)
            try:
                self.check()
                if shutil.disk_usage(self.root).free < 3 * 1024**3:
                    self.pause("Disk headroom below 3 GiB")
                self.status("draining" if self.stop.is_set() else "running")
            except Exception as error:  # noqa: BLE001 -- stop admission and drain paid work
                self.pause(type(error).__name__ + " in recovery monitor")


def collect(root: Path) -> None:
    """Hold the shared lock, verify the sandbox and run only new recovery requests."""
    manifest = verify(root)
    isolate_collection_process()
    directory = base(root)
    owner = root / "runs/extension-pilot/coordination/owner.json"
    launch = (
        directory
        / "operations/launches"
        / (time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + str(os.getpid()))
    )
    if owner.exists():
        original.freeze(launch / "previous-owner.json", read(owner))
    with collection_lock(root, owner="compact-models-v1", run_id=RUN):
        require_pilot_verification(root, manifest["oracle_image"])
        containers = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if containers.stdout.strip():
            raise RuntimeError("Existing experiment sandbox containers require review")
        image = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                manifest["oracle_image"],
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if image.stdout.strip() != manifest["oracle_image"]:
            raise ValueError("Pinned Docker image changed")
        original.freeze(
            launch / "authorized-launch.json",
            {
                "at": time.time(),
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "authorization": AUTHORIZATION,
                "manifest_sha256": digest(json_text(manifest)),
                "sandbox_clean": True,
                "oracle_image": image.stdout.strip(),
            },
        )
        seed_pacing(root, manifest)
        pipeline = RecoveryPipeline(root, manifest)
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

            asyncio.run(execute())
        verify(root)
        cleanup = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "label=grader-power.oracle=true"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if cleanup.stdout.strip():
            raise RuntimeError("Post-run sandbox cleanup requires review")
        original.freeze(
            launch / "completed.json",
            {
                "at": time.time(),
                "sandbox_clean": True,
                "all_bindings_verified": True,
                "usage": pipeline.ledger.snapshot(),
            },
        )
        analyze(root, "replacements")


def main() -> None:
    """Expose preparation, mechanical-only repair, paid collection and verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("prepare", "mechanical", "collect", "verify")
    )
    parser.add_argument("--report", action="append", type=Path, default=[])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.action == "prepare":
        result = prepare(root, [p.resolve() for p in args.report])
        print(
            json.dumps(
                {
                    "repairs": len(result["repairs"]),
                    "replacements": len(result["records"]),
                    "cap_usd": CAP,
                }
            )
        )
    elif args.action == "mechanical":
        apply_repairs(root)
    elif args.action == "collect":
        collect(root)
    else:
        verify(root)
        print("Recovery source, evidence, and report bindings verified")


if __name__ == "__main__":
    main()
