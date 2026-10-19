"""One additive, same-limit recovery across all completed main experiments.

Original collectors, results and ledgers remain immutable. Fresh attempts have
new identities; the supplemental analysis always uses the designated attempt,
including failure, rather than selecting the better of two responses.
"""

import argparse
import asyncio
import copy
import json
import os
import shutil
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from dotenv import dotenv_values
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
)

from evals import inspect_agent as agent
from evals import inspect_compact_replication as compact
from evals import inspect_extension_pilot as extension
from evals import inspect_judge_recovery as previous
from evals import inspect_outcome_recovery as prior
from evals import inspect_style_preferences as syntax
from evals.inspect_mode_a import digest, json_text
from grader_power import preference_judge
from grader_power.agent_sandbox import EpisodeLimits
from grader_power.code_extract import extract_code
from grader_power.datasets.oracle import evaluate_solution
from grader_power.inference import AccessProfile
from grader_power.main_tasks import main_tasks
from grader_power.pacing import WAIT_ALLOWANCE, WaitClock, append_event, write_json
from grader_power.preference_tasks import load_bundles
from grader_power.replication_spending import RequestLedger
from grader_power.replication_transport import (
    ModelPacer,
    RequestContext,
    concurrent_transport,
    request_scope,
)
from grader_power.spending import nanodollars, usage_cost

RUN = "all-experiments-v1"
CAP = 30
TARGET = 15
RELATIVE_RUNS = {
    **previous.RELATIVE_RUNS,
    "syntax": "style-preferences/syntax-preferences-v1",
    "observation": "reviewer-observation/reviewer-observation-v1",
    "strong": "reviewer-observation/strong-preferences-v1",
}
STUB = "dba3ee19e60dbe8c571ec67a5e513021cbf8d49020f8dc5d2921b0801e52b781"
NEW_SOURCES = (
    "evals/inspect_all_recovery.py",
    "scripts/recover_all_experiments.py",
    "tests/test_all_recovery.py",
)
read = prior.read
freeze = prior.original.freeze


def base(root: Path) -> Path:
    """Locate the independent recovery cohort, never an original run directory."""
    return root / "runs/recovery" / RUN


def catalog(root: Path) -> dict[str, Any]:
    """Reconstruct exact frozen inputs for 204 generations and 96 saved judgments."""
    directory = base(root)
    generation = read(directory / "operations/generation-audit.json")
    gaps = read(directory / "operations/initial-main-gap-audit.json")
    evidence: dict[str, str] = dict(generation["sources"])
    for filename in (
        "generation-audit.json",
        "initial-main-gap-audit.json",
        "approval.json",
        "mechanical-completion.json",
        "mechanical-source-proof.json",
    ):
        path = directory / "operations" / filename
        evidence[str(path.relative_to(root))] = digest(path.read_bytes())
    prior.verify_hashes(root, generation["sources"])
    manifests: dict[str, Any] = {}
    assignments: dict[str, dict[str, Any]] = {}
    profiles: dict[str, Any] = {}
    judges: dict[str, Any] = {}
    blocks: dict[str, Path] = {}
    images: set[str] = set()
    records: list[dict[str, Any]] = []

    def bind(path: Path) -> None:
        evidence[str(path.relative_to(root))] = digest(path.read_bytes())

    for study in (
        "compact_replication",
        "simple_replication",
        "syntax",
        "observation",
        "strong",
    ):
        parent = root / "runs" / RELATIVE_RUNS[study]
        manifest = read(parent / "manifest.json")
        bind(parent / "manifest.json")
        bind(parent / "assignments.json")
        if (
            digest((parent / "assignments.json").read_bytes())
            != manifest["assignments_sha256"]
        ):
            raise ValueError("Original assignment binding changed")
        manifests[study] = manifest
        assignments[study] = {
            r["assignment_id"]: r for r in read(parent / "assignments.json")
        }
        images.add(manifest["oracle_image"])
        for name, spec in manifest["profiles"].items():
            if name in profiles:
                old = profiles[name]
                if any(
                    old[k] != spec[k]
                    for k in (
                        "model",
                        "provider",
                        "generation",
                        "base_url",
                        "api_key_env",
                    )
                ):
                    raise ValueError("Generation settings differ across source studies")
                # Pricing is conservative accounting, not a model parameter.
                for k in ("input_rate", "output_rate"):
                    old[k] = max(old[k], spec[k])
            else:
                profiles[name] = copy.deepcopy(spec)
        current_judges = copy.deepcopy(manifest["judges"])
        if study == "syntax":
            revision = syntax.read_revision(parent)
            if revision is None:
                raise ValueError("Expected approved naming revision")
            current_judges["naming"]["system_prompt"] = revision["naming_judge_prompt"]
            revision_path = parent / syntax.REVISION_MANIFEST
            bind(revision_path)
        for design, judge in current_judges.items():
            judges[study + ":" + design] = judge
    # The current judge price is the conservative higher recorded price.
    judge_profile = copy.deepcopy(manifests["syntax"]["judge_profile"])
    profiles["sonnet"].update(
        input_rate=judge_profile["input_rate"], output_rate=judge_profile["output_rate"]
    )
    extension_root = root / "runs" / RELATIVE_RUNS["extension"]
    top = read(extension_root / "manifest.json")
    bind(extension_root / "manifest.json")
    for item in top["blocks"]:
        path = extension_root / "blocks" / item["name"] / "manifest.json"
        frozen = read(path)
        if digest(json_text(frozen)) != item["manifest_sha256"]:
            raise ValueError("Extension block changed")
        for r in frozen["assignments"]:
            blocks[r["assignment_id"]] = path

    def fresh(case: dict[str, Any], kind: str) -> dict[str, Any]:
        study, aid = case["study"], case["assignment_id"]
        bound = case.get("bindings", case.get("evidence", {}))
        prior.verify_hashes(root, bound)
        evidence.update(bound)
        if study in assignments:
            old = copy.deepcopy(assignments[study][aid])
            design = old["study"]
            judge_key = study + ":" + design
            if study in ("simple_replication", "observation", "strong"):
                design = "mapping"
        elif study == "extension":
            block_path = blocks[aid]
            block = read(block_path)
            bind(block_path)
            old = copy.deepcopy(
                next(r for r in block["assignments"] if r["assignment_id"] == aid)
            )
            old.update(profile="gpt-oss-120b")
            design, judge_key = "mapping", "extension:mapping"
            judges[judge_key] = copy.deepcopy(block["judge"])
            images.add(block["oracle_image"])
            old["agent_profile"] = block["profiles"]["gpt-oss-120b"]
        elif study == "three_style":
            old = copy.deepcopy(case.get("row", {}))
            old.update(source_task_id=case["task"], profile=case["profile"], mode="A")
            design, judge_key = "reversal", "three_style:reversal"
            last = read(prior.base(root) / "manifest.json")
            bind(prior.base(root) / "manifest.json")
            judges[judge_key] = copy.deepcopy(last["judges"]["reversal"])
        else:
            raise ValueError("Unexpected recovery source study")
        rid = digest(json_text([RUN, study, aid, kind, 1]))
        return {
            **old,
            "assignment_id": rid,
            "replaces": aid,
            "origin_study": study,
            "original_run": "runs/" + RELATIVE_RUNS[study],
            "phase": "recovery",
            "recovery_kind": kind,
            "recovery_attempt": 1,
            "study": judge_key,
            "design": design,
            "category": case["category"],
            "original_record": old,
        }

    generation_keys = set()
    for case in generation["cases"]:
        r = fresh(case, "generation")
        r["source_raw"] = case["raw_path"]
        records.append(r)
        generation_keys.add((case["study"], case["assignment_id"]))
    tasks = {t.visible.source_task_id: t for t in main_tasks(root)}
    for case in [*gaps["new_main_cases"], *gaps["old_remaining"]]:
        key = case["study"], case["assignment_id"]
        if key in generation_keys or case["assignment_id"] == STUB:
            continue
        if any(
            x["study"] == key[0] and x["assignment_id"] == key[1]
            for x in gaps["free_repairs"]
        ):
            continue
        if case["category"] == "mechanical_repair":
            continue
        r = fresh(case, "judge")
        packet_path = root / case["judge_path"] / "packet.json"
        if case["study"] in assignments:
            effective = case.get("effective_attempt_id", case["assignment_id"])
            source = (
                root
                / r["original_run"]
                / "requests"
                / effective
                / "generation/raw.json"
            )
        else:
            candidates = [
                p
                for p in case["evidence"]
                if p.endswith("/raw.json") and "/judge/" not in p
            ]
            if len(candidates) != 1:
                raise ValueError("Ambiguous saved generation")
            source = root / candidates[0]
        raw = read(source)
        packet = (
            syntax.packet_from_output(raw, {**r, "study": r["design"]})
            if r["origin_study"] == "syntax"
            else prior.original.packet_from_output(raw, tasks[r["source_task_id"]])
        )
        if packet is None:
            raise ValueError("Judgment has no unchanged parseable source")
        data = packet.model_dump(mode="json")
        if packet_path.exists():
            if read(packet_path) != data:
                raise ValueError(
                    "Packet differs from independently reconstructed source"
                )
            bind(packet_path)
        bind(source)
        r.update(
            packet=data,
            source_raw=str(source.relative_to(root)),
            source_raw_sha256=digest(json_text(raw)),
        )
        # The previous three-style failure used range-only serialization; preserve it.
        r["line_ranges"] = r["origin_study"] == "three_style"
        if r["line_ranges"]:
            judges[r["study"]]["system_prompt"] = previous.range_prompt(
                judges[r["study"]]["system_prompt"]
            )
        records.append(r)
    if Counter(r["recovery_kind"] for r in records) != {"generation": 204, "judge": 96}:
        raise ValueError("Recovery scope differs from approved audit")
    if len(images) != 1 or len({r["assignment_id"] for r in records}) != 300:
        raise ValueError("Sandbox identity or recovery identities disagree")
    locked = set(read(root / "manifests/main_selection.json")["confirmation"])
    if locked & {r["source_task_id"] for r in records}:
        raise ValueError("Locked task must remain unexposed")
    return {
        "records": records,
        "profiles": profiles,
        "judges": judges,
        "judge_profile": judge_profile,
        "oracle_image": images.pop(),
        "retained_evidence": evidence,
        "analysis_sources": generation["sources"],
    }


def prepare(root: Path, reports: list[Path]) -> dict[str, Any]:
    """Freeze the independently approved cohort with passed tests and source snapshots."""
    data = catalog(root)
    verified = {}
    for path in reports:
        cases = ET.parse(path).getroot().findall(".//testcase")
        if not cases or any(
            c.find(t) is not None
            for c in cases
            for t in ("error", "failure", "skipped")
        ):
            raise ValueError("Verification report did not pass completely")
        verified[str(path.relative_to(root))] = digest(path.read_bytes())
    if len(verified) < 3:
        raise ValueError(
            "Offline, oracle Docker and agent Docker verification required"
        )
    sources = {
        **compact.sources(root),
        **{n: digest((root / n).read_bytes()) for n in NEW_SOURCES},
    }
    sources.update(
        {
            str(p.relative_to(root)): digest(p.read_bytes())
            for p in (root / "evals").glob("*.py")
        }
    )
    manifest = {
        "run": RUN,
        "cap_usd": CAP,
        "target_usd": TARGET,
        **data,
        "authorization": read(base(root) / "operations/approval.json"),
        "source_hashes": sources,
        "verification_reports": verified,
        "policy": "One fresh same-limit attempt per audited incomplete main cell; fixed designated outcome even if failed. Retired pilots and locked tasks excluded. One fresh blinded judgment per gap; genuine residual ambiguity retained.",
    }
    freeze(base(root) / "manifest.json", manifest)
    for name, sha in sources.items():
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError("Source changed while freezing")
        target = base(root) / "source-snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.read_bytes() != payload:
            raise ValueError("Source snapshot conflict")
        target.write_bytes(payload)
    return manifest


def verify(root: Path) -> dict[str, Any]:
    """Fail closed on changed inputs, instructions, sources, ceilings or evidence."""
    manifest = read(base(root) / "manifest.json")
    if (
        manifest["run"] != RUN
        or manifest["cap_usd"] != 30
        or len(manifest["records"]) != 300
    ):
        raise ValueError("Approved recovery scope changed")
    sources = dict(manifest["source_hashes"])
    amendment_path = (
        base(root) / "operations/recoveries/agent-context-v1/amendment.json"
    )
    if amendment_path.exists():
        amendment = read(amendment_path)
        if amendment["manifest_sha256"] != digest(json_text(manifest)) or set(
            amendment["sources"]
        ) != {"evals/inspect_all_recovery.py", "tests/test_all_recovery.py"}:
            raise ValueError("Operational amendment scope changed")
        for name, sha in amendment["sources"].items():
            if (
                digest((base(root) / "source-snapshot" / name).read_bytes())
                != sources[name]
            ):
                raise ValueError("Original recovery source snapshot changed")
            sources[name] = sha
        audit_path = base(root) / amendment["audit"]
        if digest(audit_path.read_bytes()) != amendment["audit_sha256"]:
            raise ValueError("Operational failure audit changed")
        prior.verify_hashes(base(root), read(audit_path)["retained_artifacts"])
        prior.verify_hashes(root, amendment["reports"])
    policy_path = (
        base(root) / "operations/recoveries/terminal-outcomes-v1/amendment.json"
    )
    if policy_path.exists():
        policy = read(policy_path)
        if policy["manifest_sha256"] != digest(json_text(manifest)) or set(
            policy["sources"]
        ) != {"evals/inspect_all_recovery.py", "tests/test_all_recovery.py"}:
            raise ValueError("Approved terminal-outcome amendment scope changed")
        for name, sha in policy["sources"].items():
            if (
                digest((base(root) / policy["old_sources"] / name).read_bytes())
                != sources[name]
            ):
                raise ValueError("Pre-policy source snapshot changed")
            sources[name] = sha
        prior.verify_hashes(root, policy["evidence"])
        prior.verify_hashes(root, policy["reports"])
    prior.verify_hashes(root, sources)
    for key in ("verification_reports", "retained_evidence"):
        prior.verify_hashes(root, manifest[key])
    return manifest


def outcome(directory: Path, r: dict[str, Any]) -> dict[str, Any]:
    """Distinguish fresh generations from references and completed attempts from saves."""
    p = directory / "requests" / r["assignment_id"]
    gen = prior.pilot.read_optional(p / "generation/result.json") or {}
    score = prior.pilot.read_optional(p / "scoring.json") or {}
    judge = effective_judgment(p) or {}
    episode = prior.pilot.read_optional(p / "episode.json") or {}
    return {
        "started": (p / "generation/started.json").exists(),
        "attempt_complete": bool(gen),
        "raw_saved": (p / "generation/raw.json").exists(),
        "scored": bool(score),
        "full_correct": score.get("correctness", {}).get("full_correct"),
        "generation_status": gen.get("status"),
        "episode_status": episode.get("status"),
        "judge_started": (p / "judge/started.json").exists(),
        "judge_raw_saved": (p / "judge/raw.json").exists(),
        "judge_result": bool(judge),
        "judge_status": judge.get("status"),
        "label": judge.get("label"),
        "audit_label": judge.get("audit_label"),
        "audit_agrees": judge.get("audit_agrees"),
    }


def repair_reversed_ranges(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    """Swap only reversed quote-evidence endpoints and require independent agreement."""

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate judgment key")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    repaired = copy.deepcopy(value)
    changed = 0
    for evidence in repaired["evidence"]:
        if (
            type(evidence["start_line"]) is not int
            or type(evidence["end_line"]) is not int
        ):
            raise ValueError("Evidence endpoints are not integers")
        if evidence["start_line"] > evidence["end_line"]:
            evidence["start_line"], evidence["end_line"] = (
                evidence["end_line"],
                evidence["start_line"],
            )
            changed += 1
    if not changed:
        raise ValueError("No reversed ranges to repair")
    parsed = syntax.parse_packet("quotes", json_text(repaired), packet)
    audit = preference_judge.audit_style(
        preference_judge.JudgeInput.model_validate(packet)
    )
    if parsed.status != "classified" or parsed.label != audit.label:
        raise ValueError("Repaired strict judgment and independent audit disagree")
    return {
        "corrected": repaired,
        "judgment": parsed.model_dump(mode="json"),
        "label": parsed.label,
        "ranges_swapped": changed,
        "audit_label": audit.label,
    }


def effective_judgment(path: Path) -> dict[str, Any] | None:
    """Read an immutable judgment, applying only an explicitly bound free repair."""
    result = prior.pilot.read_optional(path / "judge/result.json")
    repair_path = path / "judge/reversed-ranges-repair.json"
    if not repair_path.exists():
        return result
    if result is None:
        raise ValueError("Repair lacks its original judgment")
    repair = read(repair_path)
    for filename, sha in repair["bindings"].items():
        if digest((path / "judge" / filename).read_bytes()) != sha:
            raise ValueError("Free repair source changed")
    approval_path = (
        path.parents[1] / "operations/recoveries/terminal-outcomes-v1/approval.json"
    )
    if (
        digest(approval_path.read_bytes()) != repair["approval_sha256"]
        or path.name not in read(approval_path)["range_repair_ids"]
    ):
        raise ValueError("Free repair is outside exact human approval")
    raw = ModelOutput.model_validate(read(path / "judge/raw.json"))
    checked = repair_reversed_ranges(raw.completion, read(path / "judge/packet.json"))
    if checked != repair["repair"]:
        raise ValueError("Free repair no longer validates")
    return {
        **result,
        "status": "classified",
        "label": checked["label"],
        "judgment": checked["judgment"],
        "mechanical_repair": "reversed_ranges",
        "audit_agrees": True,
    }


def expected_token_cap(directory: Path, record: dict[str, Any], stage: str) -> bool:
    """Recognize saved original-limit truncation, never transport or sandbox errors."""
    if (
        stage != "generation"
        or record.get("mode") != "A"
        or record.get("recovery_kind") != "generation"
    ):
        return False
    path = directory / "requests" / record["assignment_id"] / "generation"
    result = prior.pilot.read_optional(path / "result.json") or {}
    raw = prior.pilot.read_optional(path / "raw.json") or {}
    choices = raw.get("choices", [])
    return (
        result.get("status") == "incomplete"
        and bool(choices)
        and choices[0].get("stop_reason") == "max_tokens"
    )


def parse_judgment(r: dict[str, Any], text: str, packet: dict[str, Any]) -> Any:
    """Use the original strict schema and semantic rubric for each source study."""
    if r["origin_study"] == "syntax":
        return syntax.parse_packet(r["design"], text, packet)
    if r.get("line_ranges"):
        return previous.parse_ranges(text, packet, r["design"])
    return compact.parse_packet(r["design"], text, packet)


class Pipeline(compact.Pipeline):
    """Share the existing bounded transport; use only this recovery's fresh intents."""

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
        _, self.datasets = load_bundles(root)
        events = self.operations / "stage-events.jsonl"
        self.history = (
            [json.loads(s) for s in events.read_text().splitlines()]
            if events.exists()
            else []
        )
        self.events_seen = {(e["identity"], e["stage"]) for e in self.history}
        self.measurements = {
            r["assignment_id"]: outcome(self.base, r) for r in self.records
        }
        self.queues: dict[str, asyncio.Queue[Any]] = {
            k: asyncio.Queue() for k in manifest["profiles"]
        }
        self.scoring: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.judging: asyncio.Queue[Any] = asyncio.Queue(maxsize=64)
        self.account = prior.AccountRPM(self.operations / "pacing")
        self.models: dict[str, Any] = {}
        self.keys: set[str] = set()
        self.active: Counter[str] = Counter()
        self.started_at = time.time()
        self.targets = {k: 8 for k in (*manifest["profiles"], "judge")}
        self.targets["judge"] = 16
        self.completions: list[dict[str, Any]] = []

    def check(self) -> None:
        """Honor human stops in every source run as well as current recovery stops."""
        super().check()
        if (self.operations / "controller-stop.json").exists():
            raise RuntimeError("Recovery stop requires review")
        for run in {r["original_run"] for r in self.records}:
            if any((self.root / run / p).exists() for p in ("STOP", "operations/STOP")):
                self.pause("Original-study human STOP")
                raise RuntimeError("Original-study human STOP")

    def event(
        self, r: dict[str, Any], stage: str, failed: bool, error: str | None = None
    ) -> None:
        """Retain every stage and the original five failures in 48 stopping rule."""
        key = r["assignment_id"], stage
        if key not in self.events_seen:
            event = {
                "identity": key[0],
                "stage": stage,
                "failed": failed,
                "operational_failed": failed
                and not expected_token_cap(self.base, r, stage),
                "error_type": error,
                "profile": r["profile"],
                "at": time.time(),
            }
            self.history.append(event)
            self.events_seen.add(key)
            append_event(self.operations / "stage-events.jsonl", event)
            state = (
                prior.pilot.read_optional(self.operations / "recovery-state.json") or {}
            )
            window = self.history[state.get("error_window_origin", 0) :][-48:]
            if sum(e.get("operational_failed", e["failed"]) for e in window) >= 5:
                self.pause("Five failed stages in last 48")
        self.measurements[r["assignment_id"]] = outcome(self.base, r)

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Permit exactly one fresh Mode A intent, then score its saved response."""
        if r["recovery_kind"] != "generation" or r["mode"] != "A":
            raise ValueError("Misrouted generation")
        if r["messages"] != [{"role": "user", "content": r["messages"][0]["content"]}]:
            raise ValueError("Frozen message structure changed")
        raw = await self.request(
            r, "generation", [ChatMessageUser(content=r["messages"][0]["content"])]
        )
        if raw is not None:
            await self.scoring.put(r)

    async def scoring_stage(self, r: dict[str, Any]) -> None:
        """Run generated code only through the unchanged pinned Docker oracle."""
        if r["recovery_kind"] != "generation" or r["mode"] != "A":
            raise ValueError(
                "Saved judgments/agent episodes cannot enter Mode A scoring"
            )
        if r["origin_study"] != "syntax":
            await super().scoring_stage(r)
            return
        p = self.base / "requests" / r["assignment_id"]
        raw = read(p / "generation/raw.json")
        if not (p / "scoring-started.json").exists():
            freeze(
                p / "scoring-started.json",
                {"at": time.time(), "raw_sha256": digest(json_text(raw))},
            )
            extraction = extract_code(
                ModelOutput.model_validate(raw).completion, r["entry_point"]
            )
            correctness: dict[str, Any] = {
                "status": "not_evaluated",
                "full_correct": False,
                "reason": "No parseable code",
            }
            if extraction.code is not None and extraction.status == "ok":
                result = await asyncio.to_thread(
                    evaluate_solution,
                    self.datasets[r["dataset"]],
                    r["dataset"],
                    r["source_task_id"],
                    extraction.code,
                    mode="full",
                    image=self.manifest["oracle_image"],
                )
                correctness = {**asdict(result), "full_correct": result.full_correct}
            freeze(
                p / "scoring.json",
                {
                    "extraction": asdict(extraction),
                    "correctness": correctness,
                    "source_raw_sha256": digest(json_text(raw)),
                },
            )
        score = prior.pilot.read_optional(p / "scoring.json")
        if score is None or compact.scoring_infrastructure_issue(score):
            self.event(r, "scoring", True, "SandboxInfrastructure")
            self.pause("Sandbox scoring requires review")
            return
        self.event(r, "scoring", False)
        await self.judging.put(r)

    async def judging_stage(self, r: dict[str, Any]) -> None:
        """Issue one blinded call; never feed prior failures or treatment/model identity."""
        p = self.base / "requests" / r["assignment_id"]
        if (p / "judge/started.json").exists() or (p / "judge/result.json").exists():
            return
        if r["recovery_kind"] == "judge":
            data = r["packet"]
            source_sha = r["source_raw_sha256"]
        else:
            raw_path = p / "generation/raw.json"
            if not raw_path.exists():
                return
            raw = read(raw_path)
            source_sha = digest(json_text(raw))
            output = ModelOutput.model_validate(raw)
            if r["mode"] == "B":
                packet = extension.packet_from_final_file(
                    raw, self.tasks_by_id[r["source_task_id"]]
                )
                # Original Mode B scoring assesses the captured final file even
                # after an interrupted episode; episode status stays separate.
                complete = True
            elif r["origin_study"] == "syntax":
                packet = syntax.packet_from_output(raw, {**r, "study": r["design"]})
                complete = bool(
                    output.choices and output.choices[0].stop_reason == "stop"
                )
            else:
                packet = prior.original.packet_from_output(
                    raw, self.tasks_by_id[r["source_task_id"]]
                )
                complete = bool(
                    output.choices and output.choices[0].stop_reason == "stop"
                )
            if packet is None or not complete:
                label = (
                    "invalid"
                    if r["origin_study"] == "syntax"
                    else "invalid_or_unparseable"
                )
                freeze(
                    p / "judge/result.json",
                    {"source_raw_sha256": source_sha, "status": label, "label": label},
                )
                self.event(r, "judge", False)
                return
            data = packet.model_dump(mode="json")
        freeze(p / "judge/packet.json", data)
        rendered = (
            syntax.render_packet(r["design"], data)
            if r["origin_study"] == "syntax"
            else compact.render_packet(r["design"], data)
        )
        response = await self.request(
            r,
            "judge",
            [
                ChatMessageSystem(
                    content=self.manifest["judges"][r["study"]]["system_prompt"]
                ),
                ChatMessageUser(content=rendered),
            ],
        )
        result: dict[str, Any] = {
            "source_raw_sha256": source_sha,
            "packet_sha256": digest(json_text(data)),
            "status": "judge_error",
            "label": None,
        }
        try:
            if response is not None:
                output = ModelOutput.model_validate(response)
                if output.choices and output.choices[0].stop_reason == "stop":
                    parsed = parse_judgment(r, output.completion, data)
                    result.update(
                        status=parsed.status,
                        label=parsed.label,
                        judgment=parsed.model_dump(mode="json"),
                    )
        except (ValueError, TypeError) as error:
            result["error_type"] = type(error).__name__
        if r["origin_study"] == "syntax":
            try:
                audit = preference_judge.audit_style(
                    preference_judge.JudgeInput.model_validate(data)
                )
                freeze(p / "style-audit.json", audit.model_dump(mode="json"))
                result.update(
                    audit_label=audit.label,
                    audit_agrees=audit.label == result["label"]
                    if result["label"] is not None
                    else None,
                )
            except (ValueError, TypeError, SyntaxError):
                result["audit_label"] = None
        freeze(p / "judge/result.json", result)
        self.event(
            r,
            "judge",
            result["status"] not in ("classified", "abstain"),
            result.get("error_type"),
        )

    def status(self, state: str) -> dict[str, Any]:
        """Report counts by origin and keep unknown reservations distinct from charges."""
        by_group: dict[str, Any] = {}
        for r in self.records:
            result = outcome(self.base, r)
            key = r["origin_study"] + "/" + r["profile"] + "/" + r["recovery_kind"]
            group = by_group.setdefault(key, Counter())
            group["assigned"] += 1
            for field in (
                "started",
                "attempt_complete",
                "raw_saved",
                "scored",
                "judge_started",
                "judge_raw_saved",
                "judge_result",
            ):
                group[field] += bool(result[field])
            group["resolved"] += result["label"] is not None
            group["terminal_unresolved"] += (
                result["judge_result"] and result["label"] is None
            )
        report = {
            "at": time.time(),
            "pid": os.getpid(),
            "state": state,
            "reason": self.reason,
            "groups": by_group,
            "usage": self.ledger.snapshot(),
            "active": dict(self.active),
            "queues": {
                **{k: q.qsize() for k, q in self.queues.items()},
                "scoring": self.scoring.qsize(),
                "judge": self.judging.qsize(),
            },
            "last_minute": [e for e in self.completions if e["at"] > time.time() - 60],
            "pacing": {
                k: ModelPacer(self.operations / "pacing" / k, profile=k).utilization()
                for k in self.manifest["profiles"]
            },
        }
        write_json(self.operations / "status.json", report)
        return report

    async def monitor(self) -> None:
        """Check real progress, immutable inputs and disk headroom without paid probes."""
        while True:
            await asyncio.sleep(30)
            try:
                self.check()
                verify(self.root)
                if shutil.disk_usage(self.root).free < 3 * 1024**3:
                    self.pause("Disk headroom below 3 GiB")
                self.status("draining" if self.stop.is_set() else "running")
            except Exception as error:  # noqa: BLE001 -- stop admission and drain paid requests
                self.pause(type(error).__name__ + " in recovery monitor")

    async def run(self) -> None:
        """Run independent one-shot and judging work with bounded downstream queues."""
        self.initialize_models()
        workers = [
            asyncio.create_task(self.worker(q, "generation", i, profile))
            for profile, q in self.queues.items()
            for i in range(8)
        ]
        workers += [
            asyncio.create_task(self.worker(self.scoring, "scoring", i, "scoring"))
            for i in range(4)
        ]
        workers += [
            asyncio.create_task(self.worker(self.judging, "judge", i, "judge"))
            for i in range(16)
        ]
        watcher = asyncio.create_task(self.monitor())
        self.status("running")
        try:
            for r in self.records:
                if r["recovery_kind"] == "judge" or r["mode"] == "B":
                    await self.judging.put(r)
                else:
                    self.queues[r["profile"]].put_nowait(r)
            await asyncio.gather(*(q.join() for q in self.queues.values()))
            await self.scoring.join()
            await self.judging.join()
        finally:
            self.finishing.set()
            for q in self.queues.values():
                for _ in range(8):
                    q.put_nowait(None)
            for _ in range(4):
                await self.scoring.put(None)
            for _ in range(16):
                await self.judging.put(None)
            await asyncio.gather(*workers)
            watcher.cancel()
            try:
                await watcher
            except asyncio.CancelledError:
                pass
        self.status("paused" if self.stop.is_set() else "requests_complete")
        if self.stop.is_set():
            raise RuntimeError(str(self.reason))


def agent_bound(r: dict[str, Any]) -> int:
    """Reserve the complete original episode before creating a fresh workspace."""
    limits = r["episode_limits"]
    inputs = limits["max_model_input_bytes"] + 2048 * limits["max_model_turns"]
    outputs = r["agent_profile"]["generation"]["max_tokens"] * limits["max_model_turns"]
    return nanodollars((inputs * 0.3 + outputs * 0.6) / 1e6)


async def agent_dispatch(
    pipeline: Pipeline,
    record: dict[str, Any],
    clock: WaitClock,
    dispatch: Any,
    episode: agent.Episode,
    original: Any,
    inputs: Any,
    tools: Any,
    tool_choice: Any,
    config: Any,
) -> Any:
    """Give each distinct model turn its own non-replayable transport identity."""
    path = (
        pipeline.base
        / "requests"
        / record["assignment_id"]
        / "generation/dispatches"
        / f"{episode.calls + 1:04d}"
    )
    pacer = ModelPacer(
        pipeline.operations / "pacing/gpt-oss-120b", profile="gpt-oss-120b"
    )
    context = RequestContext(
        "gpt-oss-120b",
        "fireworks",
        record["agent_profile"]["model"].split("/", 1)[1],
        path,
        pipeline.operations,
        pacer,
        pipeline.account,
        clock,
        pipeline.check,
    )
    with request_scope(context):
        result = await dispatch(episode, original, inputs, tools, tool_choice, config)
    output = result[0] if isinstance(result, tuple) else result
    if not isinstance(output, ModelOutput):
        raise TypeError("Agent provider returned no ModelOutput")
    pacer.usage("fireworks", output.model_dump(mode="json"))
    return result


def collect_agents(pipeline: Pipeline) -> None:
    """Run each whole agent episode in a new sandbox with its original active limits.

    The tested agent adapter patches its provider instance and therefore runs one
    episode at a time. Paid one-shot/judge requests run concurrently afterward.
    """
    root, manifest, directory = pipeline.root, pipeline.manifest, pipeline.base
    for r in pipeline.records:
        if r["recovery_kind"] != "generation" or r["mode"] != "B":
            continue
        pipeline.check()
        p = directory / "requests" / r["assignment_id"]
        if (p / "generation/started.json").exists():
            continue
        verify(root)
        cohort = directory / "agent-cohorts" / r["assignment_id"]
        original_record = copy.deepcopy(r["original_record"])
        original_record.pop("agent_profile", None)
        original_record.pop("profile", None)
        original_record["assignment_id"] = r["assignment_id"]
        assigned = prior.pilot.Assignment(
            original_record, pipeline.tasks_by_id[r["source_task_id"]]
        )
        agent.validate_assignment(assigned)
        config = GenerateConfig.model_validate(r["agent_profile"]["generation"])
        frozen = {
            "purpose": "authorized_fresh_whole_episode_recovery",
            "assignments": [original_record],
            "oracle_image": manifest["oracle_image"],
            "profiles": {"gpt-oss-120b": r["agent_profile"]},
        }
        freeze(cohort / "manifest.json", frozen)
        (cohort / "gpt-oss-120b").mkdir(parents=True, exist_ok=True)
        identity = r["assignment_id"] + "/generation"
        bound = agent_bound(r)
        pipeline.ledger.reserve({identity: bound})
        freeze(
            p / "generation/started.json",
            {
                "at": time.time(),
                "reserved": bound,
                "mode": "B",
                "model": r["agent_profile"]["model"],
                "cohort_manifest_sha256": digest(json_text(frozen)),
            },
        )
        pipeline.status("agent_running")
        clock = WaitClock()
        spec = manifest["profiles"]["gpt-oss-120b"]
        access = AccessProfile(
            model=spec["model"],
            base_url=spec["base_url"],
            api_key_env=spec["api_key_env"],
            revision="provider-managed",
            max_tokens=4096,
            reasoning_effort="low",
        )
        lifetime = EpisodeLimits.container_lifetime_seconds.fget
        if lifetime is None:
            raise ValueError("Missing bounded sandbox lifetime")
        proxy = SimpleNamespace(**{**vars(asyncio), "wait_for": clock.wait_for})
        original_save = agent.save
        original_dispatch = agent.Episode.dispatch

        async def dispatch(
            episode: agent.Episode,
            original: Any,
            inputs: Any,
            tools: Any,
            tool_choice: Any,
            config: Any,
            r: dict[str, Any] = r,
            clock: WaitClock = clock,
            original_dispatch: Any = original_dispatch,
        ) -> Any:
            return await agent_dispatch(
                pipeline,
                r,
                clock,
                original_dispatch,
                episode,
                original,
                inputs,
                tools,
                tool_choice,
                config,
            )

        def save(
            path: Path,
            value: Any,
            clock: WaitClock = clock,
            original_save: Any = original_save,
        ) -> None:
            if path.name == "episode.json":
                value = {
                    **value,
                    "active_work_seconds": value["wall_seconds"],
                    "wall_seconds": value["wall_seconds"] + clock.waiting(),
                    "transport_wait_seconds": clock.waiting(),
                }
            original_save(path, value)

        def bounded_lifetime(limits: EpisodeLimits, lifetime: Any = lifetime) -> int:
            if lifetime is None:
                raise ValueError("Missing bounded sandbox lifetime")
            return lifetime(limits) + int(WAIT_ALLOWANCE)

        with (
            patch.object(
                prior.pilot,
                "load_access_profiles",
                return_value={"gpt-oss-120b": access},
            ),
            patch.object(prior.pilot, "build_task", agent.build_task),
            patch.object(agent, "asyncio", proxy),
            patch.object(prior.pilot, "asyncio", proxy),
            patch.object(agent, "time", SimpleNamespace(monotonic=clock.monotonic)),
            patch.object(agent, "save", save),
            patch.object(agent.Episode, "dispatch", dispatch),
            patch.object(
                EpisodeLimits, "container_lifetime_seconds", property(bounded_lifetime)
            ),
        ):
            prior.pilot.collect_profile(
                root,
                cohort,
                (assigned,),
                "gpt-oss-120b",
                manifest["oracle_image"],
                {**dotenv_values(root / ".env"), **os.environ},
                resume=True,
                max_new_attempts=1,
                generation_override=config,
            )
        attempt = cohort / "gpt-oss-120b/attempts" / r["assignment_id"]
        episode = prior.pilot.read_optional(attempt / "episode.json")
        if episode is None:
            freeze(
                p / "generation/result.json",
                {"status": "request_error", "error_type": "EpisodeBookkeepingMissing"},
            )
            pipeline.event(r, "generation", True, "EpisodeBookkeepingMissing")
            pipeline.pause("Episode bookkeeping requires review")
            pipeline.check()
        if episode is None:
            raise RuntimeError("Missing terminal episode")
        freeze(p / "episode.json", episode)
        intents = list((attempt / "provider").glob("*.started.json"))
        raws = list((attempt / "provider").glob("*.raw.json"))
        if len(intents) == len(raws) == episode["model_dispatches"]:
            try:
                cost = sum(
                    usage_cost(read(q)["usage"], input_rate=0.15, output_rate=0.6)
                    for q in raws
                )
            except (ValueError, KeyError, TypeError):
                pipeline.pause("Incomplete episode usage; reservation retained")
            else:
                pipeline.ledger.settle({identity: cost})
        if pipeline.ledger.snapshot()["overrun"]:
            pipeline.pause("Episode usage exceeded reservation")
        if not episode["cleanup_verified"] or episode["status"] in (
            "cleanup_unverified",
            "workspace_error",
            "sandbox_error",
            "infrastructure_error",
        ):
            pipeline.pause("Agent sandbox integrity requires review")
        if (attempt / "final.py").exists():
            derived = extension.prepare_final_file_judging(
                cohort, (assigned,), "gpt-oss-120b"
            )
            raw = read(
                derived / "gpt-oss-120b/attempts" / r["assignment_id"] / "raw.json"
            )
            freeze(p / "generation/raw.json", raw)
            freeze(
                p / "generation/provenance.json",
                {
                    "kind": "derived_final_workspace_file_not_provider_response",
                    "source_final": str((attempt / "final.py").relative_to(root)),
                    "source_sha256": digest((attempt / "final.py").read_bytes()),
                },
            )
        score = prior.pilot.read_optional(attempt / "scoring.json")
        if score is not None:
            freeze(p / "scoring.json", score)
            failed = compact.scoring_infrastructure_issue(score)
            pipeline.event(
                r, "scoring", failed, "SandboxInfrastructure" if failed else None
            )
            if failed:
                pipeline.pause("Agent scoring infrastructure failure")
        elif (attempt / "final.py").exists():
            pipeline.pause("Agent scoring missing")
        freeze(
            p / "generation/result.json",
            {
                "status": episode["status"],
                "at": time.time(),
                "episode_path": str((attempt / "episode.json").relative_to(root)),
            },
        )
        pipeline.event(
            r,
            "generation",
            episode["status"] != "completed",
            episode.get("failure_type"),
        )
        pipeline.status("agent_running")
        pipeline.check()


def sandbox_clean(image: str) -> None:
    """Check both agent and scoring sandboxes, leaving unrelated containers untouched."""
    previous.sandbox_clean(image)
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", "label=grader-power.agent=true"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.stdout.strip():
        raise RuntimeError("Existing agent containers require review")


def seed_pacing(root: Path) -> None:
    """Preserve the latest observed limits/cooldowns without inventing quota headroom."""
    parents = {
        "gpt-oss-120b": "reviewer-observation/reviewer-observation-v1",
        "sonnet": "reviewer-observation/strong-preferences-v1",
        "minimax": "reviewer-observation/strong-preferences-v1",
        "kimi": "reviewer-observation/strong-preferences-v1",
        "glm": "reviewer-observation/strong-preferences-v1",
        "deepseek": "reviewer-observation/reviewer-observation-v1",
        "qwen": "reviewer-observation/reviewer-observation-v1",
    }
    bindings = {}
    for profile, parent in parents.items():
        source = (
            root / "runs" / parent / "operations/pacing" / profile / "rate-state.json"
        )
        target = base(root) / "operations/pacing" / profile / "rate-state.json"
        if not target.exists():
            write_json(target, read(source))
            bindings[str(source.relative_to(root))] = digest(source.read_bytes())
    account = (
        root
        / "runs/reviewer-observation/strong-preferences-v1/operations/pacing/account-rpm.json"
    )
    target = base(root) / "operations/pacing/account-rpm.json"
    if not target.exists():
        write_json(target, read(account))
        bindings[str(account.relative_to(root))] = digest(account.read_bytes())
    if bindings:
        freeze(base(root) / "operations/pacing-seed.json", bindings)


def collect(root: Path) -> None:
    """Verify authorization and cleanup under the global lock before new paid work."""
    manifest = verify(root)
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
                "manifest_sha256": digest(json_text(manifest)),
                "authorization": manifest["authorization"],
                "sandbox_clean": True,
            },
        )
        seed_pacing(root)
        pipeline = Pipeline(root, manifest)
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(
                sig, lambda *_: pipeline.pause("Operator signal; drain paid work")
            )
        with prior.pilot.zero_retry_clients(), concurrent_transport():
            try:
                collect_agents(pipeline)
                with ThreadPoolExecutor(max_workers=6) as executor:

                    async def execute() -> None:
                        loop = asyncio.get_running_loop()
                        loop.set_default_executor(executor)
                        for sig in (signal.SIGINT, signal.SIGTERM):
                            loop.add_signal_handler(
                                sig, pipeline.pause, "Operator signal; drain paid work"
                            )
                        await pipeline.run()

                    asyncio.run(execute())
            finally:
                sandbox_clean(manifest["oracle_image"])
                freeze(
                    launch / "drained.json",
                    {
                        "at": time.time(),
                        "usage": pipeline.ledger.snapshot(),
                        "sandbox_clean": True,
                    },
                )
        verify(root)
        freeze(
            launch / "complete.json", {"at": time.time(), "all_bindings_verified": True}
        )
    analyze(root)


def canonical_source(root: Path, study: str) -> Path:
    """Select the authoritative pre-recovery view, including earlier fixed lineage."""
    if study in previous.RELATIVE_RUNS:
        return previous.base(root) / "analysis" / study / "measured-rows.json"
    parent = root / "runs" / RELATIVE_RUNS[study]
    return (
        parent
        / ("analysis-v4" if study == "syntax" else "analysis-v1")
        / "measured-rows.json"
    )


def overlay_row(
    root: Path, row: dict[str, Any], record: dict[str, Any]
) -> dict[str, Any]:
    """Designate the fresh outcome even when worse; verify every classified label."""
    p = base(root) / "requests" / record["assignment_id"]
    measured = outcome(base(root), record)
    result = effective_judgment(p)
    if result is not None:
        raw_path = (
            root / record["source_raw"]
            if record["recovery_kind"] == "judge"
            else p / "generation/raw.json"
        )
        raw = read(raw_path)
        if result["source_raw_sha256"] != digest(json_text(raw)):
            raise ValueError("Recovered judgment's source binding changed")
        if result["status"] in ("classified", "abstain"):
            packet = read(p / "judge/packet.json")
            if result["packet_sha256"] != digest(json_text(packet)):
                raise ValueError("Recovered packet binding changed")
            response = ModelOutput.model_validate(read(p / "judge/raw.json"))
            text = response.completion
            if result.get("mechanical_repair") == "reversed_ranges":
                text = json_text(
                    read(p / "judge/reversed-ranges-repair.json")["repair"]["corrected"]
                )
            parsed = parse_judgment(record, text, packet)
            if (
                parsed.label != result["label"]
                or parsed.model_dump(mode="json") != result["judgment"]
            ):
                raise ValueError("Recovered label does not match strict raw judgment")
    updated = copy.deepcopy(row)
    if record["recovery_kind"] == "generation":
        if not measured["attempt_complete"]:
            raise ValueError("Fresh generation has no terminal outcome")
        updated.update(measured)
        updated["prior_generation_recovery_id"] = row.get("recovery_assignment_id")
        updated["recovery_assignment_id"] = record["assignment_id"]
        if record["mode"] == "B":
            episode = prior.pilot.read_optional(p / "episode.json") or {}
            exposure = episode.get("exposure_summary", {})
            required = [
                f["path"]
                for f in record["visible_files"]
                if f["path"] == "AGENTS.md" or f["path"].startswith(".ci/")
            ]
            updated.update(
                exposure=exposure,
                all_guidance_before_edit=all(
                    exposure.get(k, {}).get("full_before_first_explicit_edit") is True
                    for k in required
                ),
                all_guidance_before_final=all(
                    exposure.get(k, {}).get("full_before_finalization") is True
                    for k in required
                ),
            )
    else:
        if not measured["judge_result"]:
            raise ValueError("Fresh judgment has no terminal outcome")
        updated.update(
            {
                k: v
                for k, v in measured.items()
                if k.startswith("judge")
                or k in ("label", "audit_label", "audit_agrees")
            }
        )
    if result is not None and result.get("mechanical_repair"):
        updated["all_judgment_repair"] = result["mechanical_repair"]
    updated.update(
        all_recovery_id=record["assignment_id"],
        all_recovery_method=record["recovery_kind"],
        pending_judgment=False,
    )
    return updated


def analyze(root: Path) -> dict[str, Any]:
    """Publish all assigned rows and the original statistical families additively."""
    from grader_power import (
        reviewer_observation,
        strong_preference_replication,
        style_preference_experiment,
    )

    manifest = verify(root)
    records = {(r["origin_study"], r["replaces"]): r for r in manifest["records"]}
    gaps = read(base(root) / "operations/initial-main-gap-audit.json")
    repairs = {(r["study"], r["assignment_id"]): r for r in gaps["free_repairs"]}
    summary, index = {}, {}
    for study, relative in RELATIVE_RUNS.items():
        source = canonical_source(root, study)
        expected = manifest["analysis_sources"][str(source.relative_to(root))]
        if digest(source.read_bytes()) != expected:
            raise ValueError("Original full-denominator analysis changed")
        original_rows = read(source)
        rows, lineage = [], []
        for row in original_rows:
            key = study, row["assignment_id"]
            updated = copy.deepcopy(row)
            repair = repairs.get(key)
            if repair is not None:
                prior.verify_hashes(root, repair["original_bindings"])
                jp = root / repair["original_judge_path"]
                fixed = prior.normalize_judgment(
                    read(jp / "raw.json")["completion"],
                    read(jp / "packet.json"),
                    "mapping",
                )
                if fixed != repair["repair"]:
                    raise ValueError("Mechanical recovery changed")
                updated.update(
                    label=fixed["label"],
                    judge_status=fixed["status"],
                    judge_result=True,
                    judgment_recovery_method="audited_mechanical_normalization",
                )
                lineage.append(
                    {
                        "assignment_id": key[1],
                        "method": "mechanical",
                        "label": fixed["label"],
                    }
                )
            r = records.get(key)
            if r is not None:
                updated = overlay_row(root, updated, r)
                lineage.append(
                    {
                        "assignment_id": key[1],
                        "method": r["recovery_kind"],
                        "recovery_id": r["assignment_id"],
                        "original_label": row["label"],
                        "label": updated["label"],
                    }
                )
            rows.append(updated)
        if len(rows) != len(original_rows) or any(
            a["assignment_id"] != b["assignment_id"]
            for a, b in zip(rows, original_rows, strict=True)
        ):
            raise ValueError("Supplement changed assigned identities or denominator")
        if study == "syntax":
            result = style_preference_experiment.analyze(rows, repetitions=3)
        elif study == "observation":
            result = reviewer_observation.analyze(rows)
        elif study == "strong":
            result = strong_preference_replication.analyze(rows)
        else:
            result = previous.statistics(study, rows)
        destination = base(root) / "analysis/final" / study
        freeze(destination / "measured-rows.json", rows)
        freeze(destination / "lineage.json", lineage)
        result.update(
            recovery_note="One designated same-limit fresh attempt or fresh blinded judgment; all original outcomes retained. Genuine ambiguity remains unknown. No correctness filtering or best-of selection.",
            measured_rows_sha256=digest(
                (destination / "measured-rows.json").read_bytes()
            ),
        )
        freeze(destination / "analysis.json", result)
        index[relative] = {
            "path": str((destination / "measured-rows.json").relative_to(root)),
            "sha256": digest((destination / "measured-rows.json").read_bytes()),
            "rows": len(rows),
            "original_source": str(source.relative_to(root)),
            "original_sha256": expected,
            "lineage": str((destination / "lineage.json").relative_to(root)),
            "lineage_sha256": digest((destination / "lineage.json").read_bytes()),
        }
        summary[study] = {
            "assigned": len(rows),
            "resolved": sum(r["label"] is not None for r in rows),
            "unresolved": sum(r["label"] is None for r in rows),
            "changes": len(lineage),
        }
    freeze(base(root) / "analysis/final/summary.json", summary)
    freeze(
        base(root) / "analysis-index.json",
        {
            "complete": True,
            "manifest_sha256": digest(json_text(manifest)),
            "sources": index,
        },
    )
    ledger = RequestLedger(
        base(root) / "spending.sqlite3",
        cap=nanodollars(CAP),
        scope=digest(json_text(manifest)),
    )
    states = [(r, outcome(base(root), r)) for r in manifest["records"]]
    report = {
        "at": time.time(),
        "generation_attempts": sum(
            s["attempt_complete"]
            for r, s in states
            if r["recovery_kind"] == "generation"
        ),
        "new_saved": sum(
            s["raw_saved"] for r, s in states if r["recovery_kind"] == "generation"
        ),
        "new_resolved": sum(s["label"] is not None for _, s in states),
        "mechanical_repairs": len(repairs),
        "usage": ledger.snapshot(),
        "studies": summary,
        "all_original_bindings_verified": True,
    }
    freeze(base(root) / "completion.json", report)
    return report


def main() -> None:
    """Expose explicit preparation, verification, collection and additive analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "verify", "collect", "catalog", "analyze")
    )
    parser.add_argument("--report", action="append", default=[], type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        prepare(root, [p.resolve() for p in args.report])
    elif args.command == "verify":
        manifest = verify(root)
        print(
            json.dumps(
                {"verified": True, "records": len(manifest["records"]), "cap": CAP}
            )
        )
    elif args.command == "catalog":
        result = catalog(root)
        print(
            json.dumps(
                {
                    "records": len(result["records"]),
                    "counts": dict(
                        Counter(r["recovery_kind"] for r in result["records"])
                    ),
                }
            )
        )
    elif args.command == "analyze":
        print(json.dumps(analyze(root)))
    else:
        collect(root)


if __name__ == "__main__":
    main()
