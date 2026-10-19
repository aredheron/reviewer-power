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
from grader_power import expanded_preference_study as study
from grader_power import style_judge
from grader_power.expanded_preference_lock import (
    collection_lock,
    isolate_collection_process,
)
from grader_power.expanded_preference_scoring import (
    packet_from_code,
    packet_from_output,
    score_task,
)
from grader_power.pacing import WaitClock, append_event, write_json
from grader_power.pilot_tasks import require_pilot_verification
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
from grader_power.reviewer_observation_followup import qualify_factual_format
from grader_power.spending import nanodollars, usage_cost

ROOT = Path(__file__).resolve().parents[1]
NEW_SOURCES = (
    "src/grader_power/expanded_preference_study.py",
    "src/grader_power/expanded_preference_tasks.py",
    "src/grader_power/expanded_preference_scoring.py",
    "src/grader_power/expanded_preference_lock.py",
    "src/grader_power/datasets/apps_source.py",
    "src/grader_power/datasets/mbpp_full.py",
    "evals/inspect_expanded_preferences.py",
    "scripts/prepare_expanded_preferences.py",
    "scripts/run_expanded_preferences.py",
    "planning/expanded_preference_protocol.md",
    "tests/test_expanded_preferences.py",
    "tests/test_expanded_preferences_runner.py",
    "tests/integration/test_expanded_preferences_docker.py",
)
ModelPacer = SharedModelPacer
REFERENCE_REVIEW = "operations/recoveries/reference-review-v1"
REFERENCE_PROPOSAL_SHA256 = (
    "1bbf4ddd78ef8827fa8e18c75a1e9a6c3ba515f01759a92d340bcac4816bcb31"
)
PILOT_CLEANUP = "operations/recoveries/pilot-cleanup-v1"
PILOT_PROPOSAL = "operations/recoveries/pilot-readiness-review-v1/cleanup-proposal.json"
PILOT_PROPOSAL_SHA256 = (
    "b4b0e21ffcd628397643e13b8f794671bf9a014f1173a220741475eec9096383"
)
MAIN_FORMAT = "operations/recoveries/main-judge-format-v1"
MAIN_FORMAT_PROPOSAL_SHA256 = (
    "90f0e5503e6f0e6dce6e4321b34a18d19e7689bc1aecadbce9b04112251c7855"
)
MAIN_FORMAT_APPROVAL_SHA256 = (
    "ca5810573c2694624846527fe495f5bbd8767ebf53a1784d71501a954587b113"
)
MAIN_FORMAT_RESULT = "judge/qualified-format-v1.json"


def main_format_approved(base: Path) -> bool:
    """Require the exact human-approved main-only mechanical cleanup proposal."""
    path = base / MAIN_FORMAT / "approval.json"
    if not path.exists():
        return False
    approval = json.loads(path.read_text())
    if (
        digest(path.read_bytes()) != MAIN_FORMAT_APPROVAL_SHA256
        or approval["proposal_sha256"] != MAIN_FORMAT_PROPOSAL_SHA256
        or digest((base / approval["proposal"]).read_bytes())
        != MAIN_FORMAT_PROPOSAL_SHA256
    ):
        raise ValueError("Main format cleanup approval or proposal changed")
    return True


def normalize_main_judge_format(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Remove approved evidence notes and identical integer start-line duplicates.

    Keep JSON object pairs until their exact location is known. Every other
    duplicate, malformed/trailing JSON, or nonfinite constant is rejected. The
    caller must still run the unchanged schema and exact source-evidence parser.
    """

    class Pairs(list[tuple[str, Any]]):
        pass

    def reject_constant(value: str) -> None:
        raise ValueError("Nonfinite JSON constant: " + value)

    raw = json.loads(text, object_pairs_hook=Pairs, parse_constant=reject_constant)
    edits: list[dict[str, Any]] = []

    def walk(value: Any, path: list[str | int]) -> Any:
        if isinstance(value, Pairs):
            out: dict[str, Any] = {}
            evidence = len(path) == 2 and path[0] == "evidence" and type(path[1]) is int
            for key, item in value:
                clean = walk(item, path + [key])
                if key in out:
                    if (
                        evidence
                        and key == "start_line"
                        and type(clean) is int
                        and type(out[key]) is int
                        and clean == out[key]
                    ):
                        edits.append(
                            {
                                "path": path + [key],
                                "operation": "collapse_identical_duplicate_start_line",
                                "value": clean,
                            }
                        )
                        continue
                    raise ValueError("Unapproved duplicate JSON field")
                out[key] = clean
            if evidence and "start_line_note" in out:
                note = out["start_line_note"]
                if note is not None and not isinstance(note, str):
                    raise ValueError("Nontext start_line_note")
                edits.append(
                    {
                        "path": path + ["start_line_note"],
                        "operation": "remove_noncontract_metadata",
                        "value": note,
                    }
                )
                del out["start_line_note"]
            return out
        if isinstance(value, list):
            return [walk(item, path + [i]) for i, item in enumerate(value)]
        return value

    return json.dumps(walk(raw, []), ensure_ascii=False), edits


def main_format_candidate(base: Path, r: dict[str, Any]) -> dict[str, Any] | None:
    """Reconstruct a qualified label from bound originals, without modifying them.

    Only a normal, completed main judgment with one of the approved formatting
    defects can qualify. Semantic abstentions and all other failures remain as
    originally recorded. A changed source binding is an integrity error.
    """
    if r["phase"] != "main" or r["kind"] != "core":
        return None
    if not main_format_approved(base):
        raise ValueError("Main format cleanup requires approval")
    path = path_for(base, r)
    result = pilot.read_optional(path / "judge/result.json") or {}
    if result.get("status") != "judge_error" or result.get("label") is not None:
        return None
    if not (path / "judge/raw.json").exists():
        return None
    names = (
        "generation/raw.json",
        "judge/raw.json",
        "judge/packet.json",
        "judge/result.json",
    )
    data = {name: json.loads((path / name).read_text()) for name in names}
    packet = data["judge/packet.json"]
    if result.get("source_raw_sha256") != digest(
        json_text(data["generation/raw.json"])
    ) or result.get("packet_sha256") != digest(json_text(packet)):
        raise ValueError("Main format cleanup source binding changed")
    output = ModelOutput.model_validate(data["judge/raw.json"])
    if not output.choices or output.choices[0].stop_reason != "stop":
        return None
    try:
        cleaned, edits = normalize_main_judge_format(output.completion)
        parsed = parse_packet(r["study"], cleaned, packet)
    except (ValueError, TypeError):
        return None
    if not edits or parsed.label is None:
        return None
    return {
        "assignment_id": r["assignment_id"],
        "approval_sha256": MAIN_FORMAT_APPROVAL_SHA256,
        "proposal_sha256": MAIN_FORMAT_PROPOSAL_SHA256,
        "label_source": "approved_main_judge_format_v1",
        "bindings": {name: digest((path / name).read_bytes()) for name in names},
        "edits": edits,
        "normalized_text": cleaned,
        "label": parsed.label,
        "status": parsed.status,
        "judgment": parsed.model_dump(mode="json"),
    }


def qualified_main_judgment(base: Path, r: dict[str, Any]) -> dict[str, Any] | None:
    """Read a separate qualified result only after revalidating all its evidence."""
    saved = pilot.read_optional(path_for(base, r) / MAIN_FORMAT_RESULT)
    if saved is not None and saved != main_format_candidate(base, r):
        raise ValueError("Saved main format qualification differs from raw evidence")
    return saved


def save_main_format_qualification(
    base: Path, r: dict[str, Any]
) -> dict[str, Any] | None:
    """Freeze one free, deterministic overlay; never replace a raw judge result."""
    candidate = main_format_candidate(base, r)
    if candidate is not None:
        original.freeze(path_for(base, r) / MAIN_FORMAT_RESULT, candidate)
    return candidate


def remove_approved_evidence_notes(text: str) -> tuple[str, list[Any]]:
    """Remove only the approved metadata; reject duplicate keys and missing notes.

    This is a pilot qualification helper, never the collection judge parser.
    Its caller must restrict it to the four approved raw-response hashes.
    """

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field in pilot qualification")
            result[key] = value
        return result

    value = json.loads(text, object_pairs_hook=unique)
    notes = []
    for evidence in value["evidence"]:
        if "start_line_note" not in evidence or not isinstance(
            evidence["start_line_note"], (str, type(None))
        ):
            raise ValueError("Pilot evidence metadata differs from approved repair")
        notes.append(evidence.pop("start_line_note"))
    if not notes:
        raise ValueError("Approved pilot repair requires evidence")
    return json_text(value), notes


def pilot_qualification(base: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    """Reconstruct the exact free cleanup from immutable, approved saved evidence."""
    proposal_path = base / PILOT_PROPOSAL
    approval = json.loads((base / PILOT_CLEANUP / "approval.json").read_text())
    if (
        digest(proposal_path.read_bytes()) != PILOT_PROPOSAL_SHA256
        or approval["proposal_sha256"] != PILOT_PROPOSAL_SHA256
        or approval["answer"]
        != "i don't see the question, but you may apply the cleanup and start glm and qwen."
        or approval["main_models"] != ["glm", "qwen"]
        or approval["held_models"] != ["gpt-oss-120b"]
        or approval["cap_usd"] != study.CAP
        or approval["paid_cleanup_calls"] != 0
    ):
        raise ValueError("Pilot cleanup lacks matching approval")
    proposal = json.loads(proposal_path.read_text())
    by_id = {r["assignment_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate assignment in pilot cleanup")
    factual_ids = {
        r["assignment_id"]
        for r in records
        if r["phase"] == "pilot" and r["kind"] == "comprehension_probe"
    }
    if len(factual_ids) != 144 or factual_ids != {
        item["assignment_id"] for item in proposal["facts"]
    }:
        raise ValueError("Pilot cleanup must cover all 144 factual responses")

    def bound_file(record: dict[str, Any], name: str, expected: str) -> dict[str, Any]:
        path = path_for(base, record) / name
        if digest(path.read_bytes()) != expected:
            raise ValueError("Approved pilot evidence changed: " + name)
        return json.loads(path.read_text())

    facts = {}
    judges = {}
    for item in proposal["facts"]:
        rec = by_id[item["assignment_id"]]
        if rec["profile"] != item["profile"] or rec["condition"] != item["condition"]:
            raise ValueError("Pilot factual designation changed")
        output = ModelOutput.model_validate(
            bound_file(rec, "generation/raw.json", item["source_raw_sha256"])
        )
        probe = bound_file(rec, "probe.json", item["original_probe_sha256"])
        qualified = qualify_factual_format(
            output.completion,
            study.expected_facts(rec),
            generation_complete=bool(
                output.choices and output.choices[0].stop_reason == "stop"
            ),
        )
        if (
            qualified != item["qualification"]
            or probe["fence_only"]["all_correct"] != item["original_pass"]
        ):
            raise ValueError(
                "Pilot factual qualification differs from approved proposal"
            )
        facts[rec["assignment_id"]] = qualified
    for item in proposal["judge_repairs"]:
        rec = by_id[item["assignment_id"]]
        if (
            rec["phase"] != "pilot"
            or rec["kind"] != "core"
            or rec["profile"] != item["profile"]
            or rec["source_task_id"] != item["task"]
        ):
            raise ValueError("Pilot judge designation changed")
        source = bound_file(rec, "generation/raw.json", item["source_raw_sha256"])
        result = bound_file(rec, "judge/result.json", item["original_result_sha256"])
        packet = bound_file(rec, "judge/packet.json", item["packet_sha256"])
        output = ModelOutput.model_validate(
            bound_file(rec, "judge/raw.json", item["judge_raw_sha256"])
        )
        if (
            result.get("label") is not None
            or result["source_raw_sha256"] != digest(json_text(source))
            or result["packet_sha256"] != digest(json_text(packet))
            or not output.choices
            or output.choices[0].stop_reason != "stop"
        ):
            raise ValueError("Pilot judge source/completeness binding changed")
        cleaned, notes = remove_approved_evidence_notes(output.completion)
        parsed = parse_packet(rec["study"], cleaned, packet)
        if (
            notes != item["removed_metadata"]
            or parsed.label != item["candidate_label"]
            or parsed.model_dump(mode="json") != item["candidate_judgment"]
        ):
            raise ValueError("Pilot judge repair differs from approved proposal")
        judges[rec["assignment_id"]] = {
            "label": parsed.label,
            "judgment": parsed.model_dump(mode="json"),
            "label_source": "approved_pilot_metadata_cleanup",
        }
    if len(judges) != 4:
        raise ValueError("Exactly four pilot judge repairs are approved")
    return {
        "proposal_sha256": PILOT_PROPOSAL_SHA256,
        "approval_sha256": digest(
            (base / PILOT_CLEANUP / "approval.json").read_bytes()
        ),
        "facts": facts,
        "judges": judges,
        "main_models": ["glm", "qwen"],
        "held_models": ["gpt-oss-120b"],
        "paid_cleanup_calls": 0,
    }


def qualified_pilot_rows(
    rows: list[dict[str, Any]], qualification: dict[str, Any]
) -> list[dict[str, Any]]:
    """Copy readiness rows only; preserve original labels and every main outcome."""
    facts, judges = qualification["facts"], qualification["judges"]
    by_id = {r["assignment_id"]: r for r in rows}
    if len(by_id) != len(rows) or not (facts.keys() | judges.keys()) <= by_id.keys():
        raise ValueError("Missing or duplicate qualified pilot designation")
    result = []
    for row in rows:
        aid = row["assignment_id"]
        copied = dict(row)
        if aid in facts or aid in judges:
            if row["phase"] != "pilot" or row["kind"] != (
                "comprehension_probe" if aid in facts else "core"
            ):
                raise ValueError("Qualification cannot modify non-pilot outcomes")
            if aid in facts:
                copied["probe_correct"] = facts[aid]["qualified_correct"]
            else:
                copied["label"] = judges[aid]["label"]
                copied["label_source"] = judges[aid]["label_source"]
        result.append(copied)
    return result


def corrected_reference_records(
    base: Path, records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive exactly the four approved fresh controls without editing old records."""
    folder = base / REFERENCE_REVIEW
    proposal_path = folder / "proposal.json"
    approval = json.loads((folder / "approval.json").read_text())
    if (
        digest(proposal_path.read_bytes()) != REFERENCE_PROPOSAL_SHA256
        or approval["proposal_sha256"] != REFERENCE_PROPOSAL_SHA256
        or approval["answer"] != "Approve four corrected reference checks (recommended)"
        or approval["cap_usd"] != study.CAP
        or approval["fresh_judge_requests"] != 4
    ):
        raise ValueError("Corrected reference batch lacks exact matching approval")
    proposal = json.loads(proposal_path.read_text())
    originals = {r["assignment_id"]: r for r in records if r["phase"] == "calibration"}
    additions = []
    for item in proposal["four_references"]:
        previous = originals[item["original_assignment_id"]]
        if (
            previous["source_task_id"] != item["source_task_id"]
            or previous["expected_label"] != item["expected_label"]
            or previous["reference_code"] != item["original_code"]
            or digest(item["proposed_code"]) != item["proposed_code_sha256"]
            or not item["correctness"]["correctness"]["full_correct"]
        ):
            raise ValueError("Approved reference/code/correctness binding changed")
        row = {k: v for k, v in previous.items() if k != "assignment_id"}
        row.update(
            phase="calibration2",
            replaces_assignment_id=previous["assignment_id"],
            reference_code=item["proposed_code"],
            reference_sha256=item["proposed_code_sha256"],
        )
        row["assignment_id"] = digest(json_text(row))
        additions.append(row)
    if len(additions) != 4 or {
        (r["source_task_id"], r["expected_label"]) for r in additions
    } != {
        (task, style)
        for task in ("Mbpp/632", "MbppFull/363")
        for style in ("loop_only", "comprehension_only")
    }:
        raise ValueError("Corrected reference scope differs from the approved four")
    return additions


def qualified_calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Always designate all four new outcomes, including missing/failed judgments."""
    old = [r for r in rows if r["phase"] == "calibration"]
    fresh = [r for r in rows if r["phase"] == "calibration2"]
    if not fresh:
        return old
    by_id = {r["assignment_id"]: r for r in old}
    replaced = {r["replaces_assignment_id"] for r in fresh}
    if len(fresh) != 4 or len(replaced) != 4 or not replaced <= by_id.keys():
        raise ValueError("Incomplete or duplicate corrected-reference designation")
    for row in fresh:
        previous = by_id[row["replaces_assignment_id"]]
        if any(
            row[k] != previous[k]
            for k in ("source_task_id", "expected_label", "dataset", "kind")
        ):
            raise ValueError("Corrected-reference designation changed its task/style")
    return [r for r in old if r["assignment_id"] not in replaced] + [
        {**r, "phase": "calibration"} for r in fresh
    ]


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
        cap=nanodollars(manifest["cap_usd"]),
        scope=digest(json_text(manifest)),
    )


def path_for(base: Path, r: dict[str, Any]) -> Path:
    """Locate only the new run's immutable outcomes."""
    return base / "requests" / r["assignment_id"]


def measure(
    base: Path, r: dict[str, Any], *, qualified: bool = False
) -> dict[str, Any]:
    """Count saved records, separating terminal labels, pending stages and probes."""
    path = path_for(base, r)
    generation = pilot.read_optional(path / "generation/result.json") or {}
    scoring = pilot.read_optional(path / "scoring.json") or {}
    judged = pilot.read_optional(path / "judge/result.json") or {}
    selected = qualified_main_judgment(base, r) if qualified else None
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
        "provider": None
        if r["kind"] == "reference_judgment"
        else study.PROFILES[r["profile"]]["provider"],
        "dataset": r.get("dataset"),
        "family": r.get("family"),
        "expected_label": r.get("expected_label"),
        "replaces_assignment_id": r.get("replaces_assignment_id"),
        "reference_code": r["kind"] == "reference_judgment",
        "reused_generation": r.get("supplement_stage") == "judge",
        "generation_model": None
        if r["kind"] == "reference_judgment"
        else study.PROFILES[r["profile"]]["model"],
        "started": (path / "generation/started.json").exists(),
        "attempt_complete": bool(generation),
        "raw_saved": (path / "generation/raw.json").exists(),
        "scored": bool(scoring),
        "full_correct": scoring.get("correctness", {}).get("full_correct"),
        "judge_started": (path / "judge/started.json").exists(),
        "judge_result": bool(judged),
        "label": (selected or judged).get("label"),
        "strict_label": judged.get("label"),
        "label_source": (selected or {}).get("label_source", "original_strict_judge"),
        "format_qualified": selected is not None,
        "probe_correct": probe.get("fence_only", {}).get("all_correct"),
        "probe_correct_strict": probe.get("strict", {}).get("all_correct"),
        "generation_status": generation.get("status"),
        "judge_status": (selected or judged).get("status"),
        "strict_judge_status": judged.get("status"),
    }


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Keep model, experiment, cohort and missingness explicit in status reports."""
    result = {}
    for phase in ("calibration", "calibration2", "pilot", "main"):
        for profile in study.PROFILES:
            for design in ("reviewer_observation",):
                if phase.startswith("calibration") and profile != "glm":
                    continue
                selected = [
                    r
                    for r in rows
                    if (r["phase"], r["profile"], r["study"])
                    == (phase, profile, design)
                ]
                code = [r for r in selected if r["kind"] != "comprehension_probe"]
                result[
                    f"{phase}/{'reference' if phase.startswith('calibration') else profile}/{design}"
                ] = {
                    "assigned": len(selected),
                    "coding_assigned": sum(r["kind"] == "core" for r in selected),
                    "reference_judgments_assigned": sum(
                        r["kind"] == "reference_judgment" for r in selected
                    ),
                    "reused_generations": sum(
                        r.get("reused_generation", False) for r in selected
                    ),
                    **{
                        k: sum(bool(r[k]) for r in selected)
                        for k in ("started", "attempt_complete", "raw_saved", "scored")
                    },
                    "resolved": sum(r["label"] is not None for r in code),
                    "format_qualified": sum(
                        r.get("format_qualified", False) for r in code
                    ),
                    "strict_resolved": sum(
                        r.get("strict_label", r["label"]) is not None for r in code
                    ),
                    "terminal_unresolved": sum(
                        r["judge_result"] and r["label"] is None for r in code
                    ),
                    "inflight_judges": sum(
                        r["judge_started"] and not r["judge_result"] for r in code
                    ),
                    "pending_judges": sum(
                        (r["raw_saved"] or r["kind"] == "reference_judgment")
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
            t["task_id"]: t
            for t in json.loads((self.base / "tasks-private.json").read_text())
        }
        self.main_format_enabled = main_format_approved(self.base)
        self.measurements = {
            r["assignment_id"]: measure(
                self.base, r, qualified=self.main_format_enabled
            )
            for r in records
        }
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
        self,
        r: dict[str, Any],
        stage: str,
        failed: bool,
        error: str | None = None,
        *,
        format_qualified: bool = False,
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
            if format_qualified:
                if (
                    stage != "judge"
                    or failed
                    or not qualified_main_judgment(self.base, r)
                ):
                    raise ValueError(
                        "Only verified main format qualifications bypass a strict failure"
                    )
                e.update(strict_failed=True, format_qualified=True)
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
        self.measurements[identity] = measure(
            self.base, r, qualified=self.main_format_enabled
        )

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
                r,
                stage,
                outcome["status"] == "request_error",
                outcome.get("error_type"),
            )
        return raw

    async def generation_stage(self, r: dict[str, Any]) -> None:
        """Never dispatch a started request again; retain incomplete/missing responses."""
        if r["kind"] == "reference_judgment":
            await self.judging.put(r)
            return
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
                score_task,
                self.root,
                task,
                output.completion,
                self.manifest["oracle_image"],
            )
            original.freeze(path / "scoring.json", score)
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
        calibration = r["kind"] == "reference_judgment"
        if calibration:
            packet = packet_from_code(
                r["reference_code"], self.tasks_by_id[r["source_task_id"]]
            )
            binding = {"source_reference_sha256": r["reference_sha256"]}
            invalid = False
        else:
            raw = json.loads(generation_path(self.base, r).read_text())
            output = ModelOutput.model_validate(raw)
            packet = packet_from_output(raw, self.tasks_by_id[r["source_task_id"]])
            binding = {"source_raw_sha256": digest(json_text(raw))}
            invalid = (
                packet is None
                or not output.choices
                or output.choices[0].stop_reason != "stop"
            )
        if invalid:
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
        if packet is None:
            raise ValueError("Missing valid judge packet")
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
        qualified = (
            save_main_format_qualification(self.base, r)
            if self.main_format_enabled and r["phase"] == "main"
            else None
        )
        self.event(
            r,
            "judge",
            qualified is None and result["status"] not in ("classified", "abstain"),
            result.get("error_type"),
            format_qualified=qualified is not None,
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
            candidates += [
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
        candidates.extend(
            root / run / "operations/pacing" / profile / "rate-state.json"
            for run in (
                "runs/reviewer-observation/strong-preferences-v1",
                "runs/recovery/all-experiments-v1",
            )
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


def verify_reports(root: Path, reports: dict[str, str]) -> None:
    """Require actual passing reports with no skipped verification cases."""
    for name, expected in reports.items():
        path = root / name
        cases = ET.parse(path).getroot().findall(".//testcase")
        if (
            digest(path.read_bytes()) != expected
            or not cases
            or any(
                c.find(tag) is not None
                for c in cases
                for tag in ("failure", "error", "skipped")
            )
        ):
            raise ValueError("Verification report changed or did not pass: " + name)


def prepare(root: Path, reports: list[Path]) -> None:
    """Freeze the approved new scope after source-faithful packet and sandbox checks."""
    from grader_power.expanded_preference_tasks import load_tasks

    base = study.base(root)
    if (base / "manifest.json").exists():
        raise ValueError("Study already frozen")
    approval = json.loads((base / "approval.json").read_text())
    proposal = base / study.PREPARATION / "budget-proposal.json"
    if (
        approval["cap_usd"] != study.CAP
        or approval["answer"] != "Approve new $275 ceiling"
        or approval["proposal_sha256"] != digest(proposal.read_bytes())
    ):
        raise ValueError("The exact new ceiling/proposal lacks matching approval")
    if len(reports) < 3:
        raise ValueError(
            "Offline, new task-adapter Docker, and nine-case oracle reports required"
        )
    verified = {
        str(p.resolve().relative_to(root)): digest(p.read_bytes()) for p in reports
    }
    verify_reports(root, verified)
    logs = {}
    for filename, expected_text in (
        ("pyright.txt", "0 errors, 0 warnings"),
        ("ruff.txt", "All checks passed!"),
    ):
        path = base / study.PREPARATION / filename
        if expected_text not in path.read_text():
            raise ValueError("Required source verification did not pass: " + filename)
        logs[str(path.relative_to(root))] = digest(path.read_bytes())
    original.verify_integration(reports[-1])
    image = (root / "data/sandbox/oracle-image.id").read_text().strip()
    require_pilot_verification(root, image)
    tasks = load_tasks(root)
    prepared = base / study.PREPARATION
    proposed = json.loads((prepared / "assignments-proposed.json").read_text())
    if study.schedule(root, tasks) != proposed or tasks != json.loads(
        (prepared / "tasks-private.json").read_text()
    ):
        raise ValueError("Reviewable task/prompt preparation changed")
    calibration = study.calibration_records(tasks)
    records = calibration + proposed
    old = study.old_coverage(root)
    original.freeze(base / "tasks-private.json", tasks)
    original.freeze(base / "retained-main.json", old)
    original.freeze(base / "assignments.json", records)
    old_manifest = json.loads((root / "runs/main/main-32-v1/manifest.json").read_text())
    judge = {
        **old_manifest["judge"],
        "generation": {**old_manifest["judge"]["generation"], "max_connections": 64},
    }
    if (
        judge["model"] != study.JUDGE_PROFILE["model"]
        or judge["generation"]["max_tokens"] != 4096
    ):
        raise ValueError("Blinded judge settings changed")
    budget = json.loads(proposal.read_text())
    floors = {
        profile: [
            next(
                r["mean_recorded_cost"]
                for r in budget["by_model_stage"]
                if (r["model"], r["stage"]) == (profile, stage)
            )
            for stage in ("generation", "judge")
        ]
        for profile in study.PROFILES
    }
    bindings = {
        **old["bindings"],
        **{n: sha for t in tasks for n, sha in t["reference_evidence"].items()},
    }
    manifest = {
        "run": study.RUN,
        "cap_usd": study.CAP,
        "authorization": study.AUTHORIZATION,
        "main_assignments": 7776,
        "pilot_assignments": 360,
        "reference_judgments": 144,
        "retained_observations": 3456,
        "profiles": {
            k: {**v, "generation": study.generation(k)}
            for k, v in study.PROFILES.items()
        },
        "judge_profile": study.JUDGE_PROFILE,
        "judges": {"reviewer_observation": judge},
        "oracle_image": image,
        "code_sha256": sources(root),
        "verification_reports": verified,
        "verification_logs": logs,
        "historical_bindings": bindings,
        "cost_floor_per_response": floors,
        "frozen_artifacts": {
            n: digest((base / n).read_bytes())
            for n in (
                "approval.json",
                "assignments.json",
                "tasks-private.json",
                "retained-main.json",
            )
        },
        "prospective_analysis": {
            "primary_tasks": 72,
            "secondary_pooled_tasks": 104,
            "repetitions": 3,
            "primary_contrasts": study.PRIMARY_COMPARISONS,
            "secondary_contrasts": study.SECONDARY_COMPARISONS,
            "holm_family_sizes": [9, 9],
            "bootstrap_draws": 10000,
            "all_assigned_rows": True,
            "correctness_filter": False,
        },
        "prospective_readiness": {
            "calibration": "No incorrect resolved reference; >=95% resolved per dataset; >=1 correct per task",
            "saved_per_model": 120,
            "coding_correct_min": 65,
            "coding_resolved_min": 69,
            "factual_min_per_arm": 7,
            "factual_denominator_per_arm": 8,
            "behavior_gate": False,
        },
        "operational_limits": {
            "network_slots_per_lane": 32,
            "initial_slots": 8,
            "docker_threads": 4,
            "downstream_queue_size": 64,
            "consecutive_window": 48,
            "operational_failure_threshold": 5,
            "saved_capped_generation_is_terminal_study_outcome": True,
        },
    }
    original.freeze(base / "manifest.json", manifest)
    for name, expected in manifest["code_sha256"].items():
        data = (root / name).read_bytes()
        if digest(data) != expected:
            raise ValueError("Source changed during freeze")
        path = base / "source-snapshot" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(data)
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
    """Fail closed on frozen science, approval, reference, report or operational drift."""
    base = study.base(root)
    manifest = json.loads((base / "manifest.json").read_text())
    if manifest["run"] != study.RUN or manifest["cap_usd"] != study.CAP:
        raise ValueError("Frozen study/spending scope changed")
    current = sources(root)
    if current != manifest["code_sha256"]:
        recovery = json.loads((base / "operations/recovery-state.json").read_text())
        path = base / recovery["source_amendment"]
        amendment = json.loads(path.read_text())
        changed = {n for n in current if current[n] != manifest["code_sha256"].get(n)}
        if (
            digest(path.read_bytes()) != recovery["source_amendment_sha256"]
            or set(current) != set(manifest["code_sha256"])
            or not changed
            <= {
                "evals/inspect_expanded_preferences.py",
                "src/grader_power/expanded_preference_lock.py",
                "tests/test_expanded_preferences_runner.py",
            }
            or amendment["manifest_sha256"] != digest(json_text(manifest))
            or amendment["code_sha256"] != current
        ):
            raise ValueError("Repair cannot change frozen science or shared sources")
        for name in changed:
            if (
                digest((base / "source-snapshot" / name).read_bytes())
                != manifest["code_sha256"][name]
            ):
                raise ValueError("Original operational source snapshot changed")
        verify_reports(root, amendment["verification_reports"])
        for name, sha in {
            **amendment["audit_bindings"],
            **(amendment["retained_artifacts"] if retained else {}),
        }.items():
            if digest((base / name).read_bytes()) != sha:
                raise ValueError("Operational repair audit changed")
    for name, sha in manifest["frozen_artifacts"].items():
        if digest((base / name).read_bytes()) != sha:
            raise ValueError("Frozen study artifact changed: " + name)
    verify_reports(root, manifest["verification_reports"])
    for name, sha in manifest["verification_logs"].items():
        if digest((root / name).read_bytes()) != sha:
            raise ValueError("Source verification log changed")
    if retained:
        for name, sha in manifest["historical_bindings"].items():
            if digest((root / name).read_bytes()) != sha:
                raise ValueError(
                    "Retained historical/reference evidence changed: " + name
                )
    records = json.loads((base / "assignments.json").read_text())
    recovery = json.loads((base / "operations/recovery-state.json").read_text())
    if recovery.get("reference_followup"):
        if current == manifest["code_sha256"]:
            raise ValueError("Reference follow-up requires a verified source amendment")
        additions = corrected_reference_records(base, records)
        path = base / REFERENCE_REVIEW / "assignments.json"
        if json.loads(path.read_text()) != additions:
            raise ValueError("Frozen corrected-reference assignments changed")
        records += additions
    if recovery.get("pilot_cleanup"):
        qualification = pilot_qualification(base, records)
        saved = json.loads((base / PILOT_CLEANUP / "qualification.json").read_text())
        rows = [measure(base, r) for r in records if r["phase"] == "pilot"]
        gate = study.pilot_gate(qualified_pilot_rows(rows, qualification))
        if (
            qualification != saved
            or gate != json.loads((base / "pilot-qualified-gate.json").read_text())
            or study.pilot_gate(rows)
            != json.loads((base / "pilot-gate.json").read_text())
        ):
            raise ValueError(
                "Approved readiness overlay or original failed gate changed"
            )
    if recovery.get("main_format_cleanup"):
        if not main_format_approved(base):
            raise ValueError("Approved main format cleanup is missing")
        for record in records:
            qualified_main_judgment(base, record)
    return manifest, records


def median_forecast(
    root: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    measurements: list[dict[str, Any]] | None = None,
    *,
    draws: int = 1000,
) -> dict[str, Any]:
    """Forecast this new ledger only, keeping uncertainty reservations nonoverlapping."""
    if draws < 100:
        raise ValueError("At least 100 forecast draws required")
    base = study.base(root)
    balance = ledger_for(base, manifest).snapshot()
    by_id = {r["assignment_id"]: r for r in records}
    samples: dict[tuple[str, str, str], list[float]] = {}
    with closing(
        sqlite3.connect(f"file:{base / 'spending.sqlite3'}?mode=ro", uri=True)
    ) as db:
        for identity, settled in db.execute(
            "SELECT identity,settled FROM entries WHERE settled IS NOT NULL"
        ):
            aid, stage = identity.rsplit("/", 1)
            row = by_id.get(aid)
            if row and row["kind"] == "core":
                samples.setdefault((row["profile"], stage, row["phase"]), []).append(
                    settled / 1e9
                )
    measured = (
        measurements
        if measurements is not None
        else [measure(base, r) for r in records]
    )
    pending = Counter()
    for row in measured:
        if row["kind"] == "reference_judgment":
            if not row["judge_started"] and not row["judge_result"]:
                pending["calibration", "judge"] += 1
            continue
        if not row["started"]:
            pending[row["profile"], "generation"] += 1
        if (
            row["kind"] == "core"
            and not row["judge_started"]
            and not row["judge_result"]
            and (not row["started"] or row["raw_saved"] or not row["attempt_complete"])
        ):
            pending[row["profile"], "judge"] += 1
    calibration_floor = max(v[1] for v in manifest["cost_floor_per_response"].values())
    simulations = [
        balance["committed"] / 1e9 + pending["calibration", "judge"] * calibration_floor
    ] * draws
    randomizer = random.Random(study.SEED)
    profiles = {}
    for profile, stage in product_profiles_stages():
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
        profiles[profile + "/" + stage] = {
            "future_requests": pending[profile, stage],
            "sample": "main" if len(main) >= 32 else "pilot/historical floor",
            "sample_size": len(main) if len(main) >= 32 else len(pilot_values),
            "sample_mean_usd": sum(values) / len(values),
            "floor_usd": floor,
        }
    simulations.sort()
    estimate = median(simulations)
    return {
        "median_total_usd": estimate,
        "range10_90_usd": [simulations[draws // 10], simulations[9 * draws // 10]],
        "recorded_usd": balance["spent"] / 1e9,
        "outstanding_usd": balance["outstanding"] / 1e9,
        "hard_cap_usd": manifest["cap_usd"],
        "budget_notification_required": estimate >= 350
        or estimate > manifest["cap_usd"],
        "profiles": profiles,
        "draws": draws,
        "seed": study.SEED,
        "at": time.time(),
    }


def product_profiles_stages() -> list[tuple[str, str]]:
    """Return the fixed generation/judging cost strata."""
    return [(p, s) for p in study.PROFILES for s in ("generation", "judge")]


def sandbox_state(image: str) -> dict[str, Any]:
    """Prove the pinned image and both sandbox adapter namespaces are clean."""
    actual = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    ).stdout.strip()
    if actual != image:
        raise ValueError("Sandbox image changed")
    state: dict[str, Any] = {"image": actual, "at": time.time()}
    for name in ("oracle", "agent"):
        result = subprocess.run(
            ["docker", "ps", "-aq", "--filter", f"label=grader-power.{name}=true"],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        state[name + "_containers"] = result.stdout.split()
        if result.stdout.strip():
            raise ValueError("Experiment sandbox is not empty: " + name)
    return state


def collect(root: Path, phase: str) -> None:
    """Admit only approved, unstarted requests after stage-specific readiness checks."""
    if phase not in ("calibration", "calibration2", "pilot", "main"):
        raise ValueError("Unknown collection phase")
    manifest, records = load(root, retained=True)
    base = study.base(root)
    guard(base)
    recovery = json.loads((base / "operations/recovery-state.json").read_text())
    if recovery.get("retry_blocked"):
        raise ValueError("Recovery blocked pending new human direction")
    if recovery.get("pilot_cleanup") and phase != "main":
        raise ValueError("Approved cleanup permits only unstarted main collection")
    if (base / "operations/controller-stop.json").exists():
        raise ValueError(
            "Prior stop requires explicit diagnosis and audited disposition"
        )
    measured = [measure(base, r) for r in records]
    eligible = list(study.PROFILES)
    if phase == "calibration2" and not recovery.get("reference_followup"):
        raise ValueError("Fresh corrected references are not authorized")
    if (
        phase not in ("calibration", "calibration2")
        and not study.calibration_gate(qualified_calibration_rows(measured))["passed"]
    ):
        raise ValueError("Blinded reference calibration did not pass")
    gate = None
    if phase == "main":
        allowed = list(study.PROFILES)
        readiness = measured
        if recovery.get("pilot_cleanup"):
            qualification = json.loads(
                (base / PILOT_CLEANUP / "qualification.json").read_text()
            )
            readiness = qualified_pilot_rows(measured, qualification)
            allowed = qualification["main_models"]
        gate = study.pilot_gate(readiness)
        eligible = [
            g["profile"]
            for g in gate["groups"]
            if g["passed"] and g["profile"] in allowed
        ]
        if not eligible:
            raise ValueError("No model passed pilot readiness")
    forecast = median_forecast(root, manifest, records, measured)
    if forecast["median_total_usd"] > manifest["cap_usd"]:
        raise ValueError("Median forecast exceeds approved new ceiling")
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
            "forecast": forecast,
            "gate": gate,
        },
    )
    owner = root / "runs/extension-pilot/coordination/owner.json"
    if owner.exists():
        original.freeze(launch / "previous-owner.json", json.loads(owner.read_text()))
    with collection_lock(root, owner=study.RUN, run_id=study.RUN):
        require_pilot_verification(root, manifest["oracle_image"])
        original.freeze(
            launch / "preflight.json", sandbox_state(manifest["oracle_image"])
        )
        initialize_pacing(root, base)
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
                original.freeze(
                    launch / "cleanup.json", sandbox_state(manifest["oracle_image"])
                )
        if phase == "calibration":
            gate = study.calibration_gate([measure(base, r) for r in records])
            original.freeze(base / "calibration-gate.json", gate)
        elif phase == "calibration2":
            gate = study.calibration_gate(
                qualified_calibration_rows([measure(base, r) for r in records])
            )
            original.freeze(base / "calibration-v2-gate.json", gate)
        elif phase == "pilot":
            gate = study.pilot_gate([measure(base, r) for r in records])
            original.freeze(base / "pilot-gate.json", gate)
        else:
            analyze(root)
            pipeline.status(
                "complete"
                if len(eligible) == len(study.PROFILES)
                else "qualified_main_complete_others_held"
            )


def analyze(root: Path) -> dict[str, Any]:
    """Verify raw label/source bindings and preserve new, historical and pooled cohorts."""
    manifest, records = load(root, retained=True)
    base = study.base(root)
    for row in records:
        path = path_for(base, row)
        judged = pilot.read_optional(path / "judge/result.json")
        if not judged:
            continue
        key = (
            "source_reference_sha256"
            if row["kind"] == "reference_judgment"
            else "source_raw_sha256"
        )
        expected = (
            row["reference_sha256"]
            if row["kind"] == "reference_judgment"
            else digest(json_text(json.loads(generation_path(base, row).read_text())))
        )
        if judged.get(key) != expected:
            raise ValueError("Judge source binding changed")
        if judged["status"] in ("classified", "abstain"):
            packet = json.loads((path / "judge/packet.json").read_text())
            raw = ModelOutput.model_validate(
                json.loads((path / "judge/raw.json").read_text())
            )
            parsed = parse_packet(row["study"], raw.completion, packet)
            if (
                parsed.label != judged["label"]
                or digest(json_text(packet)) != judged["packet_sha256"]
            ):
                raise ValueError(
                    "Saved judge result differs from strictly parsed raw output"
                )
    retained = json.loads((base / "retained-main.json").read_text())
    old = retained["rows"]
    versions = [("analysis-v1", False)]
    if main_format_approved(base):
        versions.append(("analysis-v2", True))
    result: dict[str, Any] = {}
    for version, qualified in versions:
        fresh = [
            measure(base, r, qualified=qualified)
            for r in records
            if r["phase"] == "main"
        ]
        output = base / version
        summaries = {}
        for name, rows, n in (
            ("new72", fresh, 72),
            ("original32", old, 32),
            ("pooled104_secondary", fresh + old, 104),
        ):
            analysis = study.analyze_cohort(rows, cohort=name, task_count=n)
            analysis["measured_rows_sha256"] = digest(json_text(rows))
            write_json(output / name / "measured-rows.json", rows)
            write_json(output / name / "analysis.json", analysis)
            summaries[name] = {
                "assigned": analysis["assigned"],
                "tasks": n,
                "missing_labels": analysis["missing_labels"],
            }
        write_json(output / "retained-lineage.json", retained["lineage"])
        write_json(
            output / "new72/task-sensitivity.json", study.task_sensitivity(fresh)
        )
        result = {
            "manifest_sha256": digest(json_text(manifest)),
            "cohorts": summaries,
            "ledger": ledger_for(base, manifest).snapshot(),
            "counts": counts([measure(base, r, qualified=qualified) for r in records]),
            "main_format_qualification": {
                "enabled": qualified,
                "approval_sha256": MAIN_FORMAT_APPROVAL_SHA256 if qualified else None,
                "proposal_sha256": MAIN_FORMAT_PROPOSAL_SHA256 if qualified else None,
                "recovered_labels": sum(r["format_qualified"] for r in fresh),
                "strict_analysis": "analysis-v1",
            },
            "pilot_qualification": pilot.read_optional(
                base / PILOT_CLEANUP / "qualification.json"
            ),
            "qualified_pilot_gate": pilot.read_optional(
                base / "pilot-qualified-gate.json"
            ),
        }
        write_json(output / "summary.json", result)
    return result


def main() -> None:
    """Freeze verified preparation, collect gated phases, or reconstruct offline results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "collect", "run", "status", "analyze")
    )
    parser.add_argument(
        "--phase",
        choices=("calibration", "calibration2", "pilot", "main"),
        default="calibration",
    )
    parser.add_argument("--report", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(ROOT, args.report)
    elif args.command in ("run", "collect"):
        isolate_collection_process()
        recovery = json.loads(
            (study.base(ROOT) / "operations/recovery-state.json").read_text()
        )
        first_phase = (
            "calibration2" if recovery.get("reference_followup") else "calibration"
        )
        for phase in (
            (first_phase, "pilot", "main") if args.command == "run" else (args.phase,)
        ):
            collect(ROOT, phase)
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
