"""Replace only five wrapper-interrupted recoveries under explicit human approval."""

import argparse
import asyncio
import copy
import os
import signal
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals import inspect_all_recovery as recovery
from grader_power.replication_transport import concurrent_transport

NAME = "agent-wrapper-replacements-v1"
TARGETS = frozenset(
    {
        "15627caadcf3c04f3c98c59f44f65bf6f86ff232388525f4e494a7cd1fbb16a5",
        "6eeeca86f6c8ce57692baf7acfb7106f469490e8cdb205d25a0b2073b3e93183",
        "4fe82561a6f2024938cc9bbe1ef2f2b5bab5b9e497d0bd6c1ae30501f47ef9bc",
        "7fa4cbd52326daf6ef1c1142ae6939b2f521fa9ccff84b49319d67cccd3dbbdf",
        "7167fffe9a528878792920d747d666649a59b46a8191c19a7e3317ceb7960380",
    }
)


def directory(root: Path) -> Path:
    """Keep the narrow extra approval distinct from the original recovery manifest."""
    return recovery.base(root) / "operations/recoveries" / NAME


def replacement_records(manifest: dict) -> list[dict]:
    """Assign exactly one fresh identity to each approved interrupted episode."""
    originals = {r["assignment_id"]: r for r in manifest["records"]}
    result = []
    for old in sorted(TARGETS):
        r = copy.deepcopy(originals[old])
        if r["recovery_kind"] != "generation" or r["mode"] != "B":
            raise ValueError("Replacement is not an agent generation")
        r["assignment_id"] = recovery.digest(NAME + ":" + old)
        r["supersedes_recovery_id"] = old
        result.append(r)
    return result


def prepare(root: Path, report: Path) -> None:
    """Freeze explicit approval, unchanged inputs, tests, and failed-attempt evidence."""
    manifest = recovery.verify(root)
    d = directory(root)
    approval = recovery.read(d / "approval.json")
    if approval.get("answer") != "Yes, replace those five attempts":
        raise ValueError("Exact replacement approval is missing")
    cases = recovery.ET.parse(report).getroot().findall(".//testcase")
    if not cases or any(
        c.find(t) is not None for c in cases for t in ("error", "failure", "skipped")
    ):
        raise ValueError("Replacement regression checks must pass")
    retained = {}
    for aid in sorted(TARGETS):
        p = recovery.base(root) / "requests" / aid
        episode = recovery.read(p / "episode.json")
        if episode["status"] != "generation_error" or episode["model_dispatches"] != 2:
            raise ValueError("Interrupted episode evidence differs")
        for parent in (p, recovery.base(root) / "agent-cohorts" / aid):
            for path in parent.rglob("*"):
                if path.is_file():
                    retained[str(path.relative_to(root))] = recovery.digest(
                        path.read_bytes()
                    )
    sources = {
        name: recovery.digest((root / name).read_bytes())
        for name in (
            "scripts/replace_interrupted_recoveries.py",
            "tests/test_recovery_replacements.py",
        )
    }
    recovery.freeze(
        d / "manifest.json",
        {
            "name": NAME,
            "original_manifest_sha256": recovery.digest(recovery.json_text(manifest)),
            "records": replacement_records(manifest),
            "sources": sources,
            "approval_sha256": recovery.digest((d / "approval.json").read_bytes()),
            "reports": {
                str(report.relative_to(root)): recovery.digest(report.read_bytes())
            },
            "retained_artifacts": retained,
            "cap_usd": 30,
            "policy": "One designated replacement for each of five wrapper-interrupted recoveries. Same original limits. Preserve all failures; no fallback or best-of.",
        },
    )


def verify(root: Path) -> tuple[dict, dict]:
    """Require original science, the exact five records, and every retained failure."""
    original = recovery.verify(root)
    d = directory(root)
    supplement = recovery.read(d / "manifest.json")
    if (
        supplement["original_manifest_sha256"]
        != recovery.digest(recovery.json_text(original))
        or supplement["records"] != replacement_records(original)
        or supplement["cap_usd"] != 30
    ):
        raise ValueError("Replacement scope changed")
    if supplement["approval_sha256"] != recovery.digest(
        (d / "approval.json").read_bytes()
    ):
        raise ValueError("Replacement approval changed")
    for k in ("sources", "reports", "retained_artifacts"):
        recovery.prior.verify_hashes(root, supplement[k])
    return original, supplement


def collect(root: Path) -> None:
    """Use the original ledger and transport, selecting only the approved new IDs."""
    original, supplement = verify(root)
    recovery.prior.isolate_collection_process()
    b = recovery.base(root)
    launch = (
        b
        / "operations/launches"
        / (
            time.strftime("replacement-%Y%m%dT%H%M%SZ", time.gmtime())
            + "-"
            + str(os.getpid())
        )
    )
    with recovery.prior.collection_lock(
        root, owner="compact-models-v1", run_id=recovery.RUN
    ):
        recovery.prior.require_pilot_verification(root, original["oracle_image"])
        recovery.sandbox_clean(original["oracle_image"])
        recovery.freeze(
            launch / "authorized-launch.json",
            {
                "at": time.time(),
                "pid": os.getpid(),
                "pgid": os.getpgrp(),
                "supplement_sha256": recovery.digest(recovery.json_text(supplement)),
                "sandbox_clean": True,
            },
        )
        pipeline = recovery.Pipeline(root, original)
        pipeline.records = supplement["records"]
        pipeline.measurements = {
            r["assignment_id"]: recovery.outcome(b, r) for r in pipeline.records
        }
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(
                sig, lambda *_: pipeline.pause("Operator signal; drain paid work")
            )
        with recovery.prior.pilot.zero_retry_clients(), concurrent_transport():
            try:
                recovery.collect_agents(pipeline)
                asyncio.run(pipeline.run())
            finally:
                recovery.sandbox_clean(original["oracle_image"])
                recovery.freeze(
                    launch / "drained.json",
                    {
                        "at": time.time(),
                        "usage": pipeline.ledger.snapshot(),
                        "sandbox_clean": True,
                    },
                )
        verify(root)
    if (b / "analysis-index.json").exists():
        analyze(root)


def analyze(root: Path) -> None:
    """Write a second additive export with five fixed replacements and full lineage."""
    original, supplement = verify(root)
    b = recovery.base(root)
    first_index = recovery.read(b / "analysis-index.json")
    source = first_index["sources"][recovery.RELATIVE_RUNS["extension"]]
    rows = recovery.read(root / source["path"])
    prior_lineage = recovery.read(root / source["lineage"])
    records = {r["replaces"]: r for r in supplement["records"]}
    changed, lineage = [], []
    for row in rows:
        record = records.get(row["assignment_id"])
        if record is None:
            changed.append(row)
            continue
        if not recovery.outcome(b, record)["judge_result"]:
            raise ValueError("Replacement judging is not terminal")
        new = recovery.overlay_row(root, row, record)
        changed.append(new)
        lineage.append(
            {
                "assignment_id": row["assignment_id"],
                "method": "generation",
                "recovery_id": record["assignment_id"],
                "supersedes_recovery_id": record["supersedes_recovery_id"],
                "original_label": row["label"],
                "label": new["label"],
            }
        )
    lineage = [r for r in prior_lineage if r["assignment_id"] not in records] + lineage
    if len(changed) != 12288 or len(records) != 5:
        raise ValueError("Replacement denominator or target count changed")
    destination = b / "analysis/final-v2/extension"
    recovery.freeze(destination / "measured-rows.json", changed)
    recovery.freeze(destination / "lineage.json", lineage)
    analysis = recovery.previous.statistics("extension", changed)
    analysis.update(
        recovery_note=supplement["policy"],
        measured_rows_sha256=recovery.digest(
            (destination / "measured-rows.json").read_bytes()
        ),
    )
    recovery.freeze(destination / "analysis.json", analysis)
    index = copy.deepcopy(first_index)
    index.update(
        supplement_manifest=str((directory(root) / "manifest.json").relative_to(root)),
        supplement_sha256=recovery.digest(recovery.json_text(supplement)),
        prior_index_sha256=recovery.digest((b / "analysis-index.json").read_bytes()),
    )
    binding = index["sources"][recovery.RELATIVE_RUNS["extension"]]
    binding.update(
        path=str((destination / "measured-rows.json").relative_to(root)),
        sha256=recovery.digest((destination / "measured-rows.json").read_bytes()),
        lineage=str((destination / "lineage.json").relative_to(root)),
        lineage_sha256=recovery.digest((destination / "lineage.json").read_bytes()),
    )
    recovery.freeze(b / "analysis-index-v2.json", index)
    ledger = recovery.RequestLedger(
        b / "spending.sqlite3",
        cap=recovery.nanodollars(30),
        scope=recovery.digest(recovery.json_text(original)),
    )
    recovery.freeze(
        directory(root) / "completion.json",
        {
            "at": time.time(),
            "replacement_attempts": len(records),
            "episode_statuses": dict(
                Counter(
                    recovery.outcome(b, r)["episode_status"] for r in records.values()
                )
            ),
            "resolved": sum(
                recovery.outcome(b, r)["label"] is not None for r in records.values()
            ),
            "usage": ledger.snapshot(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "collect", "analyze"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.command == "prepare":
        if args.report is None:
            raise ValueError("Passed regression report required")
        prepare(root, args.report.resolve())
    elif args.command == "verify":
        verify(root)
        print("Verified exact five replacements")
    elif args.command == "collect":
        collect(root)
    else:
        analyze(root)


if __name__ == "__main__":
    main()
