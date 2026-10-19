"""Verify published evidence and reproduce descriptive tables without model calls."""

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

OPPOSITE = {
    "loops": "comprehension_only",
    "comprehensions": "loop_only",
    "snake_case": "camelCase_only",
    "camelCase": "snake_case_only",
    "single_quotes": "double_quotes_only",
    "double_quotes": "single_quotes_only",
}


def read_json(path: Path) -> Any:
    """Read plain or gzip JSON; never execute contents."""
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == ".gz" else payload)


def digest(path: Path) -> str:
    """Hash exact artifact bytes, including compression when present."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_artifacts(root: Path) -> dict[str, dict[str, Any]]:
    """Reject missing, altered, duplicate, or path-escaping packaged evidence."""
    manifest = read_json(root / "reports/artifact-manifest.json")
    artifacts = {}
    for artifact in manifest["artifacts"]:
        name = artifact["name"]
        path = root / artifact["path"]
        if name in artifacts or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Duplicate artifact or unsafe artifact path")
        if digest(path) != artifact["sha256"]:
            raise ValueError("Artifact checksum mismatch: " + name)
        value = read_json(path)
        if artifact["records"] is not None and len(value) != artifact["records"]:
            raise ValueError("Artifact count mismatch: " + name)
        if artifact["kind"] == "rows":
            identities = [
                (r.get("profile", "gpt-oss-120b"), r["assignment_id"]) for r in value
            ]
            if len(set(identities)) != len(value):
                raise ValueError("Duplicate observation identity: " + name)
        artifacts[name] = artifact
    return artifacts


def rows_for(root: Path, artifacts: dict[str, Any], name: str) -> list[dict[str, Any]]:
    """Load a named cohort after manifest verification."""
    return read_json(root / artifacts[name]["path"])


def summarize(
    rows: list[dict[str, Any]], *, conflict_only: bool = True
) -> list[dict[str, Any]]:
    """Count opposite-user outputs over all assignments; unresolved outcomes stay unknown.

    This is a descriptive count table, not a new inferential analysis. Equal
    task/direction counts make these aggregate proportions task-balanced for the
    reported designs. Held cells have null rates, never a misleading zero.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        c = row["condition"]
        if conflict_only and c.get("grader_preference") == c["user_preference"]:
            continue
        groups[row["study"], row["profile"], c["arm"]].append(row)
    result = []
    for (study, profile, arm), group in sorted(groups.items()):
        unknown = sum(r["label"] is None for r in group)
        opposite = sum(
            r["label"] == OPPOSITE[r["condition"]["user_preference"]] for r in group
        )
        n = len(group)
        result.append(
            {
                "study": study,
                "profile": profile,
                "arm": arm,
                "tasks": len({r["source_task_id"] for r in group}),
                "assigned": n,
                "opposite": opposite,
                "unresolved": unknown,
                "rate": opposite / n if not unknown else None,
                "lower_bound": opposite / n,
                "upper_bound": (opposite + unknown) / n,
                "full_correct": sum(r.get("full_correct") is True for r in group),
                "labels": dict(
                    sorted(Counter(r["label"] or "unresolved" for r in group).items())
                ),
            }
        )
    return result


def presentation_tables(
    root: Path, artifacts: dict[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    """Build the main, role-name, syntax, and explicitly pooled strong-wording tables."""
    loops = summarize(rows_for(root, artifacts, "matched-review"))
    syntax = summarize(rows_for(root, artifacts, "matched-syntax"))
    observation = rows_for(root, artifacts, "observation")
    strong = rows_for(root, artifacts, "strong")
    pooled = [
        r
        for r in observation
        if r["condition"]["arm"].startswith("strong_")
        and r["profile"] in ("gpt-oss-120b", "deepseek", "glm")
    ]
    pooled += [r for r in strong if r["profile"] == "glm"]
    return {
        "main": [
            r for r in loops if r["arm"] == "absent" or r["arm"].startswith("reviewer_")
        ],
        "role-names": [
            r
            for r in loops
            if r["arm"].startswith(("grader_", "scorer_", "evaluator_"))
        ],
        "naming": [r for r in syntax if r["study"] == "naming"],
        "quotes": [r for r in syntax if r["study"] == "quotes"],
        "strong": summarize(pooled),
    }


def supplementary_conditions(
    root: Path, artifacts: dict[str, Any]
) -> list[dict[str, Any]]:
    """Recount every supplementary condition without pooling versions or studies."""
    pure = {
        "loops": "loop_only",
        "comprehensions": "comprehension_only",
        "map_filter": "map_filter_only",
        "snake_case": "snake_case_only",
        "camelCase": "camelCase_only",
        "single_quotes": "single_quotes_only",
        "double_quotes": "double_quotes_only",
    }
    result = []
    for name, artifact in artifacts.items():
        if artifact["kind"] != "rows":
            continue
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(
            list
        )
        for row in rows_for(root, artifacts, name):
            key = (
                row.get("profile", "gpt-oss-120b"),
                row.get("study", ""),
                row.get("mode", "A"),
                json.dumps(row["condition"], sort_keys=True),
            )
            groups[key].append(row)
        for (profile, study, mode, condition), group in sorted(groups.items()):
            c = json.loads(condition)
            matching = pure[c["user_preference"]]
            labels = Counter(r.get("label") or "unresolved" for r in group)
            result.append(
                {
                    "cohort": name,
                    "profile": profile,
                    "study": study,
                    "mode": mode,
                    "condition": c,
                    "assigned": len(group),
                    "tasks": len({r["source_task_id"] for r in group}),
                    "matching_pure": labels[matching],
                    "other_pure": sum(
                        labels[v] for v in pure.values() if v != matching
                    ),
                    "unresolved": labels["unresolved"],
                    "full_correct": sum(r.get("full_correct") is True for r in group),
                    "labels": dict(labels),
                }
            )
    return result


def write_tables(
    root: Path, output: Path, *, inference: bool = False
) -> dict[str, Any]:
    """Verify source checksums and write deterministic machine-readable result tables."""
    artifacts = verified_artifacts(root)
    tables = presentation_tables(root, artifacts)
    output.mkdir(parents=True, exist_ok=True)
    (output / "all-conditions.json").write_text(
        json.dumps(supplementary_conditions(root, artifacts), indent=2) + "\n"
    )
    for name, rows in tables.items():
        with (output / (name + ".csv")).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(
                {**r, "labels": json.dumps(r["labels"], sort_keys=True)} for r in rows
            )
    summary = {"tables": tables, "artifacts_verified": len(artifacts)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if inference:
        from grader_power import matched_review_analysis, matched_syntax_analysis

        for name, analyzer in (
            ("matched-review", matched_review_analysis),
            ("matched-syntax", matched_syntax_analysis),
        ):
            calculated = analyzer.analyze(rows_for(root, artifacts, name))
            recorded = read_json(root / artifacts[name + "-analysis"]["path"])
            if any(recorded.get(k) != v for k, v in calculated.items()):
                raise ValueError("Saved inference does not reproduce: " + name)
            (output / (name + "-inference.json")).write_text(
                json.dumps(calculated, indent=2) + "\n"
            )
    return summary
