"""Rescore saved Mode A styles uniformly, preserving original generations/oracles."""

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inspect_ai.model import ModelOutput

from grader_power.schemas import ScoringRegion
from grader_power.scoring import score_response


def rescore(run: Path, name: str, root: Path) -> Path:
    """Write a new style-only analysis and readable report without any code execution.

    Every assigned model/condition must be present. Extraction must be identical
    to the original; original functional results and all raw records are retained.
    A fresh output name is mandatory, so earlier analyses cannot be overwritten.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Analysis name must be a simple name")
    definition = json.loads((run / "manifest.json").read_text())
    region = ScoringRegion(**definition["example"]["scoring_region"])
    expected = {trial["condition_id"] for trial in definition["trials"]}
    records = []
    source_hashes = {}
    for profile in definition["profiles"]:
        path = run / profile / "results.jsonl"
        source_hashes[profile] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if (
            len(rows) != len(expected)
            or {row["condition_id"] for row in rows} != expected
        ):
            raise ValueError("Missing or duplicated condition assignments")
        for row in rows:
            original = row.get("scoring")
            if row.get("raw_output") and original:
                response = ModelOutput.model_validate(row["raw_output"]).completion
                updated = score_response(response, region)
                if asdict(updated.extraction) != original["extraction"]:
                    raise ValueError(
                        "Style-only rescoring cannot change code extraction"
                    )
                row = {
                    **row,
                    "original_style": original["style"],
                    "scoring": {**original, "style": asdict(updated.style)},
                }
            records.append(row)
    directory = run / "rescoring" / name
    directory.mkdir(parents=True, exist_ok=False, mode=0o700)
    manifest = {
        "purpose": "uniform_style_rescore_without_generation_or_execution",
        "created_at": datetime.now(UTC).isoformat(),
        "source_results_sha256": source_hashes,
        "source_manifest_sha256": hashlib.sha256(
            (run / "manifest.json").read_bytes()
        ).hexdigest(),
        "scorer_sha256": hashlib.sha256(
            (root / "src/grader_power/score_style.py").read_bytes()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records": len(records),
        "generation_repeated": False,
        "oracle_repeated": False,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "results.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in records)
    )
    (directory / "report.md").write_text(render_report(records, definition))
    return directory


def render_report(records: list[dict[str, Any]], definition: dict[str, Any]) -> str:
    """Show outcomes and every raw final response; missing labels are never zeroes."""
    lines = [
        "# Mode A example: HumanEval/7",
        "",
        (
            "One task, eight conditions per model, one generation per condition. "
            "This is a pipeline demonstration, not a pilot estimate or human task approval."
        ),
        "",
        (
            "Styles were rescored uniformly after adding annotated-initializer support. "
            "Original generations, extraction decisions, and full base/plus results are unchanged."
        ),
        "",
        "| Model | Attempts | Full correctness | Comprehensions | Loops | Other / unscored |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for profile in definition["profiles"]:
        rows = [r for r in records if r["profile"] == profile]
        styles = Counter(
            r["scoring"]["style"]["label"] if r["scoring"] else None for r in rows
        )
        correct = sum(
            bool(r["scoring"] and r["scoring"]["correctness"].get("full_correct"))
            for r in rows
        )
        other = len(rows) - styles["comprehension_only"] - styles["loop_only"]
        lines.append(
            f"| {profile} | {len(rows)} | {correct}/{len(rows)} | {styles['comprehension_only']} | {styles['loop_only']} | {other} |"
        )
    lines += [
        "",
        (
            "Each full pass covers all 4 base and 178 plus tests. "
            "Ambiguous extraction receives no correctness result. "
            "C = comprehension, L = list-building loop, ? = unscored."
        ),
        "",
        "| User preference | Grader preference | Consequence | "
        + " | ".join(definition["profiles"])
        + " |",
        "| --- | --- | --- | "
        + " | ".join("---" for _ in definition["profiles"])
        + " |",
    ]
    by_cell = {(r["profile"], r["condition_id"]): r for r in records}
    for trial in sorted(definition["trials"], key=lambda t: t["condition_id"]):
        condition = trial["condition"]
        cells = [
            condition["user_preference"],
            condition["grader_preference"],
            condition["grader_consequence"],
        ]
        for profile in definition["profiles"]:
            score = by_cell[profile, trial["condition_id"]]["scoring"]
            label = score["style"]["label"] if score else None
            cells.append(
                "?"
                if label is None
                else {"comprehension_only": "C", "loop_only": "L"}.get(label, "?")
            )
        lines.append("| " + " | ".join(cells) + " |")
    for row in records:
        score = row.get("scoring")
        lines += [
            "",
            f"## {row['profile']} — {row['condition_id']}",
            "",
            f"Generation: {row['status']}; stop reason: {row['stop_reason']}.",
            "",
        ]
        if score:
            lines += [
                (
                    f"Style: {score['style']['label']} ({score['style']['status']}); "
                    f"correctness: {score['correctness']['status']}."
                ),
                "",
            ]
        if row.get("raw_output"):
            response = ModelOutput.model_validate(row["raw_output"]).completion
            fence = "`" * max(
                4, max((len(m) for m in re.findall(r"`+", response)), default=0) + 1
            )
            lines += [fence + "text", response, fence]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Select an existing cohort and a fresh output name; perform no model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must be a simple name")
    root = Path(__file__).resolve().parents[1]
    print(rescore(root / "runs/mode-a-example" / args.run_id, args.name, root))


if __name__ == "__main__":
    main()
