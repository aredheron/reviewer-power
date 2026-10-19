"""Optional completed gap-recovery views, without duplicating any assignment."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text())


def verify(
    root: Path, directory: Path, *, endpoint: bool
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify immutable evidence and reproduce paired statistics before display."""
    from grader_power import matched_endpoint_recovery, matched_syntax_analysis

    audit = read(directory / "audit.json")
    if not audit.get("verified") or not audit.get(
        "source_and_retained_artifact_bindings_verified"
    ):
        raise ValueError("Unverified latest recovery")
    for name, expected in audit["bindings"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError("Latest recovery binding changed")
    rows = read(directory / "measured-rows.json")
    result = read(directory / "analysis.json")
    expected = (
        matched_endpoint_recovery.analyze(rows)
        if endpoint
        else matched_syntax_analysis.analyze(rows)
    )
    if any(result.get(k) != v for k, v in expected.items()):
        raise ValueError("Latest paired analysis differs from recomputation")
    if not endpoint and result.get("pending_recoveries") != []:
        raise ValueError("Latest recovery has pending stages")
    return rows, result


def apply(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    """Attach complete latest cells while retaining strict and first recovery cells."""
    from grader_power.results_dashboard import condition_info, matched_tally

    b = root / "runs/recovery/matched-gaps-v1"
    for endpoint, directory in [
        (True, b / "loop-endpoint-v1"),
        (False, b / "analysis-v1"),
    ]:
        if not (directory / "audit.json").is_file():
            continue
        rows, result = verify(root, directory, endpoint=endpoint)
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            family = (
                "MATCHED"
                if endpoint
                else {"naming": "MATCHED_NAMING", "quotes": "MATCHED_QUOTES"}[
                    row["study"]
                ]
            )
            groups[condition_info(family, row)["id"]].append(row)
        found = set()
        for condition in data["conditions"]:
            if condition["id"] not in groups:
                continue
            found.add(condition["id"])
            if endpoint and "recovery_cells" in condition:
                condition["prior_recovery_cells"] = condition["recovery_cells"]
            latest = []
            for cell in condition["cells"]:
                selected = [
                    r
                    for r in groups[condition["id"]]
                    if r["profile"] == cell["model"]
                    and (
                        cell["user"] == "all"
                        or r["condition"]["user_preference"] == cell["user"]
                    )
                ]
                if not selected:
                    raise ValueError("Latest recovery lost a dashboard cell")
                latest.append(
                    {
                        "model": cell["model"],
                        "user": cell["user"],
                        **matched_tally(
                            selected, result["by_condition"], seed=result["seed"]
                        ),
                        "sources": [
                            str((directory / "measured-rows.json").relative_to(root))
                        ],
                    }
                )
            condition["recovery_cells"] = latest
        if set(groups) != found:
            raise ValueError("Latest recovery introduced or removed condition IDs")
        new = {
            "analysis": result,
            "analysis_source": str((directory / "analysis.json").relative_to(root)),
            "adds_observations": 0,
            "note": (
                "Designated loop recovery plus one explicitly approved no-implementation endpoint; original abstention preserved."
                if endpoint
                else "197 designated fresh generations, 114 fresh blinded judgments and 62 original range repairs, with separately approved recovery format qualifications. GLM naming remains held."
            ),
        }
        if endpoint:
            data["matched_prior_recovery"] = data["matched_recovery"]
            data["matched_recovery"] = new
        else:
            data["matched_syntax_recovery"] = new
    return data
