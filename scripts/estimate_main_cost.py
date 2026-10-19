"""Estimate main inference costs from frozen prompts and pilot usage, without API calls."""

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.inspect_main import load_run
from evals.inspect_mode_a import digest, json_text, save


def estimate(root: Path, definition: dict[str, Any]) -> dict[str, Any]:
    """Project uncached-equivalent costs; scenarios are not billing guarantees."""
    rates = definition["plan"]["usd_per_million_tokens"]
    sources = {
        "gpt-oss-120b": root / "runs/sampling-check/gpt-oss-t07-20261010",
        "claude-sonnet": root / "runs/rl-condition-pilot/rl-condition-20261010",
    }
    result: dict[str, Any] = {"components": {}, "sources_sha256": {}}
    for profile, run in sources.items():
        manifest = json.loads((run / "manifest.json").read_text())
        by_id = {a["assignment_id"]: a for a in manifest["assignments"]}
        path = run / profile / "results.jsonl"
        rows = [
            r
            for r in map(json.loads, path.read_text().splitlines())
            if r["kind"] == "core" and r.get("raw_output", {}).get("usage")
        ]
        result["sources_sha256"][str(path.relative_to(root))] = digest(
            path.read_bytes()
        )
        usage = [r["raw_output"]["usage"] for r in rows]
        inputs = [
            u["input_tokens"]
            + (u.get("input_tokens_cache_read") or 0)
            + (u.get("input_tokens_cache_write") or 0)
            for u in usage
        ]
        ratio = sum(inputs) / sum(
            len(json_text(by_id[r["assignment_id"]]["messages"]).encode()) for r in rows
        )
        input_tokens = ratio * sum(
            len(json_text(a["messages"]).encode()) for a in definition["assignments"]
        )
        output_tokens = mean(u["output_tokens"] for u in usage) * len(
            definition["assignments"]
        )
        result["components"][profile] = {
            "requests": len(definition["assignments"]),
            "pilot_responses": len(rows),
            "projected_input_tokens": input_tokens,
            "projected_output_tokens": output_tokens,
            "usd": (
                input_tokens * rates[profile]["input"]
                + output_tokens * rates[profile]["output"]
            )
            / 1e6,
        }
    paths = sorted(
        (sources["gpt-oss-120b"] / "judge-with-source-contract").glob("*/raw.json")
    )
    judge_usage = [json.loads(p.read_text())["usage"] for p in paths]
    for p in paths:
        result["sources_sha256"][str(p.relative_to(root))] = digest(p.read_bytes())
    requests = 2 * len(definition["assignments"])
    input_tokens = (
        mean(
            u["input_tokens"]
            + (u.get("input_tokens_cache_read") or 0)
            + (u.get("input_tokens_cache_write") or 0)
            for u in judge_usage
        )
        * requests
    )
    output_tokens = mean(u["output_tokens"] for u in judge_usage) * requests
    result["components"]["judge"] = {
        "requests": requests,
        "pilot_responses": len(paths),
        "projected_input_tokens": input_tokens,
        "projected_output_tokens": output_tokens,
        "usd": (
            input_tokens * rates["claude-sonnet"]["input"]
            + output_tokens * rates["claude-sonnet"]["output"]
        )
        / 1e6,
    }
    result.update(
        projected_total_usd=sum(c["usd"] for c in result["components"].values()),
        double_token_scenario_usd=2
        * sum(c["usd"] for c in result["components"].values()),
        rates=rates,
        price_sources=definition["plan"]["price_sources"],
        method="Generation input scaled by exact prompt byte lengths using pilot token/byte ratios; generation outputs and judge inputs/outputs use pilot means. All input tokens charged at base uncached rates; no cache discounts assumed.",
        limitations="Not a spend cap or worst-case bound. Task/code length, cache writes, reasoning usage and billing changes may alter costs. Excludes retries, extra audits, confirmation study and existing pod charges. No new API requests made.",
    )
    return result


def main() -> None:
    """Write a reproducible estimate linked to the frozen main manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cohort, _, definition = load_run(root, args.run_id)
    result = estimate(root, definition)
    result["manifest_sha256"] = digest((cohort / "manifest.json").read_bytes())
    save(cohort / "cost-estimate.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("projected_total_usd", "double_token_scenario_usd")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
