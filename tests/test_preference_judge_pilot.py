"""Verify bounded, resumable judge piloting without credentials or network calls."""

import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from grader_power.spending import SpendingLedger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
pilot = importlib.import_module("scripts.pilot_preference_judges")


def case(identity: str = "example", *, repeat_of: str | None = None) -> dict[str, Any]:
    """Return an inert source snippet with a known naming label."""
    value = {
        "id": identity,
        "family": "naming",
        "entry_point": "solve",
        "original_parameters": ["values"],
        "code": "def solve(values):\n    total_count = len(values)\n    return total_count\n",
        "expected_label": "snake_case_only",
        "notes": "injection calibration",
    }
    if repeat_of is not None:
        value["repeat_of"] = repeat_of
    return value


def response(
    *, malformed: bool = False, usage: bool = True, structured: bool = False
) -> Any:
    """Construct a provider-shaped response; no generated code executes."""
    completion = (
        "not json"
        if malformed
        else json.dumps(
            {
                "status": "classified",
                "target_forms": ["snake_case"],
                "evidence": [
                    {
                        "start_line": 2,
                        "end_line": 2,
                        "quote": "    total_count = len(values)",
                    }
                ],
                "flags": [],
                "rationale": "One informative local binding.",
            }
        )
    )
    if structured and not malformed:
        value = json.loads(completion)
        for evidence in value["evidence"]:
            del evidence["quote"]
        completion = json.dumps(value)
    raw = {
        "completion": completion,
        "usage": {"input_tokens": 10, "output_tokens": 20} if usage else None,
    }
    return SimpleNamespace(
        completion=completion,
        choices=[SimpleNamespace(stop_reason="stop")],
        model_dump=lambda **kwargs: raw,
    )


class FakeModel:
    """Record concurrency and return deterministic responses without network access."""

    def __init__(
        self,
        *,
        error: bool = False,
        malformed: bool = False,
        usage: bool = True,
        structured: bool = False,
    ) -> None:
        self.calls = 0
        self.active = 0
        self.peak = 0
        self.error, self.malformed, self.usage = error, malformed, usage
        self.structured = structured
        self.config = None

    async def generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        self.config = kwargs.get("config")
        self.active += 1
        self.peak = max(self.peak, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        if self.error:
            raise RuntimeError("secret-credential-must-not-be-written")
        return response(
            malformed=self.malformed, usage=self.usage, structured=self.structured
        )


def run_cases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model: FakeModel,
    cases: list[dict[str, Any]],
    *,
    cap: int = pilot.CAP,
    version: str = "v1",
) -> SpendingLedger:
    """Use a real durable ledger around mocked provider dispatch."""
    monkeypatch.setattr(
        pilot, "resolved_retry_policy", lambda model: {"max_retries": 0}
    )
    ledger = SpendingLedger(tmp_path / "spending.json", cap=cap, scope="test")
    asyncio.run(
        pilot.generate_cases(
            tmp_path,
            {
                "cases": cases,
                "systems": {"naming": "Classify syntax."},
                "prompt_version": version,
                "generation": pilot.generation_for(version).model_dump(
                    mode="json", exclude_none=True
                ),
            },
            model,
            ledger,
            "secret-credential-must-not-be-written",
        )
    )
    return ledger


def test_frozen_evidence_rejects_changes(tmp_path: Path) -> None:
    path = tmp_path / "definition.json"
    pilot.freeze(path, {"prompt": "original"})
    pilot.freeze(path, {"prompt": "original"})
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        pilot.freeze(path, {"prompt": "changed"})


def test_prepare_is_offline_and_excludes_answer_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        pilot,
        "get_model",
        lambda *a, **k: pytest.fail("Offline prepare constructed a provider"),
    )
    definition = pilot.prepare(ROOT, tmp_path)
    cases = definition["cases"]
    assert (
        sum(
            row["origin"] == "agent_authored_reference" and not row.get("repeat_of")
            for row in cases
        )
        == 16
    )
    assert sum(bool(row.get("repeat_of")) for row in cases) == 8
    assert (
        sum((tmp_path / "cases" / row["id"] / "outcome.json").exists() for row in cases)
        == 2
    )
    packet = json.loads(next((tmp_path / "cases").glob("*/packet.json")).read_text())
    assert set(packet) == set(pilot.PACKET_FIELDS)
    assert "expected_label" not in packet and "notes" not in packet
    assert pilot.prepare(ROOT, tmp_path) == definition


def test_parallelism_settlement_and_no_resend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel()
    cases = [case(f"case-{index}") for index in range(9)]
    ledger = run_cases(tmp_path, monkeypatch, model, cases)
    assert model.calls == 9 and model.peak == 4
    assert ledger.snapshot()["outstanding"] == 0
    assert ledger.snapshot()["spent"] == 9 * (10 * 4000 + 20 * 20000)
    run_cases(tmp_path, monkeypatch, model, cases)
    assert model.calls == 9


def test_uncertain_start_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pilot.freeze(tmp_path / "cases/example/started.json", {"uncertain": True})
    model = FakeModel()
    run_cases(tmp_path, monkeypatch, model, [case()])
    assert model.calls == 0
    report = pilot.summarize(tmp_path, {"cases": [case()]})
    assert report["families"]["naming"]["statuses"] == {"started_without_outcome": 1}


def test_request_errors_redact_and_keep_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(error=True)
    with pytest.raises(RuntimeError, match="Initial provider request failed"):
        run_cases(tmp_path, monkeypatch, model, [case(), case("unattempted")])
    ledger = SpendingLedger(tmp_path / "spending.json", cap=pilot.CAP, scope="test")
    outcome = json.loads((tmp_path / "cases/example/outcome.json").read_text())
    assert outcome == {
        "status": "request_error",
        "label": None,
        "error_type": "RuntimeError",
    }
    assert ledger.snapshot()["outstanding"] > 0
    assert model.calls == 1
    assert not (tmp_path / "cases/unattempted/started.json").exists()
    assert "secret-credential" not in "".join(
        path.read_text() for path in tmp_path.rglob("*.json")
    )


@pytest.mark.parametrize(
    "malformed,usage,status",
    [(True, True, "invalid_response"), (False, False, "classified")],
)
def test_malformed_and_missing_usage_are_distinct(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,
    usage: bool,
    status: str,
) -> None:
    ledger = run_cases(
        tmp_path, monkeypatch, FakeModel(malformed=malformed, usage=usage), [case()]
    )
    outcome = json.loads((tmp_path / "cases/example/outcome.json").read_text())
    assert outcome["status"] == status
    assert (ledger.snapshot()["outstanding"] == 0) == usage
    if malformed:
        assert outcome["label"] is None


def test_budget_refusal_does_not_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel()
    run_cases(tmp_path, monkeypatch, model, [case()], cap=1)
    assert model.calls == 0
    assert not (tmp_path / "cases/example/started.json").exists()


def test_summary_never_counts_unresolved_as_substantive(tmp_path: Path) -> None:
    rows = [
        case(),
        case("missing"),
        case("repeat", repeat_of="example"),
        {**case("invalid"), "expected_label": "invalid"},
    ]
    for identity, status, label in [
        ("example", "classified", "snake_case_only"),
        ("repeat", "classified", "snake_case_only"),
        ("invalid", "precheck_invalid", None),
    ]:
        pilot.freeze(
            tmp_path / "cases" / identity / "outcome.json",
            {"status": status, "label": label},
        )
    result = pilot.summarize(tmp_path, {"cases": rows})["families"]["naming"]
    assert result["confusion"] == {"snake_case_only": {"snake_case_only": 1}}
    assert result["substantive_agreement"] == 1
    assert result["all_valid_agreement"] == 0.5
    assert result["repeat_stability"] == {
        "assigned": 1,
        "classified_pairs": 1,
        "same_label": 1,
    }
    assert result["prechecks_match_expected"] and not result["proposed_gate_passed"]


def test_v2_freezes_schema_and_separate_validation_cases(tmp_path: Path) -> None:
    definition = pilot.prepare(ROOT, tmp_path, version="v2")
    assert definition["prompt_version"] == "v2"
    assert definition["generation"]["reasoning_effort"] == "medium"
    schema = definition["generation"]["response_schema"]["json_schema"]
    assert schema["additionalProperties"] is False
    assert list(schema["properties"])[-1] == "target_forms"
    assert len(definition["cases"]) == 89
    assert (
        sum(row["origin"] == "independent_validation" for row in definition["cases"])
        == 16
    )
    assert (
        "fixtures/style_preferences/judge_validation_v2.json"
        in definition["source_sha256"]
    )
    assert pilot.prepare(ROOT, tmp_path, version="v2") == definition
    with pytest.raises(ValueError, match="Frozen evidence changed"):
        pilot.prepare(ROOT, tmp_path, version="v1")


def test_v3_preserves_old_cases_and_distinguishes_absent_evidence(
    tmp_path: Path,
) -> None:
    """Revised labels are explicit overlays; all current labels match source audits."""
    legacy = {row["id"]: row for row in pilot.build_cases(ROOT, version="v2")}
    definition = pilot.prepare(ROOT, tmp_path, version="v3")
    assert len(definition["cases"]) == 105
    assert (
        sum(row["origin"] == "independent_validation" for row in definition["cases"])
        == 16
    )
    for row in definition["cases"]:
        if row["expected_label"] == "invalid":
            continue
        assert pilot.audit_style(pilot.packet_for(row)).label == row["expected_label"]
        if row.get("previous_expected_label"):
            assert legacy[row["id"]]["expected_label"] == "neither"
            assert row["expected_label"] in ("no_naming_evidence", "no_quote_evidence")
    schema = definition["generation"]["response_schema"]["json_schema"]
    assert "no_naming_evidence" in json.dumps(schema)
    assert "no_quote_evidence" in json.dumps(schema)
    # Anthropic's structured output requires one enum, not string-schema unions.
    forms_schema = schema["properties"]["target_forms"]["items"]
    assert forms_schema["type"] == "string" and "anyOf" not in forms_schema
    assert set(forms_schema["enum"]) == {
        "snake_case",
        "camelCase",
        "single_quotes",
        "double_quotes",
        "other",
        "no_naming_evidence",
        "no_quote_evidence",
    }
    assert "no_naming_evidence" not in pilot.generation_for("v2").model_dump_json()


def test_summary_displays_no_evidence_separately_from_other_and_unknown(
    tmp_path: Path,
) -> None:
    rows = [case("neutral"), case("other"), case("unknown")]
    for identity, label in [("neutral", "no_naming_evidence"), ("other", "neither")]:
        pilot.freeze(
            tmp_path / "cases" / identity / "outcome.json",
            {"status": "classified", "label": label},
        )
    result = pilot.summarize(tmp_path, {"cases": rows})["families"]["naming"]
    assert result["label_counts"] == {"no_naming_evidence": 1, "neither": 1}
    assert result["no_style_evidence"] == 1
    assert result["statuses"]["not_attempted"] == 1


def test_v2_uses_frozen_config_and_hydrates_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = FakeModel(structured=True)
    run_cases(tmp_path, monkeypatch, model, [case()], version="v2")
    outcome = json.loads((tmp_path / "cases/example/outcome.json").read_text())
    assert outcome["status"] == "classified"
    assert (
        outcome["judgment"]["evidence"][0]["quote"] == "    total_count = len(values)"
    )
    assert model.config is not None
    assert model.config.reasoning_effort == "medium"
    assert model.config.response_schema is not None
    packet = pilot.packet_for(case())
    assert pilot.reservation(
        "system", packet, {"large_schema": "x" * 1000}
    ) > pilot.reservation("system", packet)


def test_fresh_validation_accuracy_is_separate(tmp_path: Path) -> None:
    rows = [
        case(),
        {**case("fresh-correct"), "origin": "independent_validation"},
        {**case("fresh-missing"), "origin": "independent_validation"},
    ]
    for identity in ("example", "fresh-correct"):
        pilot.freeze(
            tmp_path / "cases" / identity / "outcome.json",
            {"status": "classified", "label": "snake_case_only"},
        )
    report = pilot.summarize(tmp_path, {"cases": rows, "prompt_version": "v2"})
    fresh = report["families"]["naming"]["independent_validation"]
    assert fresh["assigned"] == 2 and fresh["correct"] == 1
    assert fresh["all_valid_agreement"] == 0.5
    assert fresh["substantive_agreement"] == 1
