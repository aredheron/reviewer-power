"""Observable admission, concurrency and restart properties of the spending guard."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from grader_power.spending import SpendingLedger, SpendingLimit, nanodollars, usage_cost


def test_rolling_reservations_pause_and_resume_without_losing_uncertain_cost(
    tmp_path: Path,
) -> None:
    ledger = SpendingLedger(tmp_path / "ledger.json", cap=100, scope="frozen")
    ledger.reserve({"episode1": 60, "judge1": 20})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"episode2": 60, "judge2": 20})
    ledger.settle({"episode1": 5})
    resumed = SpendingLedger(tmp_path / "ledger.json", cap=100, scope="frozen")
    assert resumed.snapshot()["outstanding"] == 20
    with pytest.raises(SpendingLimit):
        resumed.reserve({"episode2": 60, "judge2": 20})
    resumed.settle({"judge1": 3})
    resumed.reserve({"episode2": 60, "judge2": 20})
    assert resumed.snapshot()["committed"] == 88
    resumed.reserve({"episode2": 60, "judge2": 20})
    assert resumed.snapshot()["committed"] == 88


@pytest.mark.parametrize("worker_count", [4, 12])
def test_concurrent_admission_is_atomic(tmp_path: Path, worker_count: int) -> None:
    def reserve(i: int) -> bool:
        try:
            SpendingLedger(tmp_path / "ledger.json", cap=100, scope="frozen").reserve(
                {str(i): 60}
            )
        except SpendingLimit:
            return False
        return True

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        assert sum(pool.map(reserve, range(worker_count))) == 1


def test_unknown_usage_does_not_release_reservation_and_overrun_blocks(
    tmp_path: Path,
) -> None:
    ledger = SpendingLedger(tmp_path / "ledger.json", cap=100, scope="frozen")
    ledger.reserve({"episode": 60})
    with pytest.raises(ValueError):
        usage_cost({"input_tokens": 5}, input_rate=0.15, output_rate=0.6)
    assert ledger.snapshot()["outstanding"] == 60
    ledger.settle({"episode": 70})
    with pytest.raises(SpendingLimit):
        ledger.reserve({"next": 1})
    assert ledger.snapshot()["spent"] == 70


def test_scope_and_settled_values_cannot_change(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    ledger = SpendingLedger(path, cap=100, scope="original")
    ledger.reserve({"episode": 50})
    ledger.settle({"episode": 10})
    for cap, scope in ((101, "original"), (100, "different")):
        with pytest.raises(ValueError):
            SpendingLedger(path, cap=cap, scope=scope).snapshot()
    with pytest.raises(ValueError):
        ledger.settle({"episode": 9})


def test_usage_includes_cache_writes_and_rounds_up() -> None:
    assert (
        usage_cost(
            {
                "input_tokens": 100,
                "output_tokens": 10,
                "input_tokens_cache_read": 20,
                "input_tokens_cache_write": 30,
            },
            input_rate=2,
            output_rate=10,
        )
        == 460000
    )
    assert nanodollars("0.0000000001") == 1
    for value in ("NaN", "Infinity", "-1"):
        with pytest.raises(ValueError):
            nanodollars(value)
