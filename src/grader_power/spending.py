"""Durable rolling reservations for bounded inference and judging.

Amounts are integer nanodollars. Each request/episode has a stable identity;
uncertain or interrupted charges keep their full reservation. The caller must
reserve before dispatch, reconcile only complete usage, and never retry started
requests. This ledger supplements, rather than replaces, collection locks.
"""

import fcntl
import json
import os
from collections.abc import Mapping
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any


class SpendingLimit(RuntimeError):
    """The next complete unit of work cannot fit within the approved ceiling."""


def nanodollars(usd: str | float | Decimal) -> int:
    """Round a finite nonnegative dollar amount upward, never understating cost."""
    value = Decimal(str(usd))
    if not value.is_finite() or value < 0:
        raise ValueError("Cost must be finite and nonnegative")
    return int((value * 1_000_000_000).to_integral_value(rounding=ROUND_CEILING))


def usage_cost(
    usage: Mapping[str, Any], *, input_rate: float, output_rate: float
) -> int:
    """Value complete provider usage conservatively, including cache writes.

    Inspect reports cache tokens separately. Cache reads receive the full input
    price and writes twice that price; no assumed discounts enter the guard.
    Missing usage is an error so callers retain outstanding reservations.
    """
    fields = (
        "input_tokens",
        "output_tokens",
        "input_tokens_cache_read",
        "input_tokens_cache_write",
    )
    # Inspect uses null for unsupported/unused optional cache counters. Required
    # input/output totals must still be present; null totals never become zero.
    counts: dict[str, int] = {}
    for key in fields:
        value = usage.get(key)
        if "cache" in key and value is None:
            value = 0
        if type(value) is not int or value < 0:
            raise ValueError("Complete nonnegative integer token usage is required")
        counts[key] = value
    inputs = (
        counts["input_tokens"]
        + counts["input_tokens_cache_read"]
        + 2 * counts["input_tokens_cache_write"]
    )
    return nanodollars(
        (
            Decimal(inputs) * Decimal(str(input_rate))
            + Decimal(counts["output_tokens"]) * Decimal(str(output_rate))
        )
        / 1_000_000
    )


class SpendingLedger:
    """Serialize admission and settlement; bind the ceiling to one frozen scope."""

    def __init__(self, path: Path, *, cap: int, scope: str) -> None:
        if type(cap) is not int or cap <= 0 or not scope:
            raise ValueError("Require a positive integer ceiling and frozen scope")
        self.path, self.cap, self.scope = path, cap, scope

    def _change(self, action: str, values: Mapping[str, int]) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {
                    "schema_version": 1,
                    "cap": self.cap,
                    "scope": self.scope,
                    "entries": {},
                    "events": [],
                    "overrun": False,
                }
            )
            if state["cap"] != self.cap or state["scope"] != self.scope:
                raise ValueError(
                    "Spending scope/ceiling differs from the frozen ledger"
                )
            entries = state["entries"]
            for identity, amount in values.items():
                if not identity or type(amount) is not int or amount < 0:
                    raise ValueError(
                        "Reservations require stable IDs and integer costs"
                    )
                previous = entries.get(identity)
                if action == "reserve":
                    if previous is not None:
                        if previous["reserved"] != amount:
                            raise ValueError("Reservation changed for an existing ID")
                        continue
                    entries[identity] = {"reserved": amount, "settled": None}
                elif action == "settle":
                    if previous is None:
                        raise ValueError("Cannot settle work that was never reserved")
                    if (
                        previous["settled"] is not None
                        and previous["settled"] != amount
                    ):
                        raise ValueError("Cannot revise a settled usage record")
                    previous["settled"] = amount
                    if amount > previous["reserved"]:
                        state["overrun"] = True
                else:
                    raise ValueError("Unknown ledger operation")
            committed = sum(
                v["reserved"] if v["settled"] is None else v["settled"]
                for v in entries.values()
            )
            if action == "reserve" and (state["overrun"] or committed > self.cap):
                raise SpendingLimit(
                    "Spent plus outstanding plus next reservations exceeds the guard"
                )
            if values:
                state["events"].append({"action": action, "amounts": dict(values)})
            state["committed"] = committed
            state["spent"] = sum(
                v["settled"] for v in entries.values() if v["settled"] is not None
            )
            state["outstanding"] = committed - state["spent"]
            temporary = self.path.with_suffix(f".{os.getpid()}.tmp")
            try:
                with temporary.open("x") as stream:
                    json.dump(state, stream, sort_keys=True, indent=2)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self.path)
                descriptor = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            finally:
                temporary.unlink(missing_ok=True)
            return state

    def reserve(self, amounts: Mapping[str, int]) -> dict[str, Any]:
        """Atomically admit an episode plus judging; repeats do not reserve twice."""
        return self._change("reserve", amounts)

    def settle(self, amounts: Mapping[str, int]) -> dict[str, Any]:
        """Replace bounds with complete usage; overruns persist and block new work."""
        return self._change("settle", amounts)

    def snapshot(self) -> dict[str, Any]:
        """Read a consistent balance without freeing ambiguous reservations."""
        return self._change("settle", {})
