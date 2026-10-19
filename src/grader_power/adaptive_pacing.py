"""Observed-usage pacing for new experiments; historical pacing stays intact.

Each provider has one durable FIFO and token window across worker processes.
Reported limits are authoritative. Only dimensions without reported limits are
probed upward. Financial reservations remain in the separate spending ledger.
"""

import math
from pathlib import Path
from typing import Any

import httpx2 as httpx

from grader_power.pacing import HEADERS, SharedPacer, append_event

TARGET = 0.85
RAMP_SECONDS = 900
PROBE_CEILINGS = {
    "fireworks": {
        "requests": 6000.0,
        "input": 216_000_000.0,
        "uncached": 54_000_000.0,
        "output": 2_160_000.0,
    },
    "anthropic": {
        "requests": 10000.0,
        "input": 10_000_000.0,
        "output": 2_000_000.0,
        "combined": 12_000_000.0,
    },
}


class AdaptivePacer(SharedPacer):
    """Coordinate estimates, actual usage, header limits and bounded probing.

    Use one instance per sequential worker. Call ``usage`` immediately after each
    durable response. Uncertain requests retain their estimate for the token window;
    the financial ledger independently retains their full monetary reservation.
    """

    def __init__(self, directory: Path, **kwargs: Any) -> None:
        super().__init__(directory, **kwargs)
        self.current: dict[str, str] = {}
        self.current_estimate: dict[str, int] = {}
        self.settled_tickets: set[str] = set()

    def lane(self, state: dict[str, Any], provider: str) -> dict[str, Any]:
        lane = super().lane(state, provider)
        if "adaptive_version" not in lane:
            lane.update(
                adaptive_version=1,
                verified_limits={},
                fraction=TARGET,
                last_probe=self.clock(),
                last_backoff=0.0,
                last_recovery=self.clock(),
                input_ratio=1.0,
                uncached_ratio=1.0,
                window=[],
                queue=[],
            )
        return lane

    def event(self, provider: str, reason: str, **fields: Any) -> None:
        """Persist numeric limit provenance; never request bodies or headers wholesale."""
        append_event(
            self.directory / "limit-events.jsonl",
            {"provider": provider, "reason": reason, **fields},
        )

    def adjust(self, provider: str, lane: dict[str, Any], now: float) -> None:
        """Probe unverified dimensions only with sustained demand and no throttles."""
        if now - max(lane["last_429"], lane["last_recovery"]) >= 300:
            previous = lane["fraction"]
            lane["fraction"] = min(TARGET, previous * 1.1)
            lane["last_recovery"] = now
            if lane["fraction"] != previous:
                self.event(
                    provider, "healthy_fraction_recovery", fraction=lane["fraction"]
                )
        if now - lane["last_probe"] < RAMP_SECONDS:
            return
        lane["last_probe"] = now
        if now - lane["last_429"] < RAMP_SECONDS:
            return
        actual = {
            key: sum(r["demand"].get(key, 0) for r in lane["window"] if r["settled"])
            for key in lane["limits"]
        }
        for key, old in list(lane["limits"].items()):
            if key in lane["verified_limits"] or actual[key] < old * 0.5:
                continue
            new = min(old * 1.25, PROBE_CEILINGS[provider][key])
            if new > old:
                lane["limits"][key] = new
                self.event(
                    provider,
                    "unverified_probe",
                    metric=key,
                    old=old,
                    new=new,
                    actual_last_minute=actual[key],
                    verified_account_quota=False,
                )

    def admit(
        self,
        provider: str,
        input_tokens: int,
        max_output: int,
        *,
        ticket: str | None = None,
    ) -> float:
        """Admit one FIFO request against estimated inflight plus recent actual tokens."""
        if ticket is None:
            raise ValueError("Adaptive admission requires a durable request ticket")
        now = self.clock()
        with self.state() as state:
            lane = self.lane(state, provider)
            lane["window"] = [r for r in lane["window"] if r["at"] > now - 60]
            queue = lane["queue"]
            if not any(r["ticket"] == ticket for r in queue):
                queue.append({"ticket": ticket})
            wait = max(lane["next_at"], lane["cooldown_until"]) - now
            if queue[0]["ticket"] != ticket:
                return max(0.05, wait)
            self.adjust(provider, lane, now)
            demand = {
                "requests": 1.0,
                "input": max(1.0, input_tokens * lane["input_ratio"]),
                "output": min(max_output, lane["output_estimate"] * 1.1),
            }
            demand["uncached"] = max(1.0, demand["input"] * lane["uncached_ratio"])
            demand["combined"] = demand["input"] + demand["output"]
            for key, limit in lane["limits"].items():
                total = sum(r["demand"].get(key, 0) for r in lane["window"])
                # A single request larger than the target window can run alone.
                if total and total + demand[key] > max(
                    demand[key], limit * lane["fraction"]
                ):
                    wait = max(wait, min(r["at"] for r in lane["window"]) + 60 - now)
            if wait > 0:
                return wait
            interval = max(60 * demand[k] / v for k, v in lane["limits"].items())
            lane["next_at"] = now + interval / lane["fraction"]
            lane["last_send"] = now
            lane["window"].append(
                {
                    "ticket": ticket,
                    "at": now,
                    "demand": demand,
                    "settled": False,
                    "input_estimate": input_tokens,
                }
            )
            queue.pop(0)
            self.current[provider] = ticket
            self.current_estimate[provider] = input_tokens
            return 0.0

    def observe(
        self, provider: str, response: httpx.Response, *, delay: float | None = None
    ) -> None:
        """Honor provider caps and cooldowns; back off once per throttle cluster."""
        now = self.clock()
        with self.state() as state:
            lane = self.lane(state, provider)
            for header, key in HEADERS[provider].items():
                try:
                    value = float(response.headers[header])
                except (KeyError, ValueError):
                    continue
                if not math.isfinite(value) or value <= 0:
                    continue
                old = lane["limits"].get(key)
                if old != value or key not in lane["verified_limits"]:
                    self.event(
                        provider,
                        "provider_header",
                        metric=key,
                        old=old,
                        new=value,
                        header=header,
                    )
                lane["limits"][key] = value
                lane["verified_limits"][key] = {"value": value, "observed_at": now}
            if delay is not None:
                lane["cooldown_until"] = max(lane["cooldown_until"], now + delay)
                lane["throttles"] += 1
                lane["last_429"] = now
                if now - lane["last_backoff"] >= 30:
                    lane["fraction"] = max(0.1, lane["fraction"] * 0.8)
                    lane["last_backoff"] = now
                    self.event(
                        provider,
                        "transient_429_backoff",
                        fraction=lane["fraction"],
                        cooldown_until=lane["cooldown_until"],
                    )
                ticket = self.current.get(provider)
                lane["window"] = [r for r in lane["window"] if r["ticket"] != ticket]
            elif response.is_success:
                lane["accepted"] += 1
        # The historical implementation handles Anthropic reset headers. Parse
        # only the relevant numeric/date fields here, without its repeated backoff.
        if provider == "anthropic":
            from datetime import datetime

            with self.state() as state:
                lane = self.lane(state, provider)
                for metric in ("requests", "input-tokens", "output-tokens", "tokens"):
                    try:
                        remaining = float(
                            response.headers[f"anthropic-ratelimit-{metric}-remaining"]
                        )
                        reset = datetime.fromisoformat(
                            response.headers[f"anthropic-ratelimit-{metric}-reset"]
                        ).timestamp()
                    except (KeyError, ValueError):
                        continue
                    if remaining <= 0 and math.isfinite(reset):
                        lane["cooldown_until"] = max(lane["cooldown_until"], reset)

    def usage(self, provider: str, raw: dict[str, Any]) -> None:
        """Reconcile a complete saved response immediately, once per admission."""
        usage = raw.get("usage") or {}
        inp, out = usage.get("input_tokens"), usage.get("output_tokens")
        cached = usage.get("input_tokens_cache_read") or 0
        writes = usage.get("input_tokens_cache_write") or 0
        if (
            not isinstance(inp, int)
            or not isinstance(out, int)
            or not all(type(v) is int and v >= 0 for v in (inp, out, cached, writes))
        ):
            return
        # Narrow the required counts after validation.
        inp, out = int(inp), int(out)
        total = inp + cached + writes
        ticket = self.current.get(provider)
        if ticket is None or ticket in self.settled_tickets:
            return
        with self.state() as state:
            lane = self.lane(state, provider)
            record = next((r for r in lane["window"] if r["ticket"] == ticket), None)
            if record is None:
                record = {
                    "ticket": ticket,
                    "at": self.clock(),
                    "settled": False,
                    "input_estimate": self.current_estimate[provider],
                }
                lane["window"].append(record)
            if record["settled"]:
                return
            self.settled_tickets.add(ticket)
            record["demand"] = {
                "requests": 1,
                "input": total,
                "uncached": inp + writes,
                "output": out,
                "combined": total + out,
            }
            record["settled"] = True
            lane["output_estimate"] = max(
                16.0, 0.8 * lane["output_estimate"] + 0.2 * out
            )
            lane["input_ratio"] = max(
                0.05,
                min(
                    2.0,
                    0.8 * lane["input_ratio"]
                    + 0.2 * total / max(1, record["input_estimate"]),
                ),
            )
            lane["uncached_ratio"] = min(
                1.0,
                max(
                    0.01,
                    0.8 * lane["uncached_ratio"] + 0.2 * (inp + writes) / max(1, total),
                ),
            )

    def utilization(self) -> dict[str, Any]:
        """Expose recent realized use and its binding dimension for monitoring."""
        now = self.clock()
        with self.state() as state:
            result = {}
            for provider, lane in state.items():
                recent = [r for r in lane.get("window", []) if r["at"] > now - 60]
                actual = {
                    k: sum(r["demand"].get(k, 0) for r in recent if r["settled"])
                    for k in lane["limits"]
                }
                ratios = {k: actual[k] / v for k, v in lane["limits"].items()}
                result[provider] = {
                    "actual_last_minute": actual,
                    "utilization": ratios,
                    "limits": lane["limits"],
                    "fraction": lane["fraction"],
                    "queued": len(lane.get("queue", [])),
                    "throttles": lane["throttles"],
                }
            return result
