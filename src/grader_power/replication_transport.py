"""Concurrent request-local transport auditing with shared model/account pacing."""

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx2 as httpx

from grader_power.adaptive_pacing import AdaptivePacer
from grader_power.pacing import (
    WaitClock,
    append_event,
    locked,
    retry_after,
    transient_429,
    write_json,
)


class ModelPacer(AdaptivePacer):
    """Share a model lane; keep Qwen's unverified probes below its published tier."""

    def __init__(self, directory: Path, *, profile: str, **kwargs: Any) -> None:
        super().__init__(directory, **kwargs)
        self.profile = profile

    def adjust(self, provider: str, lane: dict[str, Any], now: float) -> None:
        before = dict(lane["limits"])
        super().adjust(provider, lane, now)
        # Discover an unknown starting allowance from real demand, separately
        # from the provider's slower quota growth. Stop discovery permanently
        # after any throttle; reported dimensions are never probed.
        last = lane.setdefault("last_calibration", now)
        probes = lane.setdefault("calibration_probes", 0)
        if before != lane["limits"]:
            lane["last_calibration"] = now
        elif (
            provider == "fireworks"
            and not lane["last_429"]
            and probes < 8
            and now - last >= 60
            and lane.get("queue")
        ):
            settled = [r for r in lane["window"] if r["settled"] and r["at"] > now - 60]
            changed = False
            for metric, old in list(lane["limits"].items()):
                actual = sum(r["demand"].get(metric, 0) for r in settled)
                if (
                    metric in lane["verified_limits"]
                    or len(settled) < 10
                    or actual < 0.65 * old
                ):
                    continue
                ceiling = {
                    "input": 216_000_000,
                    "uncached": 54_000_000,
                    "output": 2_160_000,
                    "requests": 6000,
                }[metric]
                if self.profile in ("qwen", "glm") and metric != "requests":
                    ceiling = ceiling // (10 if self.profile == "qwen" else 5)
                new = min(ceiling, old * 1.25)
                if new > old:
                    lane["limits"][metric] = new
                    changed = True
                    self.event(
                        provider,
                        "starting_allowance_calibration",
                        metric=metric,
                        old=old,
                        new=new,
                        actual_last_minute=actual,
                        verified_account_quota=False,
                    )
            lane["last_calibration"] = now
            if changed:
                lane["calibration_probes"] += 1
        if self.profile in ("qwen", "glm"):
            scale = 2 if self.profile == "glm" else 1
            for metric, ceiling in {
                "input": 21600000,
                "uncached": 5400000,
                "output": 216000,
            }.items():
                ceiling *= scale
                if (
                    metric not in lane["verified_limits"]
                    and lane["limits"][metric] > ceiling
                ):
                    lane["limits"][metric] = float(ceiling)
                    self.event(
                        provider,
                        "unverified_tier_ceiling",
                        metric=metric,
                        ceiling=ceiling,
                    )


class AccountRPM:
    """Reserve every Fireworks HTTP subattempt against one durable account envelope."""

    def __init__(
        self, directory: Path, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.path = directory / "account-rpm.json"
        self.clock = clock

    def admit(self, ticket: str) -> float:
        """Return a wait or reserve once; use an 85% target below the 6000 RPM maximum."""
        now = self.clock()
        with locked(self.path.with_suffix(".lock")):
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"cap": 6000, "window": []}
            )
            window = [r for r in state["window"] if r["at"] > now - 60]
            if any(r["ticket"] == ticket for r in window):
                return 0.0
            if len(window) >= max(1, math.floor(state["cap"] * 0.85)):
                return max(0.05, min(r["at"] for r in window) + 60 - now)
            window.append({"ticket": ticket, "at": now})
            write_json(self.path, {**state, "window": window})
        return 0.0

    def observe(self, response: httpx.Response) -> None:
        """Reported lower request caps constrain all model lanes together."""
        text = response.headers.get("x-ratelimit-limit-requests")
        if not text:
            return
        try:
            value = float(text)
        except ValueError:
            return
        if not math.isfinite(value) or value <= 0:
            return
        with locked(self.path.with_suffix(".lock")):
            state = (
                json.loads(self.path.read_text())
                if self.path.exists()
                else {"window": []}
            )
            write_json(
                self.path,
                {**state, "cap": min(6000, value), "reported_at": self.clock()},
            )


@dataclass
class RequestContext:
    """One paid request owns its pacing reconciliation and transport history."""

    profile: str
    provider: str
    model: str
    path: Path
    operations: Path
    pacer: ModelPacer
    account: AccountRPM
    clock: WaitClock
    guard: Callable[[], None]
    sent: bool = False


CURRENT: ContextVar[RequestContext | None] = ContextVar(
    "replication_request", default=None
)


@contextmanager
def request_scope(context: RequestContext) -> Iterator[None]:
    """Isolate overlapping requests without changing global SDK hooks per request."""
    token = CURRENT.set(context)
    try:
        yield
    finally:
        CURRENT.reset(token)


@contextmanager
def concurrent_transport() -> Iterator[None]:
    """Install one dispatcher; only complete, explicitly transient 429s may repeat.

    Every accepted/ambiguous outcome is terminal for its original request ID.
    Context-local pacer objects share durable model files but never each other's
    current ticket. No SDK, timeout, connection or 5xx replay is introduced.
    """
    original = httpx.AsyncClient.send

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        provider = {
            "api.fireworks.ai": "fireworks",
            "api.anthropic.com": "anthropic",
        }.get(request.url.host)
        if provider is None or request.method != "POST":
            return await original(client, request, **kwargs)
        c = CURRENT.get()
        if c is None:
            raise RuntimeError("Provider POST has no authorized request context")
        body = await request.aread()
        payload = json.loads(body)
        if (
            provider != c.provider
            or not isinstance(payload, dict)
            or payload.get("model") != c.model
            or "messages" not in payload
        ):
            raise ValueError("Provider request differs from frozen model")
        if c.sent or (c.path / "transport.json").exists():
            raise RuntimeError("Already-started transport cannot be replayed")
        c.sent = True
        ticket = hashlib.sha256(str(c.path.resolve()).encode()).hexdigest()
        record: dict[str, Any] = {
            "profile": c.profile,
            "provider": provider,
            "model": c.model,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "attempts": 0,
            "state": "pending_admission",
            "ticket": ticket,
        }
        rejected = 0
        try:
            while True:
                c.guard()
                wait = c.pacer.admit(
                    provider,
                    max(1, math.ceil(len(body) / 3)),
                    int(payload.get("max_tokens", 4096)),
                    ticket=ticket,
                )
                if wait > 0:
                    write_json(c.path / "transport.json", record)
                    await c.clock.sleep(min(wait, 2.0))
                    continue
                if provider == "fireworks":
                    account_ticket = ticket + "/" + str(record["attempts"] + 1)
                    while (wait := c.account.admit(account_ticket)) > 0:
                        c.guard()
                        await c.clock.sleep(min(wait, 2.0))
                c.guard()
                record.update(state="in_flight", attempts=record["attempts"] + 1)
                write_json(c.path / "transport.json", record)
                append_event(
                    c.operations / "transport-events.jsonl",
                    {**record, "event": "send_started"},
                )
                try:
                    response = await original(client, request, **kwargs)
                except BaseException:
                    record["state"] = "uncertain_failure"
                    write_json(c.path / "transport.json", record)
                    append_event(
                        c.operations / "transport-events.jsonl",
                        {**record, "event": "send_uncertain"},
                    )
                    raise
                record["http_status"] = response.status_code
                if provider == "fireworks":
                    c.account.observe(response)
                if response.status_code == 429:
                    await response.aread()
                if transient_429(response, provider, time.time()):
                    rejected += 1
                    delay = max(
                        retry_after(response.headers, time.time()) or 0,
                        min(60.0, 2.0 ** min(rejected, 6)),
                    ) + random.uniform(0.1, 1)
                    c.pacer.observe(provider, response, delay=delay)
                    record["state"] = "pending_rate_limit"
                    write_json(c.path / "transport.json", record)
                    append_event(
                        c.operations / "transport-events.jsonl",
                        {**record, "retry_after_seconds": delay},
                    )
                    await response.aclose()
                    continue
                c.pacer.observe(provider, response)
                record["state"] = "accepted" if response.is_success else "not_retried"
                write_json(c.path / "transport.json", record)
                append_event(
                    c.operations / "transport-events.jsonl",
                    {**record, "event": "response_headers"},
                )
                if response.status_code == 429:
                    write_json(
                        c.operations / "pacing/deferred" / (ticket + ".json"), record
                    )
                return response
        except BaseException:
            c.pacer.cancel_admission(provider, ticket)
            if record["state"] in ("pending_admission", "pending_rate_limit"):
                write_json(c.path / "transport.json", record)
                write_json(
                    c.operations / "pacing/deferred" / (ticket + ".json"), record
                )
            raise

    with patch.object(httpx.AsyncClient, "send", send):
        yield
