"""Shared provider pacing and explicit, auditable transport-only 429 retries.

SDK/Inspect retries remain disabled. Only a complete HTTP 429 rejection known to
mean temporary throttling can repeat; accepted or ambiguously failed sends never
repeat. State contains only numeric limits and nonsecret request fingerprints.
"""

import asyncio
import fcntl
import hashlib
import json
import math
import os
import random
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, TypeVar
from unittest.mock import patch

import httpx2 as httpx

T = TypeVar("T")
WAIT_ALLOWANCE = 600.0
DEFAULT_LIMITS = {
    "fireworks": {
        "requests": 240.0,
        "input": 200000.0,
        "uncached": 200000.0,
        "output": 40000.0,
    },
    "anthropic": {"requests": 60.0, "input": 240000.0, "output": 32000.0},
}
HEADERS = {
    "fireworks": {
        "x-ratelimit-limit-requests": "requests",
        "x-ratelimit-limit-tokens-prompt": "input",
        "x-ratelimit-limit-tokens-cache-adjusted-prompt": "uncached",
        "x-ratelimit-limit-tokens-generated": "output",
    },
    "anthropic": {
        f"anthropic-ratelimit-{name}-limit": key
        for name, key in (
            ("requests", "requests"),
            ("input-tokens", "input"),
            ("output-tokens", "output"),
            ("tokens", "combined"),
        )
    },
}


class PacingDeferred(RuntimeError):
    """A known rejected/unsent request is retained; no more work may be admitted."""


def write_json(path: Path, value: Any) -> None:
    """Atomically persist operational state, never credentials or request headers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def locked(path: Path) -> Iterator[None]:
    """Serialize short state changes across isolated worker processes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def append_event(path: Path, value: dict[str, Any]) -> None:
    """Append one complete fsynced audit line under an independent process lock."""
    with locked(path.with_suffix(".lock")), path.open("a") as stream:
        stream.write(json.dumps({"at": time.time(), **value}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def retry_after(headers: httpx.Headers, now: float) -> float | None:
    """Parse seconds or an HTTP-date, rejecting invalid/nonfinite durations."""
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            delay = parsedate_to_datetime(value).timestamp() - now
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, delay) if math.isfinite(delay) else None


def transient_429(response: httpx.Response, provider: str, now: float) -> bool:
    """Require positive throttle evidence; exclude quota/billing/unknown 429s."""
    if response.status_code != 429:
        return False
    body = response.text.lower()
    if any(
        word in body
        for word in ("credit", "balance", "billing", "spend", "payment", "quota")
    ):
        return False
    delay = retry_after(response.headers, now)
    if provider == "anthropic":
        # Anthropic's spend-cap 429 has no Retry-After. Be conservative even if
        # a future undocumented transient variant lacks it.
        return delay is not None
    return delay is not None or any(
        text in body
        for text in (
            "rate limit",
            "rate_limit",
            "too many requests",
            "tokens per minute",
            "requests per minute",
        )
    )


class SharedPacer:
    """Smooth estimated request/token demand at 80% of reported provider limits.

    Conservative startup ceilings are local pacing choices, not claims about the
    account's quota. Header limits replace them. Cache use is initially charged
    as uncached demand. Output estimates use recent usage with headroom; spending
    reservations are separate and still use worst-case token bounds.
    """

    def __init__(
        self, directory: Path, *, clock: Callable[[], float] = time.time
    ) -> None:
        self.directory, self.clock = directory, clock
        self.path = directory / "rate-state.json"

    @contextmanager
    def state(self) -> Iterator[dict[str, Any]]:
        """Read and atomically commit the shared numeric scheduling state."""
        with locked(self.path.with_suffix(".lock")):
            value = json.loads(self.path.read_text()) if self.path.exists() else {}
            yield value
            write_json(self.path, value)

    def lane(self, state: dict[str, Any], provider: str) -> dict[str, Any]:
        """Create a bounded initial lane without inspecting account credentials."""
        return state.setdefault(
            provider,
            {
                "limits": dict(DEFAULT_LIMITS[provider]),
                "next_at": 0.0,
                "cooldown_until": 0.0,
                "penalty": 1.0,
                "last_429": 0.0,
                "last_restore": self.clock(),
                "throttles": 0,
                "accepted": 0,
                "output_estimate": 512.0,
            },
        )

    def admit(
        self,
        provider: str,
        input_tokens: int,
        max_output: int,
        *,
        ticket: str | None = None,
    ) -> float:
        """Reserve spacing in arrival order, or return a bounded polling delay.

        Transport calls retain a unique ticket while waiting. A later worker
        cannot take every newly available slot and starve an earlier episode.
        Cancellation removes only its own ticket; a crashed worker's ticket
        remains fail-closed until reviewed recovery proves that worker is gone.
        """
        now = self.clock()
        with self.state() as state:
            lane = self.lane(state, provider)
            queue = lane.setdefault("queue", [])
            if ticket is not None and not any(
                entry["ticket"] == ticket for entry in queue
            ):
                queue.append({"ticket": ticket, "pid": os.getpid()})
            wait = max(lane["next_at"], lane["cooldown_until"]) - now
            if queue and queue[0]["ticket"] != ticket:
                return max(0.05, wait)
            if wait > 0:
                return wait
            if now - max(lane["last_429"], lane["last_restore"]) >= 900:
                lane["penalty"] = max(1.0, lane["penalty"] / 1.25)
                lane["last_restore"] = now
            demand = {
                "requests": 1,
                "input": input_tokens,
                "uncached": input_tokens,
                "output": min(max_output, lane["output_estimate"]),
            }
            demand["combined"] = input_tokens + demand["output"]
            interval = max(
                60 * demand[key] / limit for key, limit in lane["limits"].items()
            )
            lane["next_at"] = now + interval * lane["penalty"] / 0.8
            lane["last_send"] = now
            if queue:
                queue.pop(0)
            return 0.0

    def cancel_admission(self, provider: str, ticket: str) -> None:
        """Release only this cancelled waiter's place, without admitting a send."""
        with self.state() as state:
            lane = self.lane(state, provider)
            lane["queue"] = [
                entry for entry in lane.get("queue", []) if entry["ticket"] != ticket
            ]

    def observe(
        self, provider: str, response: httpx.Response, *, delay: float | None = None
    ) -> None:
        """Learn allowlisted numeric headers and propagate any cooldown globally."""
        now = self.clock()
        with self.state() as state:
            lane = self.lane(state, provider)
            for header, key in HEADERS[provider].items():
                try:
                    value = float(response.headers[header])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(value) and value > 0:
                    lane["limits"][key] = value
            if provider == "anthropic":
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
            if delay is not None:
                lane["cooldown_until"] = max(lane["cooldown_until"], now + delay)
                lane["penalty"] = min(16.0, lane["penalty"] * 2)
                lane["last_429"] = now
                lane["throttles"] += 1
            elif response.is_success:
                lane["accepted"] += 1

    def usage(self, provider: str, raw: dict[str, Any]) -> None:
        """Reconcile the scheduling estimate from complete saved output usage."""
        output = raw.get("usage", {}).get("output_tokens") if raw.get("usage") else None
        if type(output) is not int or output < 0:
            return
        with self.state() as state:
            lane = self.lane(state, provider)
            lane["output_estimate"] = max(
                256.0, min(4096.0, max(output * 1.5, lane["output_estimate"] * 0.98))
            )


class WaitClock:
    """Exclude only explicit pacing sleeps from active work, with a hard allowance."""

    def __init__(self, *, allowance: float = WAIT_ALLOWANCE) -> None:
        self.allowance = allowance
        self.waited = 0.0
        self.wait_started: float | None = None

    def waiting(self) -> float:
        """Include an in-progress wait so deadline enforcement remains accurate."""
        return self.waited + (
            time.monotonic() - self.wait_started
            if self.wait_started is not None
            else 0.0
        )

    def monotonic(self) -> float:
        """Use real monotonic time minus controlled queue/backoff intervals."""
        return time.monotonic() - self.waiting()

    async def sleep(self, seconds: float) -> None:
        """Bound accumulated wait per episode/judge; never grant unlimited lifetime."""
        if seconds < 0 or not math.isfinite(seconds):
            raise ValueError("Invalid pacing wait")
        if self.waiting() + seconds > self.allowance:
            raise PacingDeferred("Throttle waiting allowance exhausted")
        self.wait_started = time.monotonic()
        try:
            await asyncio.sleep(seconds)
        finally:
            self.waited += time.monotonic() - self.wait_started
            self.wait_started = None

    async def wait_for(self, awaitable: Awaitable[T], timeout: float | None) -> T:
        """Keep the original active timeout plus at most the bounded wait allowance."""
        if timeout is None:
            return await awaitable
        task = asyncio.ensure_future(awaitable)
        active_end = self.monotonic() + timeout
        hard_end = time.monotonic() + timeout + self.allowance
        try:
            while not task.done():
                remaining = min(
                    active_end - self.monotonic(), hard_end - time.monotonic()
                )
                if remaining <= 0:
                    raise TimeoutError
                await asyncio.wait({task}, timeout=min(0.1, remaining))
            return task.result()
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


@contextmanager
def paced_http(
    pacer: SharedPacer, path: Path, clock: WaitClock, guard: Callable[[], None]
) -> Iterator[None]:
    """Retry only pre-generation HTTP throttles below the existing redaction layer.

    Every network subattempt is durable before send. Uncertain failures cannot
    be retried. Exhaustion leaves a pending marker and stops collection; it does
    not authorize replay of a partially executed episode after process restart.
    """
    original = httpx.AsyncClient.send
    sequence = 0

    async def send(
        client: httpx.AsyncClient, request: httpx.Request, **kwargs: Any
    ) -> httpx.Response:
        nonlocal sequence
        provider = {
            "api.fireworks.ai": "fireworks",
            "api.anthropic.com": "anthropic",
        }.get(request.url.host)
        if provider is None or request.method != "POST":
            return await original(client, request, **kwargs)
        guard()
        body = await request.aread()
        payload = json.loads(body)
        if not isinstance(payload, dict) or "messages" not in payload:
            return await original(client, request, **kwargs)
        sequence += 1
        identity = f"{sequence:04d}"
        record_path = path / "transport" / f"{identity}.json"
        ticket = hashlib.sha256(str(record_path.resolve()).encode()).hexdigest()
        record: dict[str, Any] = {
            "provider": provider,
            "request_sha256": hashlib.sha256(body).hexdigest(),
            "attempts": 0,
            "state": "pending_admission",
        }
        rejected = 0
        while True:
            try:
                guard()
                wait = pacer.admit(
                    provider,
                    max(1, math.ceil(len(body) / 3)),
                    int(payload.get("max_tokens", 4096)),
                    ticket=ticket,
                )
                if wait > 0:
                    write_json(record_path, record)
                    await clock.sleep(min(wait, 5.0))
                    continue
            except BaseException:
                pacer.cancel_admission(provider, ticket)
                if record["state"] in ("pending_rate_limit", "pending_admission"):
                    deferred = {**record, "transport_record": str(record_path)}
                    write_json(path / "pacing-pending.json", deferred)
                    write_json(
                        pacer.directory
                        / "deferred"
                        / (
                            hashlib.sha256(str(record_path).encode()).hexdigest()
                            + ".json"
                        ),
                        deferred,
                    )
                raise
            record.update(state="in_flight", attempts=record["attempts"] + 1)
            write_json(record_path, record)
            append_event(
                pacer.directory / "transport-events.jsonl",
                {**record, "event": "send_started", "record": str(record_path)},
            )
            try:
                response = await original(client, request, **kwargs)
            except BaseException:
                record["state"] = "uncertain_failure"
                write_json(record_path, record)
                append_event(
                    pacer.directory / "transport-events.jsonl",
                    {**record, "event": "send_uncertain", "record": str(record_path)},
                )
                raise
            record["http_status"] = response.status_code
            if response.status_code == 429:
                await response.aread()
            if transient_429(response, provider, time.time()):
                rejected += 1
                floor = retry_after(response.headers, time.time())
                delay = max(
                    floor or 0.0, min(60.0, 2.0 ** min(rejected, 6))
                ) + random.uniform(0.1, 1.0)
                pacer.observe(provider, response, delay=delay)
                record["state"] = "pending_rate_limit"
                write_json(record_path, record)
                append_event(
                    pacer.directory / "transport-events.jsonl",
                    {
                        **record,
                        "retry_after_seconds": delay,
                        "record": str(
                            record_path.relative_to(pacer.directory.parent.parent)
                        )
                        if record_path.is_relative_to(pacer.directory.parent.parent)
                        else str(record_path),
                    },
                )
                await response.aclose()
                continue
            pacer.observe(provider, response)
            record["state"] = "accepted" if response.is_success else "not_retried"
            write_json(record_path, record)
            append_event(
                pacer.directory / "transport-events.jsonl",
                {**record, "event": "response_headers", "record": str(record_path)},
            )
            if response.status_code == 429:
                # Unknown/quota rejections need human review, never a retry loop.
                deferred = {
                    **record,
                    "reason": "nontransient_or_unknown_429",
                    "transport_record": str(record_path),
                }
                write_json(path / "pacing-pending.json", deferred)
                write_json(
                    pacer.directory
                    / "deferred"
                    / (hashlib.sha256(str(record_path).encode()).hexdigest() + ".json"),
                    deferred,
                )
            return response

    with patch.object(httpx.AsyncClient, "send", send):
        yield
