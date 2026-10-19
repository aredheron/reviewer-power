"""Deterministic throttle simulations; no provider requests or model execution."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx2 as httpx
import pytest

from grader_power.pacing import (
    PacingDeferred,
    SharedPacer,
    WaitClock,
    paced_http,
    retry_after,
    transient_429,
)


class VirtualClock(WaitClock):
    """Advance a synthetic wall clock without making simulations sleep."""

    def __init__(self, *, allowance: float = 600) -> None:
        super().__init__(allowance=allowance)
        self.now = 10000.0

    async def sleep(self, seconds: float) -> None:
        if self.waited + seconds > self.allowance:
            raise PacingDeferred("synthetic wait exhausted")
        self.now += seconds
        self.waited += seconds


def test_retry_after_formats_and_quota_classification() -> None:
    assert retry_after(httpx.Headers({"retry-after": "12.5"}), 0) == 12.5
    assert (
        retry_after(httpx.Headers({"retry-after": "Thu, 01 Jan 1970 00:00:10 GMT"}), 5)
        == 5
    )
    for value in ("NaN", "inf", "bad"):
        assert retry_after(httpx.Headers({"retry-after": value}), 0) is None
    assert transient_429(
        httpx.Response(429, text="rate_limit_exceeded"), "fireworks", 0
    )
    assert not transient_429(
        httpx.Response(429, text="rate_limit_error"), "anthropic", 0
    )
    assert transient_429(
        httpx.Response(429, headers={"retry-after": "1"}, text="rate_limit_error"),
        "anthropic",
        0,
    )
    for body in (
        "insufficient credits",
        "monthly spend cap",
        "billing error",
        "quota exceeded",
    ):
        assert not transient_429(
            httpx.Response(429, headers={"retry-after": "1"}, text=body), "fireworks", 0
        )
    assert not transient_429(httpx.Response(503, text="rate limit"), "fireworks", 0)


def test_shared_admission_serializes_workers_and_token_limits(tmp_path: Path) -> None:
    now = [100.0]
    pacer = SharedPacer(tmp_path, clock=lambda: now[0])
    response = httpx.Response(
        200,
        headers={
            "X-Ratelimit-Limit-Tokens-Prompt": "6000",
            "X-Ratelimit-Limit-Tokens-Generated": "6000",
        },
    )
    pacer.observe("fireworks", response)
    with ThreadPoolExecutor(max_workers=8) as workers:
        waits = list(
            workers.map(lambda _: pacer.admit("fireworks", 1000, 100), range(8))
        )
    assert waits.count(0) == 1
    assert max(waits) == pytest.approx(12.5)
    now[0] += 12.5
    assert pacer.admit("fireworks", 1000, 100) == 0
    # Distinct provider lane isn't held by Fireworks' reservations.
    assert pacer.admit("anthropic", 1000, 100) == 0


def test_shared_cooldown_survives_restart_and_headers_are_allowlisted(
    tmp_path: Path,
) -> None:
    pacer = SharedPacer(tmp_path, clock=lambda: 100)
    pacer.observe(
        "fireworks",
        httpx.Response(
            429,
            headers={
                "authorization": "DO-NOT-RECORD",
                "x-ratelimit-limit-tokens-prompt": "10000",
            },
        ),
        delay=30,
    )
    second = SharedPacer(tmp_path, clock=lambda: 110)
    assert second.admit("fireworks", 100, 100) == 20
    text = (tmp_path / "rate-state.json").read_text()
    assert "DO-NOT-RECORD" not in text
    assert json.loads(text)["fireworks"]["penalty"] == 2


def test_arrival_order_prevents_fast_polling_worker_starvation(tmp_path: Path) -> None:
    now = [100.0]
    pacer = SharedPacer(tmp_path, clock=lambda: now[0])
    assert pacer.admit("fireworks", 1000, 100, ticket="first") == 0
    assert pacer.admit("fireworks", 1000, 100, ticket="waiting") > 0
    assert pacer.admit("fireworks", 1000, 100, ticket="later") > 0
    now[0] += 60
    # Even repeated polls and an unticketed caller cannot take the older slot.
    for _ in range(20):
        assert pacer.admit("fireworks", 1000, 100, ticket="later") > 0
    assert pacer.admit("fireworks", 1000, 100) > 0
    restarted = SharedPacer(tmp_path, clock=lambda: now[0])
    assert restarted.admit("fireworks", 1000, 100, ticket="waiting") == 0
    now[0] += 60
    assert restarted.admit("fireworks", 1000, 100, ticket="later") == 0


def test_cancelled_ticket_releases_only_itself_and_keeps_cooldown(
    tmp_path: Path,
) -> None:
    now = [100.0]
    pacer = SharedPacer(tmp_path, clock=lambda: now[0])
    pacer.observe("fireworks", httpx.Response(429), delay=20)
    assert pacer.admit("fireworks", 1000, 100, ticket="cancelled") == 20
    assert pacer.admit("fireworks", 1000, 100, ticket="next") == 20
    pacer.cancel_admission("fireworks", "cancelled")
    pacer.cancel_admission("fireworks", "unrelated")
    assert pacer.admit("fireworks", 1000, 100, ticket="next") == 20
    now[0] += 20
    assert pacer.admit("fireworks", 1000, 100, ticket="next") == 0
    lane = json.loads(pacer.path.read_text())["fireworks"]
    assert lane["queue"] == []
    assert lane["throttles"] == 1 and lane["penalty"] == 2


def test_twelve_waiters_get_a_turn_under_adversarial_polling(tmp_path: Path) -> None:
    now = [100.0]
    pacer = SharedPacer(tmp_path, clock=lambda: now[0])
    assert pacer.admit("fireworks", 1000, 100, ticket="warmup") == 0
    for i in range(12):
        assert pacer.admit("fireworks", 1000, 100, ticket=f"worker-{i}") > 0
    admitted: list[int] = []
    for _ in range(12):
        now[0] += 1
        # Later workers always poll first, reproducing the unfair scheduling risk.
        for i in reversed(range(12)):
            if (
                i not in admitted
                and pacer.admit("fireworks", 1000, 100, ticket=f"worker-{i}") == 0
            ):
                admitted.append(i)
    assert admitted == list(range(12))


@pytest.mark.parametrize("host", ["api.fireworks.ai", "api.anthropic.com"])
def test_429_retries_exact_request_without_duplicate_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    clock = VirtualClock()
    monkeypatch.setattr("grader_power.pacing.time.time", lambda: clock.now)
    pacer = SharedPacer(tmp_path, clock=lambda: clock.now)
    bodies: list[bytes] = []
    times: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        times.append(clock.now)
        if len(bodies) < 3:
            return httpx.Response(
                429,
                headers={"retry-after": "10"},
                json={"error": {"type": "rate_limit_error"}},
            )
        return httpx.Response(200, json={"answer": "one accepted output"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with paced_http(pacer, tmp_path / "attempt", clock, lambda: None):
                result = await client.post(
                    f"https://{host}/v1/messages",
                    json={
                        "messages": [{"role": "user", "content": "same prompt"}],
                        "max_tokens": 4096,
                    },
                    headers={"authorization": "DO-NOT-RECORD"},
                )
                assert result.status_code == 200

    asyncio.run(run())
    assert len(bodies) == 3 and len(set(bodies)) == 1
    assert times[1] - times[0] >= 10 and times[2] - times[1] >= 10
    record = json.loads((tmp_path / "attempt/transport/0001.json").read_text())
    assert record["state"] == "accepted" and record["attempts"] == 3
    events = [
        json.loads(line)
        for line in (tmp_path / "transport-events.jsonl").read_text().splitlines()
    ]
    assert (
        len([event for event in events if event["state"] == "pending_rate_limit"]) == 2
    )
    assert len([event for event in events if event.get("event") == "send_started"]) == 3
    assert not (tmp_path / "attempt/pacing-pending.json").exists()
    assert all("DO-NOT-RECORD" not in p.read_text() for p in tmp_path.rglob("*.json*"))


@pytest.mark.parametrize(
    "failure", ["connection", "timeout", "500", "503", "quota", "unknown429"]
)
def test_ambiguous_or_permanent_failures_are_never_retried(
    tmp_path: Path, failure: str
) -> None:
    count = 0

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        if failure == "connection":
            raise httpx.ConnectError("private diagnostic", request=request)
        if failure == "timeout":
            raise httpx.ReadTimeout("private diagnostic", request=request)
        if failure == "quota":
            return httpx.Response(
                429, text="insufficient credit", headers={"retry-after": "1"}
            )
        if failure == "unknown429":
            return httpx.Response(429, text="unknown error")
        return httpx.Response(int(failure), text="provider unavailable")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with paced_http(
                SharedPacer(tmp_path), tmp_path / "attempt", WaitClock(), lambda: None
            ):
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={"messages": []},
                )

    if failure in ("connection", "timeout"):
        with pytest.raises(httpx.TransportError):
            asyncio.run(run())
    else:
        asyncio.run(run())
    assert count == 1
    assert (
        "private diagnostic"
        not in (tmp_path / "attempt/transport/0001.json").read_text()
    )


def test_persistent_throttle_is_durable_pending_not_endless_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = VirtualClock(allowance=20)
    monkeypatch.setattr("grader_power.pacing.time.time", lambda: clock.now)
    pacer = SharedPacer(tmp_path, clock=lambda: clock.now)
    count = 0

    def handle(_: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(429, headers={"retry-after": "30"}, text="rate limit")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with paced_http(pacer, tmp_path / "attempt", clock, lambda: None):
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={"messages": []},
                )

    with pytest.raises(PacingDeferred):
        asyncio.run(run())
    assert count == 1
    assert (
        json.loads((tmp_path / "attempt/pacing-pending.json").read_text())["state"]
        == "pending_rate_limit"
    )
    assert len(list((tmp_path / "deferred").glob("*.json"))) == 1
    assert json.loads(pacer.path.read_text())["fireworks"]["queue"] == []


def test_wait_clock_excludes_only_pacing_and_cancels_active_timeout() -> None:
    async def run() -> None:
        clock = WaitClock(allowance=0.2)

        async def throttled() -> int:
            await clock.sleep(0.08)
            return 7

        assert await clock.wait_for(throttled(), timeout=0.03) == 7
        cancelled = False

        async def ordinary_work() -> None:
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            finally:
                cancelled = True

        with pytest.raises(TimeoutError):
            await clock.wait_for(ordinary_work(), timeout=0.02)
        assert cancelled
        with pytest.raises(PacingDeferred):
            await clock.sleep(0.2)

    asyncio.run(run())


@pytest.mark.parametrize("worker_count", [4, 6, 12])
def test_long_run_simulation_respects_pacing(tmp_path: Path, worker_count: int) -> None:
    now = [10000.0]
    pacer = SharedPacer(tmp_path, clock=lambda: now[0])
    pacer.observe(
        "fireworks",
        httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-requests": "120",
                "x-ratelimit-limit-tokens-prompt": "120000",
                "x-ratelimit-limit-tokens-generated": "60000",
            },
        ),
    )
    sends = []
    # Ready workers contend on every tick; one dispatch at a time is admitted,
    # while accepted requests may continue running concurrently at the provider.
    for _ in range(1000):
        waits = [pacer.admit("fireworks", 1000, 512) for _ in range(worker_count)]
        assert waits.count(0) == 1
        sends.append(now[0])
        now[0] += min(w for w in waits if w > 0)
    assert len(sends) == 1000
    # Output-token demand is slightly more restrictive than the request limit.
    assert (len(sends) - 1) / (sends[-1] - sends[0]) * 60 == pytest.approx(
        60000 * 0.8 / 512
    )
