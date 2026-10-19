"""Deterministic simulations of shared adaptive pacing; no provider requests."""

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx2 as httpx
import pytest

from grader_power.adaptive_pacing import AdaptivePacer
from grader_power.pacing import WaitClock, paced_http


class MemoryPacer(AdaptivePacer):
    """Simulate time and state transitions without fsync or wall-clock sleeps."""

    def __init__(self) -> None:
        self.now = 10000.0
        super().__init__(Path("/unused"), clock=lambda: self.now)
        self.saved: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []

    @contextmanager
    def state(self) -> Iterator[dict[str, Any]]:
        yield self.saved

    def event(self, provider: str, reason: str, **fields: Any) -> None:
        self.events.append({"provider": provider, "reason": reason, **fields})


def usage(out: int = 250) -> dict[str, Any]:
    return {
        "usage": {
            "input_tokens": 400,
            "output_tokens": out,
            "input_tokens_cache_read": 0,
            "input_tokens_cache_write": 0,
        }
    }


@pytest.mark.parametrize("headers", [False, True])
def test_sustained_simulation_nears_cap_and_tracks_growth(headers: bool) -> None:
    p = MemoryPacer()
    begin = p.now
    accepted: list[tuple[float, int]] = []
    cap = 40000
    for i in range(10000):
        if p.now - begin >= 1800:
            break
        cap = 40000 * (1.25 if p.now - begin >= 900 else 1)
        if headers:
            p.observe(
                "fireworks",
                httpx.Response(
                    200, headers={"x-ratelimit-limit-tokens-generated": str(cap)}
                ),
            )
        wait = p.admit("fireworks", 1000, 4096, ticket=str(i))
        while wait > 0:
            p.now += wait + 0.001
            wait = p.admit("fireworks", 1000, 4096, ticket=str(i))
        actual = sum(n for at, n in accepted if at > p.now - 60)
        assert actual + 250 <= cap, "smooth pacing exceeded simulated provider cap"
        accepted.append((p.now, 250))
        p.usage("fireworks", usage())
    # Throughput measured over late steady-state minutes, not startup.
    for start, limit in [(begin + 600, 40000), (begin + 1500, 50000)]:
        realized = sum(n for at, n in accepted if start <= at < start + 180) / 3
        assert 0.70 < realized / limit < 0.9
    if headers:
        assert not any(
            e["reason"] == "unverified_probe" and e.get("metric") == "output"
            for e in p.events
        )
    else:
        assert any(
            e["reason"] == "unverified_probe" and e.get("metric") == "output"
            for e in p.events
        )


def test_no_growth_from_idle_time_and_cluster_backoff() -> None:
    p = MemoryPacer()
    p.admit("fireworks", 1000, 4096, ticket="first")
    p.now += 1000
    p.admit("fireworks", 1000, 4096, ticket="second")
    assert p.saved["fireworks"]["limits"]["output"] == 40000
    for _ in range(12):
        p.observe("fireworks", httpx.Response(429), delay=45)
    assert p.saved["fireworks"]["fraction"] == pytest.approx(0.85 * 0.8)
    assert p.admit("fireworks", 100, 100, ticket="waiting") >= 45
    assert p.admit("anthropic", 100, 100, ticket="judge") == 0
    assert p.saved["fireworks"]["throttles"] == 12


def test_header_provenance_actual_usage_and_restart(tmp_path: Path) -> None:
    now = [10000.0]
    p = AdaptivePacer(tmp_path, clock=lambda: now[0])
    p.observe(
        "fireworks",
        httpx.Response(
            200,
            headers={
                "x-ratelimit-limit-tokens-generated": "20000",
                "authorization": "DO-NOT-SAVE",
            },
        ),
    )
    assert p.admit("fireworks", 1000, 4096, ticket="one") == 0
    p.usage("fireworks", usage(100))
    first = json.loads(p.path.read_text())
    p.usage("fireworks", usage(100))
    assert json.loads(p.path.read_text()) == first
    p.observe("fireworks", httpx.Response(429), delay=90)
    q = AdaptivePacer(tmp_path, clock=lambda: now[0])
    assert q.admit("fireworks", 100, 100, ticket="later") == 90
    assert all("DO-NOT-SAVE" not in f.read_text() for f in tmp_path.glob("*.json*"))


@pytest.mark.parametrize("code", [429, 500, 503])
def test_transport_retries_only_confirmed_transient_rejection(
    tmp_path: Path, code: int
) -> None:
    class Clock(WaitClock):
        async def sleep(self, seconds: float) -> None:
            now[0] += seconds
            self.waited += seconds

    now = [10000.0]
    clock = Clock()
    p = AdaptivePacer(tmp_path, clock=lambda: now[0])
    bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(
                code, headers={"retry-after": "2"}, text="rate_limit_exceeded"
            )
        return httpx.Response(200, json={"ok": True})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            with paced_http(p, tmp_path / "attempt", clock, lambda: None):
                await client.post(
                    "https://api.fireworks.ai/v1/chat/completions",
                    json={
                        "messages": [{"role": "user", "content": "fixed"}],
                        "max_tokens": 4096,
                    },
                )

    asyncio.run(run())
    assert len(bodies) == (2 if code == 429 else 1)
    assert len(set(bodies)) == 1


def test_slow_response_still_updates_usage_after_window_expires() -> None:
    p = MemoryPacer()
    p.admit("fireworks", 1000, 4096, ticket="slow")
    p.now += 100
    # Another worker can expire the first worker's rolling record.
    p.saved["fireworks"]["window"] = []
    p.usage("fireworks", usage(1000))
    assert p.utilization()["fireworks"]["actual_last_minute"]["output"] == 1000
    p.usage("fireworks", usage(1000))
    assert p.utilization()["fireworks"]["actual_last_minute"]["output"] == 1000
