"""Concurrent HTTP contexts preserve identities, pacing and no ambiguous replay."""

import asyncio
import json
from pathlib import Path

import httpx2 as httpx
import pytest

from grader_power.pacing import WaitClock
from grader_power.replication_transport import (
    AccountRPM,
    ModelPacer,
    RequestContext,
    concurrent_transport,
    request_scope,
)


def context(tmp_path: Path, name: str, now: list[float]) -> RequestContext:
    class Clock(WaitClock):
        async def sleep(self, seconds: float) -> None:
            now[0] += seconds
            await asyncio.sleep(0)

    return RequestContext(
        name,
        "fireworks",
        "model-" + name,
        tmp_path / name,
        tmp_path / "operations",
        ModelPacer(tmp_path / "pacing" / name, profile=name, clock=lambda: now[0]),
        AccountRPM(tmp_path / "pacing", clock=lambda: now[0]),
        Clock(),
        lambda: None,
    )


@pytest.mark.parametrize(
    "blocked", ["none", "reported", "throttled", "no_demand", "exhausted"]
)
def test_starting_allowance_calibration_respects_caps_throttles_and_bound(
    tmp_path, blocked
):
    pacer = ModelPacer(tmp_path, profile="qwen", clock=lambda: 1000.0)
    lane = pacer.lane({}, "fireworks")
    lane["last_calibration"] = 900.0
    lane["queue"] = [{"ticket": "waiting"}]
    lane["window"] = [
        {"at": 999.0, "settled": True, "demand": {"output": 3000}} for _ in range(10)
    ]
    if blocked == "reported":
        lane["verified_limits"]["output"] = {"value": 40000, "observed_at": 999}
    elif blocked == "throttled":
        lane["last_429"] = 999
    elif blocked == "no_demand":
        lane["queue"] = []
    elif blocked == "exhausted":
        lane["calibration_probes"] = 8
    pacer.adjust("fireworks", lane, 1000)
    assert lane["limits"]["output"] == (50000 if blocked == "none" else 40000)
    pacer.adjust("fireworks", lane, 1001)
    assert lane["limits"]["output"] == (50000 if blocked == "none" else 40000)


@pytest.mark.parametrize(
    "code,transient,expected",
    [(429, True, 2), (429, False, 1), (503, False, 1), (200, False, 1)],
)
def test_only_explicit_transient_429_repeats(tmp_path, code, transient, expected):
    now = [10000.0]
    c = context(tmp_path, "qwen", now)
    calls = []

    async def handler(request):
        calls.append(request)
        return httpx.Response(
            code if len(calls) == 1 else 200,
            headers={"retry-after": "2"} if transient else {},
            json={
                "error": {"message": "rate limit" if transient else "quota exceeded"}
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with concurrent_transport(), request_scope(c):
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={"model": c.model, "messages": [], "max_tokens": 100},
                )
                with pytest.raises(RuntimeError, match="replayed"):
                    await client.post(
                        "https://api.fireworks.ai/inference/v1/chat/completions",
                        json={"model": c.model, "messages": []},
                    )

    asyncio.run(run())
    assert len(calls) == expected
    assert json.loads((c.path / "transport.json").read_text())["attempts"] == expected


def test_connection_error_never_repeats_or_releases_unknown_ticket(tmp_path):
    c = context(tmp_path, "deepseek", [10000.0])

    async def handler(request):
        raise httpx.ReadTimeout("ambiguous", request=request)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with (
                concurrent_transport(),
                request_scope(c),
                pytest.raises(httpx.ReadTimeout),
            ):
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={"model": c.model, "messages": []},
                )

    asyncio.run(run())
    record = json.loads((c.path / "transport.json").read_text())
    assert record["state"] == "uncertain_failure" and record["attempts"] == 1


def test_concurrent_contexts_and_shared_model_lane(tmp_path):
    now = [10000.0]
    first = context(tmp_path, "deepseek", now)
    second = context(tmp_path, "deepseek", now)
    second.path = tmp_path / "second-request"
    entered = 0
    both = asyncio.Event()

    async def handler(request):
        nonlocal entered
        entered += 1
        if entered == 2:
            both.set()
        await both.wait()
        return httpx.Response(200, json={})

    async def send(client, c):
        with request_scope(c):
            await client.post(
                "https://api.fireworks.ai/inference/v1/chat/completions",
                json={"model": c.model, "messages": [], "max_tokens": 100},
            )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with concurrent_transport():
                await asyncio.gather(send(client, first), send(client, second))

    asyncio.run(run())
    a = json.loads((first.path / "transport.json").read_text())
    b = json.loads((second.path / "transport.json").read_text())
    assert a["ticket"] != b["ticket"]
    assert a["state"] == b["state"] == "accepted"
    assert first.pacer.current["fireworks"] != second.pacer.current["fireworks"]


def test_unauthorized_model_fails_before_http(tmp_path):
    c = context(tmp_path, "qwen", [10000.0])

    async def handler(request):
        pytest.fail("Unauthorized HTTP reached transport")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with (
                concurrent_transport(),
                request_scope(c),
                pytest.raises(ValueError, match="frozen model"),
            ):
                await client.post(
                    "https://api.fireworks.ai/inference/v1/chat/completions",
                    json={"model": "wrong", "messages": []},
                )

    asyncio.run(run())
    assert not (c.path / "transport.json").exists()


def test_account_cap_shared_and_reported_limits_win(tmp_path):
    now = [10000.0]
    a, b = (
        AccountRPM(tmp_path, clock=lambda: now[0]),
        AccountRPM(tmp_path, clock=lambda: now[0]),
    )
    a.observe(httpx.Response(200, headers={"x-ratelimit-limit-requests": "2"}))
    assert a.admit("deepseek") == 0
    assert b.admit("qwen") > 0
    now[0] += 61
    assert b.admit("qwen") == 0
    p = ModelPacer(tmp_path / "qwen", profile="qwen", clock=lambda: now[0])
    p.observe(
        "fireworks",
        httpx.Response(200, headers={"x-ratelimit-limit-tokens-generated": "10000"}),
    )
    assert p.utilization()["fireworks"]["limits"]["output"] == 10000
