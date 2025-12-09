"""Lightweight performance sanity checks for agents."""

import time
import pytest

from backend.tests.fixtures.agent_mocks import CountingAgent


@pytest.mark.asyncio
@pytest.mark.slow
async def test_fake_agent_throughput_under_one_second():
    agent = CountingAgent(delay=0.0)
    start = time.perf_counter()

    for _ in range(200):
        await agent.run("ping")

    duration = time.perf_counter() - start
    assert agent.call_count == 200
    assert duration < 1.0
