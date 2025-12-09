"""Lightweight agent mocks for tests and benchmarks."""

from __future__ import annotations

import asyncio
from typing import Any, Callable


class FakeAgent:
    """Agent that echoes or returns a fixed response."""

    def __init__(self, response: Any = "ok") -> None:
        self.response = response
        self.calls: list[str] = []

    async def run(self, prompt: str) -> Any:
        self.calls.append(prompt)
        return self.response


class CountingAgent:
    """Agent that counts invocations and can simulate delay."""

    def __init__(self, delay: float = 0.0, responder: Callable[[str, int], Any] | None = None) -> None:
        self.delay = delay
        self.responder = responder
        self.call_count = 0
        self.last_prompt: str | None = None

    async def run(self, prompt: str) -> Any:
        self.call_count += 1
        self.last_prompt = prompt
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.responder:
            return self.responder(prompt, self.call_count)
        return {"prompt": prompt, "count": self.call_count}


__all__ = ["FakeAgent", "CountingAgent"]
