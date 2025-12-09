import asyncio
import pytest

from backend.app.core.tools.base import Tool, ToolRegistry, ToolParameter
from backend.app.core.tools.manager import ToolManager
from backend.app.core.tools.calculator import CalculatorTool
from backend.app.core.tools.web_search import WebSearchTool


class _EchoTool(Tool):
    def get_name(self) -> str:
        return "echo"

    def get_description(self) -> str:
        return "Echo back the message"

    def get_parameters(self):
        return [
            ToolParameter(
                name="message",
                type="string",
                description="Message to echo",
                required=True,
            )
        ]

    async def execute(self, message: str):
        return {"echo": message}


class _SlowTool(Tool):
    def get_name(self) -> str:
        return "slow"

    def get_description(self) -> str:
        return "Sleep for a while"

    def get_parameters(self):
        return []

    async def execute(self):
        await asyncio.sleep(2)
        return {"done": True}


@pytest.mark.asyncio
async def test_rate_limit_enforced():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    mgr = ToolManager(registry_ref=reg, rate_limit_per_minute=2, execution_timeout=1)

    first = await mgr.execute("calculator", expression="1+1")
    second = await mgr.execute("calculator", expression="2+2")
    third = await mgr.execute("calculator", expression="3+3")

    assert first.success is True
    assert second.success is True
    assert third.success is False
    assert "rate limit" in third.error.lower()


@pytest.mark.asyncio
async def test_timeout_enforced():
    reg = ToolRegistry()
    reg.register(_SlowTool())
    mgr = ToolManager(registry_ref=reg, rate_limit_per_minute=5, execution_timeout=0.1)

    result = await mgr.execute("slow")

    assert result.success is False
    assert "timed out" in result.error.lower()


@pytest.mark.asyncio
async def test_allowed_tools_filtering():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(WebSearchTool())
    mgr = ToolManager(
        registry_ref=reg,
        allowed_tools={"calculator"},
        rate_limit_per_minute=5,
        execution_timeout=1,
    )

    allowed = await mgr.execute("calculator", expression="4*4")
    blocked = await mgr.execute("web_search", query="test")

    assert allowed.success is True
    assert blocked.success is False
    assert "not permitted" in blocked.error.lower()


@pytest.mark.asyncio
async def test_web_search_caching_flag():
    tool = WebSearchTool()
    first = await tool.run(query="caching test", max_results=2)
    second = await tool.run(query="caching test", max_results=2)

    assert first.success is True
    assert second.success is True
    assert first.output["cached"] is False
    assert second.output["cached"] is True
