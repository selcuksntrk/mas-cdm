"""
Tool Manager

Provides orchestration around the tool registry including:
- Rate limiting
- Execution timeouts
- Permission checks
- Audit logging
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

from backend.app.config import get_settings
from backend.app.core.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Simple sliding-window rate limiter (per manager instance)."""

    max_calls_per_minute: int
    timestamps: list[float] = field(default_factory=list)

    def allow(self) -> bool:
        now = time.monotonic()
        window_start = now - 60
        self.timestamps = [t for t in self.timestamps if t >= window_start]
        if len(self.timestamps) >= self.max_calls_per_minute:
            return False
        self.timestamps.append(now)
        return True


class ToolManager:
    """Wraps tool registry execution with guardrails."""

    def __init__(
        self,
        *,
        registry_ref: Optional[ToolRegistry] = None,
        allowed_tools: Optional[Iterable[str]] = None,
        rate_limit_per_minute: int = 60,
        execution_timeout: float = 15.0,
        audit_logging: bool = True,
    ) -> None:
        self.registry = registry_ref
        self.allowed_tools: Optional[Set[str]] = set(allowed_tools) if allowed_tools else None
        self.rate_limiter = RateLimiter(rate_limit_per_minute)
        self.execution_timeout = execution_timeout
        self.audit_logging = audit_logging

    def list_tools(self) -> list[str]:
        if self.allowed_tools is None:
            return self.registry.list_tools()
        return [t for t in self.registry.list_tools() if t in self.allowed_tools]

    def get_tool_info(self, tool_name: str) -> Optional[dict[str, Any]]:
        tool = self.registry.get(tool_name)
        if not tool:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "schema": tool.get_schema().model_dump(),
        }

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        if self.allowed_tools is not None and tool_name not in self.allowed_tools:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' is not permitted",
                execution_time=0.0,
                metadata={"tool": tool_name},
            )

        if not self.rate_limiter.allow():
            return ToolResult(
                success=False,
                output=None,
                error="Rate limit exceeded for tools",
                execution_time=0.0,
                metadata={"tool": tool_name},
            )

        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.registry.execute(tool_name, **kwargs),
                timeout=self.execution_timeout,
            )
        except asyncio.TimeoutError:
            duration = time.monotonic() - start
            return ToolResult(
                success=False,
                output=None,
                error="Tool execution timed out",
                execution_time=duration,
                metadata={"tool": tool_name},
            )

        if self.audit_logging:
            self._audit(tool_name, result)
        return result

    def _audit(self, tool_name: str, result: ToolResult) -> None:
        logger.info(
            "Tool executed",
            extra={
                "tool": tool_name,
                "success": result.success,
                "error": result.error,
                "metadata": result.metadata,
            },
        )


__all__ = ["ToolManager"]
