"""
Tool System for Multi-Agent Framework

This module provides a tool registry and execution system for agents.
Tools allow agents to interact with external systems, perform calculations,
and execute actions beyond text generation.
"""

from .base import Tool, ToolRegistry, ToolResult, ToolError
from .calculator import CalculatorTool
from .web_search import WebSearchTool
from .manager import ToolManager
from backend.app.config import get_settings
from backend.app.core.memory.retrieval_tool import RetrievalTool

# Create global tool registry
registry = ToolRegistry()

# Register built-in tools
registry.register(CalculatorTool())
registry.register(WebSearchTool())
registry.register(RetrievalTool())

_settings = get_settings()
manager = ToolManager(
    registry_ref=registry,
    rate_limit_per_minute=_settings.tool_rate_limit_per_minute,
    execution_timeout=_settings.tool_execution_timeout,
    audit_logging=_settings.enable_tool_audit_log,
)

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolError",
    "CalculatorTool",
    "WebSearchTool",
    "RetrievalTool",
    "ToolManager",
    "manager",
    "registry",
]
