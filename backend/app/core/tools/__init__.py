"""
Tool System for Multi-Agent Framework

This module provides a tool registry and execution system for agents.
Tools allow agents to interact with external systems, perform calculations,
and execute actions beyond text generation.
"""

from .base import Tool, ToolRegistry, ToolResult, ToolError
from .calculator import CalculatorTool
from .web_search import WebSearchTool

# Create global tool registry
registry = ToolRegistry()

# Register built-in tools
registry.register(CalculatorTool())
registry.register(WebSearchTool())

__all__ = [
    "Tool",
    "ToolRegistry",
    "ToolResult",
    "ToolError",
    "CalculatorTool",
    "WebSearchTool",
    "registry",
]
