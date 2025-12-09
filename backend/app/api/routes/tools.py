"""
Tools API Routes

This module provides endpoints for tool management and execution.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from backend.app.core.tools import manager
from backend.app.core.tools.base import ToolSchema


router = APIRouter(
    prefix="/tools",
    tags=["tools"]
)


class ToolExecuteRequest(BaseModel):
    """Request to execute a tool."""
    tool_name: str = Field(description="Name of the tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class ToolExecuteResponse(BaseModel):
    """Response from tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float
    metadata: Dict[str, Any]


class ToolInfo(BaseModel):
    """Information about a tool."""
    name: str
    description: str
    schema: Dict[str, Any]


@router.get("/list", response_model=List[str])
async def list_tools():
    """
    List all available tools.
    
    Returns a list of tool names that agents can use.
    
    Example:
        ```
        GET /tools/list
        
        Response:
        ["calculator", "web_search"]
        ```
    """
    return manager.list_tools()


@router.get("/schemas", response_model=List[ToolInfo])
async def get_tool_schemas():
    """
    Get schemas for all available tools.
    
    Returns detailed information about each tool including
    parameters, descriptions, and usage examples.
    
    Example:
        ```
        GET /tools/schemas
        
        Response:
        [
            {
                "name": "calculator",
                "description": "Perform mathematical calculations",
                "schema": {...}
            }
        ]
        ```
    """
    tools = []
    for tool_name in manager.list_tools():
        info = manager.get_tool_info(tool_name)
        if info:
            tools.append(ToolInfo(**info))
    return tools


@router.get("/openai-functions", response_model=List[Dict[str, Any]])
async def get_openai_functions():
    """
    Get all tools in OpenAI function calling format.
    
    This endpoint returns tool definitions in the format expected
    by OpenAI's function calling API, making it easy to integrate
    tools with GPT models.
    
    Example:
        ```
        GET /tools/openai-functions
        
        Response:
        [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "...",
                    "parameters": {...}
                }
            }
        ]
        ```
    """
    # manager delegates to registry; keep behavior identical
    return manager.registry.get_openai_functions()


@router.get("/{tool_name}", response_model=ToolInfo)
async def get_tool_info(tool_name: str):
    """
    Get information about a specific tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Tool information including schema
        
    Raises:
        404: If tool not found
    """
    info = manager.get_tool_info(tool_name)
    if info is None:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    return ToolInfo(**info)


@router.post("/execute", response_model=ToolExecuteResponse)
async def execute_tool(request: ToolExecuteRequest):
    """
    Execute a tool with given parameters.
    
    This endpoint allows you to directly execute any registered tool.
    Useful for testing tools or building custom workflows.
    
    Args:
        request: Tool execution request with tool name and parameters
        
    Returns:
        Execution result including output, success status, and timing
        
    Example:
        ```
        POST /tools/execute
        {
            "tool_name": "calculator",
            "parameters": {"expression": "2 + 2"}
        }
        
        Response:
        {
            "success": true,
            "output": {"expression": "2 + 2", "result": 4},
            "error": null,
            "execution_time": 0.001,
            "metadata": {"tool": "calculator"}
        }
        ```
    """
    result = await manager.execute(request.tool_name, **request.parameters)
    
    return ToolExecuteResponse(
        success=result.success,
        output=result.output,
        error=result.error,
        execution_time=result.execution_time,
        metadata=result.metadata
    )


@router.get("/health")
async def tools_health():
    """
    Health check endpoint for tools service.
    
    Returns:
        Status message with number of registered tools
    """
    return {
        "status": "healthy",
        "service": "tools",
        "registered_tools": len(manager.registry),
        "tools": manager.list_tools()
    }
