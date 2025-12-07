"""
Base Tool System

This module defines the base classes for the tool system, including
Tool abstraction, ToolRegistry, and execution framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List, Type
from pydantic import BaseModel, Field
from datetime import datetime, UTC
import json


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "number", "boolean", "object", "array"
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolSchema(BaseModel):
    """Schema definition for a tool (OpenAI function calling format)."""
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema format
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolResult(BaseModel):
    """Result of a tool execution."""
    success: bool
    output: Any
    error: Optional[str] = None
    execution_time: float  # in seconds
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ToolError(Exception):
    """Exception raised when tool execution fails."""
    def __init__(self, message: str, tool_name: str, original_error: Optional[Exception] = None):
        self.message = message
        self.tool_name = tool_name
        self.original_error = original_error
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class Tool(ABC):
    """
    Base class for all tools.
    
    A tool is a capability that agents can use to interact with external systems
    or perform specific computations. Tools must:
    - Have a unique name
    - Define input/output schema
    - Implement execution logic
    - Handle errors gracefully
    - Support sandboxing/safety checks
    """
    
    def __init__(self):
        self.name = self.get_name()
        self.description = self.get_description()
        self.schema = self._generate_schema()
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the unique name of this tool."""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """Return a description of what this tool does."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> List[ToolParameter]:
        """Return the list of parameters this tool accepts."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """
        Execute the tool with the given parameters.
        
        This method should:
        - Validate inputs
        - Perform the operation
        - Return the result
        - Raise ToolError on failure
        """
        pass
    
    def validate_input(self, **kwargs) -> bool:
        """
        Validate input parameters before execution.
        
        Override this method to add custom validation logic.
        """
        parameters = {p.name: p for p in self.get_parameters()}
        
        # Check required parameters
        for param in self.get_parameters():
            if param.required and param.name not in kwargs:
                raise ToolError(
                    f"Missing required parameter: {param.name}",
                    self.name
                )
        
        # Check for unknown parameters
        for key in kwargs:
            if key not in parameters:
                raise ToolError(
                    f"Unknown parameter: {key}",
                    self.name
                )
        
        return True
    
    def is_safe(self, **kwargs) -> bool:
        """
        Safety check before execution (sandboxing).
        
        Override this method to add safety checks specific to your tool.
        Return False to prevent execution.
        """
        return True
    
    async def run(self, **kwargs) -> ToolResult:
        """
        Run the tool with error handling and timing.
        
        This is the main entry point for tool execution.
        """
        import time
        
        start_time = time.time()
        
        try:
            # Validate inputs
            self.validate_input(**kwargs)
            
            # Safety check
            if not self.is_safe(**kwargs):
                raise ToolError(
                    "Tool execution blocked by safety check",
                    self.name
                )
            
            # Execute
            output = await self.execute(**kwargs)
            
            execution_time = time.time() - start_time
            
            return ToolResult(
                success=True,
                output=output,
                execution_time=execution_time,
                metadata={"tool": self.name}
            )
            
        except ToolError as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                execution_time=execution_time,
                metadata={"tool": self.name}
            )
        except Exception as e:
            execution_time = time.time() - start_time
            return ToolResult(
                success=False,
                output=None,
                error=f"Unexpected error: {str(e)}",
                execution_time=execution_time,
                metadata={"tool": self.name, "error_type": type(e).__name__}
            )
    
    def _generate_schema(self) -> ToolSchema:
        """Generate JSON schema for this tool."""
        parameters = self.get_parameters()
        
        # Convert parameters to JSON Schema format
        properties = {}
        required = []
        
        for param in parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.default is not None:
                properties[param.name]["default"] = param.default
            if param.required:
                required.append(param.name)
        
        schema_params = {
            "type": "object",
            "properties": properties,
            "required": required
        }
        
        return ToolSchema(
            name=self.name,
            description=self.description,
            parameters=schema_params
        )
    
    def get_schema(self) -> ToolSchema:
        """Return the tool schema."""
        return self.schema
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "schema": self.schema.model_dump()
        }


class ToolRegistry:
    """
    Registry for managing available tools.
    
    The registry maintains a collection of tools and provides
    methods for tool discovery and execution.
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a new tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> None:
        """Unregister a tool."""
        if tool_name in self._tools:
            del self._tools[tool_name]
    
    def get(self, tool_name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """List all registered tool names."""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[ToolSchema]:
        """Get schemas for all registered tools."""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def get_openai_functions(self) -> List[Dict[str, Any]]:
        """Get all tools in OpenAI function calling format."""
        return [tool.get_schema().to_openai_format() for tool in self._tools.values()]
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                success=False,
                output=None,
                error=f"Tool '{tool_name}' not found",
                execution_time=0.0
            )
        
        return await tool.run(**kwargs)
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if tool is registered."""
        return tool_name in self._tools
