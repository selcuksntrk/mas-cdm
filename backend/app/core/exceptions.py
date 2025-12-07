"""
Custom Exceptions for Multi-Agent Decision Making Application

This module defines custom exceptions used throughout the application
for better error handling and debugging.
"""


class AgentExecutionError(Exception):
    """Raised when an agent fails to execute properly."""
    pass


class GraphExecutionError(Exception):
    """Raised when the graph execution fails."""
    pass


class StateManagementError(Exception):
    """Raised when there's an issue with state management."""
    pass


class LLMProviderError(Exception):
    """Base exception for LLM provider errors."""
    pass


class LLMRateLimitError(LLMProviderError):
    """Raised when the LLM provider rate limit is exceeded."""
    pass


class LLMAPIConnectionError(LLMProviderError):
    """Raised when connection to LLM provider fails."""
    pass


class LLMContextWindowError(LLMProviderError):
    """Raised when the context window is exceeded."""
    pass


class LLMContentPolicyError(LLMProviderError):
    """Raised when content violates the provider's policy."""
    pass


class ToolExecutionError(Exception):
    """Raised when a tool execution fails."""
    pass


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""
    pass


class ConcurrencyError(Exception):
    """Raised when concurrent access causes conflicts."""
    pass
