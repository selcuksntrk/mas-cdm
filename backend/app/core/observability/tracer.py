"""
Agent tracing and observability module using logfire.

This module provides comprehensive tracing, logging, and performance monitoring
for multi-agent system executions.
"""

from __future__ import annotations

import time
import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar
from contextlib import contextmanager

import logfire
from pydantic import BaseModel


# Configure logging
logger = logging.getLogger(__name__)

# Type variable for decorated functions
T = TypeVar("T")


class AgentExecutionMetrics(BaseModel):
    """Metrics captured during agent execution."""
    
    agent_name: str
    decision_id: Optional[str] = None
    duration_seconds: float
    token_usage: Optional[dict[str, int]] = None
    cost_usd: Optional[float] = None
    success: bool
    error_message: Optional[str] = None
    timestamp: float


class AgentTracer:
    """
    Central tracing and observability manager for agents.
    
    Features:
    - Distributed tracing with logfire
    - Token usage and cost tracking
    - Performance metrics collection
    - Structured logging with context
    """
    
    def __init__(self, enable_tracing: bool = True):
        """
        Initialize the tracer.
        
        Args:
            enable_tracing: Whether to enable tracing (useful for testing)
        """
        self.enable_tracing = enable_tracing
        
        if self.enable_tracing:
            # Configure logfire if not already configured
            try:
                logfire.configure(send_to_logfire="if-token-present")
                logger.info("Logfire tracing initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize logfire: {e}")
                self.enable_tracing = False
    
    @contextmanager
    def trace_agent_execution(
        self,
        agent_name: str,
        decision_id: Optional[str] = None,
        inputs: Optional[dict[str, Any]] = None
    ):
        """
        Context manager for tracing agent execution.
        
        Args:
            agent_name: Name of the agent being executed
            decision_id: Optional decision ID for correlation
            inputs: Optional input parameters
            
        Yields:
            A dictionary to store execution results
            
        Example:
            with tracer.trace_agent_execution("my_agent", decision_id="123") as ctx:
                result = agent.run(state)
                ctx["result"] = result
        """
        start_time = time.time()
        execution_context = {
            "result": None,
            "token_usage": None,
            "error": None
        }
        
        # Start trace span
        if self.enable_tracing:
            span = logfire.span(
                f"agent_execution:{agent_name}",
                agent_name=agent_name,
                decision_id=decision_id,
                inputs=inputs
            )
            span.__enter__()
        else:
            span = None
        
        try:
            yield execution_context
            success = True
            error_msg = None
            
        except Exception as e:
            success = False
            error_msg = str(e)
            execution_context["error"] = error_msg
            
            if self.enable_tracing:
                logfire.error(
                    f"Agent execution failed: {agent_name}",
                    agent_name=agent_name,
                    decision_id=decision_id,
                    error=error_msg
                )
            
            raise
            
        finally:
            duration = time.time() - start_time
            
            # Extract token usage from result if available
            token_usage = execution_context.get("token_usage")
            
            # Calculate cost (simplified - adjust based on your model pricing)
            cost = self._calculate_cost(token_usage) if token_usage else None
            
            # Create metrics
            metrics = AgentExecutionMetrics(
                agent_name=agent_name,
                decision_id=decision_id,
                duration_seconds=duration,
                token_usage=token_usage,
                cost_usd=cost,
                success=success,
                error_message=error_msg,
                timestamp=start_time
            )
            
            # Log metrics
            if self.enable_tracing and span:
                logfire.info(
                    f"Agent execution completed: {agent_name}",
                    agent_name=agent_name,
                    decision_id=decision_id,
                    duration_seconds=duration,
                    token_usage=token_usage,
                    cost_usd=cost,
                    success=success
                )
                span.__exit__(None, None, None)
            
            # Also log to standard logger
            logger.info(
                f"Agent '{agent_name}' executed in {duration:.2f}s "
                f"(success={success}, tokens={token_usage})"
            )
    
    def log_agent_decision(
        self,
        agent_name: str,
        decision_id: str,
        decision: Any,
        confidence: Optional[float] = None
    ):
        """
        Log a decision made by an agent.
        
        Args:
            agent_name: Name of the agent
            decision_id: Decision ID for correlation
            decision: The decision made
            confidence: Optional confidence score
        """
        if self.enable_tracing:
            logfire.info(
                f"Agent decision: {agent_name}",
                agent_name=agent_name,
                decision_id=decision_id,
                decision=str(decision)[:200],  # Truncate long decisions
                confidence=confidence
            )
    
    def track_agent_performance(
        self,
        agent_name: str,
        metrics: dict[str, Any]
    ):
        """
        Track custom performance metrics for an agent.
        
        Args:
            agent_name: Name of the agent
            metrics: Dictionary of metric name -> value
        """
        if self.enable_tracing:
            logfire.info(
                f"Agent performance metrics: {agent_name}",
                agent_name=agent_name,
                **metrics
            )
    
    @staticmethod
    def _calculate_cost(token_usage: dict[str, int]) -> float:
        """
        Calculate cost based on token usage.
        
        This is a simplified calculation. Adjust pricing based on your model.
        
        Args:
            token_usage: Dictionary with 'prompt_tokens' and 'completion_tokens'
            
        Returns:
            Estimated cost in USD
        """
        if not token_usage:
            return 0.0
        
        # Example pricing (GPT-4 rates - adjust as needed)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        completion_tokens = token_usage.get("completion_tokens", 0)
        
        # $0.03 per 1K prompt tokens, $0.06 per 1K completion tokens
        prompt_cost = (prompt_tokens / 1000) * 0.03
        completion_cost = (completion_tokens / 1000) * 0.06
        
        return prompt_cost + completion_cost


# Global tracer instance
_tracer: Optional[AgentTracer] = None


def get_tracer() -> AgentTracer:
    """Get or create the global tracer instance."""
    global _tracer
    if _tracer is None:
        # Import here to avoid circular dependency
        from backend.app.config import get_settings
        settings = get_settings()
        _tracer = AgentTracer(enable_tracing=settings.enable_tracing)
    return _tracer


def _extract_decision_id(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of decision_id from positional/keyword args."""
    for candidate in list(args) + list(kwargs.values()):
        if candidate is None:
            continue
        if hasattr(candidate, "decision_id"):
            return getattr(candidate, "decision_id")
        if hasattr(candidate, "state") and hasattr(candidate.state, "decision_id"):
            return getattr(candidate.state, "decision_id")
    return None


def trace_agent(
    agent_name: Optional[str] = None,
    input_extractor: Optional[Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]] = None,
):
    """
    Decorator for tracing agent execution.
    
    Args:
        agent_name: Name of the agent (defaults to function name)
        
    Example:
        @trace_agent("my_agent")
        async def run_my_agent(state: DecisionState):
            # agent logic
            return result
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        name = agent_name or func.__name__
        
        def _build_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
            if input_extractor:
                try:
                    return input_extractor(args, kwargs) or {}
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to extract trace inputs: %s", exc)
                    return {}
            return {"args_count": len(args), "kwargs_keys": list(kwargs.keys())}

        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()
            decision_id = _extract_decision_id(args, kwargs)

            with tracer.trace_agent_execution(
                agent_name=name,
                decision_id=decision_id,
                inputs=_build_inputs(args, kwargs),
            ) as ctx:
                result = await func(*args, **kwargs)

                if hasattr(result, "usage"):
                    ctx["token_usage"] = {
                        "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(result.usage, "completion_tokens", 0),
                        "total_tokens": getattr(result.usage, "total_tokens", 0),
                    }

                ctx["result"] = result
                return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> T:
            tracer = get_tracer()
            decision_id = _extract_decision_id(args, kwargs)

            with tracer.trace_agent_execution(
                agent_name=name,
                decision_id=decision_id,
                inputs=_build_inputs(args, kwargs),
            ) as ctx:
                result = func(*args, **kwargs)

                if hasattr(result, "usage"):
                    ctx["token_usage"] = {
                        "prompt_tokens": getattr(result.usage, "prompt_tokens", 0),
                        "completion_tokens": getattr(result.usage, "completion_tokens", 0),
                        "total_tokens": getattr(result.usage, "total_tokens", 0),
                    }

                ctx["result"] = result
                return result
        
        # Return appropriate wrapper based on whether function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
