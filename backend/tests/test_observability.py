"""
Tests for observability and tracing functionality.
"""

import pytest
import time
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock
from backend.app.core.observability.tracer import (
    AgentTracer,
    AgentExecutionMetrics,
    get_tracer,
    trace_agent
)


class TestAgentTracer:
    """Test suite for AgentTracer class."""
    
    def test_tracer_initialization_enabled(self):
        """Test tracer initialization with tracing enabled."""
        tracer = AgentTracer(enable_tracing=True)
        assert tracer.enable_tracing is True
    
    def test_tracer_initialization_disabled(self):
        """Test tracer initialization with tracing disabled."""
        tracer = AgentTracer(enable_tracing=False)
        assert tracer.enable_tracing is False
    
    def test_trace_agent_execution_success(self):
        """Test successful agent execution tracing."""
        tracer = AgentTracer(enable_tracing=False)  # Disable logfire for testing
        
        with tracer.trace_agent_execution(
            agent_name="test_agent",
            decision_id="test-123",
            inputs={"test": "data"}
        ) as ctx:
            ctx["result"] = "success"
            ctx["token_usage"] = {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        
        # Context should have been populated
        assert ctx["result"] == "success"
        assert ctx["token_usage"]["total_tokens"] == 150
    
    def test_trace_agent_execution_with_error(self):
        """Test agent execution tracing with error."""
        tracer = AgentTracer(enable_tracing=False)
        
        with pytest.raises(ValueError):
            with tracer.trace_agent_execution(
                agent_name="test_agent",
                decision_id="test-123"
            ) as ctx:
                raise ValueError("Test error")
        
        # Error should be captured
        assert ctx["error"] == "Test error"
    
    def test_calculate_cost(self):
        """Test token cost calculation."""
        token_usage = {
            "prompt_tokens": 1000,
            "completion_tokens": 500,
            "total_tokens": 1500
        }
        
        cost = AgentTracer._calculate_cost(token_usage)
        
        # Expected: (1000/1000 * 0.03) + (500/1000 * 0.06) = 0.03 + 0.03 = 0.06
        assert cost == pytest.approx(0.06, rel=1e-6)
    
    def test_calculate_cost_empty(self):
        """Test cost calculation with empty token usage."""
        cost = AgentTracer._calculate_cost({})
        assert cost == 0.0
    
    def test_log_agent_decision(self):
        """Test logging agent decision."""
        tracer = AgentTracer(enable_tracing=False)
        
        # Should not raise any exception
        tracer.log_agent_decision(
            agent_name="test_agent",
            decision_id="test-123",
            decision="test decision",
            confidence=0.95
        )
    
    def test_track_agent_performance(self):
        """Test tracking agent performance metrics."""
        tracer = AgentTracer(enable_tracing=False)
        
        metrics = {
            "latency_ms": 150,
            "success_rate": 0.98,
            "error_count": 2
        }
        
        # Should not raise any exception
        tracer.track_agent_performance("test_agent", metrics)


class TestTraceAgentDecorator:
    """Test suite for trace_agent decorator."""
    
    @pytest.mark.asyncio
    async def test_trace_agent_decorator_async(self):
        """Test trace_agent decorator with async function."""
        
        @trace_agent("test_async_agent")
        async def mock_async_agent(state):
            # Simulate agent execution
            await asyncio.sleep(0.01)
            result = Mock()
            result.output = "test output"
            result.usage = Mock()
            result.usage.prompt_tokens = 100
            result.usage.completion_tokens = 50
            result.usage.total_tokens = 150
            return result
        
        # Create a mock state with decision_id
        mock_state = Mock()
        mock_state.decision_id = "test-456"
        
        result = await mock_async_agent(mock_state)
        
        assert result.output == "test output"
        assert result.usage.total_tokens == 150
    
    def test_trace_agent_decorator_sync(self):
        """Test trace_agent decorator with sync function."""
        
        @trace_agent("test_sync_agent")
        def mock_sync_agent(state):
            result = Mock()
            result.output = "test output"
            result.usage = None  # No usage tracking for this test
            return result
        
        mock_state = Mock()
        mock_state.decision_id = "test-789"
        
        result = mock_sync_agent(mock_state)
        
        assert result.output == "test output"
    
    @pytest.mark.asyncio
    async def test_trace_agent_decorator_with_error(self):
        """Test trace_agent decorator handles errors correctly."""
        
        @trace_agent("error_agent")
        async def failing_agent(state):
            raise RuntimeError("Agent failure")
        
        mock_state = Mock()
        mock_state.decision_id = "test-error"
        
        with pytest.raises(RuntimeError, match="Agent failure"):
            await failing_agent(mock_state)

    def test_trace_agent_decorator_extracts_decision_id_and_inputs(self):
        """Ensure decorator pulls decision_id from state containers and passes custom inputs."""
        tracer = AgentTracer(enable_tracing=False)
        captured = {}

        def fake_trace_agent_execution(agent_name, decision_id=None, inputs=None):
            captured["agent_name"] = agent_name
            captured["decision_id"] = decision_id
            captured["inputs"] = inputs

            @contextmanager
            def _ctx():
                yield {"result": None, "token_usage": None, "error": None}

            return _ctx()

        with patch(
            "backend.app.core.observability.tracer.get_tracer",
            return_value=tracer,
        ), patch.object(tracer, "trace_agent_execution", side_effect=fake_trace_agent_execution):

            @trace_agent(
                "decorated_agent",
                input_extractor=lambda args, kwargs: {"custom": kwargs.get("marker")},
            )
            def wrapped_fn(container, *, marker: str):
                result = Mock()
                result.output = "ok"
                result.usage = None
                return result

            holder = SimpleNamespace(state=SimpleNamespace(decision_id="ctx-456"))
            wrapped_fn(holder, marker="flag")

        assert captured["agent_name"] == "decorated_agent"
        assert captured["decision_id"] == "ctx-456"
        assert captured["inputs"] == {"custom": "flag"}


class TestAgentExecutionMetrics:
    """Test suite for AgentExecutionMetrics model."""
    
    def test_metrics_creation(self):
        """Test creating execution metrics."""
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            decision_id="test-123",
            duration_seconds=1.5,
            token_usage={"total_tokens": 150},
            cost_usd=0.05,
            success=True,
            timestamp=time.time()
        )
        
        assert metrics.agent_name == "test_agent"
        assert metrics.decision_id == "test-123"
        assert metrics.duration_seconds == 1.5
        assert metrics.success is True
        assert metrics.cost_usd == 0.05
    
    def test_metrics_with_error(self):
        """Test creating metrics with error."""
        metrics = AgentExecutionMetrics(
            agent_name="test_agent",
            decision_id="test-123",
            duration_seconds=0.5,
            success=False,
            error_message="Test error",
            timestamp=time.time()
        )
        
        assert metrics.success is False
        assert metrics.error_message == "Test error"


class TestGetTracer:
    """Test suite for get_tracer function."""
    
    def test_get_tracer_singleton(self):
        """Test that get_tracer returns singleton instance."""
        tracer1 = get_tracer()
        tracer2 = get_tracer()
        
        assert tracer1 is tracer2
    
    @patch('backend.app.core.observability.tracer._tracer', None)
    def test_get_tracer_creates_new_instance(self):
        """Test that get_tracer creates new instance if none exists."""
        with patch('backend.app.config.get_settings') as mock_settings:
            mock_settings.return_value.enable_tracing = False
            
            tracer = get_tracer()
            
            assert tracer is not None
            assert isinstance(tracer, AgentTracer)


class TestTracingPerformance:
    """Test suite for tracing performance overhead."""
    
    @pytest.mark.asyncio
    async def test_tracing_overhead_minimal(self):
        """Test that tracing adds minimal performance overhead."""
        tracer = AgentTracer(enable_tracing=False)
        
        # Measure execution without tracing context
        start = time.perf_counter()
        for _ in range(100):
            await asyncio.sleep(0.001)
        no_trace_duration = time.perf_counter() - start
        
        # Measure execution with tracing context
        start = time.perf_counter()
        for _ in range(100):
            with tracer.trace_agent_execution("test_agent") as ctx:
                await asyncio.sleep(0.001)
                ctx["result"] = "done"
        with_trace_duration = time.perf_counter() - start
        
        # Calculate overhead percentage
        overhead = ((with_trace_duration - no_trace_duration) / no_trace_duration) * 100
        
        # Assert overhead is less than 10% (very conservative)
        # In practice, should be much less
        assert overhead < 10.0, f"Tracing overhead too high: {overhead:.2f}%"


# Import asyncio for async tests
import asyncio


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
