"""
Tests for Enhanced Exception Handling

This module tests the custom exception handlers in the application.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.core.exceptions import (
    AgentExecutionError,
    GraphExecutionError,
    LLMRateLimitError,
    LLMAPIConnectionError,
    LLMContextWindowError,
    LLMContentPolicyError,
    ToolExecutionError,
    VectorStoreError,
    ConcurrencyError,
)


client = TestClient(app)


class TestExceptionHandlers:
    """Test suite for exception handlers."""

    def test_health_endpoint_success(self):
        """Test that health endpoint works normally."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_validation_error_handler(self):
        """Test ValueError exception handler."""
        # This will trigger a validation error by sending invalid data
        response = client.post(
            "/decisions/run",
            json={"invalid": "data"}  # Missing required fields
        )
        # Should return 422 for validation errors from Pydantic
        # or 400 for ValueError
        assert response.status_code in [400, 422]

    def test_not_found_error(self):
        """Test 404 handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_method_not_allowed(self):
        """Test 405 handling."""
        response = client.put("/health")  # Health only accepts GET
        assert response.status_code == 405


class TestCustomExceptions:
    """Test suite for custom exceptions."""

    def test_agent_execution_error_structure(self):
        """Test AgentExecutionError exception structure."""
        error = AgentExecutionError("Test agent error")
        assert str(error) == "Test agent error"
        assert isinstance(error, Exception)

    def test_graph_execution_error_structure(self):
        """Test GraphExecutionError exception structure."""
        error = GraphExecutionError("Test graph error")
        assert str(error) == "Test graph error"
        assert isinstance(error, Exception)

    def test_llm_rate_limit_error_structure(self):
        """Test LLMRateLimitError exception structure."""
        error = LLMRateLimitError("Rate limit exceeded")
        assert str(error) == "Rate limit exceeded"
        assert isinstance(error, Exception)

    def test_llm_connection_error_structure(self):
        """Test LLMAPIConnectionError exception structure."""
        error = LLMAPIConnectionError("Connection failed")
        assert str(error) == "Connection failed"
        assert isinstance(error, Exception)

    def test_llm_context_window_error_structure(self):
        """Test LLMContextWindowError exception structure."""
        error = LLMContextWindowError("Context too long")
        assert str(error) == "Context too long"
        assert isinstance(error, Exception)

    def test_llm_content_policy_error_structure(self):
        """Test LLMContentPolicyError exception structure."""
        error = LLMContentPolicyError("Policy violation")
        assert str(error) == "Policy violation"
        assert isinstance(error, Exception)

    def test_tool_execution_error_structure(self):
        """Test ToolExecutionError exception structure."""
        error = ToolExecutionError("Tool failed")
        assert str(error) == "Tool failed"
        assert isinstance(error, Exception)

    def test_vector_store_error_structure(self):
        """Test VectorStoreError exception structure."""
        error = VectorStoreError("Vector store error")
        assert str(error) == "Vector store error"
        assert isinstance(error, Exception)

    def test_concurrency_error_structure(self):
        """Test ConcurrencyError exception structure."""
        error = ConcurrencyError("Concurrent modification")
        assert str(error) == "Concurrent modification"
        assert isinstance(error, Exception)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
