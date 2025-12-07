"""
Tests for Streaming API

This module tests the Server-Sent Events (SSE) streaming functionality.
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
import json


client = TestClient(app)


class TestStreamingAPI:
    """Test suite for streaming endpoints."""

    def test_stream_health_endpoint(self):
        """Test the streaming service health endpoint."""
        response = client.get("/stream/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "streaming"

    def test_stream_decision_endpoint_exists(self):
        """Test that the streaming decision endpoint exists."""
        # Test with invalid data to confirm endpoint exists (but validation fails)
        response = client.post("/stream/decisions/stream", json={})
        # Should return 422 for missing required field or 400 for validation
        assert response.status_code in [400, 422]

    def test_stream_decision_validation_error(self):
        """Test validation error handling in streaming endpoint."""
        response = client.post(
            "/stream/decisions/stream",
            json={"decision_query": ""}  # Empty query should fail validation
        )
        assert response.status_code in [400, 422]

    @pytest.mark.slow
    def test_stream_decision_basic_flow(self):
        """
        Test basic streaming flow with a simple decision.
        
        Note: This is a slow test as it actually runs the decision process.
        """
        # Skip this test if we don't have a valid API key
        import os
        if not os.getenv("API_KEY") and not os.getenv("OPENAI_API_KEY"):
            pytest.skip("No API key available for integration test")
        
        # Use stream=True in the request (handled differently in httpx)
        with client.stream(
            "POST",
            "/stream/decisions/stream",
            json={"decision_query": "Should I learn Python or JavaScript first?"}
        ) as response:
            # Check that we get a streaming response
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]
            
            # Collect events from the stream
            events = []
            for line in response.iter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    event = json.loads(data)
                    events.append(event)
            
            # Verify we got events
            assert len(events) > 0
            
            # Check for start event
            start_events = [e for e in events if e.get("event_type") == "start"]
            assert len(start_events) > 0
            
            # Check that events have required fields
            for event in events:
                assert "event_type" in event
                assert "data" in event
                assert "timestamp" in event


class TestStreamEvent:
    """Test suite for StreamEvent model."""

    def test_stream_event_creation(self):
        """Test StreamEvent model creation."""
        from backend.app.api.routes.stream import StreamEvent
        
        event = StreamEvent(
            event_type="test",
            data={"message": "test message"}
        )
        
        assert event.event_type == "test"
        assert event.data["message"] == "test message"
        assert event.timestamp is not None

    def test_stream_event_serialization(self):
        """Test StreamEvent JSON serialization."""
        from backend.app.api.routes.stream import StreamEvent
        
        event = StreamEvent(
            event_type="progress",
            data={"step": 1, "phase": "Initialization"}
        )
        
        json_str = event.model_dump_json()
        assert isinstance(json_str, str)
        
        # Parse back to verify structure
        parsed = json.loads(json_str)
        assert parsed["event_type"] == "progress"
        assert parsed["data"]["step"] == 1


class TestDeterminePhase:
    """Test suite for phase determination logic."""

    def test_determine_current_phase(self):
        """Test phase determination from state."""
        from backend.app.api.routes.stream import determine_current_phase
        from backend.app.models.domain import DecisionState
        
        # Test initialization phase
        state = DecisionState(decision_requested="test")
        assert determine_current_phase(state) == "Initialization"
        
        # Test trigger identification phase
        state.trigger = "Test trigger"
        assert determine_current_phase(state) == "Trigger Identification"
        
        # Test root cause phase
        state.root_cause = "Test root cause"
        assert determine_current_phase(state) == "Root Cause Analysis"
        
        # Test scope definition phase
        state.scope_definition = "Test scope"
        assert determine_current_phase(state) == "Scope Definition"
        
        # Test drafting phase
        state.decision_drafted = "Test draft"
        assert determine_current_phase(state) == "Drafting"
        
        # Test goals phase
        state.goals = "Test goals"
        assert determine_current_phase(state) == "Goal Establishment"
        
        # Test decision phase
        state.result = "Test decision"
        assert determine_current_phase(state) == "Decision"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
