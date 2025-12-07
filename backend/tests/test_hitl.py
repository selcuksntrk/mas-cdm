"""
Tests for Human-in-the-Loop (HITL) Functionality

This module tests the HITL interrupt system and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from datetime import datetime, UTC

from backend.app.main import app
from backend.app.core.hitl import (
    InterruptEvent,
    InterruptType,
    InterruptStatus,
    HITLState,
)


client = TestClient(app)


class TestInterruptEvent:
    """Test suite for InterruptEvent model."""

    def test_interrupt_event_creation(self):
        """Test creating an interrupt event."""
        event = InterruptEvent(
            type=InterruptType.APPROVAL_REQUIRED,
            node_name="TestNode",
            message="Please approve this action"
        )
        
        assert event.type == InterruptType.APPROVAL_REQUIRED
        assert event.status == InterruptStatus.PENDING
        assert event.node_name == "TestNode"
        assert event.message == "Please approve this action"
        assert event.human_response is None
        assert event.responded_at is None

    def test_interrupt_event_respond(self):
        """Test responding to an interrupt event."""
        event = InterruptEvent(
            type=InterruptType.APPROVAL_REQUIRED,
            node_name="TestNode",
            message="Test"
        )
        
        # Respond to the interrupt
        event.respond(
            status=InterruptStatus.APPROVED,
            response="Approved by user",
            data={"confidence": "high"}
        )
        
        assert event.status == InterruptStatus.APPROVED
        assert event.human_response == "Approved by user"
        assert event.human_data == {"confidence": "high"}
        assert event.responded_at is not None

    def test_interrupt_types(self):
        """Test different interrupt types."""
        types = [
            InterruptType.APPROVAL_REQUIRED,
            InterruptType.INPUT_REQUIRED,
            InterruptType.REVIEW_REQUIRED,
            InterruptType.FEEDBACK_REQUIRED,
        ]
        
        for interrupt_type in types:
            event = InterruptEvent(
                type=interrupt_type,
                node_name="TestNode",
                message="Test"
            )
            assert event.type == interrupt_type


class TestHITLState:
    """Test suite for HITLState."""

    def test_hitl_state_creation(self):
        """Test creating HITL state."""
        state = HITLState()
        
        assert state.interrupts == []
        assert state.is_suspended is False
        assert state.current_interrupt_id is None

    def test_create_interrupt(self):
        """Test creating an interrupt in state."""
        state = HITLState()
        
        interrupt = state.create_interrupt(
            interrupt_type=InterruptType.APPROVAL_REQUIRED,
            node_name="TestNode",
            message="Need approval"
        )
        
        assert len(state.interrupts) == 1
        assert state.is_suspended is True
        assert state.current_interrupt_id == interrupt.id
        assert interrupt.status == InterruptStatus.PENDING

    def test_respond_to_interrupt(self):
        """Test responding to an interrupt."""
        state = HITLState()
        
        # Create interrupt
        interrupt = state.create_interrupt(
            interrupt_type=InterruptType.APPROVAL_REQUIRED,
            node_name="TestNode",
            message="Need approval"
        )
        
        # Respond to interrupt
        success = state.respond_to_interrupt(
            interrupt_id=interrupt.id,
            status=InterruptStatus.APPROVED,
            response="Looks good"
        )
        
        assert success is True
        assert state.is_suspended is False
        assert state.current_interrupt_id is None
        assert interrupt.status == InterruptStatus.APPROVED

    def test_get_pending_interrupts(self):
        """Test getting pending interrupts."""
        state = HITLState()
        
        # Create multiple interrupts
        int1 = state.create_interrupt(
            InterruptType.APPROVAL_REQUIRED,
            "Node1",
            "Test 1"
        )
        
        state.is_suspended = False  # Reset to create another
        
        int2 = state.create_interrupt(
            InterruptType.INPUT_REQUIRED,
            "Node2",
            "Test 2"
        )
        
        # Respond to first one
        state.respond_to_interrupt(int1.id, InterruptStatus.APPROVED)
        
        # Get pending
        pending = state.get_pending_interrupts()
        assert len(pending) == 1
        assert pending[0].id == int2.id

    def test_get_current_interrupt(self):
        """Test getting the current interrupt."""
        state = HITLState()
        
        # No current interrupt initially
        assert state.get_current_interrupt() is None
        
        # Create interrupt
        interrupt = state.create_interrupt(
            InterruptType.APPROVAL_REQUIRED,
            "TestNode",
            "Test"
        )
        
        # Get current interrupt
        current = state.get_current_interrupt()
        assert current is not None
        assert current.id == interrupt.id


class TestHITLAPI:
    """Test suite for HITL API endpoints."""

    def test_hitl_health_endpoint(self):
        """Test HITL health endpoint."""
        response = client.get("/hitl/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "human-in-the-loop"

    def test_get_process_interrupts_not_implemented(self):
        """Test that get_process_interrupts returns 501 (placeholder)."""
        response = client.get("/hitl/process/test123/interrupts")
        assert response.status_code == 501

    def test_respond_to_interrupt_not_implemented(self):
        """Test that respond_to_interrupt returns 501 (placeholder)."""
        response = client.post(
            "/hitl/process/test123/interrupts/int123/respond",
            json={
                "status": "approved",
                "response": "Looks good"
            }
        )
        assert response.status_code == 501


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
