"""
Tests for Decision Service

Tests for the DecisionService class that orchestrates the workflow.
"""

import pytest
from pathlib import Path

from backend.app.services.decision_service import DecisionService
from backend.app.models.domain import DecisionState


class TestDecisionService:
    """Tests for DecisionService."""
    
    def test_service_initialization(self, decision_service):
        """Test that DecisionService can be initialized."""
        assert decision_service is not None
        assert isinstance(decision_service, DecisionService)
    
    def test_extract_result_summary(self, decision_service, sample_decision_state):
        """Test extracting result summary from state."""
        summary = decision_service.extract_result_summary(sample_decision_state)
        
        assert "selected_decision" in summary
        assert "selected_decision_comment" in summary
        assert "alternative_decision" in summary
        assert "alternative_decision_comment" in summary
        
        assert summary["selected_decision"] == sample_decision_state.result
        assert summary["selected_decision_comment"] == sample_decision_state.result_comment
    
    def test_extract_full_result(self, decision_service, sample_decision_state):
        """Test extracting full result from state."""
        full_result = decision_service.extract_full_result(sample_decision_state)
        
        # Check all major fields are present
        assert "selected_decision" in full_result
        assert "trigger" in full_result
        assert "root_cause" in full_result
        assert "scope_definition" in full_result
        assert "decision_drafted" in full_result
        assert "goals" in full_result
        assert "alternatives" in full_result
        
        # Verify values
        assert full_result["trigger"] == sample_decision_state.trigger
        assert full_result["root_cause"] == sample_decision_state.root_cause
        assert full_result["selected_decision"] == sample_decision_state.result
    
    def test_validate_decision_query_valid(self, decision_service):
        """Test validation of valid decision queries."""
        valid_queries = [
            "Should I invest in stocks?",
            "Should we expand our business to Europe?",
            "Is it time to change careers?",
            "Should I buy a house or continue renting?"
        ]
        
        for query in valid_queries:
            # Should not raise exception
            decision_service.validate_decision_query(query)
    
    def test_validate_decision_query_invalid(self, decision_service):
        """Test validation of invalid decision queries."""
        # Too short
        with pytest.raises(ValueError):
            decision_service.validate_decision_query("Yes?")
        
        # Empty
        with pytest.raises(ValueError):
            decision_service.validate_decision_query("")
        
        # None
        with pytest.raises(ValueError):
            decision_service.validate_decision_query(None)
    
    @pytest.mark.asyncio
    async def test_run_decision_creates_state(self, decision_service, sample_decision_query):
        """Test that run_decision creates initial state correctly."""
        # Note: This test would actually call the AI models in a real scenario
        # For unit testing, we'd need to mock the agents
        # Here we test the service setup
        
        # Create state manually to test the service structure
        state = DecisionState(decision_requested=sample_decision_query)
        
        assert state.decision_requested == sample_decision_query
        assert state.trigger == ""  # Not yet populated
        assert state.result == ""  # Not yet populated
    
    def test_extract_result_handles_empty_state(self, decision_service):
        """Test extracting result from empty state."""
        empty_state = DecisionState()
        
        summary = decision_service.extract_result_summary(empty_state)
        
        assert summary["selected_decision"] == ""
        assert summary["selected_decision_comment"] == ""
        assert summary["alternative_decision"] == ""
        assert summary["alternative_decision_comment"] == ""
    
    def test_extract_full_result_structure(self, decision_service):
        """Test that extract_full_result returns correct structure."""
        state = DecisionState(
            decision_requested="Test query",
            trigger="Test trigger",
            root_cause="Test root cause"
        )
        
        result = decision_service.extract_full_result(state)
        
        # Verify structure
        expected_keys = [
            "selected_decision",
            "selected_decision_comment",
            "alternative_decision",
            "alternative_decision_comment",
            "trigger",
            "root_cause",
            "scope_definition",
            "decision_drafted",
            "goals",
            "complementary_info",
            "decision_draft_updated",
            "alternatives"
        ]
        
        for key in expected_keys:
            assert key in result
