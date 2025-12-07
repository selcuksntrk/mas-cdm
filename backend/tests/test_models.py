"""
Tests for Domain Models

Tests for DecisionState, ProcessInfo, and other domain models.
"""

import pytest
from datetime import datetime, UTC
from pydantic import ValidationError

from backend.app.models.domain import DecisionState, ProcessInfo, EvaluationOutput, ResultOutput


class TestDecisionState:
    """Tests for DecisionState model."""
    
    def test_empty_decision_state_creation(self):
        """Test creating an empty DecisionState."""
        state = DecisionState()
        
        assert state.decision_requested == ""
        assert state.trigger == ""
        assert state.root_cause == ""
        assert state.complementary_info_num == 0
        
    def test_decision_state_with_values(self):
        """Test creating DecisionState with initial values."""
        state = DecisionState(
            decision_requested="Should I switch careers?",
            trigger="Job dissatisfaction",
            root_cause="Lack of growth opportunities"
        )
        
        assert state.decision_requested == "Should I switch careers?"
        assert state.trigger == "Job dissatisfaction"
        assert state.root_cause == "Lack of growth opportunities"
        
    def test_decision_state_validation(self):
        """Test that DecisionState validates complementary_info_num."""
        state = DecisionState(complementary_info_num=5)
        assert state.complementary_info_num == 5
        
        # Should accept 0
        state = DecisionState(complementary_info_num=0)
        assert state.complementary_info_num == 0
        
        # Should reject negative numbers
        with pytest.raises(ValidationError):
            DecisionState(complementary_info_num=-1)
    
    def test_decision_state_json_serialization(self):
        """Test that DecisionState can be serialized to JSON."""
        state = DecisionState(
            decision_requested="Test decision",
            trigger="Test trigger"
        )
        
        json_data = state.model_dump_json()
        assert "Test decision" in json_data
        assert "Test trigger" in json_data
        
        # Deserialize back
        state2 = DecisionState.model_validate_json(json_data)
        assert state2.decision_requested == state.decision_requested
        assert state2.trigger == state.trigger
    
    def test_decision_state_update_fields(self):
        """Test updating DecisionState fields."""
        state = DecisionState()
        
        state.decision_requested = "New decision"
        state.trigger = "New trigger"
        state.complementary_info_num = 3
        
        assert state.decision_requested == "New decision"
        assert state.trigger == "New trigger"
        assert state.complementary_info_num == 3
    
    def test_full_decision_state(self, sample_decision_state):
        """Test a fully populated DecisionState."""
        assert sample_decision_state.decision_requested != ""
        assert sample_decision_state.result != ""
        assert sample_decision_state.result_comment != ""
        assert sample_decision_state.complementary_info_num == 2


class TestProcessInfo:
    """Tests for ProcessInfo model."""
    
    def test_process_info_creation(self):
        """Test creating a ProcessInfo instance."""
        now = datetime.now(UTC)
        process = ProcessInfo(
            process_id="test-123",
            query="Test query?",
            status="pending",
            created_at=now
        )
        
        assert process.process_id == "test-123"
        assert process.query == "Test query?"
        assert process.status == "pending"
        assert process.created_at == now
        assert process.current_step is None
        assert process.completed_steps == []
    
    def test_process_info_status_values(self):
        """Test different status values."""
        statuses = ["pending", "running", "completed", "failed"]
        
        for status in statuses:
            process = ProcessInfo(
                process_id=f"test-{status}",
                query="Test",
                status=status,
                created_at=datetime.now(UTC)
            )
            assert process.status == status
    
    def test_process_info_with_result(self, sample_decision_state):
        """Test ProcessInfo with decision result."""
        process = ProcessInfo(
            process_id="test-result",
            query="Test query",
            status="completed",
            result=sample_decision_state,
            created_at=datetime.now(UTC),
            completed_at=datetime.now(UTC)
        )
        
        assert process.result is not None
        assert process.result.decision_requested == sample_decision_state.decision_requested
        assert process.status == "completed"
        assert process.completed_at is not None
    
    def test_process_info_with_error(self):
        """Test ProcessInfo with error."""
        process = ProcessInfo(
            process_id="test-error",
            query="Test query",
            status="failed",
            error="Test error message",
            created_at=datetime.now(UTC)
        )
        
        assert process.status == "failed"
        assert process.error == "Test error message"
    
    def test_process_info_completed_steps(self):
        """Test ProcessInfo with completed steps."""
        steps = ["Step 1", "Step 2", "Step 3"]
        process = ProcessInfo(
            process_id="test-steps",
            query="Test query",
            status="running",
            current_step="Step 3",
            completed_steps=steps,
            created_at=datetime.now(UTC)
        )
        
        assert process.completed_steps == steps
        assert process.current_step == "Step 3"
        assert len(process.completed_steps) == 3


class TestEvaluationOutput:
    """Tests for EvaluationOutput model."""
    
    def test_evaluation_output_correct(self):
        """Test EvaluationOutput with correct result."""
        eval_output = EvaluationOutput(
            correct=True,
            comment="The output looks good and meets all criteria."
        )
        
        assert eval_output.correct is True
        assert "good" in eval_output.comment
    
    def test_evaluation_output_incorrect(self):
        """Test EvaluationOutput with incorrect result."""
        eval_output = EvaluationOutput(
            correct=False,
            comment="The output is missing key information."
        )
        
        assert eval_output.correct is False
        assert "missing" in eval_output.comment
    
    def test_evaluation_output_validation(self):
        """Test that EvaluationOutput requires both fields."""
        with pytest.raises(ValidationError):
            EvaluationOutput(correct=True)  # Missing comment
        
        with pytest.raises(ValidationError):
            EvaluationOutput(comment="Test")  # Missing correct


class TestResultOutput:
    """Tests for ResultOutput model."""
    
    def test_result_output_creation(self):
        """Test creating a ResultOutput instance."""
        result = ResultOutput(
            result="Invest in renewable energy",
            result_comment="Best long-term growth potential",
            best_alternative_result="Invest in tech stocks",
            best_alternative_result_comment="Higher short-term returns"
        )
        
        assert result.result == "Invest in renewable energy"
        assert result.result_comment == "Best long-term growth potential"
        assert result.best_alternative_result == "Invest in tech stocks"
        assert result.best_alternative_result_comment == "Higher short-term returns"
    
    def test_result_output_validation(self):
        """Test that ResultOutput requires all fields."""
        with pytest.raises(ValidationError):
            ResultOutput(
                result="Test",
                result_comment="Comment"
                # Missing alternatives
            )
    
    def test_result_output_json_serialization(self):
        """Test ResultOutput JSON serialization."""
        result = ResultOutput(
            result="Decision A",
            result_comment="Comment A",
            best_alternative_result="Decision B",
            best_alternative_result_comment="Comment B"
        )
        
        json_data = result.model_dump_json()
        assert "Decision A" in json_data
        assert "Decision B" in json_data
        
        # Deserialize
        result2 = ResultOutput.model_validate_json(json_data)
        assert result2.result == result.result
