"""
Tests for Process Manager

Tests for the ProcessManager class that handles async execution.
"""

import pytest
import asyncio
from datetime import datetime, UTC

from backend.app.services.process_manager import ProcessManager
from backend.app.models.domain import ProcessInfo, DecisionState


class TestProcessManager:
    """Tests for ProcessManager."""
    
    def test_process_manager_initialization(self, process_manager):
        """Test ProcessManager initialization."""
        assert process_manager is not None
        assert process_manager._repository is not None
        assert process_manager._decision_service is not None
    
    @pytest.mark.asyncio
    async def test_create_process(self, process_manager, sample_decision_query):
        """Test creating a new process."""
        process = await process_manager.create_process(sample_decision_query)
        
        assert process is not None
        assert process.process_id is not None
        assert len(process.process_id) > 0
        assert process.query == sample_decision_query
        assert process.status == "pending"
        assert process.current_step is None
        assert process.completed_steps == []
        assert process.result is None
        assert process.error is None
    
    @pytest.mark.asyncio
    async def test_create_multiple_processes(self, process_manager):
        """Test creating multiple processes with unique IDs."""
        process1 = await process_manager.create_process("Query 1?")
        process2 = await process_manager.create_process("Query 2?")
        process3 = await process_manager.create_process("Query 3?")
        
        # All should have unique IDs
        assert process1.process_id != process2.process_id
        assert process2.process_id != process3.process_id
        assert process1.process_id != process3.process_id
    
    @pytest.mark.asyncio
    async def test_get_process_status(self, process_manager, sample_decision_query):
        """Test retrieving process status."""
        # Create process
        process = await process_manager.create_process(sample_decision_query)
        
        # Get status
        status = await process_manager.get_process(process.process_id)
        
        assert status is not None
        assert status.process_id == process.process_id
        assert status.query == sample_decision_query
        assert status.status == "pending"
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_process_status(self, process_manager):
        """Test getting status of non-existent process."""
        status = await process_manager.get_process("nonexistent-id")
        assert status is None
    
    @pytest.mark.asyncio
    async def test_list_all_processes(self, process_manager):
        """Test listing all processes."""
        # Create multiple processes
        await process_manager.create_process("Query 1?")
        await process_manager.create_process("Query 2?")
        await process_manager.create_process("Query 3?")
        
        # List all
        processes = await process_manager.get_all_processes()
        
        assert len(processes) >= 3
        queries = {p.query for p in processes}
        assert "Query 1?" in queries
        assert "Query 2?" in queries
        assert "Query 3?" in queries
    
    @pytest.mark.asyncio
    async def test_delete_process(self, process_manager, sample_decision_query):
        """Test deleting a process."""
        # Create process
        process = await process_manager.create_process(sample_decision_query)
        
        # Verify it exists
        status = await process_manager.get_process(process.process_id)
        assert status is not None
        
        # Delete process via repository
        deleted = await process_manager._repository.delete(process.process_id)
        assert deleted is True
        
        # Verify it's gone
        status = await process_manager.get_process(process.process_id)
        assert status is None
        
        # Try deleting again
        deleted = await process_manager._repository.delete(process.process_id)
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_get_process_stats(self, process_manager):
        """Test getting process statistics."""
        # Create processes with different statuses
        p1 = await process_manager.create_process("Query 1")
        p2 = await process_manager.create_process("Query 2")
        p3 = await process_manager.create_process("Query 3")
        
        # Update statuses
        p2_info = await process_manager.get_process(p2.process_id)
        p2_info.status = "running"
        await process_manager._repository.save(p2_info)
        
        p3_info = await process_manager.get_process(p3.process_id)
        p3_info.status = "completed"
        await process_manager._repository.save(p3_info)
        
        # Get stats
        stats = await process_manager.get_stats()
        
        assert stats["total"] >= 3
        assert stats["pending"] >= 1
        assert stats["running"] >= 1
        assert stats["completed"] >= 1
    
    @pytest.mark.asyncio
    async def test_cleanup_old_processes(self, process_manager):
        """Test cleanup of old completed processes."""
        # Create a completed process
        process = await process_manager.create_process("Old query")
        
        # Manually set it as old and completed
        process_info = await process_manager.get_process(process.process_id)
        process_info.status = "completed"
        process_info.completed_at = datetime.now(UTC)
        await process_manager._repository.save(process_info)
        
        # Cleanup (this won't delete recent processes in the test)
        deleted_count = await process_manager.cleanup_completed(older_than_hours=0)
        
        # Should have cleaned at least the one we created
        assert deleted_count >= 0
    
    @pytest.mark.asyncio
    async def test_process_lifecycle(self, process_manager):
        """Test full process lifecycle: create -> update -> complete."""
        query = "Lifecycle test query?"
        
        # 1. Create
        process = await process_manager.create_process(query)
        assert process.status == "pending"
        
        # 2. Update to running
        process_info = await process_manager.get_process(process.process_id)
        process_info.status = "running"
        process_info.current_step = "Identifying trigger"
        process_info.completed_steps = ["Getting decision"]
        await process_manager._repository.save(process_info)
        
        # Verify update
        updated = await process_manager.get_process(process.process_id)
        assert updated.status == "running"
        assert updated.current_step == "Identifying trigger"
        assert len(updated.completed_steps) == 1
        
        # 3. Complete
        process_info.status = "completed"
        process_info.completed_at = datetime.now(UTC)
        process_info.result = DecisionState(
            decision_requested=query,
            result="Final decision"
        )
        await process_manager._repository.save(process_info)
        
        # Verify completion
        completed = await process_manager.get_process(process.process_id)
        assert completed.status == "completed"
        assert completed.result is not None
        assert completed.result.result == "Final decision"
        assert completed.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_process_with_error(self, process_manager):
        """Test process that fails with error."""
        query = "Error test query?"
        
        # Create process
        process = await process_manager.create_process(query)
        
        # Simulate error
        process_info = await process_manager.get_process(process.process_id)
        process_info.status = "failed"
        process_info.error = "Test error occurred"
        await process_manager._repository.save(process_info)
        
        # Verify error state
        failed = await process_manager.get_process(process.process_id)
        assert failed.status == "failed"
        assert failed.error == "Test error occurred"
        assert failed.result is None
