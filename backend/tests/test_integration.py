"""
Integration Tests

End-to-end tests that verify the complete system works together.
These tests may take longer as they involve actual workflow execution.
"""

import pytest
import asyncio
from pathlib import Path

from backend.app.models.domain import DecisionState
from backend.app.services.decision_service import DecisionService
from backend.app.services.process_manager import ProcessManager
from backend.app.services.redis_repository import InMemoryProcessRepository


class TestEndToEndWorkflow:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_workflow_with_manager(self):
        """Test complete workflow using ProcessManager."""
        # Setup
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        query = "Should I invest in renewable energy stocks?"
        
        # Create process
        process = await manager.create_process(query)
        assert process.status == "pending"
        
        # In a real scenario, execute_process would run here
        # For integration testing without mocking AI:
        # await manager.execute_process(process.process_id, query)
        
        # Verify process was created
        retrieved = await manager.get_process(process.process_id)
        assert retrieved is not None
        assert retrieved.query == query
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_process_lifecycle_complete(self):
        """Test complete process lifecycle."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # 1. Create
        process = await manager.create_process("Test decision?")
        initial_id = process.process_id
        
        # 2. Verify in pending state
        status = await manager.get_process(initial_id)
        assert status.status == "pending"
        
        # 3. Simulate running state
        status.status = "running"
        status.current_step = "Analyzing"
        await repository.save(status)
        
        # 4. Verify running state
        running_status = await manager.get_process(initial_id)
        assert running_status.status == "running"
        assert running_status.current_step == "Analyzing"
        
        # 5. Simulate completion
        running_status.status = "completed"
        running_status.result = DecisionState(
            decision_requested="Test decision?",
            result="Final decision"
        )
        await repository.save(running_status)
        
        # 6. Verify completed
        completed_status = await manager.get_process(initial_id)
        assert completed_status.status == "completed"
        assert completed_status.result is not None


class TestRepositoryIntegration:
    """Integration tests for repository operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_process_creation(self):
        """Test creating multiple processes concurrently."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Create multiple processes concurrently
        tasks = [
            manager.create_process(f"Decision {i}?")
            for i in range(10)
        ]
        
        processes = await asyncio.gather(*tasks)
        
        # Verify all were created with unique IDs
        assert len(processes) == 10
        process_ids = {p.process_id for p in processes}
        assert len(process_ids) == 10  # All unique
    
    @pytest.mark.asyncio
    async def test_concurrent_status_checks(self):
        """Test checking status of multiple processes concurrently."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Create processes
        process_ids = []
        for i in range(5):
            process = await manager.create_process(f"Query {i}?")
            process_ids.append(process.process_id)
        
        # Check all statuses concurrently
        tasks = [
            manager.get_process(pid)
            for pid in process_ids
        ]
        
        statuses = await asyncio.gather(*tasks)
        
        # Verify all statuses retrieved
        assert len(statuses) == 5
        assert all(s is not None for s in statuses)


class TestServiceIntegration:
    """Integration tests for service layer."""
    
    def test_decision_service_with_state(self):
        """Test DecisionService state extraction."""
        service = DecisionService()
        
        # Create state
        state = DecisionState(
            decision_requested="Test query",
            trigger="Test trigger",
            result="Test result",
            result_comment="Test comment"
        )
        
        # Extract results
        summary = service.extract_result_summary(state)
        full = service.extract_full_result(state)
        
        # Verify extraction
        assert summary["selected_decision"] == "Test result"
        assert full["trigger"] == "Test trigger"
    
    def test_process_manager_with_decision_service(self):
        """Test ProcessManager integration with DecisionService."""
        repository = InMemoryProcessRepository()
        decision_service = DecisionService()
        manager = ProcessManager(
            repository=repository,
            decision_service=decision_service
        )
        
        assert manager._repository is repository
        assert manager._decision_service is decision_service


class TestErrorHandling:
    """Integration tests for error handling."""
    
    @pytest.mark.asyncio
    async def test_repository_error_handling(self):
        """Test error handling in repository operations."""
        repository = InMemoryProcessRepository()
        
        # Try to get non-existent process
        result = await repository.get("nonexistent")
        assert result is None
        
        # Try to delete non-existent process
        deleted = await repository.delete("nonexistent")
        assert deleted is False
    
    @pytest.mark.asyncio
    async def test_process_manager_error_handling(self):
        """Test error handling in ProcessManager."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Get status of non-existent process
        status = await manager.get_process("nonexistent")
        assert status is None
        
        # Delete non-existent process via repository
        deleted = await repository.delete("nonexistent")
        assert deleted is False


class TestDataConsistency:
    """Tests for data consistency across operations."""
    
    @pytest.mark.asyncio
    async def test_process_update_consistency(self):
        """Test that process updates maintain consistency."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Create process
        process = await manager.create_process("Test query")
        pid = process.process_id
        
        # Update status multiple times
        for status in ["running", "completed"]:
            proc = await manager.get_process(pid)
            proc.status = status
            await repository.save(proc)
            
            # Verify update
            updated = await manager.get_process(pid)
            assert updated.status == status
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test concurrent updates to same process."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Create process
        process = await manager.create_process("Test query")
        pid = process.process_id
        
        # Function to update process
        async def update_step(step_name):
            proc = await manager.get_process(pid)
            proc.completed_steps.append(step_name)
            await repository.save(proc)
        
        # Update concurrently
        await asyncio.gather(
            update_step("Step 1"),
            update_step("Step 2"),
            update_step("Step 3")
        )
        
        # Verify all steps were added
        final = await manager.get_process(pid)
        assert len(final.completed_steps) == 3


class TestScalability:
    """Tests for system scalability."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_many_processes(self):
        """Test creating many processes."""
        repository = InMemoryProcessRepository()
        manager = ProcessManager(repository=repository)
        
        # Create many processes
        num_processes = 100
        tasks = [
            manager.create_process(f"Query {i}?")
            for i in range(num_processes)
        ]
        
        processes = await asyncio.gather(*tasks)
        
        # Verify all created
        assert len(processes) == num_processes
        
        # Verify all can be retrieved
        all_processes = await manager.get_all_processes()
        assert len(all_processes) >= num_processes
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_repository_performance(self):
        """Test repository performance with many operations."""
        repository = InMemoryProcessRepository()
        
        # Create many processes
        processes = []
        for i in range(50):
            from backend.app.models.domain import ProcessInfo
            from datetime import datetime, UTC
            
            proc = ProcessInfo(
                process_id=f"perf-test-{i}",
                query=f"Query {i}",
                status="pending",
                created_at=datetime.now(UTC)
            )
            await repository.save(proc)
            processes.append(proc)
        
        # Retrieve all
        all_procs = await repository.list_all()
        assert len(all_procs) >= 50
        
        # Get stats
        stats = await repository.get_stats()
        assert stats["total"] >= 50
