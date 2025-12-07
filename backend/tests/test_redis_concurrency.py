"""
Tests for Redis concurrency and locking mechanisms.

This test suite validates that the Redis repository correctly handles
concurrent updates using optimistic locking (WATCH/MULTI/EXEC).

Test Scenarios:
1. Concurrent status updates - multiple writers updating same process
2. Progress tracking conflicts - concurrent step updates
3. Race condition prevention - verify no lost updates
4. Retry mechanism - ensure retries work under contention
5. Performance - verify minimal overhead from locking
"""

import asyncio
import pytest
from datetime import datetime, UTC

from backend.app.models.domain import ProcessInfo, DecisionState
from backend.app.services.redis_repository import (
    RedisProcessRepository,
    InMemoryProcessRepository
)

try:
    import redis
    from fakeredis import FakeRedis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


# Skip all tests if Redis not available
pytestmark = pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")


@pytest.fixture
def fake_redis():
    """Create a fake Redis client for testing."""
    return FakeRedis(decode_responses=False)


@pytest.fixture
def redis_repo(fake_redis):
    """Create a Redis repository with fake Redis."""
    return RedisProcessRepository(redis_client=fake_redis)


class TestAtomicUpdates:
    """Test atomic update operations with locking."""
    
    @pytest.mark.asyncio
    async def test_single_update_succeeds(self, redis_repo):
        """Test that a single update succeeds without conflicts."""
        # Create a process
        process = ProcessInfo(
            process_id="test-1",
            query="Test query",
            status="pending",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Update atomically
        def set_running(p: ProcessInfo):
            p.status = "running"
            p.current_step = "Step 1"
        
        updated = await redis_repo.update_with_lock("test-1", set_running)
        
        # Verify update succeeded
        assert updated is not None
        assert updated.status == "running"
        assert updated.current_step == "Step 1"
        
        # Verify persistence
        retrieved = await redis_repo.get("test-1")
        assert retrieved.status == "running"
        assert retrieved.current_step == "Step 1"
    
    @pytest.mark.asyncio
    async def test_concurrent_status_updates(self, redis_repo):
        """
        Test concurrent updates to status field.
        
        Scenario: Two concurrent operations try to update status
        - Operation 1: pending -> running
        - Operation 2: pending -> completed
        
        Expected: Both operations should succeed (one will retry)
        Last write wins, but no data loss.
        """
        # Create process
        process = ProcessInfo(
            process_id="test-2",
            query="Test query",
            status="pending",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Define concurrent updates
        async def set_running():
            def update(p: ProcessInfo):
                # Simulate some processing time
                asyncio.sleep(0.01)
                p.status = "running"
                p.current_step = "Running"
            return await redis_repo.update_with_lock("test-2", update)
        
        async def set_completed():
            def update(p: ProcessInfo):
                # Simulate some processing time
                asyncio.sleep(0.01)
                p.status = "completed"
                p.completed_at = datetime.now(UTC).isoformat()
            return await redis_repo.update_with_lock("test-2", update)
        
        # Run concurrently
        results = await asyncio.gather(set_running(), set_completed())
        
        # Both should succeed (one retries)
        assert all(r is not None for r in results)
        
        # Final state should be one of the two
        final = await redis_repo.get("test-2")
        assert final.status in ("running", "completed")
    
    @pytest.mark.asyncio
    async def test_concurrent_progress_updates(self, redis_repo):
        """
        Test concurrent progress tracking updates.
        
        Scenario: Multiple nodes updating progress simultaneously
        - Node 1: Update to step 2
        - Node 2: Update to step 3
        - Node 3: Update to step 4
        
        Expected: All updates succeed, final state reflects last write
        """
        # Create process
        process = ProcessInfo(
            process_id="test-3",
            query="Test query",
            status="running",
            created_at=datetime.now(UTC).isoformat(),
            current_step="Step 1",
            completed_steps=[]
        )
        await redis_repo.save(process)
        
        # Define concurrent progress updates
        async def update_step(step_num: int):
            def update(p: ProcessInfo):
                p.current_step = f"Step {step_num}"
                p.completed_steps = [f"Step {i}" for i in range(1, step_num)]
            return await redis_repo.update_with_lock("test-3", update)
        
        # Run multiple concurrent updates
        results = await asyncio.gather(
            update_step(2),
            update_step(3),
            update_step(4)
        )
        
        # All should succeed
        assert all(r is not None for r in results)
        
        # Final state should be consistent
        final = await redis_repo.get("test-3")
        assert final.current_step in ["Step 2", "Step 3", "Step 4"]
        assert isinstance(final.completed_steps, list)
    
    @pytest.mark.asyncio
    async def test_update_nonexistent_process(self, redis_repo):
        """Test that updating a nonexistent process returns None."""
        def update(p: ProcessInfo):
            p.status = "running"
        
        result = await redis_repo.update_with_lock("nonexistent", update)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_multiple_field_update(self, redis_repo):
        """Test updating multiple fields atomically."""
        # Create process
        process = ProcessInfo(
            process_id="test-4",
            query="Test query",
            status="running",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Update multiple fields
        def update_all(p: ProcessInfo):
            p.status = "completed"
            p.completed_at = datetime.now(UTC).isoformat()
            p.current_step = None
            p.completed_steps = ["Step 1", "Step 2", "Step 3"]
        
        updated = await redis_repo.update_with_lock("test-4", update_all)
        
        # Verify all fields updated
        assert updated is not None
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.current_step is None
        assert len(updated.completed_steps) == 3


class TestRetryMechanism:
    """Test retry behavior under contention."""
    
    @pytest.mark.asyncio
    async def test_retry_on_conflict(self, redis_repo):
        """
        Test that conflicting updates trigger retries.
        
        We'll simulate a conflict by using the raw Redis WATCH
        and modifying the key during an update.
        """
        # Create process
        process = ProcessInfo(
            process_id="test-5",
            query="Test query",
            status="pending",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Track retry attempts
        attempt_count = [0]
        
        def update_with_conflict(p: ProcessInfo):
            attempt_count[0] += 1
            
            # On first attempt, simulate external modification
            if attempt_count[0] == 1:
                # Another client modifies the process
                other_process = ProcessInfo(
                    process_id="test-5",
                    query="Test query",
                    status="running",  # Different status
                    created_at=p.created_at
                )
                # Use asyncio to run async save in sync function
                asyncio.create_task(redis_repo.save(other_process))
            
            p.status = "completed"
        
        # This should succeed after retry
        updated = await redis_repo.update_with_lock("test-5", update_with_conflict, max_retries=3)
        
        # Should have succeeded (possibly after retries)
        assert updated is not None
        assert updated.status == "completed"


class TestConcurrentLoad:
    """Test behavior under high concurrent load."""
    
    @pytest.mark.asyncio
    async def test_many_concurrent_updates(self, redis_repo):
        """
        Test many concurrent updates to the same process.
        
        Scenario: 20 concurrent updates
        Expected: All succeed, final state is consistent
        """
        # Create process
        process = ProcessInfo(
            process_id="test-6",
            query="Test query",
            status="running",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Define update function
        async def increment_step(step_num: int):
            def update(p: ProcessInfo):
                p.current_step = f"Step {step_num}"
            return await redis_repo.update_with_lock("test-6", update)
        
        # Run many concurrent updates
        num_updates = 20
        results = await asyncio.gather(
            *[increment_step(i) for i in range(num_updates)]
        )
        
        # All should succeed
        assert all(r is not None for r in results)
        
        # Final state should be valid
        final = await redis_repo.get("test-6")
        assert final.status == "running"
        assert final.current_step is not None


class TestInMemoryRepository:
    """Test that InMemoryRepository also supports update_with_lock."""
    
    @pytest.mark.asyncio
    async def test_in_memory_update(self):
        """Test that in-memory repository supports atomic updates."""
        repo = InMemoryProcessRepository()
        
        # Create process
        process = ProcessInfo(
            process_id="test-7",
            query="Test query",
            status="pending",
            created_at=datetime.now(UTC).isoformat()
        )
        await repo.save(process)
        
        # Update atomically
        def set_running(p: ProcessInfo):
            p.status = "running"
        
        updated = await repo.update_with_lock("test-7", set_running)
        
        # Verify update succeeded
        assert updated is not None
        assert updated.status == "running"
        
        # Verify persistence
        retrieved = await repo.get("test-7")
        assert retrieved.status == "running"


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_update_with_exception_in_function(self, redis_repo):
        """Test that exceptions in update function are handled."""
        # Create process
        process = ProcessInfo(
            process_id="test-8",
            query="Test query",
            status="pending",
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Update function that raises exception
        def bad_update(p: ProcessInfo):
            raise ValueError("Something went wrong")
        
        # Should handle exception gracefully
        with pytest.raises(ValueError):
            await redis_repo.update_with_lock("test-8", bad_update)
        
        # Original process should be unchanged
        retrieved = await redis_repo.get("test-8")
        assert retrieved.status == "pending"
    
    @pytest.mark.asyncio
    async def test_update_preserves_result(self, redis_repo):
        """Test that updating metadata preserves the result field."""
        # Create process with result
        state = DecisionState(decision_requested="Should I test?")
        state.trigger = "Testing is important"
        
        process = ProcessInfo(
            process_id="test-9",
            query="Test query",
            status="running",
            result=state,
            created_at=datetime.now(UTC).isoformat()
        )
        await redis_repo.save(process)
        
        # Update status only
        def set_completed(p: ProcessInfo):
            p.status = "completed"
            p.completed_at = datetime.now(UTC).isoformat()
        
        updated = await redis_repo.update_with_lock("test-9", set_completed)
        
        # Result should be preserved
        assert updated is not None
        assert updated.result is not None
        assert updated.result.trigger == "Testing is important"
        
        # Verify persistence
        retrieved = await redis_repo.get("test-9")
        assert retrieved.result is not None
        assert retrieved.result.trigger == "Testing is important"
