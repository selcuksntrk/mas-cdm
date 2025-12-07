"""
Process Manager Service

The Repository Pattern provides an abstraction layer between business logic
(ProcessManager) and data storage (Redis, database, etc.).

Benefits:
1. Separation of Concerns: ProcessManager focuses on process logic,
   not storage implementation
2. Testability: Easy to inject mock repositories for testing
3. Flexibility: Swap storage without changing business logic
4. Single Responsibility: Each class has one job

Example:
    # Development
    manager = ProcessManager(InMemoryProcessRepository())
    
    # Production
    manager = ProcessManager(RedisProcessRepository())
    
    # Testing
    manager = ProcessManager(MockProcessRepository())
    

DEPENDENCY INJECTION:
Instead of ProcessManager creating its own storage:
    
    BAD:
    class ProcessManager:
        def __init__(self):
            self.storage = Redis()  # Hard-coded!
    
    GOOD:
    class ProcessManager:
        def __init__(self, repository: IProcessRepository):
            self.repository = repository  # Injected!

Benefits:
- Testing: Inject fake repositories
- Flexibility: Change storage at runtime
- Loose Coupling: ProcessManager doesn't know about Redis
- Configuration: Control from outside

This follows:
- Dependency Inversion Principle (SOLID)
- Hexagonal Architecture (Ports & Adapters)
- Clean Architecture (Uncle Bob)
"""


import asyncio
from datetime import datetime, UTC
from typing import Optional
from uuid import uuid4

from backend.app.models.domain import DecisionState, ProcessInfo
from backend.app.services.decision_service import DecisionService
from backend.app.services.redis_repository import (
    IProcessRepository,
    get_process_repository
)



# Process Manager
class ProcessManager:
    """
    Manager for tracking asynchronous decision-making processes.
    
    RESPONSIBILITIES:
    1. Process lifecycle management (create, execute, track)
    2. Error handling and status updates
    3. Cleanup and statistics
    
    NOT RESPONSIBLE FOR:
    - Storage implementation (that's the repository's job)
    - Redis/database connections (abstracted away)
    - Serialization (repository handles it)
    
    This separation makes ProcessManager:
    - Easy to test (mock repository)
    - Storage-agnostic (works with any repository)
    - Focused on business logic, not infrastructure
    
    DESIGN PATTERN: Facade + Repository
    ProcessManager is a Facade that simplifies complex process operations
    while using Repository pattern for storage abstraction.
    """
    
    def __init__(
        self,
        repository: Optional[IProcessRepository] = None,
        decision_service: Optional[DecisionService] = None
    ):
        """
        Initialize the process manager.
        
        DEPENDENCY INJECTION EXPLAINED:
        We accept repository and service as parameters instead of creating them.
        
        Why?
        1. Testing: Can inject mocks
        2. Flexibility: Can use different implementations
        3. Configuration: Controlled from outside
        4. Explicit Dependencies: Clear what this class needs
        
        Args:
            repository: Storage backend for processes (defaults to auto-detect)
            decision_service: Service for running decisions (defaults to new instance)
        
        Example:
            # Auto-detect storage (Redis if available, in-memory otherwise)
            manager = ProcessManager()
            
            # Force Redis
            manager = ProcessManager(repository=RedisProcessRepository())
            
            # Force in-memory (testing)
            manager = ProcessManager(repository=InMemoryProcessRepository())
        """
        self._repository = repository or get_process_repository()
        self._decision_service = decision_service or DecisionService()
    
    async def create_process(self, decision_query: str) -> ProcessInfo:
        """
        Create a new process entry and return its info.
        
        PROCESS LIFECYCLE:
        1. Generate unique ID
        2. Create ProcessInfo metadata
        3. Store in repository
        4. Return ProcessInfo to caller
        
        WHY ASYNC?
        - Repository operations might be async (network I/O for Redis)
        - Maintains consistent API with other async methods
        - Future-proof for async validations/hooks
        
        Args:
            decision_query: The decision query for the process
            
        Returns:
            ProcessInfo: Information about the created process
            
        Example:
            >>> manager = ProcessManager()
            >>> process = await manager.create_process("Should I switch careers?")
            >>> print(process.process_id)
        """
        # Generate unique process ID
        process_id = f"process_{uuid4().hex[:12]}"

        # Create process info. Use 'pending' so callers can see the
        # process has been created but not yet executed. Store the
        # original decision query on the ProcessInfo for later retrieval
        # by background tasks.
        process_info = ProcessInfo(
            process_id=process_id,
            query=decision_query,
            status="pending",
            result=None,
            error=None,
            created_at=datetime.now(UTC).isoformat(),
            completed_at=None
        )
        
        # Store in repository (might be Redis, might be in-memory)
        await self._repository.save(process_info)
        
        return process_info
    
    async def execute_process(self, process_id: str, decision_query: Optional[str] = None):
        """
        Execute a decision-making process and update its status.
        
        This method should be called as a background task.
        
        ERROR HANDLING STRATEGY:
        We use try/except to catch failures and update process status:
        - Success: status="completed", result=DecisionState
        - Failure: status="failed", error=error message
        
        This ensures that:
        1. Failures are captured and visible to users
        2. Background tasks don't crash silently
        3. Process status always reflects reality
        
        IMPORTANT: Repository Pattern in Action
        Notice we don't care HOW the data is stored (Redis vs in-memory).
        We just call repository.get() and repository.save().
        This is the power of abstraction!
        
        Args:
            process_id: The ID of the process to execute
            decision_query: The decision query to process
            
        Example:
            >>> manager = ProcessManager()
            >>> process = await manager.create_process("Should I switch careers?")
            >>> await manager.execute_process(process.process_id, "Should I switch careers?")
        """
        try:
            # Determine decision query. If not provided, fetch it from
            # the repository (compatibility with callers that only pass
            # process_id).
            if decision_query is None:
                stored = await self._repository.get(process_id)
                if not stored:
                    return
                decision_query = stored.query

            # Update status to running
            process_info = await self._repository.get(process_id)
            if process_info:
                process_info.status = "running"
                process_info.current_step = "Initializing decision workflow"
                await self._repository.save(process_info)

            # Run the actual decision process with real-time progress tracking
            from app.core.graph import decision_graph
            from app.core.graph.nodes import GetDecision
            from app.models.domain import DecisionState
            from pydantic_graph import End
            
            # Create initial state
            state = DecisionState(decision_requested=decision_query)
            first_node = GetDecision()
            
            # Define ordered workflow steps for progress tracking
            workflow_steps = [
                "Getting decision query",
                "Identifying trigger",
                "Analyzing root cause",
                "Defining scope",
                "Drafting decision",
                "Establishing goals",
                "Identifying information needs",
                "Retrieving information",
                "Updating draft",
                "Generating alternatives",
                "Selecting final decision"
            ]
            
            # Map node names to workflow step indices
            node_to_step_index = {
                "GetDecision": 0,
                "IdentifyTrigger": 1,
                "AnalyzeRootCause": 2,
                "ScopeDefinition": 3,
                "Drafting": 4,
                "EstablishGoals": 5,
                "IdentifyInformationNeeded": 6,
                "RetrieveInformationNeeded": 7,
                "UpdateDraft": 8,
                "GenerationOfAlternatives": 9,
                "Result": 10
            }
            
            # Track highest step reached (for handling retries)
            highest_step_index = -1
            
            # Execute graph with iteration for progress tracking
            async with decision_graph.iter(first_node, state=state) as run:
                while True:
                    node = await run.next()
                    node_name = type(node).__name__
                    
                    # Skip evaluator nodes in progress display
                    if node_name.startswith("Evaluate_"):
                        continue
                    
                    # Get step index for this node
                    step_index = node_to_step_index.get(node_name)
                    
                    if step_index is not None:
                        # Update highest step if this is further in the workflow
                        if step_index > highest_step_index:
                            highest_step_index = step_index
                        
                        current_step = workflow_steps[step_index]
                        
                        # Completed steps are all steps up to (but not including) current step
                        completed_steps = workflow_steps[:highest_step_index] if highest_step_index > 0 else []
                        
                        # Update process with current step
                        process_info = await self._repository.get(process_id)
                        if process_info and process_info.status == "running":
                            process_info.current_step = current_step
                            process_info.completed_steps = completed_steps
                            await self._repository.save(process_info)
                    
                    # Check if we've reached the end
                    if isinstance(node, End):
                        break

            # Retrieve current process info from repository
            process_info = await self._repository.get(process_id)
            if process_info:
                # Update process with result - mark all workflow steps as completed
                process_info.status = "completed"
                process_info.result = state
                process_info.current_step = None
                process_info.completed_steps = workflow_steps  # All steps completed
                process_info.completed_at = datetime.now(UTC).isoformat()

                # Save back to repository
                await self._repository.save(process_info)

        except Exception as e:
            # Retrieve current process info from repository
            process_info = await self._repository.get(process_id)
            if process_info:
                # Update process with error
                process_info.status = "failed"
                process_info.error = str(e)
                process_info.current_step = None
                process_info.completed_at = datetime.now(UTC).isoformat()

                # Save back to repository
                await self._repository.save(process_info)
    
    async def get_process(self, process_id: str) -> Optional[ProcessInfo]:
        """
        Get information about a specific process.
        
        WHY ASYNC?
        Repository operations might involve:
        - Network I/O (Redis)
        - Disk I/O (future database)
        - Deserialization (pickle, JSON)
        
        Making this async ensures we don't block the event loop.
        
        Args:
            process_id: The ID of the process to retrieve
            
        Returns:
            ProcessInfo: Process information, or None if not found
            
        Example:
            >>> manager = ProcessManager()
            >>> process = await manager.get_process("process_abc123")
            >>> if process:
            ...     print(process.status)
        """
        return await self._repository.get(process_id)
    
    async def process_exists(self, process_id: str) -> bool:
        """
        Check if a process exists.
        
        This is a convenience method that's more efficient than:
            process = await manager.get_process(id)
            exists = process is not None
        
        Args:
            process_id: The ID to check
            
        Returns:
            bool: True if process exists, False otherwise
        """
        return await self._repository.exists(process_id)
    
    # Backwards-compatible alias
    async def exists(self, process_id: str) -> bool:
        return await self.process_exists(process_id)
    
    async def get_all_processes(self) -> list[ProcessInfo]:
        """
        Get all tracked processes.
        
        WHY THIS IS USEFUL:
        - Admin dashboards showing all processes
        - Monitoring and debugging
        - Audit trails
        
        PERFORMANCE NOTE:
        This could be expensive with many processes in Redis.
        Consider pagination for production:
            - Add offset/limit parameters
            - Use Redis SCAN instead of KEYS
            - Cache results with TTL
        
        Returns:
            list[ProcessInfo]: List of all process information
        """
        return await self._repository.list_all()

    # Backwards-compatible alias expected by older tests
    async def list_all(self) -> list[ProcessInfo]:
        return await self.get_all_processes()
    
    async def get_running_processes(self) -> list[ProcessInfo]:
        """
        Get all currently running processes.
        
        IMPLEMENTATION NOTE:
        We filter in-memory after fetching from repository.
        Alternative approaches:
        1. Store running processes in a Redis set (faster filtering)
        2. Use Redis sorted sets with status as score
        3. Add filter parameter to repository.list_all()
        
        Current approach is simple and works well for moderate scale.
        
        Returns:
            list[ProcessInfo]: List of running processes
        """
        all_processes = await self._repository.list_all()
        return [p for p in all_processes if p.status == "running"]
    
    async def get_completed_processes(self) -> list[ProcessInfo]:
        """
        Get all completed processes.
        
        Returns:
            list[ProcessInfo]: List of completed processes
        """
        all_processes = await self._repository.list_all()
        return [p for p in all_processes if p.status == "completed"]
    
    async def get_failed_processes(self) -> list[ProcessInfo]:
        """
        Get all failed processes.
        
        WHY TRACK FAILURES?
        - Debugging: See what went wrong
        - Monitoring: Alert on high failure rates
        - Recovery: Retry failed processes
        
        Returns:
            list[ProcessInfo]: List of failed processes
        """
        all_processes = await self._repository.list_all()
        return [p for p in all_processes if p.status == "failed"]
    
    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        """
        Remove completed and failed processes from storage.
        
        WHY CLEANUP?
        - Prevent storage bloat
        - Improve query performance
        - Manage costs (Redis memory)
        
        IMPORTANT: Production Considerations
        1. Age-based cleanup (older_than_hours parameter)
           - Keeps recent results for debugging
           - Removes old data automatically
        
        2. Run as periodic task (cron job or scheduler)
           - Don't wait for manual cleanup
           - Consider APScheduler or Celery Beat
        
        3. Monitor cleanup metrics
           - How many processes cleaned?
           - How often is cleanup running?
           - Is storage growing despite cleanup?
        
        Args:
            older_than_hours: Only clean processes older than this (default 24 hours)
        
        Returns:
            int: Number of processes cleaned up
            
        Example:
            >>> manager = ProcessManager()
            >>> # Clean processes older than 1 hour
            >>> cleaned = await manager.cleanup_completed(older_than_hours=1)
            >>> print(f"Cleaned up {cleaned} processes")
        """
        return await self._repository.cleanup_completed(older_than_hours)
    
    async def cleanup_all(self):
        """
        Remove all processes from storage.
        
        DANGER ZONE
        This is a destructive operation that removes ALL process data.
        
        Use cases:
        - Development/testing: Clear test data
        - Maintenance: Full reset of process storage
        
        DO NOT use in production without:
        1. Backup
        2. Confirmation
        3. Maintenance window
        
        Consider adding:
        - Confirmation parameter: cleanup_all(confirm="yes-delete-all")
        - Backup before delete
        - Audit logging
        """
        # Get all process IDs
        all_processes = await self._repository.list_all()
        
        # Delete each one
        for process in all_processes:
            await self._repository.delete(process.process_id)
    
    async def get_stats(self) -> dict:
        """
        Get statistics about all processes.
        
        MONITORING METRICS:
        These stats are valuable for:
        - Health dashboards
        - Alerting (high failure rate?)
        - Capacity planning (too many running?)
        
        PERFORMANCE NOTE:
        With Repository Pattern, this requires:
        1. Fetch all processes (network call for Redis)
        2. Filter in memory
        
        Optimization ideas:
        - Cache stats with short TTL
        - Use Redis counters (INCR/DECR on status changes)
        - Store counts separately
        
        Returns:
            dict: Statistics including counts by status
            
        Example:
            >>> manager = ProcessManager()
            >>> stats = await manager.get_stats()
            >>> print(f"Running: {stats['running']}, Completed: {stats['completed']}")
        """
        all_processes = await self._repository.list_all()

        pending = sum(1 for p in all_processes if p.status == "pending")
        running = sum(1 for p in all_processes if p.status == "running")
        completed = sum(1 for p in all_processes if p.status == "completed")
        failed = sum(1 for p in all_processes if p.status == "failed")

        return {
            "total": len(all_processes),
            "pending": pending,
            "running": running,
            "completed": completed,
            "failed": failed,
        }
        

# Global process manager instance (singleton pattern)
# In production, consider using dependency injection
_process_manager: Optional[ProcessManager] = None


def get_process_manager() -> ProcessManager:
    """
    Get the global process manager instance (Singleton Pattern).
    
    SINGLETON PATTERN EXPLAINED:
    ============================
    Instead of creating new ProcessManager every time, we:
    1. Create one instance (_process_manager)
    2. Reuse it across the application
    3. Ensure consistent state
    
    Why?
    - Shared state: All API endpoints see same processes
    - Efficiency: No repeated initialization
    - Simplicity: Easy to get the manager
    
    IMPORTANT: Repository Selection
    ================================
    The manager automatically uses the right repository:
    - If Redis is configured and available → RedisProcessRepository
    - Otherwise → InMemoryProcessRepository
    
    This happens transparently via get_process_repository() factory.
    
    PRODUCTION NOTE:
    ================
    For more control, consider dependency injection with FastAPI:
    
        from fastapi import Depends
        
        def get_manager() -> ProcessManager:
            return ProcessManager(
                repository=get_process_repository()
            )
        
        @app.get("/process/{id}")
        async def get_process(
            id: str,
            manager: ProcessManager = Depends(get_manager)
        ):
            return await manager.get_process(id)
    
    This allows:
    - Per-request instances (if needed)
    - Easy testing with mock repositories
    - Clear dependency graph
    
    Returns:
        ProcessManager: The global process manager
        
    Example:
        >>> manager = get_process_manager()
        >>> process = await manager.create_process("Should I switch careers?")
    """
    global _process_manager
    if _process_manager is None:
        _process_manager = ProcessManager()
    return _process_manager