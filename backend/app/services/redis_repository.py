"""
Redis Process Repository

This module implements the Repository Pattern for process storage using Redis.

WHY REDIS?
Redis is an in-memory data store that provides:
1. Persistence: Data survives server restarts (unlike in-memory dict)
2. Speed: Very fast read/write operations (~100k ops/sec)
3. Distribution: Multiple backend instances can share state
4. TTL: Automatic expiration of old data
5. Atomic operations: Thread-safe without locks


"""


import asyncio
import json
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, UTC
from typing import Optional, List, Dict

import redis
from redis.exceptions import RedisError

from backend.app.models.domain import ProcessInfo, DecisionState
from backend.app.config import get_settings


class IProcessRepository(ABC):
    """
    Interface for process storage.
    
    WHY AN INTERFACE?
    Defines a contract that all implementations must follow.
    This enables:
    - Dependency Inversion: Depend on interface, not implementation
    - Easy testing: Create mock repositories
    - Flexibility: Swap storage without changing business logic
    
    Example:
        # Production
        repo = RedisProcessRepository()
        
        # Testing
        repo = InMemoryProcessRepository()
        
        # Future
        repo = PostgreSQLProcessRepository()
        
        # ProcessManager works with all of them!
        manager = ProcessManager(repo)
    """
    
    @abstractmethod
    async def save(self, process: ProcessInfo) -> None:
        """Save a process."""
        pass
    
    @abstractmethod
    async def get(self, process_id: str) -> Optional[ProcessInfo]:
        """Get a process by ID."""
        pass
    
    @abstractmethod
    async def exists(self, process_id: str) -> bool:
        """Check if a process exists."""
        pass
    
    @abstractmethod
    async def delete(self, process_id: str) -> bool:
        """Delete a process. Returns True if deleted, False if not found."""
        pass
    
    @abstractmethod
    async def list_all(self) -> List[ProcessInfo]:
        """List all processes."""
        pass
    
    @abstractmethod
    async def get_stats(self) -> Dict[str, int]:
        """Get statistics about processes."""
        pass
    
    @abstractmethod
    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        """Clean up old completed/failed processes."""
        pass
    
    @abstractmethod
    async def update_with_lock(self, process_id: str, update_fn, max_retries: int = 5) -> Optional[ProcessInfo]:
        """
        Update a process atomically.
        
        Args:
            process_id: Process ID to update
            update_fn: Function that takes ProcessInfo and modifies it
            max_retries: Maximum retry attempts on conflict
            
        Returns:
            Updated ProcessInfo or None if update failed
        """
        pass
    

class InMemoryProcessRepository(IProcessRepository):
    """
    In-memory implementation of process repository.
    
    WHEN TO USE:
    - Development: Fast, no Redis setup needed
    - Testing: Isolated tests, no external dependencies
    - Simple deployments: Single instance, don't need persistence
    
    LIMITATIONS:
    - Data lost on restart
    - Not shared across instances
    - No persistence
    - Limited by RAM
    
    This is what we used before - simple but not production-ready.
    """
    
    def __init__(self):
        """Initialize with empty storage."""
        self._storage: Dict[str, ProcessInfo] = {}
    
    async def save(self, process: ProcessInfo) -> None:
        """Save process to memory."""
        self._storage[process.process_id] = process
    
    async def get(self, process_id: str) -> Optional[ProcessInfo]:
        """Get process from memory."""
        return self._storage.get(process_id)
    
    async def exists(self, process_id: str) -> bool:
        """Check if process exists in memory."""
        return process_id in self._storage
    
    async def delete(self, process_id: str) -> bool:
        """Delete process from memory. Returns True if deleted, False if not found."""
        if process_id in self._storage:
            del self._storage[process_id]
            return True
        return False
    
    async def list_all(self) -> List[ProcessInfo]:
        """List all processes from memory."""
        return list(self._storage.values())
    
    async def get_stats(self) -> Dict[str, int]:
        """Get statistics from memory."""
        processes = list(self._storage.values())
        return {
            "total": len(processes),
            "running": sum(1 for p in processes if p.status == "running"),
            "completed": sum(1 for p in processes if p.status == "completed"),
            "failed": sum(1 for p in processes if p.status == "failed"),
        }
    
    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        """Clean up completed processes from memory."""
        # Simple implementation: remove all completed/failed
        to_delete = [
            pid for pid, p in self._storage.items()
            if p.status in ("completed", "failed")
        ]
        for pid in to_delete:
            del self._storage[pid]
        return len(to_delete)
    
    async def update_with_lock(self, process_id: str, update_fn, max_retries: int = 5) -> Optional[ProcessInfo]:
        """
        Update a process atomically (in-memory version).
        
        For in-memory storage, no actual locking is needed since
        Python's GIL ensures atomic dict operations. But we implement
        the same interface for consistency.
        """
        process = self._storage.get(process_id)
        if not process:
            return None
        
        # Apply update function
        update_fn(process)
        
        # Save updated process
        self._storage[process_id] = process
        
        return process
    

class RedisProcessRepository(IProcessRepository):
    """
    Redis implementation of process repository.
    
    WHEN TO USE:
    - Production: Persistent, shared state
    - Multi-instance: Multiple backend servers
    - High-scale: Handles thousands of processes
    
    REDIS DATA STRUCTURES USED:
    1. Hash (process:{id}):
        - Stores process metadata as fields
        - Fast field-level updates
        - Efficient for structured data
    
    2. String (process:{id}:result):
        - Stores DecisionState (pickled)
        - Separate from metadata (can be large)
        - Optional: Don't store if process failed
    
    3. Set (processes:all):
        - Tracks all process IDs
        - Fast membership checks
        - Efficient iteration
    
    4. Sorted Set (processes:completed):
        - Tracks completed processes by timestamp
        - Enables cleanup by age
        - Score = completion timestamp
    
    SERIALIZATION STRATEGY:
    Why Pickle for DecisionState?
    - Preserves Python objects exactly
    - Handles complex nested structures
    - Fast serialization/deserialization
    - No schema needed
    
    Why JSON for metadata?
    - Human-readable in Redis
    - Easy debugging
    - Language-agnostic
    - Smaller for simple data
    
    ALTERNATIVE: Could use JSON for everything, but would need
    custom serializers for datetime, dataclass, etc.
    
    ERROR HANDLING:
    All Redis operations wrapped in try/except:
    - Network failures: Return None/default
    - Connection issues: Graceful degradation
    - Data corruption: Skip invalid entries
    - Logging: Record errors for debugging
    
    TTL STRATEGY:
    - Running processes: No TTL (manual cleanup)
    - Completed processes: 7 days TTL
    - Failed processes: 7 days TTL
    
    Why? Running processes might take hours. Don't want them
    expiring mid-execution. Completed/failed can be cleaned up.
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        key_prefix: str = "process:",
        default_ttl: int = 604800  # 7 days in seconds
    ):
        """
        Initialize Redis repository.
        
        Args:
            redis_client: Redis client instance (or create from config)
            key_prefix: Prefix for all Redis keys (namespace)
            default_ttl: TTL in seconds for completed processes
        
        DESIGN DECISION: Why accept redis_client parameter?
        ===================================================
        Dependency Injection:
        - Testing: Inject fakeredis client
        - Connection pooling: Reuse client across repositories
        - Configuration: Control connection settings externally
        - Flexibility: Use different Redis instances
        """
        if redis_client is None:
            # Create from settings if not provided
            settings = get_settings()
            self._redis = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=False,  # We handle decoding
                socket_timeout=5,  # 5 second timeout
                socket_connect_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
        else:
            self._redis = redis_client
        
        self._key_prefix = key_prefix
        self._default_ttl = default_ttl
        self._all_processes_key = f"{key_prefix}all"
        self._completed_key = f"{key_prefix}completed"
    
    def _make_key(self, process_id: str) -> str:
        """
        Create Redis key for a process.
        
        Why prefix keys?
        ================
        - Namespace: Avoid collisions with other data
        - Organization: Easy to find related keys
        - Cleanup: Can delete by pattern (DEL process:*)
        - Multi-tenancy: Different prefixes per tenant
        
        Example: process:abc123 instead of just abc123
        """
        return f"{self._key_prefix}{process_id}"
    
    def _make_result_key(self, process_id: str) -> str:
        """Create Redis key for process result."""
        return f"{self._key_prefix}{process_id}:result"
    
    async def update_with_lock(
        self,
        process_id: str,
        update_fn,
        max_retries: int = 5
    ) -> Optional[ProcessInfo]:
        """
        Update a process atomically using optimistic locking.
        
        OPTIMISTIC LOCKING EXPLAINED:
        ==============================
        Redis WATCH/MULTI/EXEC provides optimistic concurrency control:
        
        1. WATCH key - Monitor key for changes
        2. GET key - Read current value
        3. Modify value in application
        4. MULTI - Begin transaction
        5. SET key - Write new value
        6. EXEC - Commit if key unchanged since WATCH
        
        If key changed between WATCH and EXEC:
        - EXEC returns None (transaction aborted)
        - Retry the operation
        
        WHY OPTIMISTIC LOCKING?
        =======================
        Pros:
        - No lock contention under low concurrency
        - No deadlocks
        - Better performance than pessimistic locks
        - Simple implementation
        
        Cons:
        - Retries needed under high contention
        - Multiple round-trips on retry
        
        Alternative: Pessimistic locking with SET NX EX
        - Acquire lock: SET lock:{id} 1 NX EX 10
        - Do work
        - Release lock: DEL lock:{id}
        - Pros: Guaranteed single writer
        - Cons: Lock management complexity, deadlock risk
        
        WHEN TO USE:
        ============
        Use this method for read-modify-write operations:
        - Status updates
        - Progress tracking
        - Any field update that depends on current value
        
        DON'T USE for simple writes (new processes, deletes)
        
        Args:
            process_id: Process ID to update
            update_fn: Function that takes ProcessInfo and modifies it
            max_retries: Maximum retry attempts on conflict
            
        Returns:
            Updated ProcessInfo or None if update failed
            
        Example:
            async def set_completed(process):
                process.status = "completed"
                process.completed_at = datetime.now(UTC).isoformat()
            
            updated = await repo.update_with_lock(pid, set_completed)
        """
        for attempt in range(max_retries):
            try:
                key = self._make_key(process_id)
                result_key = self._make_result_key(process_id)
                
                # Start watching the keys
                pipe = self._redis.pipeline()
                pipe.watch(key, result_key)
                
                # Get current process state
                current = await self.get(process_id)
                if not current:
                    pipe.unwatch()
                    return None
                
                # Apply update function
                update_fn(current)
                
                # Prepare metadata
                metadata = {
                    "process_id": current.process_id,
                    "status": current.status,
                    "error": current.error or "",
                    "query": current.query or "",
                    "created_at": current.created_at or "",
                    "completed_at": current.completed_at or "",
                    "current_step": current.current_step or "",
                    "completed_steps": json.dumps(current.completed_steps or [])
                }
                
                # Begin transaction
                pipe.multi()
                
                # Update metadata
                pipe.hset(key, mapping={
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in metadata.items()
                })
                
                # Update result if exists
                if current.result:
                    pipe.set(result_key, pickle.dumps(current.result))
                
                # Update completed tracking
                if current.status in ("completed", "failed") and current.completed_at:
                    timestamp = datetime.fromisoformat(current.completed_at).timestamp()
                    pipe.zadd(self._completed_key, {process_id: timestamp})
                    pipe.expire(key, self._default_ttl)
                    if current.result:
                        pipe.expire(result_key, self._default_ttl)
                
                # Execute transaction
                result = pipe.execute()
                
                # If result is None, transaction was aborted (conflict)
                if result is None:
                    # Retry with exponential backoff
                    await asyncio.sleep(0.01 * (2 ** attempt))
                    continue
                
                # Success - return updated process
                return current
                
            except redis.WatchError:
                # Conflict detected - retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.01 * (2 ** attempt))
                    continue
                else:
                    print(f"Update failed after {max_retries} retries")
                    return None
                    
            except RedisError as e:
                print(f"Redis update error: {e}")
                return None
        
        return None

    async def save(self, process: ProcessInfo) -> None:
        """
        Save process to Redis.
        
        OPERATIONS PERFORMED:
        =====================
        1. HSET process:{id} - Store metadata as hash
        2. SET process:{id}:result - Store result (if exists)
        3. SADD processes:all - Add to set of all processes
        4. ZADD processes:completed - Add to sorted set (if completed)
        5. EXPIRE - Set TTL (if completed/failed)
        
        WHY HASH FOR METADATA?
        ======================
        Hash allows field-level operations:
        - HSET status "completed" (update single field)
        - HGETALL (get all fields)
        - HEXISTS (check field exists)
        
        Alternative: Store as JSON string
        - Pros: Simple, one operation
        - Cons: Must fetch/parse entire object for one field
        
        We chose hash for flexibility and efficiency.
        
        CONCURRENCY NOTE:
        =================
        This method does NOT use locking. Use update_with_lock() for
        read-modify-write operations to prevent race conditions.
        """
        try:
            key = self._make_key(process.process_id)
            result_key = self._make_result_key(process.process_id)
            
            # Prepare metadata (everything except result)
            metadata = {
                "process_id": process.process_id,
                "status": process.status,
                "error": process.error or "",
                "query": process.query or "",
                "created_at": process.created_at or "",
                "completed_at": process.completed_at or "",
                "current_step": process.current_step or "",
                "completed_steps": json.dumps(process.completed_steps or [])
            }
            
            # Use pipeline for atomic operations
            # WHY PIPELINE? All operations succeed or fail together
            with self._redis.pipeline() as pipe:
                # Store metadata as hash
                pipe.hset(key, mapping={
                    k: json.dumps(v) if not isinstance(v, str) else v
                    for k, v in metadata.items()
                })
                
                # Store result separately if exists
                if process.result:
                    # Pickle DecisionState for exact preservation
                    pipe.set(result_key, pickle.dumps(process.result))
                
                # Add to set of all processes
                pipe.sadd(self._all_processes_key, process.process_id)
                
                # If completed/failed, add to sorted set with timestamp
                if process.status in ("completed", "failed") and process.completed_at:
                    timestamp = datetime.fromisoformat(process.completed_at).timestamp()
                    pipe.zadd(self._completed_key, {process.process_id: timestamp})
                    
                    # Set TTL on completed/failed processes
                    pipe.expire(key, self._default_ttl)
                    if process.result:
                        pipe.expire(result_key, self._default_ttl)
                
                # Execute all commands atomically
                pipe.execute()
        
        except RedisError as e:
            # Log error but don't crash
            # In production, use proper logging
            print(f"Redis save error: {e}")
            # Could fallback to in-memory storage here
    
    async def get(self, process_id: str) -> Optional[ProcessInfo]:
        """
        Get process from Redis.
        
        OPERATIONS:
        ===========
        1. HGETALL process:{id} - Get metadata
        2. GET process:{id}:result - Get result
        3. Deserialize and reconstruct ProcessInfo
        
        ERROR HANDLING:
        ===============
        - Key not found: Return None
        - Deserialization error: Return partial data
        - Connection error: Return None
        """
        try:
            key = self._make_key(process_id)
            result_key = self._make_result_key(process_id)
            
            # Get metadata
            metadata = self._redis.hgetall(key)
            if not metadata:
                return None
            
            # Decode bytes to strings
            metadata = {
                k.decode(): v.decode() if isinstance(v, bytes) else v
                for k, v in metadata.items()
            }
            
            # Get result if exists
            result = None
            result_data = self._redis.get(result_key)
            if result_data:
                try:
                    result = pickle.loads(result_data)
                except Exception as e:
                    print(f"Error deserializing result: {e}")
            
            # Reconstruct ProcessInfo
            completed_steps = []
            if "completed_steps" in metadata:
                try:
                    completed_steps = json.loads(metadata["completed_steps"])
                except Exception:
                    pass
            
            return ProcessInfo(
                process_id=metadata.get("process_id", process_id),
                query=metadata.get("query", ""),
                status=metadata.get("status", "unknown"),
                result=result,
                error=metadata.get("error") or None,
                created_at=metadata.get("created_at") or None,
                completed_at=metadata.get("completed_at") or None,
                current_step=metadata.get("current_step") or None,
                completed_steps=completed_steps
            )
        
        except RedisError as e:
            print(f"Redis get error: {e}")
            return None
    
    async def exists(self, process_id: str) -> bool:
        """
        Check if process exists in Redis.
        
        OPTIMIZATION: Use SISMEMBER instead of EXISTS
        ==============================================
        SISMEMBER processes:all process_id is faster than
        EXISTS process:process_id because it checks the set
        without fetching the entire hash.
        
        Complexity: O(1) for both, but SISMEMBER is cheaper
        in memory access.
        """
        try:
            return bool(self._redis.sismember(self._all_processes_key, process_id))
        except RedisError:
            return False
    
    async def delete(self, process_id: str) -> bool:
        """
        Delete process from Redis.
        
        CLEANUP OPERATIONS:
        ===================
        1. DEL process:{id} - Delete metadata
        2. DEL process:{id}:result - Delete result
        3. SREM processes:all - Remove from set
        4. ZREM processes:completed - Remove from sorted set
        
        WHY SO MANY OPERATIONS?
        ========================
        Must clean up all data structures to avoid:
        - Memory leaks
        - Orphaned data
        - Inconsistent state
        
        Returns:
            bool: True if deleted, False if not found or error
        """
        try:
            # First check if exists
            if not await self.exists(process_id):
                return False
                
            key = self._make_key(process_id)
            result_key = self._make_result_key(process_id)
            
            with self._redis.pipeline() as pipe:
                pipe.delete(key)
                pipe.delete(result_key)
                pipe.srem(self._all_processes_key, process_id)
                pipe.zrem(self._completed_key, process_id)
                pipe.execute()
            
            return True
        
        except RedisError as e:
            print(f"Redis delete error: {e}")
            return False
    
    async def list_all(self) -> List[ProcessInfo]:
        """
        List all processes from Redis.
        
        PROCESS:
        ========
        1. SMEMBERS processes:all - Get all process IDs
        2. For each ID, call get() to retrieve full data
        
        PERFORMANCE CONSIDERATION:
        ==========================
        This could be slow with many processes. For production:
        - Add pagination: SSCAN instead of SMEMBERS
        - Add filtering: Separate sets by status
        - Add caching: Cache process list for N seconds
        
        Current implementation is simple but not optimized for scale.
        """
        try:
            # Get all process IDs
            process_ids = self._redis.smembers(self._all_processes_key)
            
            # Fetch each process
            processes = []
            for pid_bytes in process_ids:
                pid = pid_bytes.decode() if isinstance(pid_bytes, bytes) else pid_bytes
                process = await self.get(pid)
                if process:
                    processes.append(process)
            
            return processes
        
        except RedisError as e:
            print(f"Redis list_all error: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, int]:
        """
        Get process statistics from Redis.
        
        OPTIMIZATION: Count without loading data
        =========================================
        Instead of fetching all processes, we:
        1. Count total: SCARD processes:all
        2. Count completed: ZCARD processes:completed
        3. For running/failed, we need to fetch (no separate tracking)
        
        TRADE-OFF:
        ==========
        Could maintain separate sets for each status:
        - processes:running
        - processes:completed
        - processes:failed
        
        Pros: O(1) counts for all statuses
        Cons: More operations on status changes
        
        Current approach is simpler.
        """
        try:
            processes = await self.list_all()
            return {
                "total": len(processes),
                "pending": sum(1 for p in processes if p.status == "pending"),
                "running": sum(1 for p in processes if p.status == "running"),
                "completed": sum(1 for p in processes if p.status == "completed"),
                "failed": sum(1 for p in processes if p.status == "failed"),
            }
        
        except RedisError as e:
            print(f"Redis get_stats error: {e}")
            return {"total": 0, "running": 0, "completed": 0, "failed": 0}
    
    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        """
        Clean up old completed/failed processes from Redis.
        
        PROCESS:
        ========
        1. Calculate cutoff timestamp
        2. ZRANGEBYSCORE processes:completed -inf cutoff - Get old process IDs
        3. For each, call delete()
        
        WHY SORTED SET?
        ===============
        Sorted sets are perfect for time-based queries:
        - Score = timestamp
        - ZRANGEBYSCORE finds all processes completed before cutoff
        - Efficient: O(log N + M) where M is results
        
        ALTERNATIVE: Store completion time in hash, scan all processes
        - Would be O(N) to scan all processes
        - Sorted set makes this O(log N + M) where M is old processes
        
        This is why we use multiple data structures - each optimized
        for specific queries.
        """
        try:
            cutoff_time = datetime.now(UTC).timestamp() - (older_than_hours * 3600)
            
            # Find processes completed before cutoff
            old_processes = self._redis.zrangebyscore(
                self._completed_key,
                '-inf',
                cutoff_time
            )
            
            # Delete each old process
            count = 0
            for pid_bytes in old_processes:
                pid = pid_bytes.decode() if isinstance(pid_bytes, bytes) else pid_bytes
                await self.delete(pid)
                count += 1
            
            return count
        
        except RedisError as e:
            print(f"Redis cleanup error: {e}")
            return 0


def get_process_repository(use_redis: bool = None) -> IProcessRepository:
    """
    Factory function to create appropriate repository.
    
    FACTORY PATTERN:
    ================
    Creates objects without specifying exact class.
    Benefits:
    - Centralized creation logic
    - Easy to switch implementations
    - Can add configuration logic
    - Supports dependency injection
    
    Args:
        use_redis: True for Redis, False for in-memory, None for auto-detect
    
    Returns:
        IProcessRepository: Repository instance
    
    DESIGN DECISION: Auto-detect based on settings
    ===============================================
    If use_redis is None, check environment:
    - If Redis configured and reachable: Use Redis
    - Otherwise: Fall back to in-memory
    
    This enables:
    - Development: Works without Redis
    - Production: Automatically uses Redis if available
    - Testing: Explicit control via parameter
    
    Example:
        # Auto-detect
        repo = get_process_repository()
        
        # Force in-memory (testing)
        repo = get_process_repository(use_redis=False)
        
        # Force Redis (production)
        repo = get_process_repository(use_redis=True)
    """
    if use_redis is None:
        # Auto-detect: Try Redis, fall back to in-memory
        settings = get_settings()
        if settings.enable_redis_persistence:
            try:
                # Test Redis connection
                client = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=settings.redis_db,
                    socket_connect_timeout=2
                )
                client.ping()  # Test connection
                return RedisProcessRepository(redis_client=client)
            except Exception as e:
                print(f"Redis not available, using in-memory: {e}")
                return InMemoryProcessRepository()
        else:
            return InMemoryProcessRepository()
    
    elif use_redis:
        return RedisProcessRepository()
    else:
        return InMemoryProcessRepository()
