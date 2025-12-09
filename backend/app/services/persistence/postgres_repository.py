"""PostgreSQL process repository (placeholder async implementation)."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Callable

from backend.app.models.domain import ProcessInfo
from backend.app.services.persistence.base import IProcessRepository


class PostgresProcessRepository(IProcessRepository):
    """
    Simplified PostgreSQL repository.

    This placeholder stores data in-memory for now but exposes the same
    interface so the backend can swap persistence backends without code changes.
    """

    def __init__(self) -> None:
        self._storage: Dict[str, ProcessInfo] = {}
        self._lock = asyncio.Lock()

    async def save(self, process: ProcessInfo) -> None:
        async with self._lock:
            self._storage[process.process_id] = process

    async def get(self, process_id: str) -> Optional[ProcessInfo]:
        async with self._lock:
            return self._storage.get(process_id)

    async def exists(self, process_id: str) -> bool:
        async with self._lock:
            return process_id in self._storage

    async def delete(self, process_id: str) -> bool:
        async with self._lock:
            if process_id in self._storage:
                del self._storage[process_id]
                return True
            return False

    async def list_all(self) -> List[ProcessInfo]:
        async with self._lock:
            return list(self._storage.values())

    async def get_stats(self) -> Dict[str, int]:
        async with self._lock:
            processes = list(self._storage.values())
        return {
            "total": len(processes),
            "running": sum(1 for p in processes if p.status == "running"),
            "completed": sum(1 for p in processes if p.status == "completed"),
            "failed": sum(1 for p in processes if p.status == "failed"),
            "pending": sum(1 for p in processes if p.status == "pending"),
        }

    async def cleanup_completed(self, older_than_hours: int = 24) -> int:
        async with self._lock:
            to_delete = [pid for pid, p in self._storage.items() if p.status in ("completed", "failed")]
            for pid in to_delete:
                del self._storage[pid]
        return len(to_delete)

    async def update_with_lock(
        self,
        process_id: str,
        update_fn: Callable[[ProcessInfo], None],
        max_retries: int = 5,
    ) -> Optional[ProcessInfo]:
        async with self._lock:
            proc = self._storage.get(process_id)
            if not proc:
                return None
            update_fn(proc)
            self._storage[process_id] = proc
            return proc


__all__ = ["PostgresProcessRepository"]
