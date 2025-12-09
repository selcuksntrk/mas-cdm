"""Persistence repository interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Protocol, Callable

from backend.app.models.domain import ProcessInfo


class IProcessRepository(ABC):
    """Contract for process repositories."""

    @abstractmethod
    async def save(self, process: ProcessInfo) -> None: ...

    @abstractmethod
    async def get(self, process_id: str) -> Optional[ProcessInfo]: ...

    @abstractmethod
    async def exists(self, process_id: str) -> bool: ...

    @abstractmethod
    async def delete(self, process_id: str) -> bool: ...

    @abstractmethod
    async def list_all(self) -> List[ProcessInfo]: ...

    @abstractmethod
    async def get_stats(self) -> Dict[str, int]: ...

    @abstractmethod
    async def cleanup_completed(self, older_than_hours: int = 24) -> int: ...

    @abstractmethod
    async def update_with_lock(
        self,
        process_id: str,
        update_fn: Callable[[ProcessInfo], None],
        max_retries: int = 5,
    ) -> Optional[ProcessInfo]: ...


class PersistenceFactory(Protocol):
    def __call__(self, backend: str | None = None): ...


__all__ = ["IProcessRepository", "PersistenceFactory"]
