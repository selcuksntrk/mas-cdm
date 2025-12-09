"""Repository factory supporting multiple backends."""

from __future__ import annotations

from backend.app.config import get_settings
from backend.app.services.persistence.base import IProcessRepository
from backend.app.services.persistence.postgres_repository import PostgresProcessRepository
from backend.app.services.redis_repository import (
    InMemoryProcessRepository,
    RedisProcessRepository,
    get_process_repository as legacy_get_process_repository,
)


BACKENDS = {
    "memory": InMemoryProcessRepository,
    "redis": RedisProcessRepository,
    "postgres": PostgresProcessRepository,
}


def get_process_repository(backend: str | None = None) -> IProcessRepository:
    """Return repository instance for requested backend.

    If backend is None, derive from settings: persistence_backend or
    enable_redis_persistence flag (fallback to memory).
    """
    settings = get_settings()
    selected = backend or getattr(settings, "persistence_backend", None)

    if selected is None:
        selected = "redis" if settings.enable_redis_persistence else "memory"

    selected = selected.lower()
    if selected not in BACKENDS:
        # Fall back to legacy auto-detect when unknown
        return legacy_get_process_repository(None)

    if selected == "redis":
        # Let redis repo handle connection test
        return legacy_get_process_repository(True)

    repo_cls = BACKENDS[selected]
    return repo_cls()


__all__ = ["get_process_repository", "BACKENDS"]
