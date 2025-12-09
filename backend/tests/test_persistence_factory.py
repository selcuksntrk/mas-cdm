import pytest
import fakeredis

from backend.app.services.persistence.factory import get_process_repository
from backend.app.services.redis_repository import InMemoryProcessRepository, RedisProcessRepository
from backend.app.services.persistence.postgres_repository import PostgresProcessRepository


def test_factory_defaults_to_memory_when_no_redis():
    repo = get_process_repository(backend="memory")
    assert isinstance(repo, InMemoryProcessRepository)


def test_factory_selects_postgres_stub():
    repo = get_process_repository(backend="postgres")
    assert isinstance(repo, PostgresProcessRepository)


def test_unknown_backend_falls_back_to_memory():
    repo = get_process_repository(backend="unknown")
    assert isinstance(repo, InMemoryProcessRepository)


def test_redis_backend_uses_fakeredis(monkeypatch):
    class DummySettings:
        enable_redis_persistence = True
        redis_host = "localhost"
        redis_port = 6379
        redis_db = 0

    from backend.app.services import redis_repository as rr

    monkeypatch.setattr(rr, "get_settings", lambda: DummySettings())
    monkeypatch.setattr(rr.redis, "Redis", fakeredis.FakeRedis)

    repo = get_process_repository(backend="redis")
    assert isinstance(repo, RedisProcessRepository)
    assert isinstance(repo._redis, fakeredis.FakeRedis)
    assert repo._redis.ping()
