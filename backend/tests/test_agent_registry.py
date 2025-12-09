import pytest

from backend.app.core.agents.registry import AgentRegistry
from backend.app.models.domain import AgentMetadata


class DummyAgent:
    pass


def _meta(name: str) -> AgentMetadata:
    return AgentMetadata(
        name=name,
        role="test",
        description="dummy",
        model="test-model",
        tools=["calculator"],
    )


def test_register_and_get():
    registry = AgentRegistry()
    agent = DummyAgent()
    registry.register("dummy", agent, _meta("dummy"))

    assert registry.get("dummy") is agent
    md = registry.get_metadata("dummy")
    assert md is not None
    assert md.name == "dummy"


def test_duplicate_registration_raises():
    registry = AgentRegistry()
    agent = DummyAgent()
    registry.register("dummy", agent, _meta("dummy"))

    with pytest.raises(ValueError):
        registry.register("dummy", DummyAgent(), _meta("dummy"))


def test_listings():
    registry = AgentRegistry()
    registry.register("a", DummyAgent(), _meta("a"))
    registry.register("b", DummyAgent(), _meta("b"))

    names = registry.list_agents()
    assert set(names) == {"a", "b"}

    metadata = registry.list_metadata()
    assert len(metadata) == 2
    assert {m.name for m in metadata} == {"a", "b"}
