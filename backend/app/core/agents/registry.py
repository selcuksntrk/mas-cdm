"""
Dynamic agent registry for discovery and metadata.
"""

from __future__ import annotations

from typing import Dict, List

from backend.app.models.domain import AgentMetadata


class AgentRegistry:
    def __init__(self):
        self._agents: Dict[str, object] = {}
        self._metadata: Dict[str, AgentMetadata] = {}

    def register(self, name: str, agent: object, metadata: AgentMetadata) -> None:
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent
        self._metadata[name] = metadata

    def get(self, name: str) -> object | None:
        return self._agents.get(name)

    def get_metadata(self, name: str) -> AgentMetadata | None:
        return self._metadata.get(name)

    def list_agents(self) -> List[str]:
        return list(self._agents.keys())

    def list_metadata(self) -> List[AgentMetadata]:
        return list(self._metadata.values())


agent_registry = AgentRegistry()

__all__ = ["AgentRegistry", "agent_registry"]
