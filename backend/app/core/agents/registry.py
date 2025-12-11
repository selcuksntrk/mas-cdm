"""
Dynamic agent registry for discovery and metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic_ai import Agent

from backend.app.models.domain import AgentMetadata


class AgentRegistry:
    """
    Registry for managing agent instances and their metadata.
    
    Provides:
    - Agent registration with metadata
    - Agent lookup by name
    - List of all registered agents
    """
    
    def __init__(self) -> None:
        self._agents: Dict[str, Agent[Any, Any]] = {}
        self._metadata: Dict[str, AgentMetadata] = {}

    def register(self, name: str, agent: Agent[Any, Any], metadata: AgentMetadata) -> None:
        """Register an agent with its metadata. Raises ValueError if already registered."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already registered")
        self._agents[name] = agent
        self._metadata[name] = metadata

    def get(self, name: str) -> Agent[Any, Any] | None:
        """Get an agent by name, returns None if not found."""
        return self._agents.get(name)

    def get_metadata(self, name: str) -> AgentMetadata | None:
        """Get agent metadata by name, returns None if not found."""
        return self._metadata.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names."""
        return list(self._agents.keys())

    def list_metadata(self) -> List[AgentMetadata]:
        """List metadata for all registered agents."""
        return list(self._metadata.values())


agent_registry = AgentRegistry()

__all__ = ["AgentRegistry", "agent_registry"]
