"""
Agent lifecycle management utilities.

Provides pause/resume/terminate semantics and concurrency checks
for registered agents.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Dict, Optional

from backend.app.config import Settings, get_settings
from backend.app.core.agents.registry import AgentRegistry, agent_registry
from backend.app.core.exceptions import ConcurrencyError


class AgentStatus(str, Enum):
    UNINITIALIZED = "uninitialized"
    ACTIVE = "active"
    PAUSED = "paused"
    TERMINATED = "terminated"
    TIMEOUT = "timeout"


class AgentLifecycle:
    def __init__(self, registry: AgentRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings
        self._statuses: Dict[str, AgentStatus] = {}
        self._start_times: Dict[str, float] = {}

    def _check_agent_exists(self, agent_id: str) -> None:
        if self.registry.get(agent_id) is None:
            raise ValueError(f"Agent '{agent_id}' is not registered")

    def _active_count(self) -> int:
        return sum(1 for status in self._statuses.values() if status == AgentStatus.ACTIVE)

    def _ensure_capacity(self) -> None:
        if self._active_count() >= self.settings.max_concurrent_agents:
            raise ConcurrencyError("Max concurrent agents reached")

    def _enforce_timeout(self, agent_id: str) -> None:
        status = self._statuses.get(agent_id)
        if status != AgentStatus.ACTIVE:
            return
        started_at = self._start_times.get(agent_id)
        if started_at is None:
            return
        if time.monotonic() - started_at >= self.settings.agent_timeout:
            self._statuses[agent_id] = AgentStatus.TIMEOUT
            self._start_times.pop(agent_id, None)

    async def initialize_agent(self, agent_id: str, agent_config: Optional[dict] = None):
        self._check_agent_exists(agent_id)
        self._enforce_timeout(agent_id)

        # If already active, return existing agent
        current_status = self._statuses.get(agent_id)
        if current_status == AgentStatus.ACTIVE:
            return self.registry.get(agent_id)

        self._ensure_capacity()
        agent = self.registry.get(agent_id)

        # Apply dynamic config if provided
        if agent_config:
            model_override = agent_config.get("model")
            if model_override:
                self.settings.agent_model_mapping[agent_id] = model_override

            temperature_override = agent_config.get("temperature")
            if temperature_override is not None:
                self.settings.agent_temperature_mapping[agent_id] = float(temperature_override)

        self._statuses[agent_id] = AgentStatus.ACTIVE
        self._start_times[agent_id] = time.monotonic()
        return agent

    async def pause_agent(self, agent_id: str) -> AgentStatus:
        self._check_agent_exists(agent_id)
        self._enforce_timeout(agent_id)

        if self._statuses.get(agent_id) in {AgentStatus.TERMINATED, AgentStatus.TIMEOUT}:
            raise ValueError(f"Agent '{agent_id}' is terminated")

        self._statuses[agent_id] = AgentStatus.PAUSED
        self._start_times.pop(agent_id, None)
        return AgentStatus.PAUSED

    async def resume_agent(self, agent_id: str) -> AgentStatus:
        self._check_agent_exists(agent_id)
        self._enforce_timeout(agent_id)

        if self._statuses.get(agent_id) in {AgentStatus.TERMINATED, AgentStatus.TIMEOUT}:
            raise ValueError(f"Agent '{agent_id}' is terminated")

        self._ensure_capacity()
        self._statuses[agent_id] = AgentStatus.ACTIVE
        self._start_times[agent_id] = time.monotonic()
        return AgentStatus.ACTIVE

    async def terminate_agent(self, agent_id: str) -> AgentStatus:
        self._check_agent_exists(agent_id)
        self._statuses[agent_id] = AgentStatus.TERMINATED
        self._start_times.pop(agent_id, None)
        return AgentStatus.TERMINATED

    def get_agent_status(self, agent_id: str) -> AgentStatus:
        self._check_agent_exists(agent_id)
        self._enforce_timeout(agent_id)
        return self._statuses.get(agent_id, AgentStatus.UNINITIALIZED)

    def update_agent_config(self, agent_id: str, model: Optional[str] = None, temperature: Optional[float] = None) -> None:
        self._check_agent_exists(agent_id)
        if model:
            self.settings.agent_model_mapping[agent_id] = model
        if temperature is not None:
            self.settings.agent_temperature_mapping[agent_id] = float(temperature)

    def reset(self) -> None:
        """Reset lifecycle state (primarily for testing)."""
        self._statuses.clear()
        self._start_times.clear()


lifecycle_manager = AgentLifecycle(agent_registry, get_settings())

__all__ = ["AgentLifecycle", "AgentStatus", "lifecycle_manager"]
