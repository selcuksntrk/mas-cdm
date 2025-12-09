"""Simple in-memory message broker for inter-agent communication."""

from __future__ import annotations

import asyncio
from uuid import uuid4
from typing import Dict, Tuple

from backend.app.models.domain import AgentMessage


class MessageBroker:
    def __init__(self) -> None:
        # Mapping agent_id -> message_type -> asyncio.Queue[AgentMessage]
        self._subscribers: Dict[str, Dict[str, asyncio.Queue[AgentMessage]]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, agent_id: str, message_type: str) -> asyncio.Queue[AgentMessage]:
        async with self._lock:
            agent_subs = self._subscribers.setdefault(agent_id, {})
            if message_type not in agent_subs:
                agent_subs[message_type] = asyncio.Queue()
            return agent_subs[message_type]

    async def unsubscribe(self, agent_id: str, message_type: str) -> None:
        async with self._lock:
            agent_subs = self._subscribers.get(agent_id)
            if not agent_subs:
                return
            agent_subs.pop(message_type, None)
            if not agent_subs:
                self._subscribers.pop(agent_id, None)

    async def send_message(self, from_agent: str, to_agent: str, message_type: str, payload: dict) -> AgentMessage:
        msg = self._build_message(from_agent, to_agent, message_type, payload)
        await self._dispatch(msg)
        return msg

    async def broadcast(self, from_agent: str, message_type: str, payload: dict) -> AgentMessage:
        msg = self._build_message(from_agent, None, message_type, payload)
        await self._dispatch(msg)
        return msg

    def _build_message(self, from_agent: str, to_agent: str | None, message_type: str, payload: dict) -> AgentMessage:
        return AgentMessage(
            message_id=str(uuid4()),
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            payload=payload,
        )

    async def _dispatch(self, msg: AgentMessage) -> None:
        async with self._lock:
            if msg.to_agent:
                queues = [self._subscribers.get(msg.to_agent, {}).get(msg.message_type)]
            else:
                queues = [subs.get(msg.message_type) for subs in self._subscribers.values()]

        for queue in queues:
            if queue:
                await queue.put(msg)

    def snapshot_subscribers(self) -> Dict[str, Tuple[str, ...]]:
        """Return subscriber snapshot for testing or inspection."""
        return {aid: tuple(subs.keys()) for aid, subs in self._subscribers.items()}


message_broker = MessageBroker()

__all__ = ["MessageBroker", "message_broker"]
