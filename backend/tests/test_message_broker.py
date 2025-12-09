import asyncio
import pytest

from backend.app.core.communication.message_broker import MessageBroker


@pytest.mark.asyncio
async def test_send_message_reaches_subscriber():
    broker = MessageBroker()
    queue = await broker.subscribe("agent_b", "info")

    msg = await broker.send_message("agent_a", "agent_b", "info", {"x": 1})
    received = await asyncio.wait_for(queue.get(), timeout=0.1)

    assert received.message_id == msg.message_id
    assert received.payload == {"x": 1}
    assert received.to_agent == "agent_b"


@pytest.mark.asyncio
async def test_broadcast_reaches_multiple_subscribers():
    broker = MessageBroker()
    q1 = await broker.subscribe("agent_b", "notice")
    q2 = await broker.subscribe("agent_c", "notice")

    msg = await broker.broadcast("agent_a", "notice", {"hello": True})

    got1 = await asyncio.wait_for(q1.get(), timeout=0.1)
    got2 = await asyncio.wait_for(q2.get(), timeout=0.1)

    assert got1.message_id == msg.message_id
    assert got2.message_id == msg.message_id
    assert got1.to_agent is None
    assert got2.to_agent is None


@pytest.mark.asyncio
async def test_unsubscribe_stops_delivery():
    broker = MessageBroker()
    queue = await broker.subscribe("agent_b", "info")
    await broker.unsubscribe("agent_b", "info")

    await broker.send_message("agent_a", "agent_b", "info", {})

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(queue.get(), timeout=0.05)


@pytest.mark.asyncio
async def test_message_order_preserved_per_subscriber():
    broker = MessageBroker()
    queue = await broker.subscribe("agent_b", "info")

    await broker.send_message("agent_a", "agent_b", "info", {"idx": 1})
    await broker.send_message("agent_a", "agent_b", "info", {"idx": 2})

    first = await asyncio.wait_for(queue.get(), timeout=0.1)
    second = await asyncio.wait_for(queue.get(), timeout=0.1)

    assert first.payload["idx"] == 1
    assert second.payload["idx"] == 2
