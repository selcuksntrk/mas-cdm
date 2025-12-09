"""Helpers for running graphs in tests."""

from __future__ import annotations

from pydantic_graph import Graph


def run_graph_sync(graph: Graph, start_node, state):
    """Convenience wrapper for synchronous tests (runs event loop internally)."""
    import asyncio

    return asyncio.get_event_loop().run_until_complete(graph.run(start_node, state=state))


async def run_graph_async(graph: Graph, start_node, state):
    """Async wrapper to execute a graph from a given start node."""
    return await graph.run(start_node, state=state)


__all__ = ["run_graph_sync", "run_graph_async"]
