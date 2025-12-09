"""
Graph builder utilities for dynamic workflows.
Validates node sets and edges for cycles and reachability.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Iterable, List, Set, Tuple

from backend.app.core.graph.nodes import GetDecision
from backend.app.core.graph.executor import GRAPH_NODES


class GraphValidationError(ValueError):
    """Raised when a graph specification is invalid."""


class GraphBuilder:
    def __init__(self):
        self.available_nodes: Dict[str, type] = {cls.__name__: cls for cls in GRAPH_NODES}

    def validate_nodes(self, node_names: Iterable[str]) -> List[str]:
        names = list(node_names)
        unknown = [n for n in names if n not in self.available_nodes]
        if unknown:
            raise GraphValidationError(f"Unknown nodes: {unknown}")
        if GetDecision.__name__ not in names:
            raise GraphValidationError("Graph must include entry node GetDecision")
        return names

    def validate_edges(self, nodes: List[str], edges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
        node_set = set(nodes)
        edge_list = []
        for src, dst in edges:
            if src not in node_set or dst not in node_set:
                raise GraphValidationError(f"Edge references unknown node: {(src, dst)}")
            edge_list.append((src, dst))
        return edge_list

    def ensure_no_cycles(self, nodes: List[str], edges: List[Tuple[str, str]]) -> None:
        indeg = defaultdict(int)
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for src, dst in edges:
            if dst not in adj[src]:
                adj[src].add(dst)
                indeg[dst] += 1
        queue = deque([n for n in nodes if indeg[n] == 0])
        visited = 0
        while queue:
            cur = queue.popleft()
            visited += 1
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if visited != len(nodes):
            raise GraphValidationError("Graph contains a cycle")

    def ensure_reachable(self, nodes: List[str], edges: List[Tuple[str, str]], start: str = GetDecision.__name__) -> None:
        adj: Dict[str, Set[str]] = {n: set() for n in nodes}
        for src, dst in edges:
            adj[src].add(dst)
        seen: Set[str] = set()
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj[cur])
        unreachable = [n for n in nodes if n not in seen]
        if unreachable:
            raise GraphValidationError(f"Unreachable nodes: {unreachable}")

    def build(self, node_names: List[str], edges: List[Tuple[str, str]] | None = None) -> Dict[str, object]:
        nodes = self.validate_nodes(node_names)
        edges = self.validate_edges(nodes, edges or self._default_linear_edges(nodes))
        self.ensure_no_cycles(nodes, edges)
        self.ensure_reachable(nodes, edges)
        return {
            "nodes": nodes,
            "edges": edges,
            "start": GetDecision.__name__,
        }

    @staticmethod
    def _default_linear_edges(nodes: List[str]) -> List[Tuple[str, str]]:
        return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


graph_builder = GraphBuilder()

__all__ = ["GraphBuilder", "GraphValidationError", "graph_builder"]
