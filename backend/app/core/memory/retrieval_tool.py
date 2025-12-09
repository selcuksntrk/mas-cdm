"""
Retrieval tool for agents to fetch relevant context from the in-memory store.
"""

from __future__ import annotations

from typing import Any, List

from backend.app.core.memory.store import memory_store
from backend.app.core.tools.base import Tool, ToolParameter


class RetrievalTool(Tool):
    def get_name(self) -> str:
        return "retrieve_memory"

    def get_description(self) -> str:
        return "Retrieve relevant context snippets from the agent memory store."

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query for retrieval",
                required=True,
            ),
            ToolParameter(
                name="top_k",
                type="number",
                description="Maximum results to return (1-5)",
                required=False,
                default=3,
            ),
        ]

    def validate_input(self, **kwargs) -> bool:
        super().validate_input(**kwargs)
        top_k = kwargs.get("top_k", 3)
        if not isinstance(top_k, int) or top_k < 1 or top_k > 5:
            raise ValueError("top_k must be an integer between 1 and 5")
        return True

    async def execute(self, query: str, top_k: int = 3) -> Any:
        results = memory_store.search(query, top_k=top_k)
        formatted = [
            {
                "content": doc.content,
                "metadata": doc.metadata,
                "score": float(score),
            }
            for doc, score in results
        ]
        return {
            "query": query,
            "results": formatted,
            "num_results": len(formatted),
        }


__all__ = ["RetrievalTool"]
