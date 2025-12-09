"""
Web Search Tool

Provides web search capabilities for agents.
Currently returns mock data - integrate with a real search API in production.
"""

import time
from typing import Any, Dict, List, Tuple
from .base import Tool, ToolParameter, ToolError


class WebSearchTool(Tool):
    """
    A web search tool for finding information online.
    
    Note: This is a mock implementation. In production, integrate with:
    - Google Custom Search API
    - Bing Search API
    - DuckDuckGo API
    - Tavily Search API
    """

    def __init__(self):
        super().__init__()
        self._cache: Dict[Tuple[str, int], Tuple[float, Any]] = {}
        # Defaults; can be tuned via config if needed in future.
        self._cache_ttl_seconds = 300
        self._cache_max_entries = 50
    
    def get_name(self) -> str:
        return "web_search"
    
    def get_description(self) -> str:
        return "Search the web for information. Returns top search results with titles, URLs, and snippets."
    
    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="query",
                type="string",
                description="Search query string",
                required=True
            ),
            ToolParameter(
                name="max_results",
                type="number",
                description="Maximum number of results to return (1-10)",
                required=False,
                default=5
            )
        ]
    
    def validate_input(self, **kwargs) -> bool:
        """Additional validation for web search."""
        super().validate_input(**kwargs)
        
        query = kwargs.get("query", "")
        if len(query) < 2:
            raise ToolError("Query must be at least 2 characters", self.name)
        
        max_results = kwargs.get("max_results", 5)
        if not isinstance(max_results, int) or max_results < 1 or max_results > 10:
            raise ToolError("max_results must be between 1 and 10", self.name)
        
        return True
    
    def is_safe(self, **kwargs) -> bool:
        """
        Safety check for web search.
        
        Block searches for:
        - Extremely long queries (potential DoS)
        - Queries with suspicious patterns
        """
        query = kwargs.get("query", "")
        
        # Block excessively long queries
        if len(query) > 500:
            return False
        
        return True
    
    async def execute(self, query: str, max_results: int = 5) -> Any:
        """
        Execute web search.
        
        This is a MOCK implementation. Replace with actual search API.
        """
        cache_key = (query, max_results)
        now = time.monotonic()

        # Return cached result when fresh
        cached = self._cache.get(cache_key)
        if cached and (now - cached[0] <= self._cache_ttl_seconds):
            return {
                **cached[1],
                "cached": True,
            }

        # Mock search results
        mock_results = [
            {
                "title": f"Result for '{query}' - Example Site",
                "url": f"https://example.com/search?q={query.replace(' ', '+')}",
                "snippet": f"This is a mock search result for '{query}'. In production, this would return real search results from a search API.",
                "relevance_score": 0.95
            },
            {
                "title": f"Documentation: {query}",
                "url": f"https://docs.example.com/{query.replace(' ', '-')}",
                "snippet": f"Official documentation and guides related to {query}. Learn more about best practices and implementation details.",
                "relevance_score": 0.88
            },
            {
                "title": f"Forum Discussion: {query}",
                "url": f"https://forum.example.com/topic/{query.replace(' ', '-')}",
                "snippet": f"Community discussion about {query}. Users share their experiences and solutions.",
                "relevance_score": 0.75
            }
        ]
        
        # Limit results
        results = mock_results[:max_results]
        
        response = {
            "query": query,
            "num_results": len(results),
            "results": results,
            "note": "MOCK DATA - Integrate with real search API in production (Google, Bing, Tavily, etc.)",
            "cached": False,
        }

        # Cache the response
        if len(self._cache) >= self._cache_max_entries:
            # Drop oldest entry
            oldest_key = min(self._cache.items(), key=lambda item: item[1][0])[0]
            self._cache.pop(oldest_key, None)
        self._cache[cache_key] = (now, response)

        return response
