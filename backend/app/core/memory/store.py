"""
Simple in-memory vector-ish store for agent memory.
Uses term-frequency vectors with cosine similarity for lightweight semantic-ish matching
without external dependencies.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+")

# Default limits to prevent unbounded memory growth
DEFAULT_MAX_DOCUMENTS = 1000
DEFAULT_MAX_CONTENT_LENGTH = 10000  # characters per document


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _embed(text: str) -> Counter:
    return Counter(_tokenize(text))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a.keys()) & set(b.keys())
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class MemoryDocument:
    content: str
    metadata: Dict[str, str] = field(default_factory=dict)
    embedding: Counter = field(default_factory=Counter)


class MemoryStore:
    """
    In-memory document store with size limits and LRU-style eviction.
    
    Args:
        max_documents: Maximum number of documents to store. When exceeded,
                      oldest documents are evicted (FIFO).
        max_content_length: Maximum length of content per document. Longer
                           content is truncated with a warning.
    """
    
    def __init__(
        self,
        max_documents: int = DEFAULT_MAX_DOCUMENTS,
        max_content_length: int = DEFAULT_MAX_CONTENT_LENGTH,
    ):
        self._docs: List[MemoryDocument] = []
        self._max_documents = max_documents
        self._max_content_length = max_content_length
        self._eviction_count = 0

    def clear(self) -> None:
        self._docs.clear()
        self._eviction_count = 0

    def add_document(self, content: str, metadata: Dict[str, str] | None = None) -> None:
        # Truncate content if too long
        if len(content) > self._max_content_length:
            logger.warning(
                "Document content truncated from %d to %d characters",
                len(content),
                self._max_content_length,
            )
            content = content[:self._max_content_length]
        
        doc = MemoryDocument(
            content=content,
            metadata=metadata or {},
            embedding=_embed(content),
        )
        
        # Evict oldest documents if at capacity (FIFO eviction)
        while len(self._docs) >= self._max_documents:
            evicted = self._docs.pop(0)
            self._eviction_count += 1
            logger.debug(
                "Evicted document (total evictions: %d): %s...",
                self._eviction_count,
                evicted.content[:50],
            )
        
        self._docs.append(doc)

    def add_documents(self, contents: Iterable[Tuple[str, Dict[str, str] | None]]) -> None:
        for content, meta in contents:
            self.add_document(content, meta)

    def search(self, query: str, top_k: int = 3) -> List[Tuple[MemoryDocument, float]]:
        if not self._docs:
            return []
        query_vec = _embed(query)
        scored = [
            (doc, _cosine(query_vec, doc.embedding))
            for doc in self._docs
        ]
        scored = [pair for pair in scored if pair[1] > 0.0]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_k]
    
    @property
    def document_count(self) -> int:
        """Return the current number of documents in the store."""
        return len(self._docs)
    
    @property
    def eviction_count(self) -> int:
        """Return the total number of documents evicted due to capacity limits."""
        return self._eviction_count


memory_store = MemoryStore()

__all__ = ["MemoryStore", "MemoryDocument", "memory_store"]
