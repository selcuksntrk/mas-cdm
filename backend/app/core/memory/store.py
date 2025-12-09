"""
Simple in-memory vector-ish store for agent memory.
Uses term-frequency vectors with cosine similarity for lightweight semantic-ish matching
without external dependencies.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple


_TOKEN_RE = re.compile(r"\w+")


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
    def __init__(self):
        self._docs: List[MemoryDocument] = []

    def clear(self) -> None:
        self._docs.clear()

    def add_document(self, content: str, metadata: Dict[str, str] | None = None) -> None:
        doc = MemoryDocument(
            content=content,
            metadata=metadata or {},
            embedding=_embed(content),
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


memory_store = MemoryStore()

__all__ = ["MemoryStore", "MemoryDocument", "memory_store"]
