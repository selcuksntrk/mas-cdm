import pytest

from backend.app.core.memory.store import memory_store
from backend.app.core.memory.retrieval_tool import RetrievalTool


@pytest.fixture(autouse=True)
def _clear_store():
    memory_store.clear()
    yield
    memory_store.clear()


def test_memory_store_add_and_search():
    memory_store.add_document("Redis is an in-memory database", {"source": "doc1"})
    memory_store.add_document("PostgreSQL is a relational database", {"source": "doc2"})

    results = memory_store.search("in-memory database", top_k=2)

    assert results
    contents = [doc.content for doc, _ in results]
    assert "Redis is an in-memory database" in contents


@pytest.mark.asyncio
async def test_retrieval_tool_returns_results():
    memory_store.add_document("Vector search retrieves relevant context", {"tag": "search"})
    tool = RetrievalTool()

    response = await tool.run(query="vector search", top_k=3)

    assert response.success is True
    assert response.output["num_results"] >= 1
    assert any("context" in r["content"] for r in response.output["results"])


@pytest.mark.asyncio
async def test_retrieval_tool_respects_top_k():
    memory_store.add_document("Doc one about testing", {"id": "1"})
    memory_store.add_document("Doc two about testing", {"id": "2"})
    memory_store.add_document("Doc three about testing", {"id": "3"})

    tool = RetrievalTool()
    response = await tool.run(query="testing", top_k=2)

    assert response.output["num_results"] == 2
