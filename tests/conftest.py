"""
conftest.py — Shared pytest fixtures.

Uses FastAPI's dependency_overrides to swap real (slow, external)
dependencies — the RAG pipeline, vectorstore, embeddings — for
lightweight fakes, so the suite runs fast with no live Pinecone/Groq
credentials required.

IMPORTANT: fake_pipeline()'s return dict and the fake modules below
assume the same shapes used in app/routers/*.py (e.g. Source(**s)).
If your real app/models.py has different field names on Source /
QueryResponse / IngestResponse etc., adjust the fakes accordingly —
these were written against api.py's usage, not the model file itself.
"""

import types

import pytest
from httpx import ASGITransport, AsyncClient

from app.api import app
from app.config import Settings, get_settings
from app.dependencies import (
    get_bm25_module,
    get_embeddings_module,
    get_ingest_module,
    get_pipeline_fn,
    get_vectorstore_module,
)

TEST_API_KEY = "test-key-123"


def fake_pipeline(question: str) -> dict:
    return {
        "answer": f"Fake answer for: {question}",
        "sources": [{"source": "fake.pdf", "page": 1, "text": "fake chunk", "score": 0.9}],
        "phase": "test",
    }


def get_test_settings() -> Settings:
    return Settings(api_key=TEST_API_KEY, groq_api_key="fake-groq-key")


class _FakeIndex:
    def describe_index_stats(self):
        return {"total_vector_count": 1}

    def query(self, **kwargs):
        return {"matches": [{"metadata": {"source": "fake.pdf"}}]}

    def delete(self, **kwargs):
        return None


@pytest.fixture
def base_overrides():
    """Overrides needed by nearly every test: settings + pipeline."""
    app.dependency_overrides[get_settings] = get_test_settings
    app.dependency_overrides[get_pipeline_fn] = lambda: fake_pipeline
    app.dependency_overrides[get_vectorstore_module] = lambda: types.SimpleNamespace(
        collection_count=lambda: 1,
        _get_index=lambda: _FakeIndex(),
        EMBEDDING_DIM=384,
        add_documents=lambda chunks, embeddings: None,
    )
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def ingest_overrides(base_overrides):
    """Additional overrides needed for /ingest specifically."""
    app.dependency_overrides[get_ingest_module] = lambda: types.SimpleNamespace(
        load_markdown=lambda path: [{"text": "chunk one", "source": "x"}],
        load_pdf=lambda path: [{"text": "chunk one", "source": "x"}],
    )
    app.dependency_overrides[get_embeddings_module] = lambda: types.SimpleNamespace(
        embed_texts=lambda texts: ([[0.0] * 384 for _ in texts], list(range(len(texts)))),
    )
    app.dependency_overrides[get_bm25_module] = lambda: types.SimpleNamespace(
        build_bm25_index=lambda chunks: None,
    )
    yield


@pytest.fixture
async def async_client(base_overrides):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
