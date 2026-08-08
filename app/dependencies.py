"""
dependencies.py — Shared FastAPI dependencies.

Centralizes what routes need: settings, the RAG pipeline, and the
ingest/embeddings/vectorstore/bm25 modules. Everything is injected via
Depends() instead of imported ad hoc inside each endpoint body (as the
original api.py did). Two benefits:

1. Idiomatic FastAPI — dependencies are declared, not hidden inline.
2. Testable — every one of these can be swapped with
   app.dependency_overrides[...] in tests, without touching real
   Pinecone/Groq credentials. See tests/conftest.py.

NOTE: these wrap the same module-level functions your original
api.py imported inline (app.pipeline.run_rag_pipeline,
app.vectorstore, app.ingest, app.embeddings, app.bm25_retriever).
If any of those modules' function signatures differ from what's
assumed here, adjust accordingly — I only had api.py to go on, not
the module internals.
"""

from functools import lru_cache
from typing import Callable

from fastapi import Depends, Header, HTTPException, status

from app.config import Settings, get_settings


# ── Singletons, created once per process and reused ─────────────────────────

@lru_cache
def get_pipeline_fn() -> Callable[[str], dict]:
    from app.pipeline import run_rag_pipeline
    return run_rag_pipeline


@lru_cache
def get_vectorstore_module():
    from app import vectorstore
    return vectorstore


@lru_cache
def get_ingest_module():
    from app import ingest
    return ingest


@lru_cache
def get_embeddings_module():
    from app import embeddings
    return embeddings


@lru_cache
def get_bm25_module():
    from app import bm25_retriever
    return bm25_retriever


# ── Auth ─────────────────────────────────────────────────────────────────

async def verify_api_key(
    x_api_key: str = Header(..., alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    Shared-secret API key check for write endpoints (/ingest,
    DELETE /documents/{source}).

    Deliberately not OAuth2/JWT: this is a single-tenant demo API, not
    a multi-user system, so a header-checked secret demonstrates the
    same "this endpoint is protected" competency without unused
    complexity. Revisit if you build out real multi-user auth later.
    """
    if x_api_key != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
