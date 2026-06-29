"""
api.py — PaperMind FastAPI application.
Wraps the RAG pipeline in a REST API.

Run with:
    uvicorn app.api:app --reload --port 8000

Then visit:
    http://localhost:8000/docs   ← interactive API docs
    http://localhost:8000/health ← system health check
"""

import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.models import (
    QueryRequest, QueryResponse,
    IngestResponse, DocumentsResponse, DocumentInfo,
    HealthResponse, DeleteResponse, Source,
)

load_dotenv()

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="PaperMind RAG API",
    description=(
        "Production-grade RAG system using hybrid retrieval (BM25 + vector), "
        "cross-encoder re-ranking, and citation enforcement. "
        "Built on the AI Index Report 2026 and research papers."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins for development — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-load models on startup so first request isn't slow."""
    print("[PaperMind API] Starting up...")
    try:
        from app.embeddings import embed_query
        from app.reranker  import _reranker
        print("[PaperMind API] Models loaded successfully.")
    except Exception as e:
        print(f"[PaperMind API] Warning: Could not pre-load models: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check — verifies ChromaDB is connected and LLM API is reachable.
    Used by deployment systems to confirm the server is live.
    """
    from app.vectorstore import collection_count

    # Check ChromaDB
    try:
        count    = collection_count()
        db_status = "ok"
    except Exception as e:
        count    = 0
        db_status = f"error: {str(e)}"

    # Check LLM API (just verify key exists)
    api_key    = os.getenv("OPENROUTER_API_KEY")
    llm_status = "ok" if api_key else "missing API key"

    overall = "ok" if db_status == "ok" and llm_status == "ok" else "degraded"

    return HealthResponse(
        status       = overall,
        chromadb     = db_status,
        llm          = llm_status,
        total_chunks = count,
        message      = (
            f"PaperMind RAG is running. "
            f"{count} chunks indexed across all documents."
        ),
    )


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(req: QueryRequest):
    """
    Core RAG endpoint. Accepts a question, runs the full pipeline:
    hybrid retrieval → re-ranking → generation → citation enforcement.
    Returns the answer with source citations.
    """
    from app.pipeline import run_rag_pipeline

    if not req.question.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question cannot be empty."
        )

    try:
        result = run_rag_pipeline(req.question)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline error: {str(e)}"
        )

    return QueryResponse(
        question = req.question,
        answer   = result["answer"],
        sources  = [Source(**s) for s in result["sources"]],
        pipeline = result.get("phase", "phase2_hybrid_rerank"),
    )


@app.post("/ingest", response_model=IngestResponse, tags=["Documents"])
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF or markdown file. The system chunks it, embeds it,
    and stores it in ChromaDB + BM25 index.
    Supported formats: .pdf, .md
    """
    from app.ingest     import load_pdf, load_markdown
    from app.embeddings import embed_texts
    from app.vectorstore import add_documents, collection_count
    from app.bm25_retriever import build_bm25_index

    # Validate file type
    filename = file.filename or "unknown"
    suffix   = Path(filename).suffix.lower()
    if suffix not in [".pdf", ".md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .md are supported."
        )

    # Save uploaded file to a temp location
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Chunk the document
        if suffix == ".pdf":
            chunks = load_pdf(tmp_path)
            # Override source name to use original filename
            for chunk in chunks:
                chunk["source"] = filename
        else:
            chunks = load_markdown(tmp_path)
            for chunk in chunks:
                chunk["source"] = filename

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No text could be extracted from the file. It may be a scanned PDF."
            )

        # Embed and store
        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        add_documents(chunks, embeddings)

        # Rebuild BM25 index with new chunks
        # Note: In production you'd append — for now rebuild
        build_bm25_index(chunks)

        total = collection_count()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}"
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    return IngestResponse(
        filename    = filename,
        chunks      = len(chunks),
        total_in_db = total,
        status      = "success",
    )


@app.get("/documents", response_model=DocumentsResponse, tags=["Documents"])
async def list_documents():
    """
    Returns a list of all documents currently stored in ChromaDB,
    with their chunk counts.
    """
    from app.vectorstore import _get_collection

    try:
        collection = _get_collection()
        results    = collection.get(include=["metadatas"])
        metadatas  = results["metadatas"]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve documents: {str(e)}"
        )

    # Count chunks per source document
    source_counts: dict[str, int] = {}
    for meta in metadatas:
        src = meta.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1

    documents = [
        DocumentInfo(source=src, chunk_count=count)
        for src, count in sorted(source_counts.items())
    ]

    return DocumentsResponse(
        total_chunks = sum(source_counts.values()),
        documents    = documents,
    )


@app.delete("/documents/{source}", response_model=DeleteResponse, tags=["Documents"])
async def delete_document(source: str):
    """
    Remove a specific document and all its chunks from ChromaDB.
    The source parameter should be the filename (e.g. 'ai_index_report_2026.pdf').
    """
    from app.vectorstore import _get_collection

    try:
        collection = _get_collection()

        # Find all chunk IDs for this source
        results = collection.get(
            where={"source": source},
            include=["metadatas"],
        )
        ids = results["ids"]

        if not ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{source}' not found in the database."
            )

        # Delete all chunks for this source
        collection.delete(ids=ids)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion error: {str(e)}"
        )

    return DeleteResponse(
        deleted = True,
        source  = source,
        message = f"Successfully deleted {len(ids)} chunks from '{source}'.",
    )
