"""
models.py — Pydantic request and response models for PaperMind API.
These define the shape of every request and response — FastAPI validates
them automatically and shows them in the /docs UI.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to ask about the documents",
        example="What are the key AI trends in 2026?"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to retrieve and re-rank",
        example=5
    )


class IngestRequest(BaseModel):
    """Optional metadata for POST /ingest"""
    description: Optional[str] = Field(
        default=None,
        description="Optional description of the document being ingested",
        example="AI Index Report 2026 by Stanford HAI"
    )


# ── Response Models ───────────────────────────────────────────────────────────

class Source(BaseModel):
    """A single source citation"""
    source: str = Field(description="Source filename", example="ai_index_report_2026.pdf")
    page:   int = Field(description="Page number",     example=42)


class QueryResponse(BaseModel):
    """Response body for POST /query"""
    question: str   = Field(description="The original question asked")
    answer:   str   = Field(description="The generated answer with citations")
    sources:  list[Source] = Field(description="List of source documents used")
    pipeline: str   = Field(description="Pipeline phase used", example="phase2_hybrid_rerank")


class IngestResponse(BaseModel):
    """Response body for POST /ingest"""
    filename:    str = Field(description="Name of the ingested file")
    chunks:      int = Field(description="Number of chunks created")
    total_in_db: int = Field(description="Total chunks now in ChromaDB")
    status:      str = Field(description="Ingestion status", example="success")


class DocumentInfo(BaseModel):
    """Info about a single ingested document"""
    source:      str = Field(description="Source filename")
    chunk_count: int = Field(description="Number of chunks from this document")


class DocumentsResponse(BaseModel):
    """Response body for GET /documents"""
    total_chunks: int                = Field(description="Total chunks in ChromaDB")
    documents:    list[DocumentInfo] = Field(description="List of ingested documents")


class HealthResponse(BaseModel):
    """Response body for GET /health"""
    status:       str  = Field(description="Overall system status", example="ok")
    chromadb:     str  = Field(description="ChromaDB connection status", example="ok")
    llm:          str  = Field(description="LLM API status",            example="ok")
    total_chunks: int  = Field(description="Total chunks in ChromaDB")
    message:      str  = Field(description="Human readable status message")


class DeleteResponse(BaseModel):
    """Response body for DELETE /documents/{source}"""
    deleted: bool = Field(description="Whether deletion was successful")
    source:  str  = Field(description="Name of the deleted document")
    message: str  = Field(description="Status message")
