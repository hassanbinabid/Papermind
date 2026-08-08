"""
routers/documents.py — Document ingestion, listing, and deletion.

Write endpoints (/ingest, DELETE /documents/{source}) require the
X-API-Key header via verify_api_key. GET /documents is read-only and
left open.
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status

from app.dependencies import (
    get_bm25_module,
    get_embeddings_module,
    get_ingest_module,
    get_vectorstore_module,
    verify_api_key,
)
from app.models import DeleteResponse, DocumentInfo, DocumentsResponse, IngestResponse
from app.rate_limit import limiter

router = APIRouter(tags=["Documents"])


async def temp_upload_file(file: UploadFile = File(...)) -> Tuple[str, str, str]:
    """
    Yield-dependency: writes the upload to a temp path for the life of
    the request, then guarantees cleanup on the way out — success or
    exception — via the generator's teardown. Replaces the manual
    try/finally + os.unlink() from the original ingest() endpoint.

    Yields: (tmp_path, original_filename, suffix)
    """
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    if suffix not in [".pdf", ".md"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{suffix}'. Only .pdf and .md are supported.",
        )

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
        yield tmp.name, filename, suffix
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@router.post(
    "/ingest",
    response_model=IngestResponse,
    dependencies=[Depends(verify_api_key)],
)
@limiter.limit("5/minute")
async def ingest(
    request: Request,
    upload: Tuple[str, str, str] = Depends(temp_upload_file),
    ingest_module=Depends(get_ingest_module),
    embeddings_module=Depends(get_embeddings_module),
    vectorstore_module=Depends(get_vectorstore_module),
    bm25_module=Depends(get_bm25_module),
):
    """
    Upload a PDF or markdown file (requires X-API-Key). Chunks it,
    embeds it, and stores it in Pinecone + rebuilds the BM25 index.
    """
    tmp_path, filename, suffix = upload

    if suffix == ".pdf":
        chunks = ingest_module.load_pdf(tmp_path)
    else:
        chunks = ingest_module.load_markdown(tmp_path)
    for chunk in chunks:
        chunk["source"] = filename

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from the file. It may be a scanned PDF.",
        )

    try:
        texts = [c["text"] for c in chunks]
        embeddings, valid_indices = embeddings_module.embed_texts(texts)
        chunks = [chunks[i] for i in valid_indices]
        vectorstore_module.add_documents(chunks, embeddings)

        # Note: rebuilds the full BM25 index rather than appending —
        # same tradeoff as the original implementation.
        bm25_module.build_bm25_index(chunks)

        total = vectorstore_module.collection_count()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion error: {str(e)}",
        )

    return IngestResponse(
        filename=filename,
        chunks=len(chunks),
        total_in_db=total,
        status="success",
    )


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(vectorstore_module=Depends(get_vectorstore_module)):
    """
    Lists all documents currently stored in Pinecone with chunk counts.
    Pinecone's free tier has no native "list all", so this samples up
    to 10000 vectors via a neutral-vector query — a known limitation,
    not a bug.
    """
    try:
        index = vectorstore_module._get_index()
        stats = index.describe_index_stats()
        total = stats.get("total_vector_count", 0)

        if total == 0:
            return DocumentsResponse(total_chunks=0, documents=[])

        dummy_vector = [0.0] * vectorstore_module.EMBEDDING_DIM
        results = index.query(
            vector=dummy_vector,
            top_k=min(total, 10000),
            include_metadata=True,
        )

        source_counts: dict[str, int] = {}
        for match in results["matches"]:
            src = match["metadata"].get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not retrieve documents: {str(e)}",
        )

    documents = [
        DocumentInfo(source=src, chunk_count=count)
        for src, count in sorted(source_counts.items())
    ]
    return DocumentsResponse(total_chunks=total, documents=documents)


@router.delete(
    "/documents/{source}",
    response_model=DeleteResponse,
    dependencies=[Depends(verify_api_key)],
)
async def delete_document(
    source: str,
    vectorstore_module=Depends(get_vectorstore_module),
):
    """
    Remove a document and all its chunks from Pinecone (requires X-API-Key).
    """
    try:
        index = vectorstore_module._get_index()
        index.delete(filter={"source": {"$eq": source}})
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Deletion error: {str(e)}",
        )

    return DeleteResponse(
        deleted=True,
        source=source,
        message=f"Deletion request sent for '{source}'. Pinecone deletes are eventually consistent.",
    )
