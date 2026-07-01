"""
vectorstore.py — Pinecone-backed vector store (replaces local ChromaDB).
Persistent, cloud-hosted vector storage so the database survives
server restarts on platforms like Render.
"""

import hashlib
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

INDEX_NAME      = "papermind"
EMBEDDING_DIM   = 384   # all-MiniLM-L6-v2 output dimension
PINECONE_CLOUD  = "aws"
PINECONE_REGION = "us-east-1"

_pc    = None
_index = None


def _get_client() -> Pinecone:
    global _pc
    if _pc is None:
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in .env file.")
        _pc = Pinecone(api_key=api_key)
    return _pc


def _get_index():
    """Get or create the Pinecone index, return a connected Index client."""
    global _index
    if _index is not None:
        return _index

    pc = _get_client()
    existing = [idx["name"] for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"  [Pinecone] Creating index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )

    _index = pc.Index(INDEX_NAME)
    return _index


def _make_id(source: str, page: int, chunk_index: int) -> str:
    """Generate a stable unique ID for a chunk."""
    raw = f"{source}__page{page}__chunk{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _clean_text(text: str) -> str:
    """Remove characters Pinecone metadata can't handle."""
    if not text:
        return " "
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def add_documents(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Store chunks and their embeddings in Pinecone.
    Returns the number of vectors successfully upserted.
    """
    index = _get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vec_id = _make_id(chunk["source"], chunk["page"], chunk["chunk_index"])
        # Pinecone metadata values must be str/int/float/bool/list-of-str
        # Store the chunk text in metadata so we can retrieve it later
        text = _clean_text(chunk["text"])[:35000]  # stay under 40KB metadata limit
        vectors.append({
            "id": vec_id,
            "values": embedding,
            "metadata": {
                "text":        text,
                "source":      chunk["source"],
                "page":        chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }
        })

    # Upsert in batches of 100 (Pinecone recommended batch size)
    batch_size = 100
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)
        total_upserted += len(batch)

    return total_upserted


def query_documents(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Find the top_k most relevant chunks for a query embedding.
    Returns list of dicts with keys: text, source, page, chunk_index, distance.
    """
    index = _get_index()

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    output = []
    for match in results["matches"]:
        meta = match["metadata"]
        output.append({
            "text":        meta.get("text", ""),
            "source":      meta.get("source", "unknown"),
            "page":        meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            # Pinecone returns similarity score (higher = better);
            # convert to distance-like value for consistency with old code
            "distance":    round(1 - match["score"], 4),
        })

    return output


def collection_count() -> int:
    """Return total number of vectors currently stored."""
    index = _get_index()
    stats = index.describe_index_stats()
    return stats.get("total_vector_count", 0)


def _get_collection():
    """
    Compatibility shim for code that expects a ChromaDB-style collection.
    Used by api.py for listing/deleting documents.
    """
    return _get_index()
