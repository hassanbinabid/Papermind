"""
vectorstore.py — Persistent ChromaDB vector store.
"""

import hashlib
import chromadb

COLLECTION_NAME = "papermind"
CHROMA_PATH = ".chroma"


def _get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def _make_id(source: str, page: int, chunk_index: int) -> str:
    raw = f"{source}__page{page}__chunk{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def _clean_text(text: str) -> str:
    """Remove characters ChromaDB can't handle."""
    if not text:
        return " "
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def add_documents(chunks: list[dict], embeddings: list[list[float]]) -> int:
    collection = _get_collection()

    ids       = []
    texts     = []
    metadatas = []
    vecs      = []

    for chunk, embedding in zip(chunks, embeddings):
        doc_id = _make_id(chunk["source"], chunk["page"], chunk["chunk_index"])
        ids.append(doc_id)
        texts.append(_clean_text(chunk["text"]))
        metadatas.append({
            "source":      chunk["source"],
            "page":        chunk["page"],
            "chunk_index": chunk["chunk_index"],
        })
        vecs.append(embedding)

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=vecs,
        metadatas=metadatas,
    )
    return len(ids)


def query_documents(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":        text,
            "source":      meta["source"],
            "page":        meta["page"],
            "chunk_index": meta["chunk_index"],
            "distance":    round(dist, 4),
        })
    return output


def collection_count() -> int:
    return _get_collection().count()