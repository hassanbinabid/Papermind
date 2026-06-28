"""
retriever.py — Hybrid retrieval: BM25 (keyword) + Vector (semantic).
Phase 2 upgrade from pure vector search.
"""

from rank_bm25 import BM25Okapi
from app.embeddings import embed_query
from app.vectorstore import query_documents, get_all_documents


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def _bm25_search(query: str, all_chunks: list[dict], top_k: int) -> list[dict]:
    """
    Run BM25 keyword search over all stored chunks.
    Returns top_k results with a bm25_score field.
    """
    corpus = [_tokenize(c["text"]) for c in all_chunks]
    bm25   = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    # Attach scores and sort
    scored = [
        {**chunk, "bm25_score": float(scores[i])}
        for i, chunk in enumerate(all_chunks)
    ]
    scored.sort(key=lambda x: x["bm25_score"], reverse=True)
    return scored[:top_k]


def _vector_search(query: str, top_k: int) -> list[dict]:
    """Run semantic vector search via ChromaDB."""
    query_embedding = embed_query(query)
    results = query_documents(query_embedding, top_k=top_k)
    return results


def _reciprocal_rank_fusion(
    bm25_results: list[dict],
    vector_results: list[dict],
    k: int = 60
) -> list[dict]:
    """
    Combine BM25 and vector results using Reciprocal Rank Fusion (RRF).
    RRF score = 1/(k + rank). Higher is better.
    Deduplicates by chunk text so the same chunk is not returned twice.
    """
    scores = {}   # text -> rrf_score
    chunks = {}   # text -> chunk dict

    for rank, chunk in enumerate(bm25_results):
        key = chunk["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks[key] = chunk

    for rank, chunk in enumerate(vector_results):
        key = chunk["text"]
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        chunks[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [chunks[k] for k in sorted_keys]


def retrieve(query_text: str, top_k: int = 5) -> list[dict]:
    """
    Hybrid retrieval: BM25 + vector search fused with RRF.
    Returns top_k deduplicated, re-ranked results.
    """
    print(f"  [Hybrid] Running BM25 search...")
    all_chunks = get_all_documents()

    if not all_chunks:
        print("  ⚠ No documents in vector store. Run ingestion first.")
        return []

    bm25_results   = _bm25_search(query_text, all_chunks, top_k=top_k * 2)

    print(f"  [Hybrid] Running vector search...")
    vector_results = _vector_search(query_text, top_k=top_k * 2)

    print(f"  [Hybrid] Fusing results with RRF...")
    fused = _reciprocal_rank_fusion(bm25_results, vector_results)

    return fused[:top_k * 2]   # return more for re-ranker to work with
