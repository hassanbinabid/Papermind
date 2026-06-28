"""
hybrid_retriever.py — Combines BM25 keyword search + vector semantic search
using Reciprocal Rank Fusion (RRF) to produce a single ranked list.

Why RRF?
- Simple, parameter-free fusion formula
- Works well even when the two rankers have very different score scales
- Consistently outperforms score-based fusion in benchmarks
"""

from app.embeddings import embed_query
from app.vectorstore import query_documents
from app.bm25_retriever import bm25_search

# RRF constant — 60 is the standard value from the original paper
RRF_K = 60


def _reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results:   list[dict],
    top_k:          int = 10,
) -> list[dict]:
    """
    Combine two ranked lists using Reciprocal Rank Fusion.

    RRF score for a document = sum of 1 / (k + rank) across all lists.
    Documents appearing high in both lists get the highest combined scores.
    """
    # Build a unique key for deduplication: (source, page, chunk_index)
    def doc_key(chunk):
        return (chunk["source"], chunk["page"], chunk["chunk_index"])

    scores = {}   # key → rrf_score
    docs   = {}   # key → chunk dict (to return full chunk later)

    # Score from vector results
    for rank, chunk in enumerate(vector_results, start=1):
        key = doc_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        docs[key]   = chunk

    # Score from BM25 results
    for rank, chunk in enumerate(bm25_results, start=1):
        key = doc_key(chunk)
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        if key not in docs:
            docs[key] = chunk

    # Sort by combined RRF score descending
    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        chunk = docs[key].copy()
        chunk["rrf_score"] = round(scores[key], 6)
        results.append(chunk)

    return results


def hybrid_retrieve(query: str, top_k: int = 10) -> list[dict]:
    """
    Full hybrid retrieval pipeline:
    1. Vector search  — semantic similarity (top 10)
    2. BM25 search    — keyword matching   (top 10)
    3. RRF fusion     — combine both lists into one ranked list
    Returns top_k fused results, ready for re-ranking.
    """
    print(f"  [Vector Search] Searching for top 10 semantic matches...")
    query_embedding = embed_query(query)
    vector_results  = query_documents(query_embedding, top_k=10)

    print(f"  [BM25 Search]   Searching for top 10 keyword matches...")
    bm25_results = bm25_search(query, top_k=10)

    print(f"  [RRF Fusion]    Combining {len(vector_results)} vector + "
          f"{len(bm25_results)} BM25 results...")
    fused = _reciprocal_rank_fusion(vector_results, bm25_results, top_k=top_k)

    print(f"  [RRF]           {len(fused)} unique chunks after fusion.")
    return fused
