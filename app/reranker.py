"""
reranker.py — Cross-encoder re-ranker using sentence-transformers.
Takes initial retrieved chunks and rescores them by reading
query + chunk together as a pair — much more precise than vector similarity alone.
"""

from sentence_transformers import CrossEncoder

# Best open-source cross-encoder for retrieval re-ranking
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"[Reranker] Loading cross-encoder: {MODEL_NAME}...")
_reranker = CrossEncoder(MODEL_NAME)
print(f"[Reranker] Model ready.")


def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score a list of chunks against the query using a cross-encoder.

    The cross-encoder reads (query, chunk_text) as a pair and outputs
    a relevance score — far more accurate than cosine similarity alone.

    Returns top_k chunks sorted by reranker score (highest first).
    """
    if not chunks:
        return []

    # Build (query, text) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    scores = _reranker.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by reranker score descending and return top_k
    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_k]
