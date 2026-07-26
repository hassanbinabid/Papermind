"""
reranker.py — Cross-encoder re-ranker using sentence-transformers.
Takes initial retrieved chunks and rescores them by reading
query + chunk together as a pair — much more precise than vector similarity alone.
Instrumented so the before/after chunk order is visible in Langfuse.
"""

from sentence_transformers import CrossEncoder
from langfuse import observe
from app.observability import langfuse

# Best open-source cross-encoder for retrieval re-ranking
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print(f"[Reranker] Loading cross-encoder: {MODEL_NAME}...")
_reranker = CrossEncoder(MODEL_NAME)
print(f"[Reranker] Model ready.")


def _order_summary(chunks: list[dict]) -> list[dict]:
    """Compact view: rank position + source/page, for before/after comparison."""
    return [
        {"rank": i + 1, "source": c["source"], "page": c["page"]}
        for i, c in enumerate(chunks)
    ]


@observe(name="cross_encoder_rerank")
def rerank(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Re-score a list of chunks against the query using a cross-encoder.

    The cross-encoder reads (query, chunk_text) as a pair and outputs
    a relevance score — far more accurate than cosine similarity alone.

    Returns top_k chunks sorted by reranker score (highest first).
    """
    if not chunks:
        return []

    langfuse.update_current_span(
        input={"query": query, "order_before_rerank": _order_summary(chunks)}
    )

    # Build (query, text) pairs for the cross-encoder
    pairs = [(query, chunk["text"]) for chunk in chunks]

    # Score all pairs
    scores = _reranker.predict(pairs)

    # Attach scores to chunks
    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    # Sort by reranker score descending and return top_k
    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    result = reranked[:top_k]

    langfuse.update_current_span(
        output={"order_after_rerank": _order_summary(result)}
    )

    return result