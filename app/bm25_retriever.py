"""
bm25_retriever.py — Keyword-based retrieval using BM25.
Complements vector search by finding exact terms, codes, and names.
"""

import json
import os
from pathlib import Path
from rank_bm25 import BM25Okapi

BM25_INDEX_PATH = ".chroma/bm25_index.json"


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def _clean(text: str) -> str:
    """Remove surrogate characters that break JSON serialization."""
    if not text:
        return " "
    return text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def build_bm25_index(chunks: list[dict]) -> None:
    """
    Build and persist a BM25 index from a list of chunks.
    Called once during ingestion, saved to disk.
    """
    cleaned_chunks = [_clean(chunk["text"]) for chunk in chunks]
    corpus = [_tokenize(text) for text in cleaned_chunks]

    index_data = {
        "corpus": corpus,
        "metadata": [
            {
                "text":        _clean(chunk["text"]),
                "source":      chunk["source"],
                "page":        chunk["page"],
                "chunk_index": chunk["chunk_index"],
            }
            for chunk in chunks
        ]
    }
    os.makedirs(os.path.dirname(BM25_INDEX_PATH), exist_ok=True)
    with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=True)
    print(f"  ✅ BM25 index saved: {len(chunks)} documents")


def load_bm25_index() -> tuple[BM25Okapi, list[dict]]:
    """
    Load the persisted BM25 index from disk.
    Returns (bm25_model, list_of_chunk_metadata).
    """
    if not Path(BM25_INDEX_PATH).exists():
        raise FileNotFoundError(
            "BM25 index not found. Run ingestion first (python main.py)."
        )
    with open(BM25_INDEX_PATH, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    corpus   = index_data["corpus"]
    metadata = index_data["metadata"]
    bm25     = BM25Okapi(corpus)
    return bm25, metadata


def bm25_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Search using BM25 keyword matching.
    Returns top_k results with text, source, page, and bm25_score.
    """
    bm25, metadata = load_bm25_index()
    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "text":        metadata[idx]["text"],
                "source":      metadata[idx]["source"],
                "page":        metadata[idx]["page"],
                "chunk_index": metadata[idx]["chunk_index"],
                "bm25_score":  float(scores[idx]),
            })

    return results