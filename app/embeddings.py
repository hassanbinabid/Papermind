"""
embeddings.py — Local embeddings using sentence-transformers.
No API key needed. Runs fully on your machine.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[Embeddings] Loading local model: {MODEL_NAME}...")
_model = SentenceTransformer(MODEL_NAME)
print(f"[Embeddings] Model ready.")


def embed_texts(texts: list[str]) -> tuple[list[list[float]], list[int]]:
    """
    Embed a list of texts one by one to safely handle any bad chunks.
    Returns (embeddings, valid_indices) — valid_indices tells you which
    original texts were successfully embedded (skips bad ones entirely,
    no zero-vector placeholders since vector DBs reject all-zero vectors).
    """
    print(f"  Embedding {len(texts)} chunks locally...")
    embeddings = []
    valid_indices = []
    skipped = 0

    for i, text in enumerate(texts):
        try:
            clean = str(text).strip() if text else ""
            if not clean:
                skipped += 1
                continue
            vec = _model.encode(clean).tolist()
            embeddings.append(vec)
            valid_indices.append(i)
        except Exception as e:
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(texts)} chunks...")

    if skipped:
        print(f"  ⚠ Skipped {skipped} bad/empty chunks.")
    print(f"  ✅ Done embedding. {len(embeddings)} valid embeddings.")
    return embeddings, valid_indices


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    return _model.encode(str(query).strip()).tolist()
