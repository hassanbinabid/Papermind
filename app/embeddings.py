"""
embeddings.py — Local embeddings using sentence-transformers.
No API key needed. Runs fully on your machine.
"""

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

print(f"[Embeddings] Loading local model: {MODEL_NAME}...")
_model = SentenceTransformer(MODEL_NAME)
print(f"[Embeddings] Model ready.")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed one by one to safely handle any bad chunks."""
    print(f"  Embedding {len(texts)} chunks locally...")
    embeddings = []
    skipped = 0

    for i, text in enumerate(texts):
        try:
            clean = str(text).strip() if text else " "
            if not clean:
                clean = " "
            vec = _model.encode(clean).tolist()
            embeddings.append(vec)
        except Exception as e:
            embeddings.append([0.0] * 384)
            skipped += 1

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{len(texts)} chunks...")

    if skipped:
        print(f"  ⚠ Skipped {skipped} bad chunks.")
    print(f"  ✅ Done embedding.")
    return embeddings


def embed_query(query: str) -> list[float]:
    """Embed a single query string for retrieval."""
    return _model.encode(str(query).strip()).tolist()